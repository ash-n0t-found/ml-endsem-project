"""
preprocessing/ppg_prep.py
=========================
PPG signal preprocessing pipeline.

Steps:
  1. Load raw PPG from BIDS .tsv / .fif sidecar
  2. Bandpass filter (0.5–8 Hz) for pulse wave preservation
  3. Peak detection via NeuroKit2
  4. Pulse wave amplitude (PWA) extraction per beat
  5. Inter-beat interval (IBI) series construction
  6. Trial-level segmentation
  7. Quality control: reject trials with < min_valid_beats

All functions are pure / side-effect-free unless explicitly noted.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

try:
    import neurokit2 as nk
    _NK_AVAILABLE = True
except ImportError:
    _NK_AVAILABLE = False
    warnings.warn(
        "neurokit2 not installed. PPG peak detection will use fallback scipy method.",
        ImportWarning,
    )

from utils.io_utils import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

PPG_BANDPASS_LOW = 0.5   # Hz
PPG_BANDPASS_HIGH = 8.0  # Hz
MIN_IBI_S = 0.4          # minimum physiological IBI (150 bpm max)
MAX_IBI_S = 1.5          # maximum physiological IBI (40 bpm min)
MIN_BEATS_PER_TRIAL = 3  # reject trial if fewer peaks found


# ── Filtering ─────────────────────────────────────────────────────────────────

def bandpass_filter_ppg(
    ppg_raw: np.ndarray,
    sfreq: float,
    low: float = PPG_BANDPASS_LOW,
    high: float = PPG_BANDPASS_HIGH,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter for PPG.

    Parameters
    ----------
    ppg_raw : ndarray, shape (n_samples,)
    sfreq : float
    low, high : float
        Passband edges in Hz.
    order : int
        Filter order (applied twice via filtfilt → effective 2*order).

    Returns
    -------
    ppg_filtered : ndarray, shape (n_samples,)
    """
    nyq = sfreq / 2.0
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    sos = sp_signal.butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, ppg_raw)


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_ppg_peaks(
    ppg_filtered: np.ndarray,
    sfreq: float,
    method: str = "neurokit",
) -> np.ndarray:
    """
    Detect systolic peaks in filtered PPG signal.

    Parameters
    ----------
    ppg_filtered : ndarray, shape (n_samples,)
    sfreq : float
    method : str
        "neurokit" (preferred) or "scipy" (fallback).

    Returns
    -------
    peak_indices : ndarray of int
        Sample indices of detected systolic peaks.
    """
    if method == "neurokit" and _NK_AVAILABLE:
        try:
            signals, info = nk.ppg_process(ppg_filtered, sampling_rate=int(sfreq))
            peaks = info["PPG_Peaks"]
            return np.asarray(peaks, dtype=int)
        except Exception as e:
            logger.warning(f"NeuroKit2 PPG processing failed ({e}). Falling back to scipy.")

    # Scipy fallback: find local maxima with distance / prominence constraints
    min_distance = int(MIN_IBI_S * sfreq)
    peaks, properties = sp_signal.find_peaks(
        ppg_filtered,
        distance=min_distance,
        prominence=0.1 * (ppg_filtered.max() - ppg_filtered.min()),
    )
    return peaks


def clean_ibi_series(
    peak_indices: np.ndarray,
    sfreq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove physiologically implausible inter-beat intervals.

    Parameters
    ----------
    peak_indices : ndarray of int
    sfreq : float

    Returns
    -------
    clean_peaks : ndarray of int
        Peak indices after removing outliers.
    ibi_s : ndarray of float
        IBI values in seconds corresponding to clean_peaks[1:].
    """
    if len(peak_indices) < 2:
        return peak_indices, np.array([])

    ibi_s = np.diff(peak_indices) / sfreq
    valid_mask = (ibi_s >= MIN_IBI_S) & (ibi_s <= MAX_IBI_S)

    # Keep only peaks bounding valid IBIs
    # A peak is valid if IBI before or after it is valid
    n = len(peak_indices)
    keep = np.ones(n, dtype=bool)
    for i in range(len(valid_mask)):
        if not valid_mask[i]:
            # Mark the less-prominent peak of the pair for removal
            keep[i + 1] = False

    clean_peaks = peak_indices[keep]
    clean_ibi = np.diff(clean_peaks) / sfreq
    return clean_peaks, clean_ibi


# ── Pulse wave amplitude ──────────────────────────────────────────────────────

def compute_pulse_wave_amplitude(
    ppg_filtered: np.ndarray,
    peak_indices: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute trough-to-peak amplitude for each pulse.

    For each systolic peak, the preceding trough is found in the window
    [peak - 0.3s, peak]. PWA = peak_value - trough_value.

    Parameters
    ----------
    ppg_filtered : ndarray, shape (n_samples,)
    peak_indices : ndarray of int
    sfreq : float

    Returns
    -------
    pwa : ndarray, shape (n_peaks,)
        Pulse wave amplitude per beat (a.u.).
    """
    search_window = int(0.3 * sfreq)
    pwa = np.zeros(len(peak_indices))

    for i, pk in enumerate(peak_indices):
        start = max(0, pk - search_window)
        segment = ppg_filtered[start:pk]
        if len(segment) == 0:
            pwa[i] = 0.0
        else:
            trough_val = segment.min()
            pwa[i] = ppg_filtered[pk] - trough_val

    return pwa


# ── Trial-level segmentation ──────────────────────────────────────────────────

def segment_ppg_trials(
    ppg_filtered: np.ndarray,
    peak_indices: np.ndarray,
    pwa_per_beat: np.ndarray,
    ibi_per_beat: np.ndarray,
    trial_onsets_samples: np.ndarray,
    trial_duration_samples: int,
    sfreq: float,
    min_beats: int = MIN_BEATS_PER_TRIAL,
) -> List[Dict]:
    """
    Segment PPG beats into trials and compute per-trial summaries.

    Parameters
    ----------
    ppg_filtered : ndarray, shape (n_samples,)
    peak_indices : ndarray of int
        All detected systolic peaks (after cleaning).
    pwa_per_beat : ndarray
        PWA value for each peak.
    ibi_per_beat : ndarray
        IBI values — length = len(peak_indices) - 1.
        ibi_per_beat[i] is IBI between peak_indices[i] and peak_indices[i+1].
    trial_onsets_samples : ndarray of int
        Sample index of each trial onset.
    trial_duration_samples : int
        Duration of each trial in samples.
    sfreq : float
    min_beats : int
        Minimum beats required; trials below this are flagged as invalid.

    Returns
    -------
    trials : list of dict
        Each dict contains:
          - 'trial_idx': int
          - 'valid': bool
          - 'n_beats': int
          - 'pwa_values': ndarray  (per-beat PWA within trial)
          - 'ibi_values': ndarray  (per-beat IBI within trial)
          - 'mean_pwa': float
          - 'mean_ibi': float
          - 'std_pwa': float
          - 'std_ibi': float
          - 'pwa_slope': float  (linear slope across trial — temporal load accumulation)
          - 'ppg_segment': ndarray  (raw PPG segment)
    """
    trials = []
    n_peaks = len(peak_indices)

    for t_idx, onset in enumerate(trial_onsets_samples):
        end = onset + trial_duration_samples

        # Find peaks within this trial
        trial_peak_mask = (peak_indices >= onset) & (peak_indices < end)
        trial_peak_positions = np.where(trial_peak_mask)[0]

        ppg_segment = ppg_filtered[onset:end]
        n_beats = len(trial_peak_positions)
        valid = n_beats >= min_beats

        if not valid:
            logger.debug(f"Trial {t_idx}: only {n_beats} beats — flagged invalid.")
            trials.append({
                "trial_idx": t_idx,
                "valid": False,
                "n_beats": n_beats,
                "pwa_values": np.array([]),
                "ibi_values": np.array([]),
                "mean_pwa": np.nan,
                "mean_ibi": np.nan,
                "std_pwa": np.nan,
                "std_ibi": np.nan,
                "pwa_slope": np.nan,
                "ppg_segment": ppg_segment,
            })
            continue

        # Gather per-beat features
        trial_pwa = pwa_per_beat[trial_peak_positions]

        # IBI: between consecutive peaks within trial
        # ibi_per_beat[i] corresponds to peaks[i]→peaks[i+1]
        # Only include IBIs where both endpoints are in trial
        trial_ibi_positions = trial_peak_positions[trial_peak_positions < n_peaks - 1]
        # Filter: next peak also in trial
        trial_ibi_positions = [
            p for p in trial_ibi_positions
            if p + 1 < n_peaks and trial_peak_mask[p + 1]
        ]
        trial_ibi = ibi_per_beat[trial_ibi_positions] if len(trial_ibi_positions) > 0 else np.array([])

        # PWA temporal slope (linear fit across beat indices)
        if len(trial_pwa) >= 2:
            x = np.arange(len(trial_pwa))
            pwa_slope = float(np.polyfit(x, trial_pwa, 1)[0])
        else:
            pwa_slope = np.nan

        trials.append({
            "trial_idx": t_idx,
            "valid": True,
            "n_beats": n_beats,
            "pwa_values": trial_pwa,
            "ibi_values": trial_ibi,
            "mean_pwa": float(np.nanmean(trial_pwa)),
            "mean_ibi": float(np.nanmean(trial_ibi)) if len(trial_ibi) > 0 else np.nan,
            "std_pwa": float(np.nanstd(trial_pwa)),
            "std_ibi": float(np.nanstd(trial_ibi)) if len(trial_ibi) > 0 else np.nan,
            "pwa_slope": pwa_slope,
            "ppg_segment": ppg_segment,
        })

    return trials


# ── Waveform complexity ────────────────────────────────────────────────────────

def sample_entropy_ppg(
    ppg_segment: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """
    Sample entropy of PPG waveform morphology.

    Parameters
    ----------
    ppg_segment : ndarray
    m : int
        Embedding dimension.
    r_factor : float
        Tolerance = r_factor * std(ppg_segment).

    Returns
    -------
    sampen : float
        Sample entropy value (NaN if computation fails).
    """
    x = ppg_segment.copy()
    if len(x) < 2 * (m + 1):
        return np.nan

    r = r_factor * np.std(x, ddof=1)
    if r < 1e-10:
        return np.nan

    def _phi(m_val):
        count = 0
        N = len(x) - m_val
        for i in range(N):
            template = x[i: i + m_val]
            dists = np.max(
                np.abs(
                    np.lib.stride_tricks.sliding_window_view(x[:N + m_val - 1], m_val) - template
                ),
                axis=1,
            )
            count += np.sum(dists <= r) - 1  # exclude self-match
        return count

    try:
        B = _phi(m)
        A = _phi(m + 1)
        if B == 0:
            return np.nan
        return float(-np.log(A / B))
    except Exception:
        return np.nan


# ── Full subject pipeline ─────────────────────────────────────────────────────

def preprocess_ppg_subject(
    ppg_raw: np.ndarray,
    sfreq: float,
    trial_onsets_samples: np.ndarray,
    trial_duration_samples: int,
    peak_method: str = "neurokit",
    compute_entropy: bool = True,
) -> Dict:
    """
    Full PPG preprocessing pipeline for one subject.

    Parameters
    ----------
    ppg_raw : ndarray, shape (n_samples,)
    sfreq : float
    trial_onsets_samples : ndarray of int
    trial_duration_samples : int
    peak_method : str
    compute_entropy : bool
        If True, compute sample entropy per trial (slow).

    Returns
    -------
    result : dict
        - 'ppg_filtered': ndarray
        - 'peak_indices': ndarray
        - 'pwa_per_beat': ndarray
        - 'ibi_per_beat': ndarray
        - 'trials': list of trial dicts
        - 'n_valid_trials': int
        - 'rejection_rate': float
    """
    logger.info("PPG preprocessing: filtering...")
    ppg_filtered = bandpass_filter_ppg(ppg_raw, sfreq)

    logger.info("PPG preprocessing: peak detection...")
    raw_peaks = detect_ppg_peaks(ppg_filtered, sfreq, method=peak_method)
    logger.info(f"  Raw peaks detected: {len(raw_peaks)}")

    clean_peaks, ibi_series = clean_ibi_series(raw_peaks, sfreq)
    logger.info(f"  Clean peaks: {len(clean_peaks)} | IBIs: {len(ibi_series)}")

    pwa_series = compute_pulse_wave_amplitude(ppg_filtered, clean_peaks, sfreq)

    logger.info("PPG preprocessing: trial segmentation...")
    trials = segment_ppg_trials(
        ppg_filtered=ppg_filtered,
        peak_indices=clean_peaks,
        pwa_per_beat=pwa_series,
        ibi_per_beat=ibi_series,
        trial_onsets_samples=trial_onsets_samples,
        trial_duration_samples=trial_duration_samples,
        sfreq=sfreq,
    )

    if compute_entropy:
        logger.info("PPG preprocessing: computing waveform entropy...")
        for t in trials:
            if t["valid"] and len(t["ppg_segment"]) > 0:
                t["waveform_entropy"] = sample_entropy_ppg(t["ppg_segment"])
            else:
                t["waveform_entropy"] = np.nan

    n_valid = sum(t["valid"] for t in trials)
    n_total = len(trials)
    rejection_rate = 1.0 - n_valid / max(n_total, 1)
    logger.info(f"PPG preprocessing: {n_valid}/{n_total} valid trials (rejection={rejection_rate:.2%})")

    return {
        "ppg_filtered": ppg_filtered,
        "peak_indices": clean_peaks,
        "pwa_per_beat": pwa_series,
        "ibi_per_beat": ibi_series,
        "trials": trials,
        "n_valid_trials": n_valid,
        "rejection_rate": rejection_rate,
    }