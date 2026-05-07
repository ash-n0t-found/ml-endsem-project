"""
features/ppg_features.py
========================
Per-trial PPG feature extraction.

Features:
- Pulse Wave Amplitude (PWA): trough-to-peak per heartbeat
- Inter-beat interval (IBI) from PPG peaks
- PWA slope across digit sequence (temporal derivative — accumulating load)
- Waveform entropy (sample entropy of pulse morphology)
- PPG IBI RMSSD

PWA suppression with cognitive load is established in Papers 2 & 3.
The triphasic reversal at overload is the key phenomenon.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger
from features.ecg_features import filter_rr_intervals

logger = setup_logger(__name__)


# ── Peak/trough detection ─────────────────────────────────────────────────────

def detect_ppg_peaks_troughs(
    ppg_signal: np.ndarray,
    sfreq: float,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect systolic peaks and diastolic troughs in PPG signal.

    Parameters
    ----------
    ppg_signal : ndarray, shape (n_samples,)
    sfreq : float
    min_hr, max_hr : float
        Physiological bounds (bpm) for peak distance filtering.

    Returns
    -------
    peaks : ndarray
        Sample indices of systolic peaks.
    troughs : ndarray
        Sample indices of diastolic troughs (minima between peaks).
    """
    # Try NeuroKit2 first
    try:
        import neurokit2 as nk
        _, info = nk.ppg_peaks(ppg_signal, sampling_rate=int(sfreq))
        peaks = np.array(info["PPG_Peaks"], dtype=int)
    except Exception:
        # Fallback: simple peak detection
        min_dist = int(60.0 / max_hr * sfreq)
        max_dist = int(60.0 / min_hr * sfreq)
        height_thresh = np.percentile(ppg_signal, 30)
        peaks, _ = find_peaks(
            ppg_signal,
            distance=min_dist,
            height=height_thresh,
        )

    if len(peaks) < 2:
        return peaks, np.array([])

    # Find troughs as minima between consecutive peaks
    troughs = np.array([
        peaks[i] + np.argmin(ppg_signal[peaks[i]:peaks[i + 1]])
        for i in range(len(peaks) - 1)
    ])

    return peaks, troughs


# ── Pulse Wave Amplitude ──────────────────────────────────────────────────────

def compute_pwa(
    ppg_signal: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
) -> np.ndarray:
    """
    Compute pulse wave amplitude (trough-to-peak) per heartbeat.

    Parameters
    ----------
    ppg_signal : ndarray
    peaks : ndarray, shape (n_peaks,)
    troughs : ndarray, shape (n_troughs,)

    Returns
    -------
    pwa : ndarray
        Per-beat PWA values.
    """
    n_beats = min(len(peaks), len(troughs))
    if n_beats == 0:
        return np.array([])

    pwa = np.array([
        ppg_signal[peaks[i]] - ppg_signal[troughs[i]]
        for i in range(n_beats)
    ])
    return pwa


def compute_pwa_slope(pwa: np.ndarray) -> float:
    """
    Linear slope of PWA across beats in a trial.

    Captures temporal accumulation of load — suppression slope
    is stronger at 9-digit vs 5-digit (Papers 2, 3).

    Parameters
    ----------
    pwa : ndarray, shape (n_beats,)

    Returns
    -------
    slope : float (units: PWA per beat)
    """
    if len(pwa) < 3:
        return np.nan
    x = np.arange(len(pwa))
    # Least-squares slope
    slope = np.polyfit(x, pwa, 1)[0]
    return float(slope)


# ── IBI from PPG ──────────────────────────────────────────────────────────────

def ppg_peaks_to_ibi(peaks: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Convert PPG peak indices to inter-beat intervals (ms).

    Parameters
    ----------
    peaks : ndarray
    sfreq : float

    Returns
    -------
    ibi_ms : ndarray
    """
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / sfreq * 1000.0


# ── Waveform entropy ──────────────────────────────────────────────────────────

def ppg_waveform_entropy(
    ppg_signal: np.ndarray,
    peaks: np.ndarray,
    n_points: int = 50,
) -> float:
    """
    Sample entropy of normalized pulse waveform morphology.

    Extracts individual pulse waveforms, interpolates to common length,
    then computes sample entropy of the flattened morphology matrix.

    Parameters
    ----------
    ppg_signal : ndarray
    peaks : ndarray
    n_points : int
        Number of points per waveform after resampling.

    Returns
    -------
    entropy : float
    """
    from scipy.interpolate import interp1d
    from features.ecg_features import sample_entropy as sampen

    if len(peaks) < 4:
        return np.nan

    waveforms = []
    for i in range(len(peaks) - 1):
        segment = ppg_signal[peaks[i]:peaks[i + 1]]
        if len(segment) < 5:
            continue
        t_orig = np.linspace(0, 1, len(segment))
        t_new = np.linspace(0, 1, n_points)
        try:
            wf = interp1d(t_orig, segment, kind="linear")(t_new)
            # Normalize to [0, 1]
            wf_range = wf.max() - wf.min()
            if wf_range > 1e-10:
                wf = (wf - wf.min()) / wf_range
            waveforms.append(wf)
        except Exception:
            continue

    if len(waveforms) < 3:
        return np.nan

    # Flatten waveform matrix and compute sample entropy
    waveform_matrix = np.array(waveforms)  # (n_beats, n_points)
    # Use mean morphology signal as proxy
    mean_wf = np.mean(waveform_matrix, axis=0)

    # Sample entropy of mean waveform
    from features.eeg_features import sample_entropy
    return sample_entropy(mean_wf)


# ── Per-trial PPG feature extraction ─────────────────────────────────────────

def extract_ppg_features_trial(
    ppg_signal: np.ndarray,
    sfreq: float,
    cfg: Config,
) -> Dict[str, float]:
    """
    Extract all PPG features for a single trial.

    Parameters
    ----------
    ppg_signal : ndarray, shape (n_samples,)
    sfreq : float
    cfg : Config

    Returns
    -------
    features : dict
    peaks : ndarray (for downstream coupling analysis)
    """
    features = {}

    peaks, troughs = detect_ppg_peaks_troughs(ppg_signal, sfreq)

    # Beat count (quality indicator)
    features["ppg_n_beats"] = float(len(peaks))

    if len(peaks) < 3:
        features.update({
            "ppg_pwa_mean": np.nan,
            "ppg_pwa_std": np.nan,
            "ppg_pwa_slope": np.nan,
            "ppg_ibi_mean": np.nan,
            "ppg_ibi_rmssd": np.nan,
            "ppg_waveform_entropy": np.nan,
        })
        return features, peaks

    # Pulse Wave Amplitude
    pwa = compute_pwa(ppg_signal, peaks, troughs)
    if len(pwa) > 0:
        features["ppg_pwa_mean"] = float(np.nanmean(pwa))
        features["ppg_pwa_std"] = float(np.nanstd(pwa, ddof=1)) if len(pwa) > 1 else np.nan
        features["ppg_pwa_slope"] = compute_pwa_slope(pwa)
    else:
        features["ppg_pwa_mean"] = np.nan
        features["ppg_pwa_std"] = np.nan
        features["ppg_pwa_slope"] = np.nan

    # IBI
    ibi = ppg_peaks_to_ibi(peaks, sfreq)
    ibi = filter_rr_intervals(ibi)  # physiological bounds filter
    if len(ibi) > 1:
        features["ppg_ibi_mean"] = float(np.mean(ibi))
        diffs = np.diff(ibi)
        features["ppg_ibi_rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        features["ppg_ibi_mean"] = np.nan
        features["ppg_ibi_rmssd"] = np.nan

    # Waveform entropy
    features["ppg_waveform_entropy"] = ppg_waveform_entropy(ppg_signal, peaks)

    return features, peaks


def extract_ppg_features_all_trials(
    ppg_epochs: np.ndarray,
    sfreq: float,
    cfg: Config,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Extract PPG features for all trials.

    Parameters
    ----------
    ppg_epochs : ndarray, shape (n_trials, n_samples)
    sfreq : float
    cfg : Config

    Returns
    -------
    feature_matrix : ndarray, shape (n_trials, n_features)
    feature_names : list of str
    all_peaks : list of ndarray
    """
    n_trials = ppg_epochs.shape[0]
    all_features = []
    all_peaks = []

    for i in range(n_trials):
        feats, peaks = extract_ppg_features_trial(ppg_epochs[i], sfreq, cfg)
        all_features.append(feats)
        all_peaks.append(peaks)

    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([
        [f[k] for k in feature_names] for f in all_features
    ])

    if verbose:
        logger.info(
            f"PPG features: {n_trials} trials × {len(feature_names)} features"
        )

    return feature_matrix, feature_names, all_peaks