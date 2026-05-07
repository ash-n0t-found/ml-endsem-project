"""
features/ecg_features.py
========================
Per-trial ECG / HRV feature extraction.

Features:
- Time-domain HRV: RMSSD, SDNN, pNN50, MeanNN
- Frequency-domain HRV: HF-HRV (vagal), LF-HRV, LF/HF ratio
- Heartbeat-Evoked Potential (HEP) amplitude — extracted in hep_features.py
  (requires co-registered EEG)

All HRV computation via NeuroKit2 where available;
fallback to manual computation.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── R-peak detection ──────────────────────────────────────────────────────────

def detect_r_peaks(
    ecg_signal: np.ndarray,
    sfreq: float,
    method: str = "neurokit",
) -> np.ndarray:
    """
    Detect R-peaks in ECG signal.

    Parameters
    ----------
    ecg_signal : ndarray, shape (n_samples,)
    sfreq : float
    method : str
        'neurokit' (preferred) or 'pantompkins' (fallback).

    Returns
    -------
    r_peaks : ndarray, shape (n_peaks,)
        Sample indices of R-peaks.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg_signal, sampling_rate=int(sfreq), method="neurokit")
        r_peaks = info["ECG_R_Peaks"]
        return np.array(r_peaks, dtype=int)
    except ImportError:
        logger.warning("NeuroKit2 not available. Using pan-tompkins fallback.")
        return _pantompkins_fallback(ecg_signal, sfreq)
    except Exception as e:
        logger.warning(f"R-peak detection failed: {e}. Using fallback.")
        return _pantompkins_fallback(ecg_signal, sfreq)


def _pantompkins_fallback(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Simplified Pan-Tompkins R-peak detection.
    Used when NeuroKit2 unavailable.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    # Bandpass 5-15 Hz
    b, a = butter(2, [5.0 / (sfreq / 2), 15.0 / (sfreq / 2)], btype="band")
    filtered = filtfilt(b, a, signal)

    # Differentiate + square
    diff = np.diff(filtered)
    squared = diff ** 2

    # Integrate (moving window ~150ms)
    win = int(0.150 * sfreq)
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # Find peaks with minimum distance ~400ms
    min_dist = int(0.400 * sfreq)
    threshold = 0.5 * np.max(integrated)
    peaks, _ = find_peaks(integrated, height=threshold, distance=min_dist)

    return peaks


# ── RR interval series ────────────────────────────────────────────────────────

def r_peaks_to_rr(r_peaks: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Convert R-peak sample indices to RR intervals in milliseconds.

    Parameters
    ----------
    r_peaks : ndarray, shape (n_peaks,)
    sfreq : float

    Returns
    -------
    rr_ms : ndarray, shape (n_peaks - 1,)
        RR intervals in ms.
    """
    if len(r_peaks) < 2:
        return np.array([])
    rr_ms = np.diff(r_peaks) / sfreq * 1000.0
    return rr_ms


def filter_rr_intervals(rr_ms: np.ndarray, min_ms: float = 300.0, max_ms: float = 2000.0) -> np.ndarray:
    """
    Remove physiologically implausible RR intervals.
    300ms = 200 bpm max; 2000ms = 30 bpm min.
    """
    return rr_ms[(rr_ms >= min_ms) & (rr_ms <= max_ms)]


# ── Time-domain HRV ───────────────────────────────────────────────────────────

def hrv_time_domain(rr_ms: np.ndarray) -> Dict[str, float]:
    """
    Compute standard time-domain HRV features.

    Parameters
    ----------
    rr_ms : ndarray
        RR intervals in milliseconds.

    Returns
    -------
    dict with keys: RMSSD, SDNN, pNN50, MeanNN
    """
    features = {
        "hrv_rmssd": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_pnn50": np.nan,
        "hrv_mean_nn": np.nan,
    }

    rr = filter_rr_intervals(rr_ms)

    if len(rr) < 3:
        return features

    features["hrv_mean_nn"] = float(np.mean(rr))
    features["hrv_sdnn"] = float(np.std(rr, ddof=1))

    # RMSSD: root mean square of successive differences
    diffs = np.diff(rr)
    features["hrv_rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))

    # pNN50: proportion of successive differences > 50ms
    features["hrv_pnn50"] = float(np.mean(np.abs(diffs) > 50.0))

    return features


# ── Frequency-domain HRV ──────────────────────────────────────────────────────

def hrv_frequency_domain(
    rr_ms: np.ndarray,
    sfreq_interp: float = 4.0,
    hf_band: Tuple[float, float] = (0.15, 0.40),
    lf_band: Tuple[float, float] = (0.04, 0.15),
    nperseg: int = 256,
) -> Dict[str, float]:
    """
    Frequency-domain HRV via Welch's method on interpolated RR series.

    Interpolates irregularly-sampled RR intervals to uniform grid.

    Parameters
    ----------
    rr_ms : ndarray
        RR intervals in ms.
    sfreq_interp : float
        Resampling frequency (Hz). Default 4 Hz standard.
    hf_band, lf_band : tuple
        Frequency band limits (Hz).
    nperseg : int

    Returns
    -------
    dict with keys: hrv_hf, hrv_lf, hrv_lf_hf_ratio
    """
    features = {
        "hrv_hf": np.nan,
        "hrv_lf": np.nan,
        "hrv_lf_hf_ratio": np.nan,
    }

    rr = filter_rr_intervals(rr_ms)
    if len(rr) < 8:
        return features

    # Cumulative time axis (seconds)
    t_rr = np.cumsum(rr) / 1000.0  # ms → s
    t_rr -= t_rr[0]

    # Interpolate to uniform grid
    t_uniform = np.arange(0, t_rr[-1], 1.0 / sfreq_interp)
    if len(t_uniform) < nperseg:
        return features

    try:
        from scipy.interpolate import interp1d
        interp_fn = interp1d(t_rr, rr, kind="cubic", bounds_error=False,
                             fill_value="extrapolate")
        rr_uniform = interp_fn(t_uniform)
    except Exception:
        return features

    # Welch PSD
    freqs, psd = welch(rr_uniform, fs=sfreq_interp,
                        nperseg=min(nperseg, len(rr_uniform)))

    def _band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() == 0:
            return np.nan
        df = freqs[1] - freqs[0]
        return float(np.trapz(psd[mask], freqs[mask]))

    hf = _band_power(*hf_band)
    lf = _band_power(*lf_band)

    features["hrv_hf"] = hf
    features["hrv_lf"] = lf

    if hf is not None and hf > 1e-12:
        features["hrv_lf_hf_ratio"] = float(lf / hf)

    return features


# ── Per-trial ECG feature extraction ─────────────────────────────────────────

def extract_ecg_features_trial(
    ecg_signal: np.ndarray,
    sfreq: float,
    cfg: Config,
) -> Dict[str, float]:
    """
    Extract all ECG/HRV features for a single trial.

    Parameters
    ----------
    ecg_signal : ndarray, shape (n_samples,)
        ECG time series for the trial epoch.
    sfreq : float
    cfg : Config

    Returns
    -------
    features : dict
    """
    features = {}

    # R-peak detection
    r_peaks = detect_r_peaks(ecg_signal, sfreq, method="neurokit")
    rr_ms = r_peaks_to_rr(r_peaks, sfreq)

    # Time-domain HRV
    td = hrv_time_domain(rr_ms)
    features.update(td)

    # Frequency-domain HRV
    fd = hrv_frequency_domain(rr_ms)
    features.update(fd)

    # Store r_peaks count (data quality indicator)
    features["ecg_n_rpeaks"] = float(len(r_peaks))

    return features, r_peaks


def extract_ecg_features_all_trials(
    ecg_epochs: np.ndarray,
    sfreq: float,
    cfg: Config,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Extract ECG features for all trials.

    Parameters
    ----------
    ecg_epochs : ndarray, shape (n_trials, n_samples)
    sfreq : float
    cfg : Config

    Returns
    -------
    feature_matrix : ndarray, shape (n_trials, n_features)
    feature_names : list of str
    all_r_peaks : list of ndarray
        R-peak indices per trial (needed for HEP computation).
    """
    n_trials = ecg_epochs.shape[0]
    all_features = []
    all_r_peaks = []

    for i in range(n_trials):
        feats, r_peaks = extract_ecg_features_trial(ecg_epochs[i], sfreq, cfg)
        all_features.append(feats)
        all_r_peaks.append(r_peaks)

    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([
        [f[k] for k in feature_names] for f in all_features
    ])

    if verbose:
        logger.info(
            f"ECG features: {n_trials} trials × {len(feature_names)} features"
        )

    return feature_matrix, feature_names, all_r_peaks