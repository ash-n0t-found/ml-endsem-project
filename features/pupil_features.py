"""
features/pupil_features.py
==========================
Trial-level pupillometry feature extraction.

Features computed per trial:
  - mean_diameter        : baseline-normalized mean during encoding window
  - peak_dilation        : max diameter in encoding window
  - peak_latency_s       : time of peak (s from trial onset)
  - dilation_rate        : slope of dilation curve in first 1 s (linear regression)
  - dfa_alpha            : detrended fluctuation analysis exponent
  - pct_valid            : fraction of non-NaN samples (quality flag)

All features returned as dict → float.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import detrend


# ── DFA ───────────────────────────────────────────────────────────────────────

def _dfa(x: np.ndarray, min_scale: int = 4, max_scale: Optional[int] = None,
         n_scales: int = 10) -> float:
    """
    Detrended Fluctuation Analysis — return scaling exponent alpha.

    Parameters
    ----------
    x : 1-D array (no NaNs)
    min_scale : minimum window size (samples)
    max_scale : maximum window size (samples); defaults to len(x) // 4
    n_scales  : number of log-spaced window sizes

    Returns
    -------
    alpha : float  (alpha ~ 0.5 = uncorrelated, > 0.5 = persistent)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_scale:
        return np.nan

    if max_scale is None:
        max_scale = max(min_scale + 1, n // 4)

    scales = np.unique(
        np.round(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)).astype(int)
    )
    scales = scales[scales >= min_scale]

    # Cumulative sum (integrated series)
    y = np.cumsum(x - np.mean(x))

    fluctuations = []
    valid_scales = []
    for s in scales:
        n_segments = n // s
        if n_segments < 2:
            continue
        F2 = []
        for k in range(n_segments):
            seg = y[k * s: (k + 1) * s]
            t = np.arange(s)
            # linear detrend
            coef = np.polyfit(t, seg, 1)
            trend = np.polyval(coef, t)
            F2.append(np.mean((seg - trend) ** 2))
        fluctuations.append(np.sqrt(np.mean(F2)))
        valid_scales.append(s)

    if len(valid_scales) < 2:
        return np.nan

    log_s = np.log10(valid_scales)
    log_f = np.log10(np.maximum(fluctuations, 1e-12))
    slope, _, _, _, _ = sp_stats.linregress(log_s, log_f)
    return float(slope)


# ── Core feature extractor ────────────────────────────────────────────────────

def extract_pupil_features(
    pupil_signal: np.ndarray,
    sfreq: float,
    encoding_tmin: float = 0.0,
    encoding_tmax: float = 4.0,
    dilation_rate_window: float = 1.0,
    min_valid_proportion: float = 0.7,
) -> Dict[str, float]:
    """
    Extract trial-level pupillometry features from a baseline-normalized signal.

    Parameters
    ----------
    pupil_signal : ndarray, shape (n_samples,)
        Baseline-normalized pupil diameter trace for one trial.
        NaNs indicate blinks / invalid samples (already handled upstream).
    sfreq : float
        Sampling frequency (Hz).
    encoding_tmin : float
        Start of encoding window relative to trial onset (s).
    encoding_tmax : float
        End of encoding window (s).
    dilation_rate_window : float
        Duration (s) for computing onset dilation rate.
    min_valid_proportion : float
        Minimum fraction of non-NaN samples; if below, return NaN features.

    Returns
    -------
    dict : feature_name → float
    """
    features: Dict[str, float] = {}

    n_samples = len(pupil_signal)
    # Sample indices for encoding window
    i_start = int(encoding_tmin * sfreq)
    i_end = min(int(encoding_tmax * sfreq), n_samples)

    encoding = pupil_signal[i_start:i_end]
    time_s = np.arange(len(encoding)) / sfreq + encoding_tmin

    # Valid sample fraction
    valid_mask = ~np.isnan(encoding)
    pct_valid = valid_mask.mean()
    features["pct_valid"] = float(pct_valid)

    if pct_valid < min_valid_proportion:
        # Insufficient valid data — return NaN for all features
        for k in ["mean_diameter", "peak_dilation", "peak_latency_s",
                  "dilation_rate", "dfa_alpha"]:
            features[k] = np.nan
        return features

    # Interpolate NaNs for feature computation
    if np.any(~valid_mask):
        encoding = _interpolate_nans(encoding)

    # Mean diameter
    features["mean_diameter"] = float(np.nanmean(encoding))

    # Peak dilation
    peak_idx = int(np.nanargmax(encoding))
    features["peak_dilation"] = float(encoding[peak_idx])
    features["peak_latency_s"] = float(time_s[peak_idx]) if peak_idx < len(time_s) else np.nan

    # Dilation rate: slope in first `dilation_rate_window` seconds
    n_rate = min(int(dilation_rate_window * sfreq), len(encoding))
    if n_rate >= 4:
        t_rate = np.arange(n_rate) / sfreq
        slope, _, _, _, _ = sp_stats.linregress(t_rate, encoding[:n_rate])
        features["dilation_rate"] = float(slope)
    else:
        features["dilation_rate"] = np.nan

    # DFA exponent
    features["dfa_alpha"] = _dfa(encoding)

    return features


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN spans."""
    x = x.copy().astype(float)
    nans = np.isnan(x)
    not_nans = ~nans
    if not_nans.sum() < 2:
        return x
    idxs = np.arange(len(x))
    x[nans] = np.interp(idxs[nans], idxs[not_nans], x[not_nans])
    return x


# ── Task-evoked pupil response (TEPR) per digit ───────────────────────────────

def extract_tepr_per_digit(
    pupil_signal: np.ndarray,
    digit_onsets_s: np.ndarray,
    sfreq: float,
    post_onset_window: float = 1.5,
    min_valid: float = 0.7,
) -> np.ndarray:
    """
    Task-evoked pupil response amplitude for each digit stimulus.

    Parameters
    ----------
    pupil_signal : ndarray, shape (n_samples,)
        Full trial pupil trace (baseline-normalized).
    digit_onsets_s : ndarray, shape (n_digits,)
        Onset time (s from trial start) of each digit stimulus.
    sfreq : float
    post_onset_window : float
        Window (s) post-onset for peak TEPR.
    min_valid : float
        Min valid proportion; returns NaN for digit if below.

    Returns
    -------
    tepr : ndarray, shape (n_digits,)
        Peak baseline-corrected pupil response per digit.
    """
    n_digits = len(digit_onsets_s)
    tepr = np.full(n_digits, np.nan)

    n_win = int(post_onset_window * sfreq)

    for i, onset_s in enumerate(digit_onsets_s):
        i0 = int(onset_s * sfreq)
        i1 = min(i0 + n_win, len(pupil_signal))
        if i1 <= i0:
            continue
        seg = pupil_signal[i0:i1]
        valid = ~np.isnan(seg)
        if valid.mean() < min_valid:
            continue
        tepr[i] = float(np.nanmax(seg))

    return tepr


# ── Subject-level aggregation ─────────────────────────────────────────────────

def aggregate_pupil_features(
    trial_features: list[Dict[str, float]],
    condition_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Aggregate per-trial pupil features into arrays indexed by trial.

    Parameters
    ----------
    trial_features : list of dicts, length n_trials
    condition_labels : ndarray, shape (n_trials,)

    Returns
    -------
    dict : feature_name → ndarray (n_trials,)
    """
    if not trial_features:
        return {}

    keys = list(trial_features[0].keys())
    out = {}
    for k in keys:
        out[k] = np.array([tf.get(k, np.nan) for tf in trial_features])
    out["condition"] = condition_labels
    return out


# ── Resting-state pupil features ──────────────────────────────────────────────

def extract_resting_pupil_features(
    pupil_signal: np.ndarray,
    sfreq: float,
    min_valid: float = 0.7,
) -> Dict[str, float]:
    """
    Resting-state baseline pupil features (computed once per subject).

    Parameters
    ----------
    pupil_signal : ndarray, shape (n_samples,)
        Full resting-state pupil recording (raw, not baseline-normalized).
    sfreq : float
    min_valid : float

    Returns
    -------
    dict : feature_name → float
    """
    features: Dict[str, float] = {}

    valid = ~np.isnan(pupil_signal)
    pct_valid = valid.mean()
    features["resting_pupil_pct_valid"] = float(pct_valid)

    if pct_valid < min_valid:
        for k in ["resting_pupil_mean", "resting_pupil_std",
                  "resting_pupil_dfa"]:
            features[k] = np.nan
        return features

    sig = pupil_signal.copy()
    if np.any(~valid):
        sig = _interpolate_nans(sig)

    features["resting_pupil_mean"] = float(np.mean(sig))
    features["resting_pupil_std"] = float(np.std(sig, ddof=1))
    features["resting_pupil_dfa"] = _dfa(sig)

    return features