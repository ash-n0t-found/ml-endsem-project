"""
preprocessing/pupil_prep.py
===========================
Pupillometry preprocessing pipeline for ds003838.

Steps:
  1. Load raw pupil diameter timeseries (120 Hz Pupil Labs)
  2. Blink detection + linear interpolation
  3. Baseline normalization (percent change from pre-stimulus window)
  4. Trial segmentation
  5. Per-trial feature extraction:
       - Mean diameter (encoding window)
       - Peak dilation latency
       - Dilation rate (onset slope)
       - Task-Evoked Pupil Response (TEPR) per digit
       - Detrended Fluctuation Analysis (DFA) exponent
  6. Quality control: reject trials with < min_valid_proportion valid samples

All functions pure / side-effect-free.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import linregress

from utils.io_utils import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

BLINK_SD_THRESHOLD = 3.0     # SD threshold for blink detection
MIN_VALID_PROPORTION = 0.70  # reject trial if < 70% valid samples
BASELINE_WINDOW_S = (-0.5, 0.0)   # pre-stimulus baseline (seconds)
MEAN_WINDOW_S = (0.0, 4.0)        # encoding window for mean diameter
DILATION_RATE_WINDOW_S = (0.0, 1.0)  # onset window for dilation slope


# ── Blink detection & interpolation ──────────────────────────────────────────

def detect_blinks(
    pupil: np.ndarray,
    sfreq: float,
    threshold_sd: float = BLINK_SD_THRESHOLD,
) -> np.ndarray:
    """
    Detect blinks / missing data in pupil timeseries.

    Blinks are identified as:
      1. Explicit zeros / negative values (missing marker from Pupil Labs)
      2. Samples where absolute derivative > threshold_sd * std(derivative)
         (rapid velocity changes characteristic of blink onset/offset)

    Parameters
    ----------
    pupil : ndarray, shape (n_samples,)
    sfreq : float
    threshold_sd : float

    Returns
    -------
    blink_mask : ndarray of bool, shape (n_samples,)
        True where sample is invalid (blink / artifact).
    """
    blink_mask = np.zeros(len(pupil), dtype=bool)

    # Missing data markers
    blink_mask |= (pupil <= 0)
    blink_mask |= ~np.isfinite(pupil)

    # Velocity-based detection
    deriv = np.abs(np.gradient(pupil, 1.0 / sfreq))
    deriv_mean = np.nanmean(deriv[~blink_mask])
    deriv_std = np.nanstd(deriv[~blink_mask])
    if deriv_std > 1e-10:
        blink_mask |= deriv > (deriv_mean + threshold_sd * deriv_std)

    # Expand blink windows by 50ms on each side
    expansion = int(0.05 * sfreq)
    blink_mask = _expand_mask(blink_mask, expansion)

    return blink_mask


def _expand_mask(mask: np.ndarray, n_samples: int) -> np.ndarray:
    """Expand boolean mask by n_samples on each side via dilation."""
    kernel = np.ones(2 * n_samples + 1, dtype=bool)
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, structure=kernel)


def interpolate_blinks(
    pupil: np.ndarray,
    blink_mask: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Linear interpolation over blink periods.

    Parameters
    ----------
    pupil : ndarray
    blink_mask : ndarray of bool
    method : str
        Currently only "linear" supported.

    Returns
    -------
    pupil_interp : ndarray
        Pupil signal with blinks replaced by interpolated values.
        NaN where blinks at edges (cannot extrapolate).
    """
    pupil_interp = pupil.copy().astype(float)
    indices = np.arange(len(pupil))
    valid_idx = indices[~blink_mask]

    if len(valid_idx) < 2:
        logger.warning("Too few valid samples for blink interpolation.")
        return pupil_interp

    pupil_interp = np.interp(indices, valid_idx, pupil[valid_idx])
    return pupil_interp


# ── Baseline normalization ─────────────────────────────────────────────────────

def baseline_normalize(
    pupil_trial: np.ndarray,
    sfreq: float,
    baseline_window_s: Tuple[float, float] = BASELINE_WINDOW_S,
    trial_tmin_s: float = -0.5,
    method: str = "percent_change",
) -> np.ndarray:
    """
    Normalize pupil trial by baseline window.

    Parameters
    ----------
    pupil_trial : ndarray, shape (n_samples_trial,)
        Single trial pupil signal starting at trial_tmin_s.
    sfreq : float
    baseline_window_s : tuple
        (start, end) in seconds relative to stimulus onset (t=0).
    trial_tmin_s : float
        Time (seconds) of first sample in pupil_trial relative to stimulus.
    method : str
        "percent_change": (x - baseline_mean) / baseline_mean * 100
        "subtract_mean": x - baseline_mean
        "z_score": (x - baseline_mean) / baseline_std

    Returns
    -------
    pupil_norm : ndarray
    """
    t = np.arange(len(pupil_trial)) / sfreq + trial_tmin_s
    bl_start, bl_end = baseline_window_s
    bl_mask = (t >= bl_start) & (t < bl_end)

    if bl_mask.sum() == 0:
        logger.warning("Baseline window contains no samples. Returning unnormalized.")
        return pupil_trial.copy().astype(float)

    bl_mean = np.nanmean(pupil_trial[bl_mask])
    bl_std = np.nanstd(pupil_trial[bl_mask])

    if abs(bl_mean) < 1e-10:
        return np.zeros_like(pupil_trial, dtype=float)

    if method == "percent_change":
        return (pupil_trial - bl_mean) / bl_mean * 100.0
    elif method == "subtract_mean":
        return pupil_trial - bl_mean
    elif method == "z_score":
        if bl_std < 1e-10:
            return np.zeros_like(pupil_trial, dtype=float)
        return (pupil_trial - bl_mean) / bl_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ── Per-trial feature extraction ──────────────────────────────────────────────

def extract_mean_diameter(
    pupil_norm: np.ndarray,
    sfreq: float,
    window_s: Tuple[float, float] = MEAN_WINDOW_S,
    trial_tmin_s: float = 0.0,
) -> float:
    """Mean normalized pupil diameter over encoding window."""
    t = np.arange(len(pupil_norm)) / sfreq + trial_tmin_s
    mask = (t >= window_s[0]) & (t < window_s[1])
    if mask.sum() == 0:
        return np.nan
    return float(np.nanmean(pupil_norm[mask]))


def extract_peak_latency(
    pupil_norm: np.ndarray,
    sfreq: float,
    trial_tmin_s: float = 0.0,
) -> float:
    """
    Latency (seconds) of maximum pupil dilation relative to stimulus onset.
    """
    t = np.arange(len(pupil_norm)) / sfreq + trial_tmin_s
    # Only look from stimulus onset
    onset_mask = t >= 0.0
    if onset_mask.sum() == 0:
        return np.nan
    peak_rel_idx = np.nanargmax(pupil_norm[onset_mask])
    return float(t[onset_mask][peak_rel_idx])


def extract_dilation_rate(
    pupil_norm: np.ndarray,
    sfreq: float,
    window_s: Tuple[float, float] = DILATION_RATE_WINDOW_S,
    trial_tmin_s: float = 0.0,
) -> float:
    """
    Dilation rate: linear slope of pupil signal during onset window (% / s).
    """
    t = np.arange(len(pupil_norm)) / sfreq + trial_tmin_s
    mask = (t >= window_s[0]) & (t < window_s[1])
    if mask.sum() < 2:
        return np.nan
    t_seg = t[mask]
    p_seg = pupil_norm[mask]
    valid = np.isfinite(p_seg)
    if valid.sum() < 2:
        return np.nan
    slope, _, _, _, _ = linregress(t_seg[valid], p_seg[valid])
    return float(slope)


def extract_tepr_per_digit(
    pupil_norm: np.ndarray,
    sfreq: float,
    digit_onset_times_s: np.ndarray,
    trial_tmin_s: float,
    tepr_window_s: Tuple[float, float] = (0.2, 1.5),
) -> np.ndarray:
    """
    Task-Evoked Pupil Response (TEPR) amplitude for each digit presentation.

    Parameters
    ----------
    pupil_norm : ndarray
        Baseline-normalized trial pupil signal.
    sfreq : float
    digit_onset_times_s : ndarray
        Times of each digit onset (seconds, relative to trial start).
    trial_tmin_s : float
        Time of first pupil_norm sample relative to trial start.
    tepr_window_s : tuple
        (start, end) seconds post-digit onset to measure TEPR.

    Returns
    -------
    tepr : ndarray, shape (n_digits,)
        Mean pupil dilation in TEPR window per digit.
    """
    t = np.arange(len(pupil_norm)) / sfreq + trial_tmin_s
    tepr = np.full(len(digit_onset_times_s), np.nan)

    for i, d_onset in enumerate(digit_onset_times_s):
        w_start = d_onset + tepr_window_s[0]
        w_end = d_onset + tepr_window_s[1]
        mask = (t >= w_start) & (t < w_end)
        if mask.sum() > 0:
            tepr[i] = float(np.nanmean(pupil_norm[mask]))

    return tepr


# ── Detrended Fluctuation Analysis ───────────────────────────────────────────

def dfa_exponent(x: np.ndarray, scales: Optional[np.ndarray] = None) -> float:
    """
    Compute DFA (Detrended Fluctuation Analysis) scaling exponent alpha.

    Alpha ~ 0.5: uncorrelated noise
    Alpha ~ 1.0: 1/f noise (typical for healthy pupil dynamics)
    Alpha > 1.0: non-stationary / long memory

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
    scales : ndarray, optional
        Window sizes to use. Defaults to log-spaced 10..len(x)//4.

    Returns
    -------
    alpha : float
        DFA scaling exponent.
    """
    n = len(x)
    if n < 20:
        return np.nan

    # Cumulative sum of mean-removed signal
    y = np.cumsum(x - np.nanmean(x))

    if scales is None:
        scales = np.unique(
            np.round(np.logspace(np.log10(10), np.log10(n // 4), 12)).astype(int)
        )
        scales = scales[scales >= 4]

    fluctuations = np.zeros(len(scales))
    valid_scales = []

    for i, s in enumerate(scales):
        n_windows = n // s
        if n_windows < 2:
            continue
        rms_vals = []
        for w in range(n_windows):
            segment = y[w * s: (w + 1) * s]
            t_seg = np.arange(len(segment), dtype=float)
            if len(segment) < 2:
                continue
            slope, intercept, _, _, _ = linregress(t_seg, segment)
            trend = slope * t_seg + intercept
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_vals:
            fluctuations[i] = np.mean(rms_vals)
            valid_scales.append(s)

    valid_scales = np.array(valid_scales)
    valid_fluct = fluctuations[: len(valid_scales)]

    if len(valid_scales) < 2 or np.any(valid_fluct <= 0):
        return np.nan

    log_s = np.log10(valid_scales)
    log_f = np.log10(valid_fluct)
    alpha, _, _, _, _ = linregress(log_s, log_f)
    return float(alpha)


# ── Full trial feature dict ───────────────────────────────────────────────────

def compute_trial_features(
    pupil_norm: np.ndarray,
    blink_mask_trial: np.ndarray,
    sfreq: float,
    trial_tmin_s: float,
    digit_onset_times_s: Optional[np.ndarray] = None,
    min_valid_proportion: float = MIN_VALID_PROPORTION,
) -> Dict:
    """
    Compute all pupil features for one baseline-normalized trial.

    Parameters
    ----------
    pupil_norm : ndarray
        Baseline-normalized pupil signal for this trial.
    blink_mask_trial : ndarray of bool
        True where sample is invalid.
    sfreq : float
    trial_tmin_s : float
        Time of first sample relative to stimulus onset.
    digit_onset_times_s : ndarray, optional
        Times of digit onsets (for TEPR). If None, TEPR skipped.
    min_valid_proportion : float

    Returns
    -------
    features : dict
    """
    valid_proportion = 1.0 - blink_mask_trial.mean()
    valid = valid_proportion >= min_valid_proportion

    if not valid:
        return {
            "valid": False,
            "valid_proportion": valid_proportion,
            "mean_diameter": np.nan,
            "peak_latency_s": np.nan,
            "dilation_rate": np.nan,
            "dfa_alpha": np.nan,
            "tepr_mean": np.nan,
            "tepr_per_digit": np.full(
                len(digit_onset_times_s) if digit_onset_times_s is not None else 0,
                np.nan
            ),
        }

    mean_diam = extract_mean_diameter(pupil_norm, sfreq, trial_tmin_s=trial_tmin_s)
    peak_lat = extract_peak_latency(pupil_norm, sfreq, trial_tmin_s=trial_tmin_s)
    dil_rate = extract_dilation_rate(pupil_norm, sfreq, trial_tmin_s=trial_tmin_s)
    alpha = dfa_exponent(pupil_norm)

    tepr = None
    if digit_onset_times_s is not None and len(digit_onset_times_s) > 0:
        tepr = extract_tepr_per_digit(
            pupil_norm, sfreq, digit_onset_times_s, trial_tmin_s
        )

    return {
        "valid": True,
        "valid_proportion": valid_proportion,
        "mean_diameter": mean_diam,
        "peak_latency_s": peak_lat,
        "dilation_rate": dil_rate,
        "dfa_alpha": alpha,
        "tepr_mean": float(np.nanmean(tepr)) if tepr is not None else np.nan,
        "tepr_per_digit": tepr if tepr is not None else np.array([]),
    }


# ── Full subject pipeline ─────────────────────────────────────────────────────

def preprocess_pupil_subject(
    pupil_raw: np.ndarray,
    sfreq: float,
    trial_onsets_samples: np.ndarray,
    trial_duration_samples: int,
    trial_tmin_s: float,
    digit_onset_times_per_trial: Optional[List[np.ndarray]] = None,
    blink_threshold_sd: float = BLINK_SD_THRESHOLD,
    baseline_method: str = "percent_change",
) -> Dict:
    """
    Full pupil preprocessing pipeline for one subject.

    Parameters
    ----------
    pupil_raw : ndarray, shape (n_samples,)
    sfreq : float
    trial_onsets_samples : ndarray of int
    trial_duration_samples : int
    trial_tmin_s : float
        Time of trial onset relative to stimulus (typically includes baseline).
    digit_onset_times_per_trial : list of ndarray, optional
        Per-trial digit onset times in seconds (for TEPR).
    blink_threshold_sd : float
    baseline_method : str

    Returns
    -------
    result : dict
        - 'pupil_interp': ndarray  (blink-interpolated full signal)
        - 'blink_mask': ndarray
        - 'trials': list of trial feature dicts
        - 'n_valid_trials': int
        - 'rejection_rate': float
    """
    logger.info("Pupil preprocessing: blink detection...")
    blink_mask = detect_blinks(pupil_raw, sfreq, threshold_sd=blink_threshold_sd)
    logger.info(f"  Blink proportion: {blink_mask.mean():.2%}")

    logger.info("Pupil preprocessing: blink interpolation...")
    pupil_interp = interpolate_blinks(pupil_raw, blink_mask)

    trials = []
    n_trials = len(trial_onsets_samples)

    for t_idx in range(n_trials):
        onset = trial_onsets_samples[t_idx]
        end = onset + trial_duration_samples

        # Extract trial segment
        pupil_trial = pupil_interp[onset:end]
        blink_trial = blink_mask[onset:end]

        if len(pupil_trial) == 0:
            logger.warning(f"Trial {t_idx}: empty pupil segment.")
            trials.append({"trial_idx": t_idx, "valid": False})
            continue

        # Baseline normalization
        pupil_norm = baseline_normalize(
            pupil_trial,
            sfreq,
            trial_tmin_s=trial_tmin_s,
            method=baseline_method,
        )

        # Digit onsets for this trial
        d_onsets = None
        if digit_onset_times_per_trial is not None and t_idx < len(digit_onset_times_per_trial):
            d_onsets = digit_onset_times_per_trial[t_idx]

        feats = compute_trial_features(
            pupil_norm=pupil_norm,
            blink_mask_trial=blink_trial,
            sfreq=sfreq,
            trial_tmin_s=trial_tmin_s,
            digit_onset_times_s=d_onsets,
        )
        feats["trial_idx"] = t_idx
        feats["pupil_norm"] = pupil_norm
        trials.append(feats)

    n_valid = sum(t.get("valid", False) for t in trials)
    rejection_rate = 1.0 - n_valid / max(n_trials, 1)
    logger.info(f"Pupil preprocessing: {n_valid}/{n_trials} valid trials (rejection={rejection_rate:.2%})")

    return {
        "pupil_interp": pupil_interp,
        "blink_mask": blink_mask,
        "trials": trials,
        "n_valid_trials": n_valid,
        "rejection_rate": rejection_rate,
    }