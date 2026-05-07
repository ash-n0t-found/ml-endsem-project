"""
utils/normalization.py
======================
Feature normalization utilities for multimodal physiological data.

Per-subject, per-condition, and resting-state-relative normalization.
All functions are pure (no side effects) and return new arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import zscore


# ── Per-subject z-score ───────────────────────────────────────────────────────

def zscore_per_subject(
    X: np.ndarray,
    subject_ids: np.ndarray,
    ddof: int = 1,
) -> np.ndarray:
    """
    Z-score each feature within each subject (across that subject's trials).

    Removes individual mean differences without touching between-subject
    variance structure. Required before condition-level GGM fitting.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    subject_ids : ndarray, shape (n_trials,)
    ddof : int
        Degrees of freedom for std (default 1 = unbiased).

    Returns
    -------
    X_norm : ndarray, shape (n_trials, n_features)
    """
    X_norm = X.copy().astype(float)
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        mu = X_norm[mask].mean(axis=0)
        sd = X_norm[mask].std(axis=0, ddof=ddof)
        sd = np.where(sd < 1e-10, 1.0, sd)
        X_norm[mask] = (X_norm[mask] - mu) / sd
    return X_norm


# ── Unit variance per condition ───────────────────────────────────────────────

def unit_variance_per_condition(
    X: np.ndarray,
    condition_ids: np.ndarray,
    ddof: int = 1,
) -> np.ndarray:
    """
    Scale each feature to unit variance within each condition.

    Applied after per-subject z-score. Prevents GGM coupling estimates
    from being dominated by variance differences across modalities.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    condition_ids : ndarray, shape (n_trials,)

    Returns
    -------
    X_scaled : ndarray, shape (n_trials, n_features)
    """
    X_scaled = X.copy().astype(float)
    for cond in np.unique(condition_ids):
        mask = condition_ids == cond
        sd = X_scaled[mask].std(axis=0, ddof=ddof)
        sd = np.where(sd < 1e-10, 1.0, sd)
        X_scaled[mask] = X_scaled[mask] / sd
    return X_scaled


# ── Resting-state delta normalization ────────────────────────────────────────

def compute_resting_delta(
    X_task: np.ndarray,
    X_rest: np.ndarray,
    subject_ids_task: np.ndarray,
    subject_ids_rest: np.ndarray,
) -> np.ndarray:
    """
    Subtract per-subject resting-state mean from task features.

    ΔX_s = X_task_s - mean(X_rest_s)

    Controls for stable trait-level physiological coupling.
    Isolates load-specific changes from resting baseline.

    Parameters
    ----------
    X_task : ndarray, shape (n_task_trials, n_features)
    X_rest : ndarray, shape (n_rest_trials, n_features)
    subject_ids_task : ndarray, shape (n_task_trials,)
    subject_ids_rest : ndarray, shape (n_rest_trials,)

    Returns
    -------
    X_delta : ndarray, shape (n_task_trials, n_features)
    """
    X_delta = X_task.copy().astype(float)
    subjects = np.unique(subject_ids_task)
    for subj in subjects:
        task_mask = subject_ids_task == subj
        rest_mask = subject_ids_rest == subj
        if rest_mask.sum() == 0:
            continue  # no rest data for this subject — skip
        rest_mean = X_rest[rest_mask].mean(axis=0)
        X_delta[task_mask] = X_delta[task_mask] - rest_mean
    return X_delta


# ── Min-max normalization ─────────────────────────────────────────────────────

def minmax_normalize(
    X: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    axis: int = 0,
) -> np.ndarray:
    """
    Scale features to [feature_range[0], feature_range[1]].

    Parameters
    ----------
    X : ndarray
    feature_range : tuple (min, max)
    axis : int
        Axis along which to compute min/max.

    Returns
    -------
    X_scaled : ndarray
    """
    lo, hi = feature_range
    xmin = X.min(axis=axis, keepdims=True)
    xmax = X.max(axis=axis, keepdims=True)
    denom = np.where(xmax - xmin < 1e-10, 1.0, xmax - xmin)
    return lo + (X - xmin) / denom * (hi - lo)


# ── Robust (median/IQR) scaling ───────────────────────────────────────────────

def robust_scale(
    X: np.ndarray,
    quantile_range: Tuple[float, float] = (25.0, 75.0),
) -> np.ndarray:
    """
    Median / IQR scaling — robust to outliers.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    quantile_range : tuple

    Returns
    -------
    X_scaled : ndarray
    """
    from scipy.stats import iqr
    med = np.median(X, axis=0)
    iqr_val = iqr(X, axis=0, rng=quantile_range)
    iqr_val = np.where(iqr_val < 1e-10, 1.0, iqr_val)
    return (X - med) / iqr_val


# ── Percent change normalization (pupil) ──────────────────────────────────────

def percent_change_baseline(
    signal: np.ndarray,
    baseline_mean: float,
) -> np.ndarray:
    """
    Normalize signal as percent change from baseline.

    (signal - baseline) / baseline * 100

    Standard pupillometry normalization.

    Parameters
    ----------
    signal : ndarray
    baseline_mean : float
        Mean signal value during baseline window.

    Returns
    -------
    signal_pct : ndarray
    """
    if abs(baseline_mean) < 1e-10:
        return signal - baseline_mean
    return (signal - baseline_mean) / abs(baseline_mean) * 100.0


# ── Residualize confounds ─────────────────────────────────────────────────────

def residualize(
    X: np.ndarray,
    confounds: np.ndarray,
) -> np.ndarray:
    """
    Regress out confound variables from feature matrix.

    Used to partial out trial order, block index, etc. before GGM fitting.
    Fits OLS per feature, returns residuals.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    confounds : ndarray, shape (n_trials, n_confounds)
        Confound regressors (trial order, block index, etc.)

    Returns
    -------
    X_residual : ndarray, shape (n_trials, n_features)
    """
    # Add intercept
    C = np.column_stack([np.ones(len(confounds)), confounds])
    # OLS: beta = (C'C)^{-1} C' X
    try:
        beta, _, _, _ = np.linalg.lstsq(C, X, rcond=None)
        X_pred = C @ beta
        return X - X_pred + X.mean(axis=0)  # add back mean to preserve scale
    except np.linalg.LinAlgError:
        return X  # fallback: no residualization


# ── PCA dimensionality reduction ──────────────────────────────────────────────

def pca_reduce(
    X: np.ndarray,
    n_components: int,
    return_explained: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Reduce dimensionality via PCA.

    Used to compress each modality to n_components PCs before PID analysis.
    Ensures Gaussian approximation is reasonable and sample sizes adequate.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    n_components : int
    return_explained : bool
        If True, also return explained variance ratios.

    Returns
    -------
    X_reduced : ndarray, shape (n_samples, n_components)
    explained_variance_ratio : ndarray, shape (n_components,) [if return_explained]
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    if return_explained:
        return X_reduced, pca.explained_variance_ratio_
    return X_reduced


# ── Feature block extraction ──────────────────────────────────────────────────

def extract_modality_block(
    X: np.ndarray,
    block_idx: int,
    block_sizes: List[int],
) -> np.ndarray:
    """
    Extract features for a single modality block from concatenated matrix.

    Parameters
    ----------
    X : ndarray, shape (n_samples, sum(block_sizes))
    block_idx : int
        Index into block_sizes list.
    block_sizes : list of int

    Returns
    -------
    X_block : ndarray, shape (n_samples, block_sizes[block_idx])
    """
    boundaries = np.cumsum([0] + block_sizes)
    start = boundaries[block_idx]
    end = boundaries[block_idx + 1]
    return X[:, start:end]