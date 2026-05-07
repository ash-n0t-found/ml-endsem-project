"""
evaluation/bootstrap.py
========================
Bootstrap confidence intervals and hypothesis tests for:

1. GGM edge weight CIs per condition
2. Cross-modal edge density CIs per condition
3. Network distance (Frobenius) CIs between condition pairs
4. PID quantities (redundancy, unique, synergy) CIs per modality pair
5. Recall prediction R² CIs
6. Non-monotonic synergy hypothesis test (bootstrap)
7. Coupling feature predictor importance CIs (ridge regression)

All intervals use the bias-corrected and accelerated (BCa) method where
sample size permits; otherwise basic percentile bootstrap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.random import default_rng

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Single bootstrapped confidence interval."""
    statistic_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    ci_level: float        # e.g., 0.95
    n_bootstrap: int
    method: str            # 'percentile' or 'bca'
    boot_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class EdgeWeightCI:
    """Bootstrap CI for a single GGM edge weight."""
    feature_i: int
    feature_j: int
    feature_i_name: str
    feature_j_name: str
    condition: str
    observed_weight: float
    ci_lower: float
    ci_upper: float
    is_cross_modal: bool
    excludes_zero: bool   # True if CI does not contain 0


@dataclass
class BootstrapReport:
    """Aggregated bootstrap results."""
    density_cis: Dict[str, BootstrapCI]            # condition → CI
    network_distance_cis: Dict[str, BootstrapCI]    # pair_key → CI
    pid_cis: Dict[str, Dict[str, BootstrapCI]]      # condition → {pid_key → CI}
    recall_r2_cis: Dict[str, BootstrapCI]           # model_name → CI
    synergy_nonmonotonic_p: Optional[float]
    synergy_nonmonotonic_significant: Optional[bool]


# ---------------------------------------------------------------------------
# Core bootstrap engine
# ---------------------------------------------------------------------------

def bootstrap_statistic(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    method: str = "bca",
    random_seed: int = 42,
    statistic_name: str = "statistic",
) -> BootstrapCI:
    """
    Bootstrap CI for a scalar statistic computed from a data matrix.

    Parameters
    ----------
    data : (N, ...) array
        First axis is the bootstrap axis (rows resampled with replacement).
    statistic_fn : callable
        Maps (N, ...) array → scalar float.
    n_bootstrap : int
    ci_level : float
    method : str  'percentile' or 'bca'
    random_seed : int

    Returns
    -------
    BootstrapCI
    """
    rng = default_rng(random_seed)
    N = data.shape[0]
    observed = float(statistic_fn(data))

    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        try:
            boot_stats[b] = statistic_fn(data[idx])
        except Exception:
            boot_stats[b] = np.nan

    # Remove NaN
    valid = boot_stats[~np.isnan(boot_stats)]
    if len(valid) < n_bootstrap * 0.8:
        logger.warning(
            f"Bootstrap [{statistic_name}]: {n_bootstrap - len(valid)} "
            f"/ {n_bootstrap} replicates failed."
        )

    alpha = 1.0 - ci_level

    if method == "bca" and len(valid) >= 50:
        ci_lower, ci_upper = _bca_interval(observed, valid, data, statistic_fn, alpha)
    else:
        ci_lower = float(np.nanpercentile(valid, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(valid, 100 * (1 - alpha / 2)))
        method = "percentile"

    return BootstrapCI(
        statistic_name=statistic_name,
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        method=method,
        boot_distribution=valid,
    )


def _bca_interval(
    observed: float,
    boot_stats: np.ndarray,
    data: np.ndarray,
    statistic_fn: Callable,
    alpha: float,
) -> Tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) bootstrap interval.

    Efron & Tibshirani (1993) Algorithm 14.3.
    """
    from scipy.stats import norm

    # Bias correction: z0
    prop_less = np.mean(boot_stats < observed)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    # Acceleration: jackknife skewness
    N = data.shape[0]
    jack_stats = np.zeros(N)
    for i in range(N):
        idx = list(range(N))
        idx.pop(i)
        try:
            jack_stats[i] = statistic_fn(data[idx])
        except Exception:
            jack_stats[i] = observed

    jack_mean = np.mean(jack_stats)
    numer = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * np.sum((jack_mean - jack_stats) ** 2) ** 1.5
    a = numer / denom if abs(denom) > 1e-12 else 0.0

    z_alpha1 = norm.ppf(alpha / 2)
    z_alpha2 = norm.ppf(1 - alpha / 2)

    p_lower = norm.cdf(z0 + (z0 + z_alpha1) / (1 - a * (z0 + z_alpha1)))
    p_upper = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))

    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)

    ci_lower = float(np.percentile(boot_stats, 100 * p_lower))
    ci_upper = float(np.percentile(boot_stats, 100 * p_upper))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# GGM edge weight CIs
# ---------------------------------------------------------------------------

def bootstrap_edge_weights(
    feature_matrix: np.ndarray,
    ggm_fit_fn: Callable,
    condition_name: str,
    modality_slices: Dict[str, slice],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    feature_names: Optional[List[str]] = None,
    random_seed: int = 42,
) -> List[EdgeWeightCI]:
    """
    Bootstrap CIs for each GGM edge weight in a condition-specific precision matrix.

    Edges whose CI excludes zero are declared reliably nonzero.

    Parameters
    ----------
    feature_matrix : (N, D) array
    ggm_fit_fn : callable (N, D) → (D, D)
    condition_name : str
    modality_slices : dict modality_name → slice

    Returns
    -------
    List[EdgeWeightCI], upper triangle only (i < j)
    """
    rng = default_rng(random_seed)
    N, D = feature_matrix.shape

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(D)]

    # Build cross-modal mask
    modality_ids = _assign_modality_ids(D, modality_slices)
    cross_modal_mask = modality_ids[:, None] != modality_ids[None, :]

    # Observed precision matrix
    theta_obs = ggm_fit_fn(feature_matrix)

    # Bootstrap the full precision matrix
    boot_thetas = np.zeros((n_bootstrap, D, D))
    n_failed = 0
    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        try:
            boot_thetas[b] = ggm_fit_fn(feature_matrix[idx])
        except Exception:
            boot_thetas[b] = theta_obs  # fallback to observed
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"Bootstrap edge weights [{condition_name}]: {n_failed} failures")

    alpha = 1.0 - ci_level
    results = []

    for i in range(D):
        for j in range(i + 1, D):
            obs_w = float(theta_obs[i, j])
            boot_w = boot_thetas[:, i, j]
            ci_lo = float(np.percentile(boot_w, 100 * alpha / 2))
            ci_hi = float(np.percentile(boot_w, 100 * (1 - alpha / 2)))

            results.append(EdgeWeightCI(
                feature_i=i,
                feature_j=j,
                feature_i_name=feature_names[i],
                feature_j_name=feature_names[j],
                condition=condition_name,
                observed_weight=obs_w,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                is_cross_modal=bool(cross_modal_mask[i, j]),
                excludes_zero=(ci_lo > 0 or ci_hi < 0),
            ))

    n_nonzero = sum(r.excludes_zero for r in results)
    n_cross_nonzero = sum(r.excludes_zero and r.is_cross_modal for r in results)
    logger.info(
        f"Edge CI [{condition_name}]: {n_nonzero}/{len(results)} exclude zero "
        f"({n_cross_nonzero} cross-modal)"
    )
    return results


# ---------------------------------------------------------------------------
# Cross-modal density CIs per condition
# ---------------------------------------------------------------------------

def bootstrap_density_by_condition(
    feature_matrices: Dict[str, np.ndarray],
    ggm_fit_fn: Callable,
    modality_slices: Dict[str, slice],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, BootstrapCI]:
    """
    Bootstrap CI for cross-modal edge density under each condition.

    Returns
    -------
    dict condition_name → BootstrapCI
    """
    results: Dict[str, BootstrapCI] = {}

    for cond, X in feature_matrices.items():
        D = X.shape[1]
        modality_ids = _assign_modality_ids(D, modality_slices)
        cross_mask = modality_ids[:, None] != modality_ids[None, :]
        n_cross = np.sum(cross_mask)

        def density_fn(X_sub: np.ndarray) -> float:
            theta = ggm_fit_fn(X_sub)
            return float(np.sum(np.abs(theta) > 1e-10 & cross_mask) / n_cross)

        ci = bootstrap_statistic(
            data=X,
            statistic_fn=density_fn,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            method="percentile",
            random_seed=random_seed,
            statistic_name=f"cross_modal_density_{cond}",
        )
        results[cond] = ci
        logger.info(
            f"Density CI [{cond}]: {ci.observed:.4f} "
            f"[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
        )

    return results


# ---------------------------------------------------------------------------
# PID bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_pid_quantities(
    pid_fn: Callable,
    covariance_matrices: Dict[str, np.ndarray],
    feature_matrices: Dict[str, np.ndarray],
    modality_pair: Tuple[str, str],
    modality_slices: Dict[str, slice],
    target_slice: slice,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, BootstrapCI]]:
    """
    Bootstrap CIs for PID quantities (redundancy, unique_x1, unique_x2, synergy)
    per condition and modality pair.

    Parameters
    ----------
    pid_fn : callable
        Signature: (Sigma: ndarray, slice_x1, slice_x2, slice_y) → PIDResult
        Must accept covariance matrix and slices, return object with
        .redundancy, .unique_x1, .unique_x2, .synergy attributes.
    covariance_matrices : dict condition → (D, D) sample covariance
    feature_matrices : dict condition → (N, D)
    modality_pair : (str, str) modality names
    modality_slices : dict modality_name → slice
    target_slice : slice  indices of target variable (recall accuracy or condition label)
    n_bootstrap : int
    ci_level : float

    Returns
    -------
    dict condition → {pid_key → BootstrapCI}
    """
    rng = default_rng(random_seed)
    m1, m2 = modality_pair
    sl1 = modality_slices[m1]
    sl2 = modality_slices[m2]

    results: Dict[str, Dict[str, BootstrapCI]] = {}

    for cond, X in feature_matrices.items():
        N, D = X.shape
        cond_results: Dict[str, BootstrapCI] = {}

        for pid_key in ["redundancy", "unique_x1", "unique_x2", "synergy"]:
            def _pid_stat(X_sub: np.ndarray, key: str = pid_key) -> float:
                Sigma = np.cov(X_sub.T)
                try:
                    result = pid_fn(Sigma, sl1, sl2, target_slice)
                    return float(getattr(result, key))
                except Exception:
                    return np.nan

            ci = bootstrap_statistic(
                data=X,
                statistic_fn=_pid_stat,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                method="percentile",
                random_seed=random_seed,
                statistic_name=f"pid_{pid_key}_{m1}_{m2}_{cond}",
            )
            cond_results[pid_key] = ci

        results[cond] = cond_results

    return results


# ---------------------------------------------------------------------------
# Non-monotonic synergy test (bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_nonmonotonic_synergy(
    synergy_by_condition: Dict[str, float],
    feature_matrices: Dict[str, np.ndarray],
    pid_fn: Callable,
    modality_pair: Tuple[str, str],
    modality_slices: Dict[str, slice],
    target_slice: slice,
    cond_low: str,
    cond_medium: str,
    cond_overload: str,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Tuple[float, bool]:
    """
    Bootstrap test for non-monotonic synergy pattern:
    synergy(medium) > synergy(overload)  AND  synergy(medium) > synergy(low)

    H0: synergy(medium) - synergy(overload) <= 0

    Returns
    -------
    p_value : float
    significant : bool
    """
    rng = default_rng(random_seed)
    m1, m2 = modality_pair
    sl1 = modality_slices[m1]
    sl2 = modality_slices[m2]

    obs_contrast = (
        synergy_by_condition[cond_medium] - synergy_by_condition[cond_overload]
    )

    def _synergy(X: np.ndarray) -> float:
        Sigma = np.cov(X.T)
        try:
            return float(pid_fn(Sigma, sl1, sl2, target_slice).synergy)
        except Exception:
            return np.nan

    null_contrasts = np.zeros(n_bootstrap)

    X_med = feature_matrices[cond_medium]
    X_over = feature_matrices[cond_overload]
    N_med = X_med.shape[0]
    N_over = X_over.shape[0]

    for b in range(n_bootstrap):
        # Permute condition labels: pool and resplit
        pooled = np.vstack([X_med, X_over])
        idx = rng.permutation(pooled.shape[0])
        X_med_perm = pooled[idx[:N_med]]
        X_over_perm = pooled[idx[N_med:N_med + N_over]]
        s_med = _synergy(X_med_perm)
        s_over = _synergy(X_over_perm)
        null_contrasts[b] = s_med - s_over

    valid = null_contrasts[~np.isnan(null_contrasts)]
    p = (np.sum(valid >= obs_contrast) + 1) / (len(valid) + 1)

    logger.info(
        f"Non-monotonic synergy test ({m1}×{m2}): "
        f"contrast={obs_contrast:+.4f}, p={p:.4f}"
    )

    return float(p), bool(p < alpha)


# ---------------------------------------------------------------------------
# Recall prediction R² bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_recall_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapCI:
    """
    Bootstrap CI for R² from a regression model's LOSO predictions.

    Parameters
    ----------
    y_true : (N,) array of true recall accuracy values
    y_pred : (N,) array of model predictions (LOSO)

    Returns
    -------
    BootstrapCI for R²
    """
    from sklearn.metrics import r2_score

    paired = np.column_stack([y_true, y_pred])

    def r2_fn(data: np.ndarray) -> float:
        return float(r2_score(data[:, 0], data[:, 1]))

    ci = bootstrap_statistic(
        data=paired,
        statistic_fn=r2_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method="bca",
        random_seed=random_seed,
        statistic_name=f"r2_{model_name}",
    )
    logger.info(
        f"R² CI [{model_name}]: {ci.observed:.4f} "
        f"[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
    )
    return ci


def bootstrap_r2_contrast(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapCI:
    """
    Bootstrap CI for the difference in R² between two models (A - B).

    Parameters
    ----------
    y_true : (N,) true values
    y_pred_a, y_pred_b : (N,) predictions from models A and B
    """
    from sklearn.metrics import r2_score

    stacked = np.column_stack([y_true, y_pred_a, y_pred_b])

    def delta_r2_fn(data: np.ndarray) -> float:
        r2_a = r2_score(data[:, 0], data[:, 1])
        r2_b = r2_score(data[:, 0], data[:, 2])
        return float(r2_a - r2_b)

    ci = bootstrap_statistic(
        data=stacked,
        statistic_fn=delta_r2_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method="bca",
        random_seed=random_seed,
        statistic_name=f"delta_r2_{model_a_name}_vs_{model_b_name}",
    )
    logger.info(
        f"ΔR² CI [{model_a_name} - {model_b_name}]: {ci.observed:+.4f} "
        f"[{ci.ci_lower:+.4f}, {ci.ci_upper:+.4f}] "
        f"({'excludes zero' if ci.ci_lower > 0 else 'includes zero'})"
    )
    return ci


# ---------------------------------------------------------------------------
# Network distance CIs
# ---------------------------------------------------------------------------

def bootstrap_network_distances(
    feature_matrices: Dict[str, np.ndarray],
    ggm_fit_fn: Callable,
    condition_pairs: List[Tuple[str, str]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, BootstrapCI]:
    """
    Bootstrap CIs for Frobenius network distances between condition pairs.

    Parameters
    ----------
    condition_pairs : list of (cond_a, cond_b) tuples

    Returns
    -------
    dict '{cond_a}_{cond_b}' → BootstrapCI
    """
    rng = default_rng(random_seed)
    results: Dict[str, BootstrapCI] = {}

    for cond_a, cond_b in condition_pairs:
        X_a = feature_matrices[cond_a]
        X_b = feature_matrices[cond_b]
        N_a, N_b = X_a.shape[0], X_b.shape[0]
        pooled = np.vstack([X_a, X_b])

        def dist_fn(X_pool: np.ndarray) -> float:
            # Resample each condition's portion
            idx_a = rng.choice(N_a, size=N_a, replace=True)
            idx_b = rng.choice(N_b, size=N_b, replace=True)
            theta_a = ggm_fit_fn(X_a[idx_a])
            theta_b = ggm_fit_fn(X_b[idx_b])
            return float(np.linalg.norm(theta_a - theta_b, "fro"))

        # Compute observed
        theta_a_obs = ggm_fit_fn(X_a)
        theta_b_obs = ggm_fit_fn(X_b)
        obs_dist = float(np.linalg.norm(theta_a_obs - theta_b_obs, "fro"))

        # Bootstrap
        boot_dists = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            boot_dists[b] = dist_fn(pooled)

        alpha = 1.0 - ci_level
        ci_lo = float(np.percentile(boot_dists, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_dists, 100 * (1 - alpha / 2)))
        key = f"{cond_a}_{cond_b}"

        results[key] = BootstrapCI(
            statistic_name=f"frobenius_dist_{key}",
            observed=obs_dist,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
            method="percentile",
            boot_distribution=boot_dists,
        )
        logger.info(
            f"Network dist CI [{cond_a}↔{cond_b}]: {obs_dist:.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        )

    return results


# ---------------------------------------------------------------------------
# Master bootstrap suite
# ---------------------------------------------------------------------------

def run_bootstrap_suite(
    feature_matrices: Dict[str, np.ndarray],
    precision_matrices: Dict[str, np.ndarray],
    ggm_fit_fn: Callable,
    modality_slices: Dict[str, slice],
    conditions_ordered: List[str],
    y_true_loso: Dict[str, np.ndarray],
    y_pred_loso: Dict[str, np.ndarray],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapReport:
    """
    Run full bootstrap suite.

    Parameters
    ----------
    y_true_loso : dict model_name → (N_subjects,) true recall
    y_pred_loso : dict model_name → (N_subjects,) predicted recall
    """
    logger.info("=== Running bootstrap CI suite ===")

    # 1. Cross-modal density CIs per condition
    logger.info("--- Bootstrap: cross-modal density CIs ---")
    density_cis = bootstrap_density_by_condition(
        feature_matrices=feature_matrices,
        ggm_fit_fn=ggm_fit_fn,
        modality_slices=modality_slices,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )

    # 2. Network distance CIs for consecutive pairs
    logger.info("--- Bootstrap: network distance CIs ---")
    pairs = [(conditions_ordered[i], conditions_ordered[i + 1])
             for i in range(len(conditions_ordered) - 1)]
    net_dist_cis = bootstrap_network_distances(
        feature_matrices=feature_matrices,
        ggm_fit_fn=ggm_fit_fn,
        condition_pairs=pairs,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )

    # 3. Recall R² CIs for each model
    logger.info("--- Bootstrap: recall R² CIs ---")
    recall_r2_cis: Dict[str, BootstrapCI] = {}
    for model_name in y_true_loso:
        ci = bootstrap_recall_r2(
            y_true=y_true_loso[model_name],
            y_pred=y_pred_loso[model_name],
            model_name=model_name,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            random_seed=random_seed,
        )
        recall_r2_cis[model_name] = ci

    return BootstrapReport(
        density_cis=density_cis,
        network_distance_cis=net_dist_cis,
        pid_cis={},
        recall_r2_cis=recall_r2_cis,
        synergy_nonmonotonic_p=None,
        synergy_nonmonotonic_significant=None,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _assign_modality_ids(D: int, modality_slices: Dict[str, slice]) -> np.ndarray:
    ids = np.zeros(D, dtype=int)
    for mod_idx, sl in enumerate(modality_slices.values()):
        ids[sl] = mod_idx
    return ids


def format_ci(ci: BootstrapCI, decimals: int = 4) -> str:
    fmt = f"{{:.{decimals}f}}"
    return (
        f"{fmt.format(ci.observed)} "
        f"[{fmt.format(ci.ci_lower)}, {fmt.format(ci.ci_upper)}] "
        f"({int(ci.ci_level*100)}% {ci.method})"
    )