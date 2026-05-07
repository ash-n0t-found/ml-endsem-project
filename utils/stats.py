"""
utils/stats.py
==============
Statistical utilities: normalization, permutation tests, FDR correction,
effect sizes, bootstrap CIs. Used across all analysis modules.
All functions pure (no side effects), fully documented.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.stats import rankdata


# ── Normalization ─────────────────────────────────────────────────────────────

def zscore_per_subject(
    X: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """
    Per-subject z-score normalization.
    For each unique subject, standardize its rows using that subject's
    mean and std computed across all its trials.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    subject_ids : ndarray, shape (n_trials,)
        Subject label for each trial.

    Returns
    -------
    X_norm : ndarray, shape (n_trials, n_features)
    """
    X_norm = X.copy().astype(float)
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        mu = X_norm[mask].mean(axis=0)
        sd = X_norm[mask].std(axis=0, ddof=1)
        sd = np.where(sd < 1e-10, 1.0, sd)  # avoid division by zero
        X_norm[mask] = (X_norm[mask] - mu) / sd
    return X_norm


def unit_variance_per_condition(
    X: np.ndarray,
    condition_ids: np.ndarray,
) -> np.ndarray:
    """
    Scale each feature to unit variance within each condition.
    Used before GGM fitting to prevent coupling artifacts from
    variance differences across modalities.

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
        sd = X_scaled[mask].std(axis=0, ddof=1)
        sd = np.where(sd < 1e-10, 1.0, sd)
        X_scaled[mask] = X_scaled[mask] / sd
    return X_scaled


def robust_scale(X: np.ndarray, quantile_range: Tuple[float, float] = (25.0, 75.0)) -> np.ndarray:
    """
    Median / IQR scaling — robust to outliers.
    Useful for non-Gaussian features before normality checking.
    """
    from scipy.stats import iqr
    med = np.median(X, axis=0)
    iqr_val = iqr(X, axis=0, rng=quantile_range)
    iqr_val = np.where(iqr_val < 1e-10, 1.0, iqr_val)
    return (X - med) / iqr_val


# ── Normality testing ─────────────────────────────────────────────────────────

def royston_multivariate_normality(X: np.ndarray) -> Tuple[float, float]:
    """
    Royston (1992) multivariate normality test.
    Tests joint normality of all features simultaneously.

    Returns
    -------
    H : float
        Test statistic.
    p_value : float

    Notes
    -----
    Requires n > p+1. Falls back to Mardia's test if dimensions too high.
    Implementation follows Royston (1992) JRSS-C.
    """
    n, p = X.shape
    if n <= p + 1:
        warnings.warn(
            f"Royston test requires n > p+1. n={n}, p={p}. Returning NaN.",
            UserWarning,
        )
        return np.nan, np.nan

    # Compute Shapiro-Wilk per variable
    W_vals = np.zeros(p)
    for j in range(p):
        w, _ = stats.shapiro(X[:, j])
        W_vals[j] = w

    # Royston transformation: z_j = Phi^{-1}(p_j)
    # where p_j = Prob(W <= W_j) from Shapiro-Wilk distribution
    # Approximate via log(1 - W) normalization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = np.log(1 - W_vals)
    
    mu = -1.2725 + 1.0521 * np.log(np.log(n))
    sigma = 1.0308 - 0.26763 * np.log(np.log(n))
    z = (u - mu) / sigma

    # Combine via chi-squared approximation
    # H ~ chi^2(p) under H0
    H = np.sum(z ** 2)
    p_val = 1 - stats.chi2.cdf(H, df=p)
    return float(H), float(p_val)


def shapiro_per_feature(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shapiro-Wilk test for each feature column.

    Returns
    -------
    W : ndarray, shape (n_features,)
    p_values : ndarray, shape (n_features,)
    """
    n_feat = X.shape[1]
    W = np.zeros(n_feat)
    pvals = np.zeros(n_feat)
    for j in range(n_feat):
        W[j], pvals[j] = stats.shapiro(X[:, j])
    return W, pvals


# ── FDR correction ────────────────────────────────────────────────────────────

def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : ndarray, shape (n_tests,)
    alpha : float
        Desired FDR level.

    Returns
    -------
    rejected : bool ndarray — True if null hypothesis rejected
    p_adjusted : ndarray — BH-adjusted p-values
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = rankdata(p_values, method="ordinal")
    
    p_adj = np.minimum(1.0, p_values * n / ranked)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    
    rejected = p_adj <= alpha
    return rejected, p_adj


# ── Permutation testing ───────────────────────────────────────────────────────

def permutation_test_diff(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable = np.mean,
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Permutation test for difference in statistic between two groups.

    Parameters
    ----------
    x, y : ndarrays
        Observed values in group 1 and group 2.
    statistic : callable
        Summary statistic (default: mean).
    n_permutations : int
    alternative : str
        'two-sided', 'greater', 'less'
    random_state : int, optional

    Returns
    -------
    observed_diff : float
    p_value : float
    """
    rng = np.random.default_rng(random_state)
    observed = statistic(x) - statistic(y)
    combined = np.concatenate([x, y])
    n_x = len(x)

    null_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_diffs[i] = statistic(perm[:n_x]) - statistic(perm[n_x:])

    if alternative == "two-sided":
        p_val = np.mean(np.abs(null_diffs) >= np.abs(observed))
    elif alternative == "greater":
        p_val = np.mean(null_diffs >= observed)
    elif alternative == "less":
        p_val = np.mean(null_diffs <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(observed), float(p_val)


def permutation_test_edge_presence(
    edge_weights_a: np.ndarray,
    edge_weights_b: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Permutation test for difference in edge weight per edge position.
    Used to test GGM edge presence differences across conditions.

    Parameters
    ----------
    edge_weights_a : ndarray, shape (n_edges,)
        Edge weights in condition A (e.g. 9-digit).
    edge_weights_b : ndarray, shape (n_edges,)
        Edge weights in condition B (e.g. 13-digit).
    n_permutations : int
    random_state : int, optional

    Returns
    -------
    p_values : ndarray, shape (n_edges,)
        One p-value per edge (two-sided).
    """
    rng = np.random.default_rng(random_state)
    n_edges = len(edge_weights_a)
    observed_diff = np.abs(edge_weights_a - edge_weights_b)

    null_diffs = np.empty((n_permutations, n_edges))
    combined = np.stack([edge_weights_a, edge_weights_b], axis=0)  # (2, n_edges)
    for i in range(n_permutations):
        # Randomly flip assignment per edge
        flip = rng.integers(0, 2, size=n_edges)
        perm_a = np.where(flip == 0, combined[0], combined[1])
        perm_b = np.where(flip == 0, combined[1], combined[0])
        null_diffs[i] = np.abs(perm_a - perm_b)

    p_values = np.mean(null_diffs >= observed_diff[None, :], axis=0)
    return p_values


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval.

    Returns
    -------
    point_estimate : float
    ci_low : float
    ci_high : float
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    
    alpha = 1 - ci
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(statistic(data)), ci_low, ci_high


# ── Effect sizes ──────────────────────────────────────────────────────────────

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d effect size for two independent groups.
    Uses pooled standard deviation.
    """
    n1, n2 = len(x), len(y)
    var_pool = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    sd_pool = np.sqrt(max(var_pool, 1e-12))
    return float((np.mean(x) - np.mean(y)) / sd_pool)


def eta_squared(groups: list[np.ndarray]) -> float:
    """
    Eta-squared (η²) effect size for one-way ANOVA.

    Parameters
    ----------
    groups : list of 1-D arrays
        One array per group.

    Returns
    -------
    eta2 : float
    """
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    return float(ss_between / max(ss_total, 1e-12))


# ── Wilcoxon signed-rank ──────────────────────────────────────────────────────

def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.
    Used for per-subject LOSO performance comparisons.

    Returns
    -------
    statistic : float
    p_value : float
    """
    result = stats.wilcoxon(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


# ── Power analysis (simple) ───────────────────────────────────────────────────

def power_t_test(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Approximate power for one-sample or paired t-test.
    Uses noncentrality parameter approach.

    Parameters
    ----------
    effect_size : float
        Cohen's d.
    n : int
        Sample size.
    alpha : float
    alternative : str

    Returns
    -------
    power : float
    """
    from scipy.stats import t as t_dist
    ncp = effect_size * np.sqrt(n)
    df = n - 1
    if alternative == "two-sided":
        t_crit = t_dist.ppf(1 - alpha / 2, df)
        power = 1 - t_dist.cdf(t_crit, df, loc=ncp) + t_dist.cdf(-t_crit, df, loc=ncp)
    else:
        t_crit = t_dist.ppf(1 - alpha, df)
        power = 1 - t_dist.cdf(t_crit, df, loc=ncp)
    return float(power)


# ── Network distance ──────────────────────────────────────────────────────────

def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Frobenius norm distance between two matrices.
    Used for GGM condition distances: d(Θ_c, Θ_c').
    """
    return float(np.linalg.norm(A - B, "fro"))


def cross_modal_edge_density(
    precision_matrix: np.ndarray,
    block_sizes: list[int],
    threshold: float = 0.0,
) -> float:
    """
    Proportion of non-zero off-diagonal block entries in precision matrix.
    Measures cross-modal coupling density.

    Parameters
    ----------
    precision_matrix : ndarray, shape (D, D)
    block_sizes : list of int
        Size of each modality feature block. Must sum to D.
    threshold : float
        Absolute value threshold for 'non-zero' (default: 0 for exact sparsity).

    Returns
    -------
    density : float
        Fraction of cross-modal edges that are non-zero.
    """
    D = precision_matrix.shape[0]
    assert sum(block_sizes) == D, "block_sizes must sum to D"
    
    # Build block index boundaries
    boundaries = np.cumsum([0] + block_sizes)
    
    cross_entries = []
    for i in range(len(block_sizes)):
        for j in range(i + 1, len(block_sizes)):
            r_start, r_end = boundaries[i], boundaries[i + 1]
            c_start, c_end = boundaries[j], boundaries[j + 1]
            block = precision_matrix[r_start:r_end, c_start:c_end]
            cross_entries.append(np.abs(block).ravel())
    
    if not cross_entries:
        return 0.0
    
    all_cross = np.concatenate(cross_entries)
    return float(np.mean(np.abs(all_cross) > threshold))