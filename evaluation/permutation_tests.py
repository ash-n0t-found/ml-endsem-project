"""
evaluation/permutation_tests.py
================================
Permutation-based statistical tests for GGM edge significance and
cross-modal coupling structure.

Tests implemented
-----------------
1. Edge presence permutation test:
   H0: edge (i,j) in Θ̂_c is no stronger than chance
   → permute condition labels, refit GGM, compare edge weights

2. Cross-modal edge density test:
   H0: density(Θ̂_9digit cross-block) == density(Θ̂_13digit cross-block)
   → permute condition labels within subjects

3. Non-monotonic coupling test:
   H0: edge_density(9) - edge_density(13) <= 0
   → one-sided permutation test on the contrast

4. Network distance test:
   H0: ||Θ̂_9 - Θ̂_13||_F == ||Θ̂_5 - Θ̂_9||_F (distances equal)
   → permute condition labels, compare Frobenius distances

All p-values corrected via Benjamini-Hochberg FDR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EdgePermResult:
    """Result for a single edge permutation test."""
    feature_i: int
    feature_j: int
    feature_i_name: str
    feature_j_name: str
    observed_weight: float
    observed_abs_weight: float
    p_value: float
    p_value_fdr: float
    significant: bool
    n_permutations: int
    is_cross_modal: bool


@dataclass
class DensityContrastResult:
    """Result for a condition-pair density contrast test."""
    condition_a: str
    condition_b: str
    observed_density_a: float
    observed_density_b: float
    observed_contrast: float   # density_a - density_b
    p_value: float
    significant: bool
    n_permutations: int
    null_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class NetworkDistanceResult:
    """Result for Frobenius distance comparison between condition pairs."""
    pair_ab: Tuple[str, str]
    pair_cd: Tuple[str, str]
    dist_ab: float
    dist_cd: float
    observed_contrast: float    # dist_ab - dist_cd
    p_value: float
    significant: bool
    n_permutations: int


@dataclass
class PermutationTestSuite:
    """Aggregated results from all permutation tests."""
    edge_tests: List[EdgePermResult]
    density_contrasts: List[DensityContrastResult]
    network_distances: List[NetworkDistanceResult]
    n_significant_edges: int
    fdr_threshold: float


# ---------------------------------------------------------------------------
# Core permutation engine
# ---------------------------------------------------------------------------

def _permute_condition_labels(
    feature_matrices: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Permute condition labels within the pooled feature matrix.

    Concatenates all conditions, shuffles rows randomly, then splits
    back to original condition sizes. Preserves within-condition sample
    sizes — destroys condition-feature associations.

    Parameters
    ----------
    feature_matrices : dict condition_name → (N_c, D) array
    rng : numpy Generator

    Returns
    -------
    permuted_matrices : dict condition_name → (N_c, D) array
    """
    conditions = list(feature_matrices.keys())
    sizes = [feature_matrices[c].shape[0] for c in conditions]
    pooled = np.vstack([feature_matrices[c] for c in conditions])

    # Shuffle rows
    idx = rng.permutation(pooled.shape[0])
    pooled_perm = pooled[idx]

    # Split back
    perm_matrices: Dict[str, np.ndarray] = {}
    start = 0
    for cond, size in zip(conditions, sizes):
        perm_matrices[cond] = pooled_perm[start : start + size]
        start += size

    return perm_matrices


# ---------------------------------------------------------------------------
# Edge permutation test
# ---------------------------------------------------------------------------

def permutation_test_edges(
    feature_matrices: Dict[str, np.ndarray],
    precision_matrices: Dict[str, np.ndarray],
    condition_a: str,
    condition_b: str,
    ggm_fit_fn,
    modality_slices: Dict[str, slice],
    n_permutations: int = 1000,
    alpha_fdr: float = 0.05,
    feature_names: Optional[List[str]] = None,
    random_seed: int = 42,
) -> List[EdgePermResult]:
    """
    Permutation test for each off-diagonal edge in the GGM precision matrix.

    For each feature pair (i, j), tests whether the observed difference in
    edge weight between condition_a and condition_b exceeds chance.

    H0: |Θ̂_a[i,j] - Θ̂_b[i,j]| = 0 (no condition effect on edge)

    Parameters
    ----------
    feature_matrices : dict condition → (N_c, D)
    precision_matrices : dict condition → (D, D) observed precision matrices
    condition_a, condition_b : str
        Conditions to compare.
    ggm_fit_fn : callable
        Signature: (X: ndarray) → precision_matrix: ndarray
        Must accept (N, D) feature matrix, return (D, D) precision matrix.
    modality_slices : dict modality_name → slice
        Slice objects indicating which feature indices belong to each modality.
        Used to flag cross-modal edges.
    n_permutations : int
    alpha_fdr : float
    feature_names : list of str or None
    random_seed : int

    Returns
    -------
    List[EdgePermResult], one per unique pair (i < j)
    """
    rng = default_rng(random_seed)
    D = precision_matrices[condition_a].shape[0]

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(D)]

    # Build cross-modal mask: True if (i,j) spans different modalities
    cross_modal_mask = _build_cross_modal_mask(D, modality_slices)

    # Observed edge differences
    Theta_a = precision_matrices[condition_a]
    Theta_b = precision_matrices[condition_b]
    observed_diff = np.abs(Theta_a - Theta_b)

    logger.info(
        f"Edge permutation test: {condition_a} vs {condition_b}, "
        f"D={D}, n_perm={n_permutations}"
    )

    # Null distributions for each edge (upper triangle only)
    n_edges = D * (D - 1) // 2
    null_diffs = np.zeros((n_permutations, n_edges))

    for perm_idx in range(n_permutations):
        perm = _permute_condition_labels(
            {condition_a: feature_matrices[condition_a],
             condition_b: feature_matrices[condition_b]},
            rng,
        )
        Theta_a_perm = ggm_fit_fn(perm[condition_a])
        Theta_b_perm = ggm_fit_fn(perm[condition_b])
        perm_diff = np.abs(Theta_a_perm - Theta_b_perm)

        edge_idx = 0
        for i in range(D):
            for j in range(i + 1, D):
                null_diffs[perm_idx, edge_idx] = perm_diff[i, j]
                edge_idx += 1

        if (perm_idx + 1) % 100 == 0:
            logger.debug(f"  permutation {perm_idx + 1}/{n_permutations}")

    # Compute p-values
    raw_p_values = []
    edge_coords = []
    edge_idx = 0
    for i in range(D):
        for j in range(i + 1, D):
            obs = observed_diff[i, j]
            null = null_diffs[:, edge_idx]
            p = (np.sum(null >= obs) + 1) / (n_permutations + 1)
            raw_p_values.append(p)
            edge_coords.append((i, j))
            edge_idx += 1

    # BH-FDR correction
    _, p_fdr, _, _ = multipletests(raw_p_values, alpha=alpha_fdr, method="fdr_bh")

    results = []
    for k, ((i, j), p_raw, p_adj) in enumerate(
        zip(edge_coords, raw_p_values, p_fdr)
    ):
        results.append(EdgePermResult(
            feature_i=i,
            feature_j=j,
            feature_i_name=feature_names[i],
            feature_j_name=feature_names[j],
            observed_weight=Theta_a[i, j] - Theta_b[i, j],
            observed_abs_weight=observed_diff[i, j],
            p_value=float(p_raw),
            p_value_fdr=float(p_adj),
            significant=bool(p_adj < alpha_fdr),
            n_permutations=n_permutations,
            is_cross_modal=bool(cross_modal_mask[i, j]),
        ))

    n_sig = sum(r.significant for r in results)
    n_cross = sum(r.significant and r.is_cross_modal for r in results)
    logger.info(
        f"Significant edges (FDR<{alpha_fdr}): {n_sig}/{len(results)} "
        f"({n_cross} cross-modal)"
    )

    return results


# ---------------------------------------------------------------------------
# Density contrast permutation test
# ---------------------------------------------------------------------------

def permutation_test_density_contrast(
    feature_matrices: Dict[str, np.ndarray],
    precision_matrices: Dict[str, np.ndarray],
    condition_a: str,
    condition_b: str,
    ggm_fit_fn,
    modality_slices: Dict[str, slice],
    n_permutations: int = 1000,
    alternative: str = "greater",
    alpha: float = 0.05,
    random_seed: int = 42,
) -> DensityContrastResult:
    """
    Permutation test for difference in cross-modal edge density between two conditions.

    Primary hypothesis test:
    Cross-modal edge density is non-monotonic → density(9-digit) > density(13-digit).

    H0: density(a) - density(b) <= 0   [one-sided, alternative='greater']
    H0: density(a) == density(b)       [two-sided, alternative='two-sided']

    Parameters
    ----------
    alternative : str
        'greater'  : H1: density(a) > density(b)
        'less'     : H1: density(a) < density(b)
        'two-sided': H1: density(a) != density(b)
    """
    rng = default_rng(random_seed)
    D = precision_matrices[condition_a].shape[0]

    cross_modal_mask = _build_cross_modal_mask(D, modality_slices)
    n_cross_edges = cross_modal_mask.sum()

    def _cross_modal_density(theta: np.ndarray) -> float:
        """Proportion of cross-modal edges with nonzero precision."""
        nonzero = np.abs(theta) > 1e-10
        return float(np.sum(nonzero & cross_modal_mask) / n_cross_edges)

    obs_density_a = _cross_modal_density(precision_matrices[condition_a])
    obs_density_b = _cross_modal_density(precision_matrices[condition_b])
    obs_contrast = obs_density_a - obs_density_b

    logger.info(
        f"Density contrast test: {condition_a} vs {condition_b}, "
        f"observed densities: {obs_density_a:.4f} vs {obs_density_b:.4f}, "
        f"contrast={obs_contrast:+.4f}"
    )

    null_contrasts = np.zeros(n_permutations)

    for perm_idx in range(n_permutations):
        perm = _permute_condition_labels(
            {condition_a: feature_matrices[condition_a],
             condition_b: feature_matrices[condition_b]},
            rng,
        )
        theta_a_perm = ggm_fit_fn(perm[condition_a])
        theta_b_perm = ggm_fit_fn(perm[condition_b])
        null_contrasts[perm_idx] = (
            _cross_modal_density(theta_a_perm)
            - _cross_modal_density(theta_b_perm)
        )

        if (perm_idx + 1) % 200 == 0:
            logger.debug(f"  permutation {perm_idx + 1}/{n_permutations}")

    # P-value
    if alternative == "greater":
        p = (np.sum(null_contrasts >= obs_contrast) + 1) / (n_permutations + 1)
    elif alternative == "less":
        p = (np.sum(null_contrasts <= obs_contrast) + 1) / (n_permutations + 1)
    else:
        p = (np.sum(np.abs(null_contrasts) >= abs(obs_contrast)) + 1) / (n_permutations + 1)

    logger.info(f"Density contrast p={p:.4f} ({alternative})")

    return DensityContrastResult(
        condition_a=condition_a,
        condition_b=condition_b,
        observed_density_a=obs_density_a,
        observed_density_b=obs_density_b,
        observed_contrast=obs_contrast,
        p_value=float(p),
        significant=bool(p < alpha),
        n_permutations=n_permutations,
        null_distribution=null_contrasts,
    )


# ---------------------------------------------------------------------------
# Network distance permutation test
# ---------------------------------------------------------------------------

def permutation_test_network_distances(
    feature_matrices: Dict[str, np.ndarray],
    precision_matrices: Dict[str, np.ndarray],
    conditions: List[str],
    ggm_fit_fn,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> List[NetworkDistanceResult]:
    """
    Test whether distance(9-digit, 13-digit) > distance(5-digit, 9-digit).

    Prediction: Overload causes a larger topological reorganization than the
    low→medium transition. Frobenius norm between precision matrices measures
    this distance.

    Tests all consecutive condition pairs if len(conditions) > 2.
    """
    rng = default_rng(random_seed)

    def _frobenius_dist(theta1: np.ndarray, theta2: np.ndarray) -> float:
        return float(np.linalg.norm(theta1 - theta2, "fro"))

    results = []

    # Test each consecutive pair vs every other consecutive pair
    for i in range(len(conditions) - 2):
        cond_a, cond_b = conditions[i], conditions[i + 1]
        cond_c, cond_d = conditions[i + 1], conditions[i + 2]

        obs_dist_ab = _frobenius_dist(
            precision_matrices[cond_a], precision_matrices[cond_b]
        )
        obs_dist_cd = _frobenius_dist(
            precision_matrices[cond_c], precision_matrices[cond_d]
        )
        obs_contrast = obs_dist_cd - obs_dist_ab  # we predict cd > ab for 9→13 vs 5→9

        logger.info(
            f"Network distance: ||{cond_a}-{cond_b}||={obs_dist_ab:.4f}, "
            f"||{cond_c}-{cond_d}||={obs_dist_cd:.4f}, "
            f"contrast={obs_contrast:+.4f}"
        )

        null_contrasts = np.zeros(n_permutations)
        all_feat = {c: feature_matrices[c] for c in [cond_a, cond_b, cond_c, cond_d]}

        for perm_idx in range(n_permutations):
            perm = _permute_condition_labels(all_feat, rng)
            theta_perm = {c: ggm_fit_fn(perm[c]) for c in [cond_a, cond_b, cond_c, cond_d]}
            perm_dist_ab = _frobenius_dist(theta_perm[cond_a], theta_perm[cond_b])
            perm_dist_cd = _frobenius_dist(theta_perm[cond_c], theta_perm[cond_d])
            null_contrasts[perm_idx] = perm_dist_cd - perm_dist_ab

        # One-sided: H1: cd > ab
        p = (np.sum(null_contrasts >= obs_contrast) + 1) / (n_permutations + 1)

        results.append(NetworkDistanceResult(
            pair_ab=(cond_a, cond_b),
            pair_cd=(cond_c, cond_d),
            dist_ab=obs_dist_ab,
            dist_cd=obs_dist_cd,
            observed_contrast=obs_contrast,
            p_value=float(p),
            significant=bool(p < alpha),
            n_permutations=n_permutations,
        ))
        logger.info(f"  p={p:.4f}")

    return results


# ---------------------------------------------------------------------------
# Stability selection-based edge reliability
# ---------------------------------------------------------------------------

def stability_selection_edges(
    feature_matrix: np.ndarray,
    ggm_fit_fn,
    condition_name: str,
    modality_slices: Dict[str, slice],
    n_subsamples: int = 100,
    subsample_fraction: float = 0.5,
    stability_threshold: float = 0.6,
    random_seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Meinshausen-Bühlmann stability selection for reliable edge identification.

    Each edge is assigned a stability score = fraction of subsamples in which
    that edge is nonzero. Edges with stability >= stability_threshold are
    declared reliably present.

    Parameters
    ----------
    feature_matrix : (N, D)
    ggm_fit_fn : callable (N, D) → (D, D) precision matrix
    condition_name : str for logging
    modality_slices : dict
    n_subsamples : int
    subsample_fraction : float  (Meinshausen & Bühlmann recommend 0.5)
    stability_threshold : float (typically 0.6–0.8)
    random_seed : int
    feature_names : list or None

    Returns
    -------
    dict with keys:
        'stability_matrix': (D, D) float, proportion of subsamples edge nonzero
        'selected_edges': (D, D) bool, stable edges
        'cross_modal_stable': (D, D) bool, stable AND cross-modal
        'stability_scores_cross': 1D array of cross-modal stability values
    """
    rng = default_rng(random_seed)
    N, D = feature_matrix.shape
    subsample_size = int(N * subsample_fraction)

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(D)]

    edge_counts = np.zeros((D, D), dtype=float)

    logger.info(
        f"Stability selection [{condition_name}]: "
        f"{n_subsamples} subsamples, fraction={subsample_fraction}, "
        f"threshold={stability_threshold}"
    )

    for sub_idx in range(n_subsamples):
        idx = rng.choice(N, size=subsample_size, replace=False)
        X_sub = feature_matrix[idx]
        try:
            theta_sub = ggm_fit_fn(X_sub)
            nonzero = np.abs(theta_sub) > 1e-10
            edge_counts += nonzero.astype(float)
        except Exception as e:
            logger.debug(f"  Subsample {sub_idx} failed: {e}")
            continue

        if (sub_idx + 1) % 25 == 0:
            logger.debug(f"  subsample {sub_idx + 1}/{n_subsamples}")

    stability_matrix = edge_counts / n_subsamples
    selected_edges = stability_matrix >= stability_threshold
    np.fill_diagonal(selected_edges, False)

    cross_modal_mask = _build_cross_modal_mask(D, modality_slices)
    cross_modal_stable = selected_edges & cross_modal_mask

    # Extract upper-triangle cross-modal stability scores
    upper = np.triu(np.ones((D, D), dtype=bool), k=1)
    cross_upper = cross_modal_mask & upper
    stability_scores_cross = stability_matrix[cross_upper]

    n_stable = np.sum(selected_edges & upper)
    n_cross_stable = np.sum(cross_modal_stable & upper)
    logger.info(
        f"Stable edges (>=threshold): {n_stable}, "
        f"cross-modal stable: {n_cross_stable}"
    )

    return {
        "stability_matrix": stability_matrix,
        "selected_edges": selected_edges,
        "cross_modal_stable": cross_modal_stable,
        "stability_scores_cross": stability_scores_cross,
        "feature_names": feature_names,
        "condition": condition_name,
        "threshold": stability_threshold,
    }


# ---------------------------------------------------------------------------
# Helper: build cross-modal mask
# ---------------------------------------------------------------------------

def _build_cross_modal_mask(
    D: int,
    modality_slices: Dict[str, slice],
) -> np.ndarray:
    """
    Build boolean (D, D) mask where True = features from different modalities.

    Parameters
    ----------
    D : int
        Total feature dimension.
    modality_slices : dict modality_name → slice
        E.g. {'eeg': slice(0,20), 'ecg': slice(20,30), ...}

    Returns
    -------
    mask : (D, D) bool array, symmetric, diagonal=False
    """
    modality_ids = np.zeros(D, dtype=int)
    for mod_idx, (_, sl) in enumerate(modality_slices.items()):
        modality_ids[sl] = mod_idx

    mask = modality_ids[:, None] != modality_ids[None, :]
    np.fill_diagonal(mask, False)
    return mask


# ---------------------------------------------------------------------------
# Master test suite runner
# ---------------------------------------------------------------------------

def run_full_permutation_suite(
    feature_matrices: Dict[str, np.ndarray],
    precision_matrices: Dict[str, np.ndarray],
    ggm_fit_fn,
    modality_slices: Dict[str, slice],
    conditions_ordered: List[str],
    n_permutations: int = 1000,
    alpha_fdr: float = 0.05,
    feature_names: Optional[List[str]] = None,
    random_seed: int = 42,
) -> PermutationTestSuite:
    """
    Run all permutation tests in the evaluation protocol.

    Runs:
    - Edge permutation test: 9-digit vs 13-digit (primary hypothesis)
    - Density contrast test: 9-digit > 13-digit (non-monotonic prediction)
    - Network distance test: ||9-13|| > ||5-9||

    Parameters
    ----------
    conditions_ordered : list of str
        In load order, e.g. ['control', 'load_5', 'load_9', 'load_13']

    Returns
    -------
    PermutationTestSuite
    """
    logger.info("=== Running full permutation test suite ===")

    # Find the 9-digit and 13-digit condition keys
    cond_medium = _find_condition(conditions_ordered, "9")
    cond_overload = _find_condition(conditions_ordered, "13")
    cond_low = _find_condition(conditions_ordered, "5")

    # 1. Edge permutation test (medium vs overload)
    logger.info("--- Test 1: Edge permutation test (9-digit vs 13-digit) ---")
    edge_results = permutation_test_edges(
        feature_matrices=feature_matrices,
        precision_matrices=precision_matrices,
        condition_a=cond_medium,
        condition_b=cond_overload,
        ggm_fit_fn=ggm_fit_fn,
        modality_slices=modality_slices,
        n_permutations=n_permutations,
        alpha_fdr=alpha_fdr,
        feature_names=feature_names,
        random_seed=random_seed,
    )

    # 2. Density contrast: 9 > 13 (primary non-monotonic prediction)
    logger.info("--- Test 2: Density contrast (9-digit > 13-digit) ---")
    density_results = []
    if cond_medium and cond_overload:
        dc = permutation_test_density_contrast(
            feature_matrices=feature_matrices,
            precision_matrices=precision_matrices,
            condition_a=cond_medium,
            condition_b=cond_overload,
            ggm_fit_fn=ggm_fit_fn,
            modality_slices=modality_slices,
            n_permutations=n_permutations,
            alternative="greater",
            alpha=alpha_fdr,
            random_seed=random_seed,
        )
        density_results.append(dc)

    # Also test 5 < 9 (confirming the rise)
    if cond_low and cond_medium:
        dc2 = permutation_test_density_contrast(
            feature_matrices=feature_matrices,
            precision_matrices=precision_matrices,
            condition_a=cond_medium,
            condition_b=cond_low,
            ggm_fit_fn=ggm_fit_fn,
            modality_slices=modality_slices,
            n_permutations=n_permutations,
            alternative="greater",
            alpha=alpha_fdr,
            random_seed=random_seed,
        )
        density_results.append(dc2)

    # 3. Network distance test
    logger.info("--- Test 3: Network distance (||9-13|| > ||5-9||) ---")
    dist_results = []
    if cond_low and cond_medium and cond_overload:
        dist_results = permutation_test_network_distances(
            feature_matrices=feature_matrices,
            precision_matrices=precision_matrices,
            conditions=[cond_low, cond_medium, cond_overload],
            ggm_fit_fn=ggm_fit_fn,
            n_permutations=n_permutations,
            alpha=alpha_fdr,
            random_seed=random_seed,
        )

    n_sig = sum(r.significant for r in edge_results)

    return PermutationTestSuite(
        edge_tests=edge_results,
        density_contrasts=density_results,
        network_distances=dist_results,
        n_significant_edges=n_sig,
        fdr_threshold=alpha_fdr,
    )


def _find_condition(conditions: List[str], digit_str: str) -> Optional[str]:
    """Find condition key containing a digit substring."""
    for c in conditions:
        if digit_str in c:
            return c
    logger.warning(f"No condition found containing '{digit_str}' in {conditions}")
    return None


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def summarize_permutation_results(suite: PermutationTestSuite) -> str:
    """Generate a human-readable summary of permutation test results."""
    lines = [
        "=" * 60,
        "PERMUTATION TEST SUITE SUMMARY",
        "=" * 60,
        f"FDR threshold: {suite.fdr_threshold}",
        f"Significant edges (9-digit vs 13-digit): {suite.n_significant_edges}",
        "",
        "--- CROSS-MODAL DENSITY CONTRASTS ---",
    ]

    for dc in suite.density_contrasts:
        sig_str = "SIGNIFICANT" if dc.significant else "not significant"
        lines.append(
            f"  {dc.condition_a} vs {dc.condition_b}: "
            f"density {dc.observed_density_a:.4f} vs {dc.observed_density_b:.4f}, "
            f"contrast={dc.observed_contrast:+.4f}, p={dc.p_value:.4f} [{sig_str}]"
        )

    lines += ["", "--- NETWORK DISTANCES ---"]
    for nd in suite.network_distances:
        sig_str = "SIGNIFICANT" if nd.significant else "not significant"
        lines.append(
            f"  ||{nd.pair_ab[0]}-{nd.pair_ab[1]}||={nd.dist_ab:.4f}, "
            f"||{nd.pair_cd[0]}-{nd.pair_cd[1]}||={nd.dist_cd:.4f}, "
            f"contrast={nd.observed_contrast:+.4f}, p={nd.p_value:.4f} [{sig_str}]"
        )

    lines += ["", "--- TOP SIGNIFICANT EDGES (FDR-corrected) ---"]
    sig_edges = sorted(
        [e for e in suite.edge_tests if e.significant],
        key=lambda e: e.p_value_fdr,
    )[:20]

    for e in sig_edges:
        modal_str = "[cross-modal]" if e.is_cross_modal else "[within-modal]"
        lines.append(
            f"  ({e.feature_i_name}, {e.feature_j_name}): "
            f"Δw={e.observed_weight:+.4f}, p_fdr={e.p_value_fdr:.4f} {modal_str}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)