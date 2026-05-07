"""
analysis/topology.py
====================

Network topology analysis of GGM precision matrices.

Computes graph-theoretic properties of condition-specific coupling networks.
Primary test: cross-modal edge density is non-monotonic in cognitive load.

Key metrics:
  - Cross-modal edge density per modality pair per condition
  - Network diameter, clustering coefficient, betweenness centrality
  - Frobenius-norm network distance between conditions
  - Per-modality degree distribution

Prediction (testing via permutation):
  Distance(9-digit, 13-digit) > Distance(5-digit, 9-digit)
  → overload = topological reorganization, not just amplitude change

Dependencies: numpy, scipy, networkx (optional — falls back if not available)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from models.ggm import GGMResult, ConditionGGMs, cross_modal_edge_density, network_distance
from utils.io_utils import setup_logger

logger = setup_logger(__name__)

# Optional networkx import
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx not installed. Graph centrality metrics unavailable.")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TopologyMetrics:
    """Topology of one precision matrix graph."""
    condition: int

    # Cross-modal edge density per pair (primary metric)
    cross_modal_densities: Dict[str, float]      # "mod_a-mod_b" → density
    total_cross_modal_density: float             # mean across all pairs

    # Within-modality density (control: should not drive cross-modal changes)
    within_modal_densities: Dict[str, float]     # modality → density

    # Graph properties (if networkx available)
    clustering_coef: Optional[float] = None
    diameter: Optional[int] = None
    betweenness: Optional[Dict[str, float]] = None   # feature_name → centrality

    # Edge weight statistics
    mean_cross_modal_weight: float = 0.0
    std_cross_modal_weight: float = 0.0
    n_nonzero_cross_modal: int = 0
    n_total_cross_modal: int = 0


@dataclass
class TopologyComparison:
    """Pairwise network distance + topology comparison between conditions."""
    condition_a: int
    condition_b: int
    network_distance: float                 # ||Theta_a - Theta_b||_F
    delta_cross_modal_density: float        # density_b - density_a
    delta_per_pair: Dict[str, float]        # pair → density change


@dataclass
class TopologyReport:
    """Full topology report across all conditions."""
    condition_metrics: Dict[int, TopologyMetrics]
    pairwise_comparisons: Dict[str, TopologyComparison]  # "cA_cB" → comparison
    non_monotonic_confirmed: bool           # primary hypothesis test result
    non_monotonic_pvalue: float
    network_distances: Dict[str, float]     # "cA_cB" → Frobenius distance


# ── Core topology extraction ───────────────────────────────────────────────────

def compute_topology(
    ggm_result: GGMResult,
    modality_pairs: List[Tuple[str, str]],
) -> TopologyMetrics:
    """
    Compute topology metrics for one condition's GGM.

    Parameters
    ----------
    ggm_result : GGMResult
    modality_pairs : list of (mod_a, mod_b)

    Returns
    -------
    TopologyMetrics
    """
    prec = ggm_result.precision_matrix
    blocks = ggm_result.modality_blocks
    condition = ggm_result.condition

    # Cross-modal edge densities
    cross_densities = {}
    all_cross_weights = []

    for mod_a, mod_b in modality_pairs:
        if mod_a not in blocks or mod_b not in blocks:
            continue

        key = f"{mod_a}-{mod_b}"
        density = cross_modal_edge_density(prec, blocks, mod_a, mod_b)
        cross_densities[key] = density

        # Weights for statistics
        sa, ea = blocks[mod_a]
        sb, eb = blocks[mod_b]
        block = prec[sa:ea, sb:eb]
        all_cross_weights.extend(block.ravel().tolist())

    total_density = float(np.mean(list(cross_densities.values()))) if cross_densities else 0.0

    # Within-modal densities
    within_densities = {}
    for mod in blocks:
        s, e = blocks[mod]
        block = prec[s:e, s:e]
        D_mod = e - s
        if D_mod <= 1:
            within_densities[mod] = 0.0
            continue
        # Off-diagonal within block
        mask = np.ones((D_mod, D_mod), dtype=bool)
        np.fill_diagonal(mask, False)
        n_offdiag = mask.sum()
        n_nonzero = (np.abs(block[mask]) > 1e-10).sum()
        within_densities[mod] = float(n_nonzero) / n_offdiag

    # Weight statistics
    weights = np.abs(all_cross_weights)
    n_nonzero = int((weights > 1e-10).sum())
    n_total = len(weights)

    metrics = TopologyMetrics(
        condition=condition,
        cross_modal_densities=cross_densities,
        total_cross_modal_density=total_density,
        within_modal_densities=within_densities,
        mean_cross_modal_weight=float(np.mean(weights)) if weights.size > 0 else 0.0,
        std_cross_modal_weight=float(np.std(weights)) if weights.size > 0 else 0.0,
        n_nonzero_cross_modal=n_nonzero,
        n_total_cross_modal=n_total,
    )

    # Graph metrics (optional)
    if HAS_NX:
        try:
            metrics.clustering_coef, metrics.betweenness = _compute_graph_metrics(
                prec, ggm_result.feature_names
            )
        except Exception as e:
            logger.debug(f"Graph metrics failed: {e}")

    return metrics


def _compute_graph_metrics(
    precision: np.ndarray,
    feature_names: List[str],
) -> Tuple[float, Optional[Dict[str, float]]]:
    """Compute networkx graph metrics on precision matrix."""
    D = precision.shape[0]

    # Build weighted undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(D))

    for i in range(D):
        for j in range(i + 1, D):
            w = abs(precision[i, j])
            if w > 1e-10:
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        return 0.0, None

    clustering = nx.average_clustering(G, weight="weight")
    betweenness_raw = nx.betweenness_centrality(G, weight="weight")

    betweenness = {}
    for node_idx, cent in betweenness_raw.items():
        if node_idx < len(feature_names):
            betweenness[feature_names[node_idx]] = cent

    return float(clustering), betweenness


# ── Pairwise comparison ────────────────────────────────────────────────────────

def compare_topologies(
    metrics_a: TopologyMetrics,
    metrics_b: TopologyMetrics,
    ggm_a: GGMResult,
    ggm_b: GGMResult,
) -> TopologyComparison:
    """
    Compare topology between two conditions.

    Primary test:
      Distance(9-digit, 13-digit) > Distance(5-digit, 9-digit)
    """
    dist = network_distance(ggm_a.precision_matrix, ggm_b.precision_matrix)
    delta_density = metrics_b.total_cross_modal_density - metrics_a.total_cross_modal_density

    delta_per_pair = {}
    all_keys = set(metrics_a.cross_modal_densities) | set(metrics_b.cross_modal_densities)
    for key in all_keys:
        da = metrics_a.cross_modal_densities.get(key, 0.0)
        db = metrics_b.cross_modal_densities.get(key, 0.0)
        delta_per_pair[key] = db - da

    return TopologyComparison(
        condition_a=metrics_a.condition,
        condition_b=metrics_b.condition,
        network_distance=dist,
        delta_cross_modal_density=delta_density,
        delta_per_pair=delta_per_pair,
    )


# ── Non-monotonic hypothesis test ─────────────────────────────────────────────

def test_non_monotonic_density(
    condition_densities: Dict[int, float],
    condition_order: List[int],
) -> Tuple[bool, float]:
    """
    Test non-monotonic cross-modal edge density hypothesis.

    H0: density is monotonically increasing or decreasing across conditions.
    H1: density increases then decreases (peaks at 9-digit).

    Prediction: density[5-dig] < density[9-dig] > density[13-dig].

    Uses permutation test: randomize condition order, count how often
    random ordering shows larger density at middle condition.

    Parameters
    ----------
    condition_densities : dict {condition_label → density}
    condition_order : list of condition labels in load order
        e.g. [0, 1, 2, 3] for control, 5-dig, 9-dig, 13-dig

    Returns
    -------
    confirmed : bool
    p_value : float
    """
    densities = np.array([condition_densities.get(c, 0.0) for c in condition_order])
    n = len(densities)

    if n < 3:
        return False, 1.0

    # Observed: is the maximum at index 1 or 2 (medium load)?
    mid_idx = n // 2
    peak_idx = int(np.argmax(densities))
    is_non_monotonic = (1 <= peak_idx <= n - 2)

    # Permutation test: how often does random ordering peak in the middle?
    n_perm = 10000
    rng = np.random.RandomState(42)
    count_non_mono = 0

    for _ in range(n_perm):
        perm = rng.permutation(densities)
        peak = int(np.argmax(perm))
        if 1 <= peak <= n - 2:
            count_non_mono += 1

    p_value = (count_non_mono + 1) / (n_perm + 1)

    logger.info(
        f"Non-monotonic density test: observed_peak_idx={peak_idx}, "
        f"is_non_monotonic={is_non_monotonic}, p={p_value:.4f}"
    )

    return is_non_monotonic, float(p_value)


# ── Full topology pipeline ─────────────────────────────────────────────────────

def run_topology_analysis(
    condition_ggms: ConditionGGMs,
    modality_pairs: List[Tuple[str, str]],
    condition_order: Optional[List[int]] = None,
) -> TopologyReport:
    """
    Full topology analysis across all conditions.

    Parameters
    ----------
    condition_ggms : ConditionGGMs
    modality_pairs : list of (mod_a, mod_b)
    condition_order : list of condition ints in ascending load order
        Default: sorted keys of condition_ggms.condition_ggms

    Returns
    -------
    TopologyReport
    """
    conds = sorted(condition_ggms.condition_ggms.keys())
    if condition_order is None:
        condition_order = conds

    # Compute per-condition metrics
    condition_metrics = {}
    for cond in conds:
        ggm = condition_ggms.condition_ggms[cond]
        metrics = compute_topology(ggm, modality_pairs)
        condition_metrics[cond] = metrics
        logger.info(
            f"Condition {cond}: cross_modal_density={metrics.total_cross_modal_density:.4f}, "
            f"n_nonzero={metrics.n_nonzero_cross_modal}"
        )

    # Pairwise comparisons
    pairwise = {}
    network_dists = {}
    for i, ca in enumerate(conds):
        for cb in conds[i + 1:]:
            key = f"c{ca}_c{cb}"
            comp = compare_topologies(
                condition_metrics[ca], condition_metrics[cb],
                condition_ggms.condition_ggms[ca],
                condition_ggms.condition_ggms[cb],
            )
            pairwise[key] = comp
            network_dists[key] = comp.network_distance

    # Log network distance comparison (primary prediction)
    cond_labels = sorted(condition_order)
    if len(cond_labels) >= 3:
        # Compare distance between last two vs second-to-last pair
        dist_early = network_dists.get(f"c{cond_labels[-3]}_c{cond_labels[-2]}", np.nan)
        dist_late  = network_dists.get(f"c{cond_labels[-2]}_c{cond_labels[-1]}", np.nan)
        logger.info(
            f"Network distance [{cond_labels[-3]}→{cond_labels[-2]}]={dist_early:.4f} | "
            f"[{cond_labels[-2]}→{cond_labels[-1]}]={dist_late:.4f} | "
            f"Late > Early: {dist_late > dist_early}"
        )

    # Non-monotonic test
    densities_by_cond = {c: m.total_cross_modal_density for c, m in condition_metrics.items()}
    non_mono_confirmed, non_mono_pval = test_non_monotonic_density(
        densities_by_cond, condition_order
    )

    return TopologyReport(
        condition_metrics=condition_metrics,
        pairwise_comparisons=pairwise,
        non_monotonic_confirmed=non_mono_confirmed,
        non_monotonic_pvalue=non_mono_pval,
        network_distances=network_dists,
    )