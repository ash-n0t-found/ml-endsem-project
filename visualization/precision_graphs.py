"""
visualization/precision_graphs.py
===================================
Figure 1 from the paper: Four precision matrix network graphs (one per condition).

Visualizations:
1. Heatmap of precision matrix Θ_c per condition
2. Network graph: nodes = physiological features, edges = nonzero precision entries
   colored by modality pair (within-modal vs. cross-modal)
3. Cross-modal edge density bar chart across conditions (the non-monotonic curve)
4. Frobenius distance matrix (condition × condition)
5. Stability selection matrix heatmap

Design principles:
- Modality nodes colored distinctly (EEG=blue, ECG=red, PPG=green, pupil=orange)
- Cross-modal edges in a distinct color, within-modal in gray
- Edge width proportional to |precision weight|
- Publication quality: 300 dpi, no chartjunk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

MODALITY_COLORS = {
    "eeg": "#2166ac",      # blue
    "ecg": "#d6604d",      # red
    "ppg": "#4dac26",      # green
    "pupil": "#e08214",    # orange
}

CONDITION_COLORS = {
    "control": "#bdbdbd",
    "load_5": "#74c476",
    "load_9": "#fd8d3c",
    "load_13": "#d94801",
}

CONDITION_LABELS = {
    "control": "Control",
    "load_5": "5-digit (low)",
    "load_9": "9-digit (peak)",
    "load_13": "13-digit (overload)",
}

WITHIN_MODAL_COLOR = "#999999"
CROSS_MODAL_COLOR = "#762a83"


# ---------------------------------------------------------------------------
# Helper: assign modality to each feature
# ---------------------------------------------------------------------------

def _build_modality_labels(
    D: int,
    modality_slices: Dict[str, slice],
) -> List[str]:
    labels = ["unknown"] * D
    for mod, sl in modality_slices.items():
        for i in range(*sl.indices(D)):
            labels[i] = mod
    return labels


def _build_cross_modal_mask(D: int, modality_slices: Dict[str, slice]) -> np.ndarray:
    labels = _build_modality_labels(D, modality_slices)
    mask = np.zeros((D, D), dtype=bool)
    for i in range(D):
        for j in range(D):
            mask[i, j] = (labels[i] != labels[j]) and (i != j)
    return mask


# ---------------------------------------------------------------------------
# Figure 1: Precision matrix heatmaps (2×2 grid, one per condition)
# ---------------------------------------------------------------------------

def plot_precision_heatmaps(
    precision_matrices: Dict[str, np.ndarray],
    modality_slices: Dict[str, slice],
    feature_names: Optional[List[str]] = None,
    conditions_order: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    vmax_percentile: float = 95,
) -> plt.Figure:
    """
    2×2 grid of precision matrix heatmaps, one per condition.

    Cross-modal blocks are framed with a colored border to highlight
    between-modality precision entries.

    Parameters
    ----------
    precision_matrices : dict condition → (D, D)
    modality_slices : dict modality_name → slice
    feature_names : list of D feature names
    conditions_order : list of condition keys for subplot ordering
    output_path : str path to save figure
    vmax_percentile : float percentile for colormap saturation

    Returns
    -------
    matplotlib Figure
    """
    if conditions_order is None:
        conditions_order = list(precision_matrices.keys())

    D = list(precision_matrices.values())[0].shape[0]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(D)]

    # Compute common color scale
    all_vals = np.concatenate([
        np.abs(precision_matrices[c]).ravel()
        for c in conditions_order if c in precision_matrices
    ])
    vmax = float(np.percentile(all_vals[all_vals > 1e-10], vmax_percentile))

    n_conds = min(len(conditions_order), 4)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
    axes = axes.ravel()

    for ax_idx, cond in enumerate(conditions_order[:n_conds]):
        ax = axes[ax_idx]
        theta = precision_matrices[cond]

        # Plot heatmap
        im = ax.imshow(
            theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest", aspect="equal"
        )

        ax.set_title(
            CONDITION_LABELS.get(cond, cond),
            fontsize=13, fontweight="bold",
            pad=8
        )

        # Modality block boundaries as tick groups
        _add_modality_separators(ax, modality_slices, D)

        # Remove dense tick labels if D > 20
        if D > 20:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks(range(D))
            ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
            ax.set_yticks(range(D))
            ax.set_yticklabels(feature_names, fontsize=6)

        # Modality block labels on left axis
        _add_modality_block_labels(ax, modality_slices, D, side="left")
        _add_modality_block_labels(ax, modality_slices, D, side="bottom")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Partial correlation")

    # Hide unused subplots
    for i in range(n_conds, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Condition-Specific Precision Matrices (Θ_c)\n"
        "Cross-Modal Coupling Structure by Cognitive Load",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved precision heatmaps: {output_path}")

    return fig


def _add_modality_separators(ax, modality_slices: Dict[str, slice], D: int):
    """Draw grid lines at modality block boundaries."""
    boundaries = set()
    for sl in modality_slices.values():
        start, stop, _ = sl.indices(D)
        boundaries.add(start - 0.5)
        boundaries.add(stop - 0.5)

    for b in sorted(boundaries):
        if 0 < b < D - 1:
            ax.axhline(b, color="black", linewidth=1.5, alpha=0.6)
            ax.axvline(b, color="black", linewidth=1.5, alpha=0.6)


def _add_modality_block_labels(ax, modality_slices, D, side="left"):
    """Add modality name labels at the midpoint of each block."""
    for mod, sl in modality_slices.items():
        start, stop, _ = sl.indices(D)
        mid = (start + stop - 1) / 2.0
        color = MODALITY_COLORS.get(mod, "#444444")

        if side == "left":
            ax.annotate(
                mod.upper(), xy=(0, mid), xycoords=("axes fraction", "data"),
                fontsize=8, color=color, fontweight="bold",
                ha="right", va="center", xytext=(-5, 0),
                textcoords="offset points",
            )
        elif side == "bottom":
            ax.annotate(
                mod.upper(), xy=(mid, 0), xycoords=("data", "axes fraction"),
                fontsize=8, color=color, fontweight="bold",
                ha="center", va="top", xytext=(0, -5),
                textcoords="offset points",
                rotation=45,
            )


# ---------------------------------------------------------------------------
# Figure 2: Network graph visualization
# ---------------------------------------------------------------------------

def plot_precision_network(
    precision_matrix: np.ndarray,
    modality_slices: Dict[str, slice],
    condition_name: str,
    feature_names: Optional[List[str]] = None,
    edge_threshold: float = 0.01,
    max_edges: int = 80,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Network graph where nodes = features, edges = nonzero precision entries.

    Node color = modality. Edge color = within-modal (gray) vs cross-modal (purple).
    Edge width proportional to |weight|.

    Parameters
    ----------
    edge_threshold : float
        Minimum |precision| to draw an edge.
    max_edges : int
        Draw at most this many edges (top by |weight|).
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed. Skipping network graph.")
        return None

    D = precision_matrix.shape[0]
    labels = _build_modality_labels(D, modality_slices)

    if feature_names is None:
        feature_names = [f"{labels[i]}_{i}" for i in range(D)]

    G = nx.Graph()

    # Add nodes
    for i in range(D):
        G.add_node(i, modality=labels[i], name=feature_names[i])

    # Add edges (upper triangle only)
    edges_to_add = []
    for i in range(D):
        for j in range(i + 1, D):
            w = float(precision_matrix[i, j])
            if abs(w) >= edge_threshold:
                edges_to_add.append((i, j, w))

    # Sort by |weight|, keep top max_edges
    edges_to_add.sort(key=lambda e: abs(e[2]), reverse=True)
    edges_to_add = edges_to_add[:max_edges]

    for i, j, w in edges_to_add:
        cross = labels[i] != labels[j]
        G.add_edge(i, j, weight=w, cross_modal=cross)

    # Layout: group nodes by modality
    pos = _modality_circle_layout(D, modality_slices)

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # Draw nodes
    for mod, sl in modality_slices.items():
        node_ids = list(range(*sl.indices(D)))
        nx.draw_networkx_nodes(
            G, pos, nodelist=node_ids,
            node_color=MODALITY_COLORS.get(mod, "#888888"),
            node_size=120, ax=ax, alpha=0.9,
        )

    # Draw edges
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("cross_modal")]
    within_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("cross_modal")]

    max_w = max((abs(precision_matrix[u, v]) for u, v in cross_edges + within_edges), default=1.0)

    for edge_list, color in [(within_edges, WITHIN_MODAL_COLOR), (cross_edges, CROSS_MODAL_COLOR)]:
        widths = [3.0 * abs(precision_matrix[u, v]) / max_w for u, v in edge_list]
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list,
            edge_color=color, width=widths, alpha=0.7, ax=ax,
        )

    ax.set_title(
        f"{CONDITION_LABELS.get(condition_name, condition_name)}\n"
        f"n_edges={len(cross_edges)+len(within_edges)} "
        f"(cross-modal: {len(cross_edges)})",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=m.upper())
               for m, c in MODALITY_COLORS.items()]
    patches += [
        mpatches.Patch(color=CROSS_MODAL_COLOR, label="Cross-modal edge"),
        mpatches.Patch(color=WITHIN_MODAL_COLOR, label="Within-modal edge"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.8)

    if output_path and fig_created:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved network graph [{condition_name}]: {output_path}")

    return fig


def _modality_circle_layout(
    D: int,
    modality_slices: Dict[str, slice],
) -> Dict[int, Tuple[float, float]]:
    """
    Circular layout grouping nodes by modality.
    Each modality occupies a sector of the circle.
    """
    pos = {}
    modalities = list(modality_slices.keys())
    n_mod = len(modalities)

    for mod_idx, (mod, sl) in enumerate(modality_slices.items()):
        nodes = list(range(*sl.indices(D)))
        n_nodes = len(nodes)

        # Sector: [start_angle, end_angle]
        sector_start = 2 * np.pi * mod_idx / n_mod
        sector_end = 2 * np.pi * (mod_idx + 1) / n_mod
        sector_mid = (sector_start + sector_end) / 2

        # Place nodes on a small arc within the sector
        if n_nodes == 1:
            angles = [sector_mid]
        else:
            angles = np.linspace(sector_start + 0.1, sector_end - 0.1, n_nodes)

        # Outer radius for modality, with slight jitter
        r = 1.0 + 0.05 * mod_idx
        for node, angle in zip(nodes, angles):
            pos[node] = (r * np.cos(angle), r * np.sin(angle))

    return pos


# ---------------------------------------------------------------------------
# Figure 2 (Paper): 4-panel network graph grid
# ---------------------------------------------------------------------------

def plot_four_condition_networks(
    precision_matrices: Dict[str, np.ndarray],
    modality_slices: Dict[str, slice],
    conditions_order: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    edge_threshold: float = 0.01,
    max_edges: int = 60,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel grid of network graphs (one per condition).
    Primary paper figure for network topology visualization.
    """
    if conditions_order is None:
        conditions_order = list(precision_matrices.keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    axes = axes.ravel()

    for ax_idx, cond in enumerate(conditions_order[:4]):
        plot_precision_network(
            precision_matrix=precision_matrices[cond],
            modality_slices=modality_slices,
            condition_name=cond,
            feature_names=feature_names,
            edge_threshold=edge_threshold,
            max_edges=max_edges,
            ax=axes[ax_idx],
        )

    for i in range(len(conditions_order), 4):
        axes[i].set_visible(False)

    fig.suptitle(
        "Physiological Coupling Network by Cognitive Load Condition\n"
        "(Edge color: purple=cross-modal, gray=within-modal; width=|precision weight|)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved 4-condition network grid: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 2b: Cross-modal edge density vs condition (non-monotonic curve)
# ---------------------------------------------------------------------------

def plot_edge_density_curve(
    precision_matrices: Dict[str, np.ndarray],
    modality_slices: Dict[str, slice],
    conditions_order: Optional[List[str]] = None,
    density_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart / line plot of cross-modal edge density across conditions.

    This is the primary finding plot: non-monotonic density curve
    (increases control→9-digit, decreases at 13-digit).

    Parameters
    ----------
    density_cis : dict condition → (ci_lower, ci_upper) bootstrap CIs
    """
    if conditions_order is None:
        conditions_order = list(precision_matrices.keys())

    D = list(precision_matrices.values())[0].shape[0]
    cross_mask = _build_cross_modal_mask(D, modality_slices)
    n_cross = cross_mask.sum()

    densities = []
    for cond in conditions_order:
        theta = precision_matrices[cond]
        nonzero = np.abs(theta) > 1e-10
        density = float(np.sum(nonzero & cross_mask) / n_cross)
        densities.append(density)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions_order))
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conditions_order]
    bars = ax.bar(x, densities, color=colors, edgecolor="black", linewidth=1.2, width=0.5)

    # Bootstrap CI error bars
    if density_cis:
        ci_lo = [densities[i] - density_cis.get(c, (densities[i], densities[i]))[0]
                 for i, c in enumerate(conditions_order)]
        ci_hi = [density_cis.get(c, (densities[i], densities[i]))[1] - densities[i]
                 for i, c in enumerate(conditions_order)]
        ax.errorbar(
            x, densities,
            yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=6, linewidth=2,
        )

    # Connect bars with line
    ax.plot(x, densities, "ko-", linewidth=2, markersize=8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions_order],
        fontsize=11
    )
    ax.set_ylabel("Cross-modal edge density", fontsize=12)
    ax.set_title(
        "Cross-Modal Physiological Coupling by Condition\n"
        "(Non-monotonic: coupling peaks at 9-digit, decouples at overload)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, min(1.0, max(densities) * 1.35))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate peak
    peak_idx = np.argmax(densities)
    ax.annotate(
        "Peak coupling\n(LC-NE fully engaged)",
        xy=(peak_idx, densities[peak_idx]),
        xytext=(peak_idx - 0.6, densities[peak_idx] + 0.04),
        fontsize=9, color="#d94801",
        arrowprops=dict(arrowstyle="->", color="#d94801"),
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved edge density curve: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Frobenius distance matrix
# ---------------------------------------------------------------------------

def plot_frobenius_distance_matrix(
    precision_matrices: Dict[str, np.ndarray],
    conditions_order: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of pairwise Frobenius distances between condition precision matrices."""
    if conditions_order is None:
        conditions_order = list(precision_matrices.keys())

    n = len(conditions_order)
    dist_mat = np.zeros((n, n))

    for i, ca in enumerate(conditions_order):
        for j, cb in enumerate(conditions_order):
            if ca in precision_matrices and cb in precision_matrices:
                dist_mat[i, j] = np.linalg.norm(
                    precision_matrices[ca] - precision_matrices[cb], "fro"
                )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_mat, cmap="YlOrRd", aspect="equal")
    plt.colorbar(im, ax=ax, label="Frobenius distance")

    labels = [CONDITION_LABELS.get(c, c) for c in conditions_order]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if dist_mat[i,j] < dist_mat.max()*0.6 else "white")

    ax.set_title("Pairwise Network Distance (Frobenius)\nbetween Condition Precision Matrices",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Frobenius distance matrix: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Stability selection heatmap
# ---------------------------------------------------------------------------

def plot_stability_heatmap(
    stability_matrix: np.ndarray,
    modality_slices: Dict[str, slice],
    condition_name: str,
    stability_threshold: float = 0.6,
    feature_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of stability scores (0–1) from Meinshausen-Bühlmann stability selection.
    Highlights reliably nonzero edges.
    """
    D = stability_matrix.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(D)]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(stability_matrix, cmap="hot_r", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Stability score (proportion nonzero across subsamples)")

    # Draw threshold contour
    ax.contour(stability_matrix, levels=[stability_threshold],
               colors=["cyan"], linewidths=1.5)

    _add_modality_separators(ax, modality_slices, D)

    if D <= 30:
        ax.set_xticks(range(D))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
        ax.set_yticks(range(D))
        ax.set_yticklabels(feature_names, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    _add_modality_block_labels(ax, modality_slices, D, side="left")
    _add_modality_block_labels(ax, modality_slices, D, side="bottom")

    ax.set_title(
        f"Stability Selection Matrix — {CONDITION_LABELS.get(condition_name, condition_name)}\n"
        f"(Cyan contour = stability threshold {stability_threshold})",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved stability heatmap [{condition_name}]: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Master figure generator
# ---------------------------------------------------------------------------

def generate_all_precision_figures(
    precision_matrices: Dict[str, np.ndarray],
    modality_slices: Dict[str, slice],
    output_dir: str,
    conditions_order: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    density_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    stability_results: Optional[Dict[str, dict]] = None,
):
    """Generate and save all precision-related figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating precision matrix figures...")

    plot_precision_heatmaps(
        precision_matrices, modality_slices, feature_names, conditions_order,
        output_path=str(out / "fig1a_precision_heatmaps.png"),
    )

    plot_four_condition_networks(
        precision_matrices, modality_slices, conditions_order, feature_names,
        output_path=str(out / "fig1b_network_graphs.png"),
    )

    plot_edge_density_curve(
        precision_matrices, modality_slices, conditions_order, density_cis,
        output_path=str(out / "fig2_edge_density_curve.png"),
    )

    plot_frobenius_distance_matrix(
        precision_matrices, conditions_order,
        output_path=str(out / "fig_frobenius_distances.png"),
    )

    if stability_results:
        for cond, stab in stability_results.items():
            if "stability_matrix" in stab:
                plot_stability_heatmap(
                    stab["stability_matrix"], modality_slices, cond,
                    stab.get("threshold", 0.6), feature_names,
                    output_path=str(out / f"fig_stability_{cond}.png"),
                )

    logger.info(f"All precision figures saved to {output_dir}")