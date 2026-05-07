"""
visualization/pid_plots.py
===========================
Figure 3 from the paper: Partial Information Decomposition (PID) plots.

Visualizations:
1. Stacked bar chart: redundancy / unique_X1 / unique_X2 / synergy per modality pair
   across conditions (the non-monotonic synergy pattern)
2. Synergy heatmap: synergy(pair, condition) as 2D heatmap
3. Redundancy-synergy ratio per condition (coupling diagnostic)
4. PID trajectory: how synergy evolves across load conditions per pair
5. PID comparison: cross-modal vs within-modal pairs

PID paper reference:
Williams & Beer (2010), Nonnegative Decomposition of Multivariate Information
Gaussian closed-form: Bertschinger et al. (2014)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme for PID components
# ---------------------------------------------------------------------------

PID_COLORS = {
    "redundancy": "#4393c3",    # blue
    "unique_x1": "#92c5de",     # light blue
    "unique_x2": "#d1e5f0",     # very light blue
    "synergy": "#d6604d",       # red-orange (highlight: the key quantity)
}

PID_LABELS = {
    "redundancy": "Redundancy (shared LC-NE info)",
    "unique_x1": "Unique X₁",
    "unique_x2": "Unique X₂",
    "synergy": "Synergy (cross-modal exclusive)",
}

CONDITION_ORDER = ["control", "load_5", "load_9", "load_13"]
CONDITION_LABELS = {
    "control": "Control",
    "load_5": "5-digit\n(low)",
    "load_9": "9-digit\n(peak)",
    "load_13": "13-digit\n(overload)",
}


# ---------------------------------------------------------------------------
# Data structure for PID results
# ---------------------------------------------------------------------------

def _get_conditions_for_pair(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    pair_key: str,
) -> List[str]:
    """Extract ordered conditions for a modality pair, matching CONDITION_ORDER."""
    available = list(pid_results.get(pair_key, {}).keys())
    ordered = [c for c in CONDITION_ORDER if c in available]
    rest = [c for c in available if c not in ordered]
    return ordered + rest


# ---------------------------------------------------------------------------
# Figure 3a: Stacked bar chart of PID components per condition
# ---------------------------------------------------------------------------

def plot_pid_stacked_bars(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    modality_pairs: Optional[List[str]] = None,
    conditions_order: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Stacked bar chart of PID decomposition for each modality pair.
    Each pair gets one subplot. Bars = conditions, segments = PID components.

    H0 prediction:
    - Redundancy peaks at 9-digit (maximum LC-NE co-activation)
    - Synergy peaks at 9-digit (complementary information, highest WM engagement)
    - Synergy collapses at 13-digit (decoupling)

    Parameters
    ----------
    pid_results : dict pair_key → {condition → {pid_component → float}}
    modality_pairs : list of pair keys to include (None = all)
    conditions_order : list of condition keys (None = use CONDITION_ORDER)
    """
    if modality_pairs is None:
        modality_pairs = list(pid_results.keys())

    if conditions_order is None:
        # Use intersection with CONDITION_ORDER, maintaining order
        all_conds = set()
        for pair in modality_pairs:
            all_conds.update(pid_results.get(pair, {}).keys())
        conditions_order = [c for c in CONDITION_ORDER if c in all_conds]
        conditions_order += [c for c in all_conds if c not in conditions_order]

    n_pairs = len(modality_pairs)
    if figsize is None:
        figsize = (5 * n_pairs, 5)

    fig, axes = plt.subplots(1, n_pairs, figsize=figsize, sharey=False)
    if n_pairs == 1:
        axes = [axes]

    pid_components = ["redundancy", "unique_x1", "unique_x2", "synergy"]

    for pair_idx, pair_key in enumerate(modality_pairs):
        ax = axes[pair_idx]
        pair_data = pid_results.get(pair_key, {})

        # Build matrix: conditions × PID components
        n_conds = len(conditions_order)
        matrix = np.zeros((n_conds, len(pid_components)))

        for ci, cond in enumerate(conditions_order):
            if cond in pair_data:
                for pi, comp in enumerate(pid_components):
                    matrix[ci, pi] = float(pair_data[cond].get(comp, 0.0))

        x = np.arange(n_conds)
        bottoms = np.zeros(n_conds)

        for pi, comp in enumerate(pid_components):
            heights = matrix[:, pi]
            bars = ax.bar(
                x, heights, bottom=bottoms,
                color=PID_COLORS[comp], edgecolor="white", linewidth=0.5,
                label=PID_LABELS[comp] if pair_idx == 0 else "_nolegend_",
                width=0.6,
            )
            # Annotate synergy bars
            if comp == "synergy":
                for xi, (h, b) in enumerate(zip(heights, bottoms)):
                    if h > 0.005:
                        ax.text(xi, b + h / 2, f"{h:.3f}",
                                ha="center", va="center", fontsize=7,
                                color="white", fontweight="bold")
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in conditions_order],
            fontsize=9
        )
        ax.set_title(
            _format_pair_title(pair_key),
            fontsize=10, fontweight="bold"
        )
        ax.set_ylabel("Information (bits)" if pair_idx == 0 else "", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # Highlight synergy peak
        synergy_vals = matrix[:, pid_components.index("synergy")]
        if synergy_vals.max() > 0:
            peak_cond_idx = np.argmax(synergy_vals)
            ax.axvline(peak_cond_idx, color="#d6604d", linewidth=2,
                       linestyle="--", alpha=0.5, zorder=0)

    # Legend outside
    handles = [
        mpatches.Patch(color=PID_COLORS[c], label=PID_LABELS[c])
        for c in pid_components
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 1.02), fontsize=9, framealpha=0.9,
    )

    fig.suptitle(
        "Partial Information Decomposition by Modality Pair and Condition\n"
        "Synergy peaks at 9-digit, collapses at overload (decoupling signature)",
        fontsize=11, fontweight="bold", y=1.06,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PID stacked bars: {output_path}")

    return fig


def _format_pair_title(pair_key: str) -> str:
    """Convert 'eeg_pupil' → 'EEG × Pupil'."""
    parts = pair_key.upper().split("_")
    if len(parts) >= 2:
        return f"{parts[0]} × {parts[1]}"
    return pair_key.upper()


# ---------------------------------------------------------------------------
# Figure 3b: Synergy heatmap (pair × condition)
# ---------------------------------------------------------------------------

def plot_synergy_heatmap(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    conditions_order: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D heatmap: rows = modality pairs, columns = conditions, value = synergy.

    Prediction: 9-digit column has highest values in the cross-modal rows.
    """
    pairs = list(pid_results.keys())
    if conditions_order is None:
        all_conds = set()
        for p in pairs:
            all_conds.update(pid_results[p].keys())
        conditions_order = [c for c in CONDITION_ORDER if c in all_conds]

    n_pairs = len(pairs)
    n_conds = len(conditions_order)
    synergy_mat = np.zeros((n_pairs, n_conds))

    for ri, pair in enumerate(pairs):
        for ci, cond in enumerate(conditions_order):
            synergy_mat[ri, ci] = float(
                pid_results[pair].get(cond, {}).get("synergy", 0.0)
            )

    fig, ax = plt.subplots(figsize=(max(6, n_conds * 1.5), max(4, n_pairs * 0.7)))
    im = ax.imshow(synergy_mat, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=synergy_mat.max())

    ax.set_xticks(range(n_conds))
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions_order],
        fontsize=10
    )
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([_format_pair_title(p) for p in pairs], fontsize=9)

    # Annotate cells
    for ri in range(n_pairs):
        for ci in range(n_conds):
            v = synergy_mat[ri, ci]
            text_color = "white" if v > synergy_mat.max() * 0.6 else "black"
            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Synergy (bits)")
    ax.set_title(
        "Synergy (Cross-Modal Exclusive Information) by Condition\n"
        "Prediction: peaks at 9-digit, collapses at 13-digit",
        fontsize=11, fontweight="bold"
    )

    # Highlight 9-digit column
    peak_ci = [c for c in conditions_order].index("load_9") if "load_9" in conditions_order else -1
    if peak_ci >= 0:
        ax.add_patch(
            plt.Rectangle(
                (peak_ci - 0.5, -0.5), 1, n_pairs,
                fill=False, edgecolor="#d94801", linewidth=3, zorder=5
            )
        )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved synergy heatmap: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 3c: Redundancy-synergy ratio across conditions
# ---------------------------------------------------------------------------

def plot_redundancy_synergy_ratio(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    pair_keys: Optional[List[str]] = None,
    conditions_order: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Redundancy / (Redundancy + Synergy) ratio across conditions per modality pair.

    Ratio near 1.0 → modalities carry same (LC-NE) information (redundant)
    Ratio near 0.5 → balanced; cross-modal combination adds new information
    Low ratio → high synergy; large gain from multimodal observation
    """
    if pair_keys is None:
        pair_keys = list(pid_results.keys())
    if conditions_order is None:
        all_conds = set()
        for p in pair_keys:
            all_conds.update(pid_results[p].keys())
        conditions_order = [c for c in CONDITION_ORDER if c in all_conds]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions_order))

    colors_cycle = plt.cm.Set2(np.linspace(0, 1, len(pair_keys)))

    for pair_idx, pair_key in enumerate(pair_keys):
        pair_data = pid_results.get(pair_key, {})
        ratios = []
        for cond in conditions_order:
            red = float(pair_data.get(cond, {}).get("redundancy", 0.0))
            syn = float(pair_data.get(cond, {}).get("synergy", 0.0))
            total = red + syn
            ratios.append(red / total if total > 1e-10 else 0.5)

        ax.plot(
            x, ratios, "o-",
            color=colors_cycle[pair_idx], linewidth=2, markersize=8,
            label=_format_pair_title(pair_key),
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Balanced (ratio=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions_order], fontsize=10
    )
    ax.set_ylabel("Redundancy / (Redundancy + Synergy)", fontsize=11)
    ax.set_title(
        "Information Sharing Mode by Condition\n"
        "High ratio → LC-NE-driven redundancy; Low ratio → synergistic cross-modal coupling",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved redundancy-synergy ratio: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 3d: Per-pair PID full decomposition line plot
# ---------------------------------------------------------------------------

def plot_pid_trajectory(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    pair_key: str,
    conditions_order: Optional[List[str]] = None,
    pid_cis: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of all 4 PID components across conditions for a single modality pair.
    Optional bootstrap CI bands.

    Parameters
    ----------
    pid_cis : dict condition → {pid_component → (ci_lower, ci_upper)}
    """
    if conditions_order is None:
        all_conds = list(pid_results.get(pair_key, {}).keys())
        conditions_order = [c for c in CONDITION_ORDER if c in all_conds]

    pair_data = pid_results.get(pair_key, {})
    x = np.arange(len(conditions_order))

    fig, ax = plt.subplots(figsize=(8, 5))

    for comp, color in PID_COLORS.items():
        vals = [float(pair_data.get(c, {}).get(comp, 0.0)) for c in conditions_order]
        lw = 3 if comp == "synergy" else 1.5
        ls = "-" if comp == "synergy" else "--"
        ax.plot(x, vals, f"o{ls}", color=color, linewidth=lw, markersize=8,
                label=PID_LABELS[comp])

        if pid_cis:
            ci_lo = [pid_cis.get(c, {}).get(comp, (vals[i], vals[i]))[0]
                     for i, c in enumerate(conditions_order)]
            ci_hi = [pid_cis.get(c, {}).get(comp, (vals[i], vals[i]))[1]
                     for i, c in enumerate(conditions_order)]
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions_order], fontsize=10
    )
    ax.set_ylabel("Information (bits)", fontsize=11)
    ax.set_title(
        f"PID Trajectory — {_format_pair_title(pair_key)}\n"
        "(Synergy predicted to peak at 9-digit, collapse at overload)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PID trajectory [{pair_key}]: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Master PID figure generator
# ---------------------------------------------------------------------------

def generate_all_pid_figures(
    pid_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    conditions_order: Optional[List[str]] = None,
    pid_cis: Optional[Dict] = None,
    cross_modal_pairs: Optional[List[str]] = None,
):
    """Generate and save all PID figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating PID figures...")

    if cross_modal_pairs is None:
        cross_modal_pairs = list(pid_results.keys())

    plot_pid_stacked_bars(
        pid_results, cross_modal_pairs, conditions_order,
        output_path=str(out / "fig3a_pid_stacked_bars.png"),
    )

    plot_synergy_heatmap(
        pid_results, conditions_order,
        output_path=str(out / "fig3b_synergy_heatmap.png"),
    )

    plot_redundancy_synergy_ratio(
        pid_results, cross_modal_pairs, conditions_order,
        output_path=str(out / "fig3c_redundancy_synergy_ratio.png"),
    )

    for pair_key in cross_modal_pairs:
        pair_cis = pid_cis.get(pair_key) if pid_cis else None
        plot_pid_trajectory(
            pid_results, pair_key, conditions_order, pair_cis,
            output_path=str(out / f"fig3d_pid_trajectory_{pair_key}.png"),
        )

    logger.info(f"All PID figures saved to {output_dir}")