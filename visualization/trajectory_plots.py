"""
visualization/trajectory_plots.py
===================================
Figures 5–6 from the paper:

Fig 5: Per-digit LGSSM latent load trajectory examples
   - Show estimated z_t across digit positions for representative subjects
   - Illustrate participant-specific overload onset
   - Compare high-WM vs low-WM capacity subjects

Fig 6: Resting-state coupling vs WM capacity scatter
   - x = resting GGM coupling strength
   - y = behavioral WM span
   - Regression line + CI + r / p annotation

Additional plots:
- BOCPD changepoint detection: posterior hazard rate per subject
- Coupling strength trajectory per subject (sliding-window GGM)
- Recall prediction model comparison bar chart (R² across feature sets)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

CONDITION_COLORS = {
    "control": "#bdbdbd",
    "load_5": "#74c476",
    "load_9": "#fd8d3c",
    "load_13": "#d94801",
}

MODEL_COLORS = {
    "marginal_only": "#4393c3",
    "coupling_only": "#d6604d",
    "marginal_plus_coupling": "#762a83",
    "resting_coupling_only": "#74c476",
    "B1_Catch22_XGBoost": "#999999",
    "B3_Early_Fusion": "#bdbdbd",
}

MODEL_LABELS = {
    "marginal_only": "Marginal features\n(single-channel)",
    "coupling_only": "Coupling features\n(GGM only)",
    "marginal_plus_coupling": "Marginal + Coupling\n(full model)",
    "resting_coupling_only": "Resting-state\ncoupling (zero-shot)",
    "B1_Catch22_XGBoost": "B1: Catch22+XGB\n(baseline)",
    "B3_Early_Fusion": "B3: Early fusion\n(ablation)",
}


# ---------------------------------------------------------------------------
# Figure 5: LGSSM latent load trajectories
# ---------------------------------------------------------------------------

def plot_lgssm_trajectories(
    lgssm_results: Dict[str, Dict[str, Any]],
    subject_ids: Optional[List[str]] = None,
    conditions_to_show: Optional[List[str]] = None,
    wm_capacity: Optional[Dict[str, float]] = None,
    n_subjects_to_plot: int = 6,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot estimated LGSSM latent load trajectories across digit positions.

    Each subplot = one subject, one trial (or condition-averaged).
    Color-coded by condition.

    Parameters
    ----------
    lgssm_results : dict subject_id → {condition → {
        'latent_trajectory': (T,) array of z_t estimates,
        'trajectory_variance': (T,) array of Var(z_t),
        'overload_onset_digit': int or None,
    }}
    wm_capacity : dict subject_id → float behavioral WM span
    """
    if subject_ids is None:
        subject_ids = list(lgssm_results.keys())[:n_subjects_to_plot]

    if conditions_to_show is None:
        # Infer from first subject
        first = lgssm_results[subject_ids[0]]
        conditions_to_show = list(first.keys())

    n_subs = min(n_subjects_to_plot, len(subject_ids))
    ncols = min(3, n_subs)
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for sub_idx, sub_id in enumerate(subject_ids[:n_subs]):
        ax = axes_flat[sub_idx]
        sub_results = lgssm_results.get(sub_id, {})
        wm = wm_capacity.get(sub_id, None) if wm_capacity else None

        for cond, cond_results in sub_results.items():
            if cond not in conditions_to_show:
                continue

            z = np.array(cond_results.get("latent_trajectory", []))
            var = np.array(cond_results.get("trajectory_variance", np.zeros_like(z)))
            T = len(z)
            if T == 0:
                continue

            digit_positions = np.arange(1, T + 1)
            color = CONDITION_COLORS.get(cond, "#888888")

            ax.plot(digit_positions, z, "-o", color=color, linewidth=2,
                    markersize=5, label=_format_cond_label(cond))

            # CI band: ±1 std
            std = np.sqrt(np.clip(var, 0, np.inf))
            ax.fill_between(digit_positions, z - std, z + std,
                            alpha=0.15, color=color)

            # Mark overload onset
            onset = cond_results.get("overload_onset_digit")
            if onset is not None and 1 <= onset <= T:
                ax.axvline(onset, color=color, linestyle=":", linewidth=2, alpha=0.8)
                ax.annotate(
                    f"onset\ndigit {onset}",
                    xy=(onset, z[onset - 1]),
                    xytext=(onset + 0.3, z[onset - 1] + 0.1 * (z.max() - z.min())),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

        title = sub_id
        if wm is not None:
            title += f"\nWM span={wm:.1f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Digit position", fontsize=9)
        ax.set_ylabel("Latent load (z_t)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(n_subs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "LGSSM Estimated Latent Cognitive Load Trajectories per Digit Position\n"
        "(Dashed vertical = estimated overload onset; band = ±1 SD uncertainty)",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved LGSSM trajectories: {output_path}")

    return fig


def _format_cond_label(cond: str) -> str:
    return {"control": "Control", "load_5": "5-digit",
            "load_9": "9-digit", "load_13": "13-digit"}.get(cond, cond)


# ---------------------------------------------------------------------------
# Figure 5b: High-WM vs Low-WM capacity comparison
# ---------------------------------------------------------------------------

def plot_wm_capacity_trajectory_comparison(
    lgssm_results: Dict[str, Dict[str, Any]],
    wm_capacity: Dict[str, float],
    condition: str = "load_13",
    n_high: int = 5,
    n_low: int = 5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare LGSSM trajectories between high and low WM capacity subjects.

    Prediction:
    - High WM: overload onset at digit ~10-11
    - Low WM: overload onset at digit ~7-8
    """
    subjects_wm = [(s, w) for s, w in wm_capacity.items() if s in lgssm_results]
    subjects_wm.sort(key=lambda x: x[1])

    low_wm_subs = [s for s, _ in subjects_wm[:n_low]]
    high_wm_subs = [s for s, _ in subjects_wm[-n_high:]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, group_subs, group_label, base_color in [
        (axes[0], low_wm_subs, "Low WM Capacity", "#d6604d"),
        (axes[1], high_wm_subs, "High WM Capacity", "#2166ac"),
    ]:
        all_trajectories = []
        for sub_id in group_subs:
            cond_data = lgssm_results.get(sub_id, {}).get(condition, {})
            z = np.array(cond_data.get("latent_trajectory", []))
            if len(z) == 0:
                continue
            T = len(z)
            all_trajectories.append(z)
            ax.plot(
                np.arange(1, T + 1), z,
                color=base_color, alpha=0.4, linewidth=1,
            )
            onset = cond_data.get("overload_onset_digit")
            if onset is not None:
                ax.axvline(onset, color=base_color, alpha=0.2, linewidth=1)

        if all_trajectories:
            # Plot mean trajectory
            max_T = max(len(z) for z in all_trajectories)
            padded = np.full((len(all_trajectories), max_T), np.nan)
            for i, z in enumerate(all_trajectories):
                padded[i, :len(z)] = z
            mean_z = np.nanmean(padded, axis=0)
            ax.plot(
                np.arange(1, max_T + 1), mean_z,
                color=base_color, linewidth=3, label="Mean trajectory",
            )
            mean_onset = np.mean([
                lgssm_results[s].get(condition, {}).get("overload_onset_digit", np.nan)
                for s in group_subs
                if lgssm_results.get(s, {}).get(condition, {}).get("overload_onset_digit") is not None
            ])
            if not np.isnan(mean_onset):
                ax.axvline(mean_onset, color=base_color, linewidth=2.5,
                           linestyle="--", label=f"Mean onset (digit {mean_onset:.1f})")

        wm_vals = [wm_capacity[s] for s in group_subs if s in wm_capacity]
        ax.set_title(
            f"{group_label}\n(WM span: {np.mean(wm_vals):.1f} ± {np.std(wm_vals):.1f}, n={len(group_subs)})",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Digit position", fontsize=10)
        ax.set_ylabel("Latent load (z_t)", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Latent Load Trajectories ({_format_cond_label(condition)})\n"
        "High-WM subjects overload later than low-WM subjects",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved WM capacity trajectory comparison: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 6: Resting-state coupling vs WM capacity scatter
# ---------------------------------------------------------------------------

def plot_resting_coupling_vs_wm(
    resting_coupling: np.ndarray,
    wm_capacity: np.ndarray,
    coupling_feature_name: str = "Resting GGM coupling strength",
    subject_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot: resting cross-modal coupling strength vs behavioral WM span.

    Regression line, 95% CI band, Pearson r and p-value annotation.

    Parameters
    ----------
    resting_coupling : (N_subjects,) coupling strength at rest
    wm_capacity : (N_subjects,) behavioral WM span
    """
    N = len(resting_coupling)
    assert len(wm_capacity) == N, "Lengths must match"

    # Linear regression
    slope, intercept, r_val, p_val, se = scipy_stats.linregress(
        resting_coupling, wm_capacity
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter
    ax.scatter(
        resting_coupling, wm_capacity,
        color="#2166ac", alpha=0.7, s=80, edgecolors="white", linewidths=0.8,
        zorder=3,
    )

    # Label subjects if provided
    if subject_ids and N <= 20:
        for i, sid in enumerate(subject_ids):
            ax.annotate(
                sid.replace("sub-", ""),
                (resting_coupling[i], wm_capacity[i]),
                fontsize=7, color="#555555",
                xytext=(3, 3), textcoords="offset points",
            )

    # Regression line
    x_range = np.linspace(resting_coupling.min(), resting_coupling.max(), 100)
    y_fit = slope * x_range + intercept
    ax.plot(x_range, y_fit, color="#d94801", linewidth=2.5, label="Regression", zorder=4)

    # 95% CI band
    x_mean = resting_coupling.mean()
    se_fit = se * np.sqrt(
        1.0 / N + (x_range - x_mean) ** 2 / np.sum((resting_coupling - x_mean) ** 2)
    )
    t_crit = scipy_stats.t.ppf(0.975, df=N - 2)
    ax.fill_between(
        x_range, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit,
        alpha=0.15, color="#d94801", label="95% CI"
    )

    # Annotation
    p_str = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"
    ax.annotate(
        f"r = {r_val:.3f}\n{p_str}\nN = {N}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=11, color="#d94801" if p_val < 0.05 else "#555555",
        fontweight="bold" if p_val < 0.05 else "normal",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(coupling_feature_name, fontsize=11)
    ax.set_ylabel("Behavioral WM Span", fontsize=11)
    ax.set_title(
        "Resting-State Physiological Coupling Predicts Working Memory Capacity\n"
        "(Stronger resting cross-modal coupling → higher WM span)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved resting coupling vs WM: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 4: Recall prediction R² comparison
# ---------------------------------------------------------------------------

def plot_recall_r2_comparison(
    model_r2_scores: Dict[str, float],
    model_r2_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    coupling_model_key: str = "marginal_plus_coupling",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing R² scores across model feature sets.

    Parameters
    ----------
    model_r2_scores : dict model_name → R² (float)
    model_r2_cis : dict model_name → (ci_lower, ci_upper) bootstrap CIs
    coupling_model_key : str key of the main coupling model (highlighted)
    """
    model_names = list(model_r2_scores.keys())
    r2_vals = [model_r2_scores[m] for m in model_names]

    # Sort by R²
    order = np.argsort(r2_vals)
    model_names = [model_names[i] for i in order]
    r2_vals = [r2_vals[i] for i in order]

    n = len(model_names)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.7)))

    colors = []
    for m in model_names:
        if m == coupling_model_key:
            colors.append("#762a83")  # highlight coupling model
        elif "B" in m and any(c.isdigit() for c in m):
            colors.append("#bdbdbd")  # baseline
        elif "coupling" in m.lower():
            colors.append("#d6604d")
        else:
            colors.append("#4393c3")

    y = np.arange(n)
    bars = ax.barh(y, r2_vals, color=colors, edgecolor="black", linewidth=0.8, height=0.6)

    # Error bars
    if model_r2_cis:
        for yi, m in enumerate(model_names):
            if m in model_r2_cis:
                ci_lo, ci_hi = model_r2_cis[m]
                ax.errorbar(
                    r2_vals[yi], yi,
                    xerr=[[r2_vals[yi] - ci_lo], [ci_hi - r2_vals[yi]]],
                    fmt="none", color="black", capsize=4, linewidth=1.5,
                )

    # Value labels
    for bar, r2 in zip(bars, r2_vals):
        ax.text(
            max(r2 + 0.005, 0.005), bar.get_y() + bar.get_height() / 2,
            f"{r2:.4f}", va="center", fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [MODEL_LABELS.get(m, m.replace("_", "\n")) for m in model_names],
        fontsize=9
    )
    ax.set_xlabel("LOSO R² (recall accuracy prediction)", fontsize=11)
    ax.set_title(
        "Recall Accuracy Prediction: Feature Set Comparison\n"
        "Coupling features add R² beyond single-channel marginal features",
        fontsize=11, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved recall R² comparison: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# BOCPD changepoint visualization
# ---------------------------------------------------------------------------

def plot_bocpd_changepoints(
    coupling_trajectories: Dict[str, np.ndarray],
    changepoint_probs: Dict[str, np.ndarray],
    wm_capacity: Optional[Dict[str, float]] = None,
    behavioral_onset: Optional[Dict[str, int]] = None,
    n_subjects: int = 6,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Per-subject BOCPD changepoint posterior over coupling strength trajectory.

    Parameters
    ----------
    coupling_trajectories : dict subject_id → (T,) coupling strength across conditions
    changepoint_probs : dict subject_id → (T,) posterior probability of changepoint at t
    behavioral_onset : dict subject_id → digit position of behavioral performance drop
    """
    subject_ids = list(coupling_trajectories.keys())[:n_subjects]
    ncols = min(3, n_subjects)
    nrows = (n_subjects + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(14, 4 * nrows))
    axes = axes.reshape(nrows, -1)

    for si, sub_id in enumerate(subject_ids):
        row = si // ncols
        col_base = (si % ncols) * 2

        traj = np.array(coupling_trajectories.get(sub_id, []))
        cp_prob = np.array(changepoint_probs.get(sub_id, []))
        T = len(traj)
        x = np.arange(1, T + 1)

        # Left: coupling trajectory
        ax1 = axes[row, col_base]
        ax1.plot(x, traj, "b-o", linewidth=2, markersize=6)
        if sub_id in (behavioral_onset or {}):
            ax1.axvline(behavioral_onset[sub_id], color="red", linewidth=2,
                        linestyle="--", label="Behavioral onset")

        wm = wm_capacity.get(sub_id) if wm_capacity else None
        title = sub_id + (f"\nWM={wm:.1f}" if wm else "")
        ax1.set_title(title, fontsize=8)
        ax1.set_xlabel("Condition index", fontsize=8)
        ax1.set_ylabel("Coupling strength", fontsize=8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(labelsize=7)

        # Right: changepoint probability
        ax2 = axes[row, col_base + 1]
        if len(cp_prob) > 0:
            ax2.bar(x, cp_prob, color="#d6604d", alpha=0.8, width=0.6)
            ax2.set_xlabel("Condition index", fontsize=8)
            ax2.set_ylabel("CP probability", fontsize=8)
            ax2.set_ylim(0, 1)
        ax2.set_title("BOCPD posterior", fontsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(labelsize=7)

    fig.suptitle(
        "Bayesian Changepoint Detection on Coupling Strength Trajectory\n"
        "(Red dashed = behavioral performance onset)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved BOCPD changepoints: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Master trajectory figure generator
# ---------------------------------------------------------------------------

def generate_all_trajectory_figures(
    output_dir: str,
    lgssm_results: Optional[Dict] = None,
    wm_capacity: Optional[Dict] = None,
    resting_coupling: Optional[np.ndarray] = None,
    wm_capacity_array: Optional[np.ndarray] = None,
    subject_ids_array: Optional[List[str]] = None,
    model_r2_scores: Optional[Dict] = None,
    model_r2_cis: Optional[Dict] = None,
    coupling_trajectories: Optional[Dict] = None,
    changepoint_probs: Optional[Dict] = None,
    behavioral_onsets: Optional[Dict] = None,
):
    """Generate and save all trajectory and scatter figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating trajectory figures...")

    if lgssm_results and wm_capacity:
        plot_lgssm_trajectories(
            lgssm_results, wm_capacity=wm_capacity,
            output_path=str(out / "fig5a_lgssm_trajectories.png"),
        )
        plot_wm_capacity_trajectory_comparison(
            lgssm_results, wm_capacity,
            output_path=str(out / "fig5b_wm_trajectory_comparison.png"),
        )

    if resting_coupling is not None and wm_capacity_array is not None:
        plot_resting_coupling_vs_wm(
            resting_coupling, wm_capacity_array,
            subject_ids=subject_ids_array,
            output_path=str(out / "fig6_resting_coupling_vs_wm.png"),
        )

    if model_r2_scores:
        plot_recall_r2_comparison(
            model_r2_scores, model_r2_cis,
            output_path=str(out / "fig4_recall_r2_comparison.png"),
        )

    if coupling_trajectories and changepoint_probs:
        plot_bocpd_changepoints(
            coupling_trajectories, changepoint_probs,
            wm_capacity=wm_capacity, behavioral_onset=behavioral_onsets,
            output_path=str(out / "fig_bocpd_changepoints.png"),
        )

    logger.info(f"All trajectory figures saved to {output_dir}")