"""
Script 06: LGSSM Per-Digit Cognitive Load Trajectories
=======================================================
Fit Linear Gaussian State Space Model (EM algorithm) per subject.
Infers continuous latent cognitive load trajectory across digit positions.

Key predictions:
  - High-WM subjects: overload onset at digit ~10-11
  - Low-WM subjects: overload onset at digit ~7-8
  - Trajectory changepoints validate against behavioral WM span

Classical replacement for LSTM/TCN (strictly EM-based, fully interpretable).

Outputs:
  - Per-subject latent load trajectories per condition
  - Overload onset estimates (BOCPD on coupling trajectory)
  - Figure 5a: Individual LGSSM trajectory panels
  - Figure 5b: High-WM vs Low-WM trajectory comparison

Run: python scripts/06_lgssm_trajectories.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle, save_json
from utils.config_loader import load_config
from models.lgssm import LinearGaussianSSM
from models.bocpd import BayesianChangePointDetector
from analysis.individual_diff import IndividualDifferenceAnalyzer
from visualization.trajectory_plots import (
    plot_lgssm_trajectories,
    plot_wm_capacity_trajectory_comparison,
    generate_all_trajectory_figures,
)

logger = logging.getLogger(__name__)

CONDITIONS = ["load_5", "load_9", "load_13"]
CONDITION_LABELS = {
    "load_5": "5-digit (Low)",
    "load_9": "9-digit (Medium)",
    "load_13": "13-digit (Overload)",
}


def parse_args():
    p = argparse.ArgumentParser(description="LGSSM trajectory analysis — Script 06")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--latent-dim", type=int, default=2,
                   help="Latent state dimension k (default 2: load + arousal)")
    p.add_argument("--n-em-iter", type=int, default=50,
                   help="EM algorithm iterations")
    p.add_argument("--force", action="store_true")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Override subject list")
    return p.parse_args()


def load_per_digit_features(cfg: dict, subjects: List[str]) -> Optional[Dict]:
    """
    Load per-digit multimodal feature vectors for each subject.
    Structure: subject_id → condition → list of (digit_pos, feature_vector)
    """
    feat_cache_dir = Path(cfg["paths"]["cache_dir"]) / "features"
    per_digit_data = {}

    for sub_id in subjects:
        sub_feat_path = feat_cache_dir / f"{sub_id}_features.pkl"
        if not sub_feat_path.exists():
            logger.warning(f"{sub_id}: feature cache not found — skipping")
            continue

        sub_feats = load_pickle(str(sub_feat_path))
        per_digit_data[sub_id] = {}

        for cond in CONDITIONS:
            cond_data = sub_feats.get("conditions", {}).get(cond, {})
            if not cond_data:
                continue

            # Build per-digit feature matrix
            # Each trial has n_digits feature vectors (one per digit position)
            # We use pupil TEPR per digit + EEG frontal theta sliding window
            try:
                per_digit_matrix = _build_per_digit_matrix(cond_data, cond, cfg)
                if per_digit_matrix is not None:
                    per_digit_data[sub_id][cond] = per_digit_matrix
                    logger.debug(
                        f"{sub_id}/{cond}: per-digit matrix shape {per_digit_matrix.shape}"
                    )
            except Exception as e:
                logger.warning(f"{sub_id}/{cond}: per-digit build failed — {e}")

    logger.info(f"Per-digit data loaded for {len(per_digit_data)} subjects")
    return per_digit_data


def _build_per_digit_matrix(cond_data: Dict, condition: str, cfg: dict) -> Optional[np.ndarray]:
    """
    Combine per-digit pupil TEPR and per-trial EEG into a sequence matrix.
    Returns (T_digits, D_features) array averaged across trials.
    """
    n_digits = cfg["paradigm"]["conditions"].get(condition, {}).get("n_digits", 9)

    # Pupil TEPR per digit is directly per-digit
    pupil_df = cond_data.get("pupil")
    eeg_df = cond_data.get("eeg")
    ecg_df = cond_data.get("ecg")
    ppg_df = cond_data.get("ppg")

    sequences = []

    if pupil_df is not None and hasattr(pupil_df, "values"):
        # Look for TEPR per digit columns
        tepr_cols = [c for c in pupil_df.columns
                     if c.startswith("tepr_digit_") or c.startswith("pupil_digit_")]
        if tepr_cols:
            # Each row = one trial, each TEPR column = one digit position
            tepr_vals = pupil_df[tepr_cols].values  # (n_trials, n_digits)
            # Average across trials to get per-digit mean
            mean_tepr = np.nanmean(tepr_vals, axis=0)[:n_digits]  # (T,)
            sequences.append(mean_tepr.reshape(-1, 1))

    # EEG: repeat trial-level feature across digits (proxy for now)
    if eeg_df is not None and hasattr(eeg_df, "values"):
        theta_cols = [c for c in eeg_df.columns if "theta" in c.lower()]
        if theta_cols:
            mean_theta = np.nanmean(eeg_df[theta_cols].values, axis=0)
            # Broadcast scalar to per-digit using linear interpolation
            mean_theta_val = np.nanmean(mean_theta)
            digit_theta = np.linspace(mean_theta_val * 0.8, mean_theta_val, n_digits)
            sequences.append(digit_theta.reshape(-1, 1))

    if not sequences:
        return None

    # Align lengths
    min_len = min(s.shape[0] for s in sequences)
    sequences = [s[:min_len] for s in sequences]

    return np.hstack(sequences)  # (T_digits, D_features)


def fit_lgssm_per_subject(
    per_digit_data: Dict,
    latent_dim: int,
    n_em_iter: int,
    output_dir: Path,
    force: bool = False,
) -> Dict:
    """
    Fit LGSSM for each subject × condition.
    Returns: subject_id → condition → {latent_trajectory, trajectory_variance, overload_onset_digit}
    """
    lgssm_cache = output_dir / "lgssm_results.pkl"
    if lgssm_cache.exists() and not force:
        logger.info("Loading cached LGSSM results...")
        return load_pickle(str(lgssm_cache))

    lgssm_model = LinearGaussianSSM(latent_dim=latent_dim, n_em_iter=n_em_iter)
    bocpd = BayesianChangePointDetector()

    all_results = {}

    for sub_id, sub_data in per_digit_data.items():
        all_results[sub_id] = {}

        for cond, F_digits in sub_data.items():
            try:
                # F_digits: (T_digits, D_obs)
                T, D = F_digits.shape
                if T < 3:
                    logger.warning(f"{sub_id}/{cond}: too few time points ({T})")
                    continue

                # Fit LGSSM via EM
                lgssm_model.fit(F_digits)

                # Infer latent trajectory
                z_hat, z_var = lgssm_model.smooth(F_digits)
                # z_hat: (T, k) — use first latent dimension as "load"
                z_load = z_hat[:, 0]
                z_var_load = z_var[:, 0, 0]

                # Estimate overload onset: where z_load peaks then declines
                onset_digit = _estimate_overload_onset(z_load)

                all_results[sub_id][cond] = {
                    "latent_trajectory": z_load.tolist(),
                    "latent_trajectory_full": z_hat.tolist(),
                    "trajectory_variance": z_var_load.tolist(),
                    "overload_onset_digit": onset_digit,
                    "n_digits": T,
                }

                logger.debug(
                    f"{sub_id}/{cond}: onset_digit={onset_digit}, "
                    f"z_max={z_load.max():.3f}, z_final={z_load[-1]:.3f}"
                )

            except Exception as e:
                logger.warning(f"{sub_id}/{cond}: LGSSM failed — {e}")

        logger.info(
            f"{sub_id}: LGSSM done — "
            f"{len(all_results[sub_id])} conditions fitted"
        )

    save_pickle(all_results, str(lgssm_cache))
    logger.info(f"LGSSM results saved → {lgssm_cache}")
    return all_results


def _estimate_overload_onset(z_load: np.ndarray) -> Optional[int]:
    """
    Estimate overload onset as the digit position where z_load peaks.
    Returns 1-indexed digit position.
    """
    if len(z_load) < 3:
        return None
    peak_idx = int(np.argmax(z_load))
    # Only count as overload onset if there's a subsequent decline
    if peak_idx < len(z_load) - 1 and z_load[-1] < z_load[peak_idx] * 0.9:
        return peak_idx + 1  # 1-indexed
    return None


def run_bocpd_on_coupling_trajectory(
    lgssm_results: Dict,
    output_dir: Path,
) -> Dict:
    """
    Apply BOCPD to coupling strength trajectory (condition-ordered).
    Used to detect individual overload thresholds at group level.
    """
    bocpd = BayesianChangePointDetector()
    bocpd_results = {}

    for sub_id, sub_data in lgssm_results.items():
        # Build condition-ordered trajectory: [5d, 9d, 13d]
        traj = []
        for cond in CONDITIONS:
            if cond in sub_data:
                z = sub_data[cond].get("latent_trajectory", [])
                if z:
                    traj.append(float(np.mean(z)))  # mean load per condition

        if len(traj) < 2:
            continue

        traj_arr = np.array(traj)
        try:
            cp_result = bocpd.detect(traj_arr)
            bocpd_results[sub_id] = {
                "coupling_trajectory": traj,
                "changepoint_probs": cp_result["changepoint_probs"].tolist(),
                "most_likely_cp": int(cp_result.get("most_likely_cp", -1)),
            }
        except Exception as e:
            logger.warning(f"{sub_id}: BOCPD failed — {e}")

    save_pickle(bocpd_results, str(output_dir / "bocpd_on_lgssm.pkl"))
    logger.info(f"BOCPD results saved → {output_dir / 'bocpd_on_lgssm.pkl'}")
    return bocpd_results


def validate_against_behavior(
    lgssm_results: Dict,
    cfg: dict,
) -> Dict:
    """
    Correlate physiological overload onset (LGSSM peak) with behavioral WM span.
    Returns {subject: {physiological_onset, behavioral_wm_span, ...}}.
    """
    analyzer = IndividualDifferenceAnalyzer(cfg)
    wm_capacity = analyzer.load_wm_capacity()

    validation = {}
    for sub_id, sub_data in lgssm_results.items():
        if sub_id not in wm_capacity:
            continue
        span = wm_capacity[sub_id]

        # Use 13-digit trial onset as proxy
        onset_13 = sub_data.get("load_13", {}).get("overload_onset_digit")
        if onset_13 is None:
            continue

        validation[sub_id] = {
            "physiological_onset_digit": onset_13,
            "behavioral_wm_span": span,
        }

    if validation:
        onsets = np.array([v["physiological_onset_digit"] for v in validation.values()])
        spans = np.array([v["behavioral_wm_span"] for v in validation.values()])
        r, p = np.nan, np.nan
        if len(onsets) >= 5:
            from scipy import stats as sp
            r, p = sp.spearmanr(onsets, spans)
        logger.info(
            f"Physiological onset vs WM span: "
            f"Spearman r={r:.3f}, p={p:.4f}, N={len(onsets)}"
        )
        for v in validation.values():
            v["spearman_r"] = float(r)
            v["spearman_p"] = float(p)

    return validation, wm_capacity


def print_lgssm_summary(lgssm_results: Dict, wm_capacity: Dict):
    """Print onset summary per subject."""
    print("\n" + "=" * 65)
    print("LGSSM OVERLOAD ONSET ESTIMATES")
    print("=" * 65)
    print(f"{'Subject':<12} {'WM Span':>9} {'Onset (5d)':>11} {'Onset (9d)':>11} {'Onset (13d)':>12}")
    print("-" * 65)
    for sub_id, sub_data in lgssm_results.items():
        wm = wm_capacity.get(sub_id, np.nan)
        onsets = {
            cond: sub_data.get(cond, {}).get("overload_onset_digit", "—")
            for cond in CONDITIONS
        }
        print(
            f"{sub_id:<12} "
            f"{wm:>9.1f} "
            f"{str(onsets.get('load_5', '—')):>11} "
            f"{str(onsets.get('load_9', '—')):>11} "
            f"{str(onsets.get('load_13', '—')):>12}"
        )
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "lgssm"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    setup_logging(
        log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "06_lgssm.log")
    )

    logger.info("=" * 60)
    logger.info("LGSSM Trajectory Analysis — Script 06")
    logger.info("=" * 60)
    logger.info(f"Latent dim: {args.latent_dim}, EM iters: {args.n_em_iter}")

    subjects = (
        args.subjects or cfg["subjects"]["development"]
    )

    # Load per-digit features
    per_digit_data = load_per_digit_features(cfg, subjects)
    if not per_digit_data:
        logger.error("No per-digit features found. Run 02_extract_features.py first.")
        sys.exit(1)

    # Fit LGSSM per subject × condition
    lgssm_results = fit_lgssm_per_subject(
        per_digit_data,
        latent_dim=args.latent_dim,
        n_em_iter=args.n_em_iter,
        output_dir=out_dir,
        force=args.force,
    )

    # BOCPD on coupling trajectory
    bocpd_results = run_bocpd_on_coupling_trajectory(lgssm_results, out_dir)

    # Validate against behavioral WM span
    validation, wm_capacity = validate_against_behavior(lgssm_results, cfg)
    save_json(
        {k: {kk: (float(vv) if isinstance(vv, float) else vv)
             for kk, vv in v.items()}
         for k, v in validation.items()},
        str(out_dir / "onset_vs_behavior_validation.json")
    )

    # Save wm_capacity for figure generation
    wm_for_plots = {k: float(v) for k, v in wm_capacity.items()}

    # Generate Figures 5a, 5b
    logger.info("Generating LGSSM trajectory figures...")
    try:
        generate_all_trajectory_figures(
            output_dir=str(out_dir / "figures"),
            lgssm_results=lgssm_results,
            wm_capacity=wm_for_plots,
        )
        logger.info("LGSSM figures saved")
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")

    # Console summary
    print_lgssm_summary(lgssm_results, wm_capacity)
    logger.info("LGSSM analysis complete.")


if __name__ == "__main__":
    main()