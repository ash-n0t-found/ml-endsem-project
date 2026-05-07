"""
Script 07: Resting-State Coupling Analysis
==========================================
Fit per-subject resting-state GGMs. Test whether resting cross-modal
coupling predicts:
  1. Behavioral WM span (capacity predictor)
  2. Task-state coupling strength (calibration of physiological response)
  3. Coupling decrement from 9d→13d (individual overload threshold)

Key finding predicted:
  Subjects with stronger resting EEG-ECG coupling show larger
  task-induced coupling changes — resting coupling calibrates
  sensitivity to cognitive demand.

Outputs:
  - Per-subject resting GGM (Θ̂_s_0)
  - Delta coupling: ΔΘ_c = Θ̂_c − Θ̂_s_0
  - Regression: resting coupling → WM span (R², Pearson r)
  - Figure 6: Resting coupling vs WM capacity scatter plot

Run: python scripts/07_resting_analysis.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle, save_json
from utils.config_loader import load_config
from utils.normalization import SubjectNormalizer
from models.ggm import GGMEstimator
from analysis.individual_diff import IndividualDifferenceAnalyzer
from analysis.topology import TopologyAnalyzer
from evaluation.bootstrap import BootstrapAnalyzer
from visualization.trajectory_plots import (
    plot_resting_coupling_vs_wm,
    generate_all_trajectory_figures,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Resting-state coupling analysis — Script 07")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--lambda-method", default="cv", choices=["cv", "bic", "stability"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--subjects", nargs="+", default=None)
    return p.parse_args()


def load_resting_features(cfg: dict, subjects: List[str]) -> Dict[str, Dict]:
    """
    Load resting-state feature vectors for each subject.
    Returns: subject_id → {feature_name: value, ...}
    """
    feat_cache_dir = Path(cfg["paths"]["cache_dir"]) / "features"
    resting_features = {}

    for sub_id in subjects:
        sub_feat_path = feat_cache_dir / f"{sub_id}_features.pkl"
        if not sub_feat_path.exists():
            logger.warning(f"{sub_id}: feature cache not found")
            continue
        sub_feats = load_pickle(str(sub_feat_path))
        resting = sub_feats.get("resting", {})
        if resting:
            resting_features[sub_id] = resting
        else:
            logger.warning(f"{sub_id}: no resting features found")

    logger.info(f"Resting features loaded for {len(resting_features)} subjects")
    return resting_features


def build_resting_feature_matrix(
    resting_features: Dict[str, Dict],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build (N_subjects, D_resting) matrix from per-subject resting feature dicts.
    Align feature names across subjects (inner join: only features present in all).
    Returns: (matrix, feature_names, subject_ids)
    """
    if not resting_features:
        return np.array([]), [], []

    # Get common features across all subjects
    feature_sets = [set(v.keys()) for v in resting_features.values()]
    common_features = sorted(feature_sets[0].intersection(*feature_sets[1:]))

    if not common_features:
        logger.warning("No common resting features across subjects — using union with NaN fill")
        all_features = sorted(set().union(*feature_sets))
        common_features = all_features

    subject_ids = list(resting_features.keys())
    matrix = np.array([
        [resting_features[s].get(f, np.nan) for f in common_features]
        for s in subject_ids
    ], dtype=float)

    # Drop columns with >50% NaN
    nan_frac = np.isnan(matrix).mean(axis=0)
    keep = nan_frac < 0.5
    matrix = matrix[:, keep]
    common_features = [f for f, k in zip(common_features, keep) if k]

    # Impute remaining NaN with column median
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_median = np.nanmedian(col)
        matrix[np.isnan(matrix[:, j]), j] = col_median

    logger.info(
        f"Resting feature matrix: {matrix.shape[0]} subjects × {matrix.shape[1]} features"
    )
    return matrix, common_features, subject_ids


def fit_resting_ggms_per_subject(
    resting_features: Dict[str, Dict],
    cfg: dict,
    output_dir: Path,
    lambda_method: str,
    force: bool = False,
) -> Dict:
    """
    Fit individual GGM per subject using their resting-state feature vector.
    Note: with only 1 vector per subject for resting, we use bootstrap resampling
    from resting epoch segments to build a feature matrix per subject.
    """
    resting_ggm_cache = output_dir / "resting_ggms.pkl"
    if resting_ggm_cache.exists() and not force:
        logger.info("Loading cached resting GGMs...")
        return load_pickle(str(resting_ggm_cache))

    estimator = GGMEstimator(lambda_method=lambda_method, cfg=cfg)
    resting_ggms = {}

    # Load resting epoch features (multi-epoch resting data needed for GGM)
    feat_cache_dir = Path(cfg["paths"]["cache_dir"]) / "features"

    for sub_id in resting_features:
        try:
            # Load multi-epoch resting feature matrix if available
            rest_epoch_path = feat_cache_dir / f"{sub_id}_resting_epochs.pkl"
            if rest_epoch_path.exists():
                F_rest = load_pickle(str(rest_epoch_path))
            else:
                # Fall back: use single resting feature vector
                # This limits GGM precision but provides a coupling baseline
                feat_vec = np.array(list(resting_features[sub_id].values()))
                feat_names = list(resting_features[sub_id].keys())
                # Cannot fit GGM from single vector — use correlation proxy
                logger.warning(
                    f"{sub_id}: no multi-epoch resting data — "
                    f"using scalar resting features as coupling proxy"
                )
                resting_ggms[sub_id] = {
                    "precision_matrix": None,
                    "resting_feature_vector": feat_vec,
                    "feature_names": feat_names,
                    "coupling_proxy": True,
                }
                continue

            result = estimator.fit(
                F_rest["matrix"],
                feature_names=F_rest["feature_names"],
                condition="resting",
            )
            resting_ggms[sub_id] = result
            n_edges = result.get("n_stable_edges", 0)
            logger.info(f"{sub_id}: resting GGM — {n_edges} stable edges")

        except Exception as e:
            logger.warning(f"{sub_id}: resting GGM failed — {e}")

    save_pickle(resting_ggms, str(resting_ggm_cache))
    logger.info(f"Resting GGMs saved → {resting_ggm_cache}")
    return resting_ggms


def compute_resting_coupling_strength(
    resting_features: Dict[str, Dict],
    resting_ggms: Dict,
    topology_analyzer: TopologyAnalyzer,
) -> Dict[str, float]:
    """
    Extract scalar resting cross-modal coupling strength per subject.
    Uses Frobenius norm of cross-block precision matrix entries,
    or resting EEG-ECG coherence as fallback.
    """
    coupling_strengths = {}

    for sub_id in resting_features:
        try:
            ggm = resting_ggms.get(sub_id, {})

            if ggm.get("precision_matrix") is not None:
                # Full GGM available: use cross-modal Frobenius norm
                theta = ggm["precision_matrix"]
                feature_names = ggm.get("feature_names", [])
                metrics = topology_analyzer.compute_metrics(
                    theta, feature_names=feature_names, condition="resting"
                )
                coupling_strengths[sub_id] = float(metrics.get("cross_modal_coupling_strength", 0))

            else:
                # Fallback: use resting EEG-ECG coherence feature
                rest_feats = resting_features[sub_id]
                coherence_keys = [
                    k for k in rest_feats
                    if "coherence" in k.lower() or "coupling" in k.lower()
                ]
                if coherence_keys:
                    coupling_strengths[sub_id] = float(
                        np.mean([rest_feats[k] for k in coherence_keys])
                    )
                elif "resting_hf_hrv" in rest_feats and "resting_frontal_theta" in rest_feats:
                    # Proxy: HRV × theta product (shared autonomic-cortical activity)
                    coupling_strengths[sub_id] = float(
                        rest_feats["resting_hf_hrv"] * rest_feats["resting_frontal_theta"]
                    )
                else:
                    logger.warning(f"{sub_id}: no resting coupling proxy available")

        except Exception as e:
            logger.warning(f"{sub_id}: coupling strength computation failed — {e}")

    logger.info(
        f"Resting coupling strengths computed for {len(coupling_strengths)} subjects: "
        f"mean={np.mean(list(coupling_strengths.values())):.4f}"
    )
    return coupling_strengths


def compute_delta_coupling(
    coupling_strengths: Dict[str, float],
    task_ggm_results: Optional[Dict],
    topology_analyzer: TopologyAnalyzer,
) -> Dict[str, Dict]:
    """
    Compute ΔΘ per subject per condition: task coupling − resting coupling.
    Controls for individual differences in baseline physiological coupling.
    """
    if task_ggm_results is None:
        return {}

    delta_coupling = {}
    for sub_id, resting_strength in coupling_strengths.items():
        delta_coupling[sub_id] = {}
        for cond, ggm_res in task_ggm_results.items():
            theta_task = ggm_res.get("precision_matrix")
            if theta_task is None:
                continue
            feature_names = ggm_res.get("feature_names", [])
            metrics = topology_analyzer.compute_metrics(
                theta_task, feature_names=feature_names, condition=cond
            )
            task_strength = float(metrics.get("cross_modal_coupling_strength", 0))
            delta = task_strength - resting_strength
            delta_coupling[sub_id][cond] = {
                "resting_strength": resting_strength,
                "task_strength": task_strength,
                "delta": delta,
            }

    return delta_coupling


def regress_resting_coupling_on_wm(
    coupling_strengths: Dict[str, float],
    wm_capacity: Dict[str, float],
    output_dir: Path,
) -> Dict:
    """
    Linear regression: resting cross-modal coupling → WM span.
    Returns regression stats + per-subject data for Figure 6.
    """
    common_subs = sorted(
        s for s in coupling_strengths if s in wm_capacity
    )
    if len(common_subs) < 5:
        logger.warning(f"Too few subjects for regression: {len(common_subs)}")
        return {}

    X = np.array([coupling_strengths[s] for s in common_subs])
    Y = np.array([wm_capacity[s] for s in common_subs])

    slope, intercept, r_val, p_val, se = scipy_stats.linregress(X, Y)
    r2 = r_val ** 2

    result = {
        "n_subjects": len(common_subs),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_val),
        "r2": float(r2),
        "p_value": float(p_val),
        "se": float(se),
        "significant": bool(p_val < 0.05),
        "subject_ids": common_subs,
        "coupling_values": X.tolist(),
        "wm_values": Y.tolist(),
    }

    logger.info(
        f"Resting coupling → WM span: "
        f"r={r_val:.3f}, R²={r2:.3f}, p={p_val:.4f}, N={len(common_subs)}"
    )
    save_json(result, str(output_dir / "resting_coupling_wm_regression.json"))
    return result


def run_additional_regressions(
    coupling_strengths: Dict[str, float],
    delta_coupling: Dict[str, Dict],
    wm_capacity: Dict[str, float],
    output_dir: Path,
) -> Dict:
    """
    Additional regressions:
    1. Task coupling at 9d → WM span (beyond resting)
    2. Delta coupling (9d→13d) → individual overload threshold
    """
    results = {}

    # Regression 1: 9d coupling strength → WM span
    subs_9d = [s for s in delta_coupling if "load_9" in delta_coupling[s] and s in wm_capacity]
    if len(subs_9d) >= 5:
        X_9d = np.array([delta_coupling[s]["load_9"]["task_strength"] for s in subs_9d])
        Y_wm = np.array([wm_capacity[s] for s in subs_9d])
        slope, intercept, r, p, se = scipy_stats.linregress(X_9d, Y_wm)
        results["task_9d_coupling_vs_wm"] = {
            "r": float(r), "r2": float(r**2), "p_value": float(p),
            "n_subjects": len(subs_9d),
        }
        logger.info(
            f"9d task coupling → WM span: r={r:.3f}, R²={r**2:.3f}, p={p:.4f}"
        )

    # Regression 2: Delta coupling (9d→13d) as overload predictor
    subs_delta = [
        s for s in delta_coupling
        if "load_9" in delta_coupling[s] and "load_13" in delta_coupling[s]
        and s in wm_capacity
    ]
    if len(subs_delta) >= 5:
        X_delta = np.array([
            delta_coupling[s]["load_9"]["task_strength"] -
            delta_coupling[s]["load_13"]["task_strength"]
            for s in subs_delta
        ])  # Positive = more decoupling at 13d (lower WM capacity expected)
        Y_wm = np.array([wm_capacity[s] for s in subs_delta])
        slope, intercept, r, p, se = scipy_stats.linregress(X_delta, Y_wm)
        results["coupling_decrement_vs_wm"] = {
            "r": float(r), "r2": float(r**2), "p_value": float(p),
            "n_subjects": len(subs_delta),
        }
        logger.info(
            f"Coupling decrement (9d→13d) → WM span: r={r:.3f}, R²={r**2:.3f}, p={p:.4f}"
        )

    save_json(results, str(output_dir / "additional_regressions.json"))
    return results


def print_resting_summary(
    regression_result: Dict,
    additional_regressions: Dict,
):
    print("\n" + "=" * 60)
    print("RESTING-STATE COUPLING ANALYSIS SUMMARY")
    print("=" * 60)
    if regression_result:
        r2 = regression_result.get("r2", np.nan)
        r = regression_result.get("r", np.nan)
        p = regression_result.get("p_value", np.nan)
        n = regression_result.get("n_subjects", "?")
        sig = "SIGNIFICANT" if regression_result.get("significant") else "NOT significant"
        print(f"Resting coupling → WM span:")
        print(f"  r={r:.3f}, R²={r2:.3f}, p={p:.4f}, N={n} [{sig}]")
        print()

    for key, res in additional_regressions.items():
        r2 = res.get("r2", np.nan)
        r = res.get("r", np.nan)
        p = res.get("p_value", np.nan)
        n = res.get("n_subjects", "?")
        print(f"{key}:")
        print(f"  r={r:.3f}, R²={r2:.3f}, p={p:.4f}, N={n}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "resting_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    setup_logging(
        log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "07_resting_analysis.log")
    )

    logger.info("=" * 60)
    logger.info("Resting-State Coupling Analysis — Script 07")
    logger.info("=" * 60)

    subjects = args.subjects or cfg["subjects"]["development"]

    # Load resting features
    resting_features = load_resting_features(cfg, subjects)
    if not resting_features:
        logger.error("No resting features. Run 02_extract_features.py first.")
        sys.exit(1)

    # Build resting feature matrix
    R, resting_feat_names, sub_ids = build_resting_feature_matrix(resting_features)
    if R.size == 0:
        logger.error("Empty resting feature matrix.")
        sys.exit(1)

    # Fit per-subject resting GGMs
    resting_ggms = fit_resting_ggms_per_subject(
        resting_features, cfg, out_dir, args.lambda_method, args.force
    )

    # Load behavioral WM capacity
    analyzer = IndividualDifferenceAnalyzer(cfg)
    wm_capacity = analyzer.load_wm_capacity()

    # Topology analysis
    topology_analyzer = TopologyAnalyzer(cfg)

    # Compute scalar coupling strength per subject
    coupling_strengths = compute_resting_coupling_strength(
        resting_features, resting_ggms, topology_analyzer
    )
    save_json(
        {k: float(v) for k, v in coupling_strengths.items()},
        str(out_dir / "resting_coupling_strengths.json")
    )

    # Load task-level GGM results for delta computation
    task_ggm_path = Path(cfg["paths"]["output_root"]) / "ggm" / "ggm_results.pkl"
    task_ggm_results = load_pickle(str(task_ggm_path)) if task_ggm_path.exists() else None

    # Delta coupling: task − resting per subject
    delta_coupling = compute_delta_coupling(
        coupling_strengths, task_ggm_results, topology_analyzer
    )
    save_pickle(delta_coupling, str(out_dir / "delta_coupling.pkl"))

    # Regression: resting coupling → WM span
    regression_result = regress_resting_coupling_on_wm(
        coupling_strengths, wm_capacity, out_dir
    )

    # Additional regressions
    additional_regressions = run_additional_regressions(
        coupling_strengths, delta_coupling, wm_capacity, out_dir
    )

    # Bootstrap CI for r
    if regression_result and len(regression_result.get("coupling_values", [])) >= 10:
        bootstrapper = BootstrapAnalyzer(n_bootstrap=1000)
        X = np.array(regression_result["coupling_values"])
        Y = np.array(regression_result["wm_values"])
        try:
            r_ci = bootstrapper.correlation_ci(X, Y, alpha=0.05)
            regression_result["r_ci_95"] = r_ci
            logger.info(f"Bootstrap r 95% CI: [{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
        except Exception as e:
            logger.warning(f"Bootstrap CI failed: {e}")
        save_json(regression_result, str(out_dir / "resting_coupling_wm_regression.json"))

    # Figure 6: Resting coupling vs WM scatter
    if regression_result and coupling_strengths and wm_capacity:
        logger.info("Generating Figure 6: Resting coupling vs WM capacity...")
        common_subs = regression_result.get("subject_ids", [])
        X_arr = np.array(regression_result.get("coupling_values", []))
        Y_arr = np.array(regression_result.get("wm_values", []))
        try:
            fig6 = plot_resting_coupling_vs_wm(
                X_arr, Y_arr,
                coupling_feature_name="Resting Cross-Modal Coupling Strength",
                subject_ids=common_subs,
                output_path=str(out_dir / "figures" / "fig6_resting_coupling_vs_wm.png"),
            )
            plt.close(fig6)
            logger.info("Figure 6 saved")
        except Exception as e:
            logger.error(f"Figure 6 failed: {e}")

    print_resting_summary(regression_result, additional_regressions)
    logger.info("Resting-state analysis complete.")


if __name__ == "__main__":
    main()