"""
Script 05: Recall Accuracy Prediction
======================================
Predict individual recall accuracy using multimodal features.
Primary ML claim: coupling-derived features from Θ_c outperform
single-modality features for predicting recall failure.

Feature sets compared (B1-B6 baselines + novel coupling):
  B1: Per-modality Catch22 + XGBoost (replicates Papers 5/6)
  B2: Late fusion of per-modality classifiers
  B3: Early fusion: concatenated features, NO coupling (critical ablation)
  B4: Full covariance (not precision matrix) features
  B5: EEG-pupil Pearson correlation (simplest coupling)
  B6: PCA of full multimodal feature vector
  Novel: Coupling features from GGM precision matrix
  Combined: Per-modality + coupling features

Validation: LOSO cross-validation (leave-one-subject-out).
Target: continuous recall accuracy (fraction correct).
Significance: Wilcoxon signed-rank test on per-subject R² differences.

Run: python scripts/05_recall_prediction.py
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
from analysis.coupling_features import CouplingFeatureExtractor
from analysis.recall_prediction import RecallPredictor
from evaluation.baselines import BaselineEvaluator
from evaluation.bootstrap import BootstrapAnalyzer
from visualization.trajectory_plots import plot_recall_r2_comparison

logger = logging.getLogger(__name__)

CONDITIONS = ["load_5", "load_9", "load_13"]

# Model keys in order for display
MODEL_ORDER = [
    "B1_catch22_xgb",
    "B2_late_fusion",
    "B3_early_fusion",
    "B4_covariance",
    "B5_eeg_pupil_corr",
    "B6_pca",
    "coupling_only",
    "marginal_only",
    "marginal_plus_coupling",
    "resting_coupling_only",
]


def parse_args():
    p = argparse.ArgumentParser(description="Recall accuracy prediction — Script 05")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--condition", default="all",
                   help="Condition to analyze (all / load_5 / load_9 / load_13)")
    return p.parse_args()


def load_inputs(cfg: dict) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Load feature matrices, GGM results, and recall accuracy labels."""
    feat_dir = Path(cfg["paths"]["output_root"]) / "features"
    ggm_dir = Path(cfg["paths"]["output_root"]) / "ggm"

    mat_path = feat_dir / "condition_feature_matrices.pkl"
    ggm_path = ggm_dir / "ggm_results.pkl"

    if not mat_path.exists():
        logger.error("Feature matrices not found. Run 02_extract_features.py.")
        return None, None, None

    if not ggm_path.exists():
        logger.error("GGM results not found. Run 03_fit_ggm.py.")
        return None, None, None

    condition_matrices = load_pickle(str(mat_path))
    ggm_results = load_pickle(str(ggm_path))

    # Load recall accuracy from feature matrices trial index
    # trial_index stores (subject_id, trial_num) per row
    # recall accuracy must come from behavioral data
    recall_accuracy = {}
    for cond, mat_info in condition_matrices.items():
        trial_index = mat_info.get("trial_index", [])
        # Placeholder: actual recall scores loaded from behavioral files
        # Structure: {(subject_id, trial_idx): recall_fraction}
        recall_accuracy[cond] = trial_index  # populated in RecallPredictor

    return condition_matrices, ggm_results, recall_accuracy


def extract_coupling_features(
    condition_matrices: Dict,
    ggm_results: Dict,
    cfg: dict,
) -> Dict[str, np.ndarray]:
    """
    Extract precision matrix-derived coupling features per trial/subject.
    Returns dict: condition → (N_trials × D_coupling) array.
    """
    extractor = CouplingFeatureExtractor(cfg)
    coupling_features = {}

    for cond, ggm_res in ggm_results.items():
        if cond not in condition_matrices:
            continue
        F_c = condition_matrices[cond]["matrix"]
        theta = ggm_res["precision_matrix"]
        feature_names = ggm_res["feature_names"]

        try:
            C_c, coupling_names = extractor.extract(
                F_c,
                precision_matrix=theta,
                feature_names=feature_names,
                condition=cond,
            )
            coupling_features[cond] = {
                "matrix": C_c,
                "feature_names": coupling_names,
            }
            logger.info(
                f"{cond}: coupling features extracted — "
                f"shape {C_c.shape}, {len(coupling_names)} features"
            )
        except Exception as e:
            logger.error(f"Coupling feature extraction failed for {cond}: {e}")

    return coupling_features


def run_loso_evaluation(
    condition_matrices: Dict,
    coupling_features: Dict,
    ggm_results: Dict,
    cfg: dict,
    n_jobs: int,
    output_dir: Path,
    force: bool,
    target_condition: str = "all",
) -> Dict:
    """
    LOSO cross-validation across all feature set variants.
    Returns nested dict: model_key → condition → {r2, rmse, per_subject_r2}
    """
    results_cache = output_dir / "loso_results.pkl"
    if results_cache.exists() and not force:
        logger.info("Loading cached LOSO results...")
        return load_pickle(str(results_cache))

    predictor = RecallPredictor(cfg, n_jobs=n_jobs)
    baseline_eval = BaselineEvaluator(cfg, n_jobs=n_jobs)

    conditions_to_eval = CONDITIONS if target_condition == "all" else [target_condition]
    all_results = {}

    for cond in conditions_to_eval:
        if cond not in condition_matrices:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating condition: {cond}")
        logger.info(f"{'='*50}")

        F_marginal = condition_matrices[cond]["matrix"]
        feature_names = condition_matrices[cond]["feature_names"]
        trial_index = condition_matrices[cond]["trial_index"]

        C_coupling = coupling_features.get(cond, {}).get("matrix", None)
        coupling_names = coupling_features.get(cond, {}).get("feature_names", [])

        # ---- Baselines B1-B6 -----------------------------------------------
        for b_key, b_label in [
            ("B1_catch22_xgb", "Catch22 + XGBoost"),
            ("B2_late_fusion", "Late fusion"),
            ("B3_early_fusion", "Early fusion (concatenated, no coupling)"),
            ("B4_covariance", "Full covariance features"),
            ("B5_eeg_pupil_corr", "EEG-pupil Pearson correlation"),
            ("B6_pca", "PCA multimodal"),
        ]:
            logger.info(f"Running {b_label}...")
            try:
                b_result = baseline_eval.run(
                    F_marginal, trial_index, feature_names,
                    baseline_key=b_key, condition=cond,
                )
                if b_key not in all_results:
                    all_results[b_key] = {}
                all_results[b_key][cond] = b_result
                logger.info(
                    f"  {b_label}: R²={b_result.get('mean_r2', np.nan):.4f} "
                    f"± {b_result.get('std_r2', np.nan):.4f}"
                )
            except Exception as e:
                logger.warning(f"  {b_label} failed: {e}")

        # ---- Novel coupling models ------------------------------------------
        if C_coupling is not None:

            # Coupling features only
            logger.info("Running coupling-only model...")
            try:
                coup_result = predictor.run_loso(
                    C_coupling, trial_index, coupling_names,
                    model_key="coupling_only", condition=cond,
                )
                all_results.setdefault("coupling_only", {})[cond] = coup_result
                logger.info(
                    f"  Coupling-only: R²={coup_result.get('mean_r2', np.nan):.4f}"
                )
            except Exception as e:
                logger.warning(f"  Coupling-only failed: {e}")

            # Marginal features only (for fair comparison)
            logger.info("Running marginal-only model...")
            try:
                marg_result = predictor.run_loso(
                    F_marginal, trial_index, feature_names,
                    model_key="marginal_only", condition=cond,
                )
                all_results.setdefault("marginal_only", {})[cond] = marg_result
                logger.info(
                    f"  Marginal-only: R²={marg_result.get('mean_r2', np.nan):.4f}"
                )
            except Exception as e:
                logger.warning(f"  Marginal-only failed: {e}")

            # Combined: marginal + coupling
            logger.info("Running marginal + coupling model...")
            try:
                F_combined = np.hstack([F_marginal, C_coupling])
                combined_names = feature_names + coupling_names
                comb_result = predictor.run_loso(
                    F_combined, trial_index, combined_names,
                    model_key="marginal_plus_coupling", condition=cond,
                )
                all_results.setdefault("marginal_plus_coupling", {})[cond] = comb_result
                logger.info(
                    f"  Marginal + coupling: R²={comb_result.get('mean_r2', np.nan):.4f}"
                )
            except Exception as e:
                logger.warning(f"  Combined failed: {e}")

        # Resting-state coupling only (zero-shot)
        logger.info("Running resting-state coupling-only model...")
        try:
            rest_result = predictor.run_loso_resting(
                trial_index, condition=cond,
                cfg_paths=cfg["paths"],
            )
            if rest_result:
                all_results.setdefault("resting_coupling_only", {})[cond] = rest_result
                logger.info(
                    f"  Resting coupling: R²={rest_result.get('mean_r2', np.nan):.4f}"
                )
        except Exception as e:
            logger.warning(f"  Resting coupling failed: {e}")

    save_pickle(all_results, str(results_cache))
    logger.info(f"\nLOSO results saved → {results_cache}")
    return all_results


def run_wilcoxon_tests(loso_results: Dict, output_dir: Path) -> Dict:
    """
    Wilcoxon signed-rank test: per-subject R² for coupling models vs baselines.
    Tests whether coupling adds significant predictive value.
    """
    logger.info("Running Wilcoxon signed-rank tests...")
    stat_results = {}

    coupling_key = "marginal_plus_coupling"
    baseline_keys = ["B3_early_fusion", "marginal_only", "B1_catch22_xgb"]

    for cond in CONDITIONS:
        for b_key in baseline_keys:
            test_key = f"{coupling_key}_vs_{b_key}_{cond}"
            try:
                coup_r2 = np.array(
                    loso_results.get(coupling_key, {})
                    .get(cond, {})
                    .get("per_subject_r2", [])
                )
                base_r2 = np.array(
                    loso_results.get(b_key, {})
                    .get(cond, {})
                    .get("per_subject_r2", [])
                )

                if len(coup_r2) < 5 or len(base_r2) < 5:
                    continue

                min_len = min(len(coup_r2), len(base_r2))
                w_stat, p_val = scipy_stats.wilcoxon(
                    coup_r2[:min_len], base_r2[:min_len],
                    alternative="greater",
                )
                stat_results[test_key] = {
                    "coupling_mean_r2": float(coup_r2.mean()),
                    "baseline_mean_r2": float(base_r2.mean()),
                    "w_statistic": float(w_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "condition": cond,
                    "baseline": b_key,
                }
                sig = "***" if p_val < 0.05 else "ns"
                logger.info(
                    f"  {coupling_key} vs {b_key} ({cond}): "
                    f"p={p_val:.4f} {sig} | "
                    f"ΔR²={coup_r2.mean() - base_r2.mean():.4f}"
                )
            except Exception as e:
                logger.warning(f"Wilcoxon test failed for {test_key}: {e}")

    save_json(stat_results, str(output_dir / "wilcoxon_tests.json"))
    return stat_results


def compute_bootstrap_cis(loso_results: Dict, output_dir: Path) -> Dict:
    """Bootstrap 95% CIs for mean R² per model per condition."""
    bootstrapper = BootstrapAnalyzer(n_bootstrap=1000)
    ci_results = {}

    for model_key, cond_data in loso_results.items():
        ci_results[model_key] = {}
        for cond, result in cond_data.items():
            per_sub_r2 = result.get("per_subject_r2", [])
            if len(per_sub_r2) < 3:
                continue
            try:
                ci = bootstrapper.mean_ci(np.array(per_sub_r2), alpha=0.05)
                ci_results[model_key][cond] = ci
            except Exception:
                pass

    return ci_results


def print_results_table(loso_results: Dict, stat_results: Dict):
    """Print final results table."""
    print("\n" + "=" * 75)
    print("RECALL PREDICTION RESULTS (LOSO Cross-Validation)")
    print("=" * 75)

    header = f"{'Model':<28} {'Load-5 R²':>10} {'Load-9 R²':>10} {'Load-13 R²':>12} {'Mean R²':>9}"
    print(header)
    print("-" * 75)

    for model_key in MODEL_ORDER:
        if model_key not in loso_results:
            continue
        r2s = []
        row = f"{model_key:<28}"
        for cond in CONDITIONS:
            r2 = loso_results[model_key].get(cond, {}).get("mean_r2", np.nan)
            r2s.append(r2)
            row += f" {r2:>10.4f}" if not np.isnan(r2) else f" {'--':>10}"
        mean_r2 = np.nanmean(r2s) if r2s else np.nan
        row += f" {mean_r2:>9.4f}" if not np.isnan(mean_r2) else f" {'--':>9}"
        print(row)

    print()
    print("Key Wilcoxon tests (coupling_plus_marginal > baseline):")
    for test_key, res in stat_results.items():
        if "load_9" in test_key:
            sig = "***" if res.get("significant") else "ns"
            print(
                f"  {test_key[:60]}: p={res.get('p_value', '?'):.4f} {sig} "
                f"ΔR²={res['coupling_mean_r2'] - res['baseline_mean_r2']:.4f}"
            )
    print("=" * 75 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "recall_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "05_recall_prediction.log")
    )

    logger.info("=" * 60)
    logger.info("Recall Prediction — Script 05")
    logger.info("=" * 60)

    # Load inputs
    condition_matrices, ggm_results, _ = load_inputs(cfg)
    if condition_matrices is None:
        sys.exit(1)

    # Extract coupling features
    logger.info("Extracting coupling features from GGM precision matrices...")
    coupling_features = extract_coupling_features(condition_matrices, ggm_results, cfg)

    # LOSO evaluation
    loso_results = run_loso_evaluation(
        condition_matrices, coupling_features, ggm_results,
        cfg, args.n_jobs, out_dir, args.force, args.condition
    )

    # Statistical tests
    stat_results = run_wilcoxon_tests(loso_results, out_dir)

    # Bootstrap CIs
    ci_results = compute_bootstrap_cis(loso_results, out_dir)
    save_pickle(ci_results, str(out_dir / "r2_bootstrap_cis.pkl"))

    # Figure 4: R² comparison bar chart
    logger.info("Generating Figure 4: Recall R² comparison...")
    try:
        # Use load_9 as the primary condition for the figure
        r2_for_fig = {
            m: loso_results[m].get("load_9", {}).get("mean_r2", np.nan)
            for m in loso_results
        }
        ci_for_fig = {
            m: ci_results.get(m, {}).get("load_9", (np.nan, np.nan))
            for m in loso_results
        }
        fig4 = plot_recall_r2_comparison(
            model_r2_scores={k: v for k, v in r2_for_fig.items() if not np.isnan(v)},
            model_r2_cis={k: v for k, v in ci_for_fig.items()
                          if not any(np.isnan(x) for x in v)},
            coupling_model_key="marginal_plus_coupling",
            output_path=str(out_dir / "figures" / "fig4_recall_r2.png"),
        )
        plt.close(fig4)
        logger.info("Figure 4 saved")
    except Exception as e:
        logger.error(f"Figure 4 failed: {e}")

    # Results table
    print_results_table(loso_results, stat_results)

    # Serializable summary
    summary = {
        model_key: {
            cond: {
                "mean_r2": float(res.get("mean_r2", np.nan)),
                "std_r2": float(res.get("std_r2", np.nan)),
                "rmse": float(res.get("rmse", np.nan)),
            }
            for cond, res in cond_data.items()
        }
        for model_key, cond_data in loso_results.items()
    }
    save_json(summary, str(out_dir / "05_recall_prediction_summary.json"))
    logger.info("Recall prediction complete.")


if __name__ == "__main__":
    main()