"""
Script 04: Partial Information Decomposition Analysis
=====================================================
Compute Gaussian PID for all pairwise modality combinations per condition.
Tests the LC-NE hypothesis:
  - Low load: EEG-pupil redundancy high (shared LC-NE driver)
  - Medium load (9d): synergy peaks (complementary cognitive systems)
  - Overload (13d): synergy collapses (decoupling)

PID quantities (exact closed-form for Gaussian variables):
  I(X1,X2;Y) = Unique(X1) + Unique(X2) + Redundancy + Synergy

Uses 2-3 PCs per modality for numerical stability and adequate sample size.

Outputs:
  - PID decomposition per modality pair per condition
  - Non-monotonic synergy hypothesis test (bootstrap)
  - Figure 3: PID decomposition plots

Run: python scripts/04_pid_analysis.py
"""

import argparse
import logging
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle, save_json
from utils.config_loader import load_config
from models.pid_gaussian import GaussianPID
from evaluation.bootstrap import BootstrapAnalyzer
from visualization.pid_plots import (
    plot_pid_decomposition,
    plot_pid_synergy_comparison,
    plot_pid_heatmap,
    generate_all_pid_figures,
)

logger = logging.getLogger(__name__)

CONDITIONS = ["load_5", "load_9", "load_13"]
CONDITION_LABELS = {
    "load_5": "5-digit (Low)",
    "load_9": "9-digit (Medium)",
    "load_13": "13-digit (Overload)",
}

# Modality pairs to analyze
MODALITY_PAIRS = [
    ("eeg", "pupil"),    # LC-NE hypothesis: should show high redundancy
    ("ppg", "pupil"),    # Sympathetic co-activation
    ("eeg", "ecg"),      # HEP-mediated coupling (vagal-cortical loop)
    ("ecg", "pupil"),    # Autonomic co-regulation
    ("eeg", "ppg"),      # Cortical-peripheral sympathetic
    ("ecg", "ppg"),      # Cardiac-peripheral
]


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian PID analysis for ds003838")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--n-pcs", type=int, default=3,
                   help="PCs per modality for PID (reduces dimensionality)")
    p.add_argument("--n-bootstrap", type=int, default=500,
                   help="Bootstrap iterations for synergy hypothesis test")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def reduce_modality_features(
    F_c: np.ndarray,
    feature_names: List[str],
    modality: str,
    n_pcs: int,
) -> Tuple[np.ndarray, str]:
    """
    Extract features for a given modality from the full matrix F_c,
    then reduce to n_pcs principal components.
    Returns (reduced_matrix, modality_label).
    """
    # Identify feature columns belonging to this modality
    modality_prefixes = {
        "eeg": ("eeg_", "frontal_theta", "alpha_", "iaf_", "csp_", "p300_", "n200_"),
        "ecg": ("ecg_", "hrv_", "hep_", "rmssd", "sdnn", "lf_", "hf_"),
        "ppg": ("ppg_", "pwa_", "pulse_"),
        "pupil": ("pupil_", "tepr_", "dilation_"),
    }

    prefixes = modality_prefixes.get(modality, (modality,))
    col_idx = [
        i for i, name in enumerate(feature_names)
        if any(name.lower().startswith(p) for p in prefixes)
    ]

    if not col_idx:
        # Fallback: use all features (should not happen with correct naming)
        logger.warning(f"No features matched for modality '{modality}' — using all")
        col_idx = list(range(F_c.shape[1]))

    X_mod = F_c[:, col_idx]

    # Remove constant features
    std = X_mod.std(axis=0)
    X_mod = X_mod[:, std > 1e-10]

    if X_mod.shape[1] == 0:
        logger.error(f"All features constant for modality {modality}")
        return np.zeros((F_c.shape[0], n_pcs)), f"{modality}_PC"

    # PCA reduction
    n_pcs_actual = min(n_pcs, X_mod.shape[1], X_mod.shape[0] - 1)
    pca = PCA(n_components=n_pcs_actual)
    X_reduced = pca.fit_transform(X_mod)
    var_explained = pca.explained_variance_ratio_.sum()
    logger.debug(
        f"  {modality}: {X_mod.shape[1]} features → {n_pcs_actual} PCs "
        f"(var explained: {var_explained:.1%})"
    )

    return X_reduced, f"{modality}_PC"


def build_pid_target(
    condition_matrices: Dict,
    condition: str,
) -> np.ndarray:
    """
    Build target variable Y for PID.
    Y = condition label (as ordinal: 1, 2, 3 for 5d, 9d, 13d).
    For within-condition analysis, use recall accuracy if available.
    """
    cond_to_label = {"load_5": 1, "load_9": 2, "load_13": 3}
    mat_info = condition_matrices[condition]
    N = mat_info["matrix"].shape[0]
    Y = np.full(N, cond_to_label[condition], dtype=float)
    return Y


def run_pid_for_condition(
    F_c: np.ndarray,
    feature_names: List[str],
    condition: str,
    n_pcs: int,
    pid_model: GaussianPID,
) -> Dict:
    """
    Run PID analysis for all modality pairs in one condition.
    Returns dict: modality_pair_str → PID decomposition dict.
    """
    condition_results = {}

    # Build target: concatenate all conditions for multi-condition target
    # For single-condition: use within-condition trial index as proxy (continuous)
    N = F_c.shape[0]
    Y = np.arange(N, dtype=float)  # Trial index as proxy; override with recall accuracy if available

    for mod1, mod2 in MODALITY_PAIRS:
        pair_key = f"{mod1}_{mod2}"
        try:
            X1, _ = reduce_modality_features(F_c, feature_names, mod1, n_pcs)
            X2, _ = reduce_modality_features(F_c, feature_names, mod2, n_pcs)

            if X1.shape[0] != X2.shape[0]:
                logger.warning(f"{pair_key}: sample count mismatch, skipping")
                continue

            pid_result = pid_model.compute(X1, X2, Y)
            condition_results[pair_key] = pid_result

            logger.debug(
                f"  {condition}/{pair_key}: "
                f"R={pid_result['redundancy']:.4f}, "
                f"U1={pid_result['unique_x1']:.4f}, "
                f"U2={pid_result['unique_x2']:.4f}, "
                f"S={pid_result['synergy']:.4f}"
            )

        except Exception as e:
            logger.warning(f"PID failed for {condition}/{pair_key}: {e}")

    return condition_results


def run_cross_condition_pid(
    condition_matrices: Dict,
    n_pcs: int,
    pid_model: GaussianPID,
) -> Dict:
    """
    Run PID for all conditions using combined condition as target Y.
    Y = condition label (1/2/3) — tests load-discriminating synergy.
    """
    logger.info("Running cross-condition PID (Y = condition label)...")

    # Pool all conditions
    all_matrices = []
    all_labels = []
    cond_to_label = {"load_5": 1, "load_9": 2, "load_13": 3}

    for cond in CONDITIONS:
        if cond not in condition_matrices:
            continue
        F_c = condition_matrices[cond]["matrix"]
        all_matrices.append(F_c)
        all_labels.extend([cond_to_label[cond]] * len(F_c))

    if not all_matrices:
        return {}

    F_all = np.vstack(all_matrices)
    Y_all = np.array(all_labels, dtype=float)
    feature_names = condition_matrices[CONDITIONS[0]]["feature_names"]

    cross_cond_results = {}
    for mod1, mod2 in MODALITY_PAIRS:
        pair_key = f"{mod1}_{mod2}"
        try:
            X1, _ = reduce_modality_features(F_all, feature_names, mod1, n_pcs)
            X2, _ = reduce_modality_features(F_all, feature_names, mod2, n_pcs)
            pid_result = pid_model.compute(X1, X2, Y_all)
            cross_cond_results[pair_key] = pid_result
            logger.info(
                f"  Cross-condition {pair_key}: "
                f"Synergy={pid_result['synergy']:.4f}, "
                f"Redundancy={pid_result['redundancy']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Cross-condition PID failed for {pair_key}: {e}")

    return cross_cond_results


def test_synergy_hypothesis(
    pid_results: Dict,  # condition → pair → PID dict
    n_bootstrap: int,
    condition_matrices: Dict,
    n_pcs: int,
) -> Dict:
    """
    Bootstrap test: synergy peaks at 9-digit, collapses at 13-digit.
    H0: synergy(load_9) <= synergy(load_13)
    H1: synergy(load_9) > synergy(load_13)
    """
    logger.info("Testing non-monotonic synergy hypothesis via bootstrap...")
    bootstrapper = BootstrapAnalyzer(n_bootstrap=n_bootstrap)
    hypothesis_results = {}

    for mod1, mod2 in MODALITY_PAIRS:
        pair_key = f"{mod1}_{mod2}"

        if ("load_9" not in pid_results or "load_13" not in pid_results):
            continue
        if pair_key not in pid_results.get("load_9", {}):
            continue

        try:
            F_9 = condition_matrices["load_9"]["matrix"]
            F_13 = condition_matrices["load_13"]["matrix"]
            feature_names = condition_matrices["load_9"]["feature_names"]

            X1_9, _ = reduce_modality_features(F_9, feature_names, mod1, n_pcs)
            X2_9, _ = reduce_modality_features(F_9, feature_names, mod2, n_pcs)
            X1_13, _ = reduce_modality_features(F_13, feature_names, mod1, n_pcs)
            X2_13, _ = reduce_modality_features(F_13, feature_names, mod2, n_pcs)

            result = bootstrapper.test_pid_synergy_difference(
                X1_9, X2_9, X1_13, X2_13,
                alternative="greater",
            )
            hypothesis_results[pair_key] = result

            sig = "SIG" if result.get("significant", False) else "ns"
            logger.info(
                f"  {pair_key}: synergy_9d={result.get('synergy_a', '?'):.4f}, "
                f"synergy_13d={result.get('synergy_b', '?'):.4f}, "
                f"p={result.get('p_value', '?'):.4f} [{sig}]"
            )

        except Exception as e:
            logger.warning(f"Bootstrap synergy test failed for {pair_key}: {e}")

    return hypothesis_results


def print_pid_summary(pid_results: Dict, hypothesis_results: Dict):
    """Print PID summary table."""
    print("\n" + "=" * 70)
    print("PID ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<18} {'Cond':<12} {'Redundancy':>12} {'Unique X1':>10} {'Unique X2':>10} {'Synergy':>10}")
    print("-" * 70)
    for cond in CONDITIONS:
        if cond not in pid_results:
            continue
        for pair_key, pid in pid_results[cond].items():
            print(
                f"{pair_key:<18} {CONDITION_LABELS.get(cond, cond):<12} "
                f"{pid.get('redundancy', 0):>12.4f} "
                f"{pid.get('unique_x1', 0):>10.4f} "
                f"{pid.get('unique_x2', 0):>10.4f} "
                f"{pid.get('synergy', 0):>10.4f}"
            )
    print()
    print("Synergy hypothesis test (9d > 13d):")
    for pair_key, res in hypothesis_results.items():
        sig_str = "***" if res.get("significant") else "ns"
        print(f"  {pair_key:<18}: p={res.get('p_value', '?'):.4f} {sig_str}")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "pid"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(cfg["paths"]["output_root"]) / "features"

    setup_logging(
        log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "04_pid_analysis.log")
    )

    logger.info("=" * 60)
    logger.info("Gaussian PID Analysis — Script 04")
    logger.info("=" * 60)
    logger.info(f"PCs per modality: {args.n_pcs}")
    logger.info(f"Bootstrap iterations: {args.n_bootstrap}")

    # Load feature matrices
    mat_path = feat_dir / "condition_feature_matrices.pkl"
    if not mat_path.exists():
        logger.error("Feature matrices not found. Run 02_extract_features.py first.")
        sys.exit(1)
    condition_matrices = load_pickle(str(mat_path))

    # Cache
    pid_cache = out_dir / "pid_results.pkl"
    if pid_cache.exists() and not args.force:
        logger.info("Loading cached PID results...")
        pid_results = load_pickle(str(pid_cache))
    else:
        # Initialize PID model
        pid_model = GaussianPID()
        pid_results = {}

        for cond in CONDITIONS:
            if cond not in condition_matrices:
                logger.warning(f"{cond} not in condition matrices — skipping")
                continue

            logger.info(f"\nRunning PID for {CONDITION_LABELS.get(cond, cond)}...")
            F_c = condition_matrices[cond]["matrix"]
            feature_names = condition_matrices[cond]["feature_names"]

            cond_pid = run_pid_for_condition(
                F_c, feature_names, cond, args.n_pcs, pid_model
            )
            pid_results[cond] = cond_pid
            logger.info(f"  {cond}: {len(cond_pid)} pairs analyzed")

        save_pickle(pid_results, str(pid_cache))
        logger.info(f"PID results saved → {pid_cache}")

    # Cross-condition PID (Y = condition label)
    cross_cond_cache = out_dir / "pid_cross_condition.pkl"
    if cross_cond_cache.exists() and not args.force:
        cross_cond_results = load_pickle(str(cross_cond_cache))
    else:
        pid_model = GaussianPID()
        cross_cond_results = run_cross_condition_pid(condition_matrices, args.n_pcs, pid_model)
        save_pickle(cross_cond_results, str(cross_cond_cache))

    # Bootstrap synergy hypothesis test
    hyp_cache = out_dir / "synergy_hypothesis_results.pkl"
    if hyp_cache.exists() and not args.force:
        hypothesis_results = load_pickle(str(hyp_cache))
    else:
        hypothesis_results = test_synergy_hypothesis(
            pid_results, args.n_bootstrap, condition_matrices, args.n_pcs
        )
        save_pickle(hypothesis_results, str(hyp_cache))

    # Serializable summary
    hyp_summary = {
        k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
        for k, v in hypothesis_results.items()
    }
    save_json(hyp_summary, str(out_dir / "synergy_hypothesis_summary.json"))

    # Generate Figure 3
    logger.info("Generating PID figures...")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_all_pid_figures(
            pid_results=pid_results,
            cross_condition_results=cross_cond_results,
            hypothesis_results=hypothesis_results,
            modality_pairs=MODALITY_PAIRS,
            condition_labels=CONDITION_LABELS,
            output_dir=str(fig_dir),
        )
        logger.info("PID figures saved")
    except Exception as e:
        logger.error(f"PID figure generation failed: {e}")

    print_pid_summary(pid_results, hypothesis_results)
    logger.info("PID analysis complete.")


if __name__ == "__main__":
    main()