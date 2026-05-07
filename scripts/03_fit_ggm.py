"""
Script 03: GGM Estimation and Topology Analysis
================================================
Fit sparse Gaussian Graphical Models (GraphLasso with stability selection)
per condition. Compute cross-modal coupling topology and test the primary
scientific hypothesis:

    Cross-modal edge density: control < 5-digit < 9-digit > 13-digit
    (non-monotonic: peaks at medium load, drops at overload)

Outputs:
  - Precision matrices per condition (Θ̂_c)
  - Network topology metrics (edge density, clustering, centrality)
  - Primary permutation test: 9-digit vs 13-digit cross-modal edges
  - Figure 1: Precision matrix network graphs (4 conditions)
  - Figure 2: Cross-modal edge density vs condition curve

Run: python scripts/03_fit_ggm.py [--subjects ...] [--lambda-method stability]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle, save_json
from utils.config_loader import load_config
from models.ggm import GGMEstimator
from analysis.topology import TopologyAnalyzer
from evaluation.permutation_tests import PermutationTester
from evaluation.bootstrap import BootstrapAnalyzer
from visualization.precision_graphs import plot_precision_networks, plot_edge_density_curve

logger = logging.getLogger(__name__)

CONDITIONS = ["load_5", "load_9", "load_13"]
CONDITION_LABELS = {
    "control": "Control",
    "load_5": "5-digit (Low)",
    "load_9": "9-digit (Medium)",
    "load_13": "13-digit (Overload)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Fit GGMs and analyze coupling topology")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--lambda-method", default="stability",
                   choices=["cv", "stability", "bic"],
                   help="Regularization selection method")
    p.add_argument("--n-subsamples", type=int, default=100,
                   help="Stability selection subsamples")
    p.add_argument("--pi-threshold", type=float, default=0.6,
                   help="Stability selection inclusion threshold")
    p.add_argument("--n-permutations", type=int, default=1000,
                   help="Permutation test iterations")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_condition_matrices(features_dir: Path) -> Optional[Dict]:
    """Load precomputed condition feature matrices."""
    mat_path = features_dir / "condition_feature_matrices.pkl"
    if not mat_path.exists():
        logger.error(f"Feature matrices not found: {mat_path}")
        logger.error("Run 02_extract_features.py first.")
        return None
    data = load_pickle(str(mat_path))
    logger.info(f"Loaded condition matrices: {list(data.keys())}")
    for cond, info in data.items():
        logger.info(f"  {cond}: {info['shape']}")
    return data


def fit_condition_ggms(
    condition_matrices: Dict,
    cfg: dict,
    args: argparse.Namespace,
    output_dir: Path,
    force: bool = False,
) -> Dict:
    """
    Fit GGM for each condition using GraphLasso + stability selection.
    Returns dict: condition → GGM result dict.
    """
    ggm_cache = output_dir / "ggm_results.pkl"
    if ggm_cache.exists() and not force:
        logger.info("Loading cached GGM results...")
        return load_pickle(str(ggm_cache))

    estimator = GGMEstimator(
        lambda_method=args.lambda_method,
        n_subsamples=args.n_subsamples,
        pi_threshold=args.pi_threshold,
        cfg=cfg,
    )

    ggm_results = {}
    for cond in CONDITIONS:
        if cond not in condition_matrices:
            logger.warning(f"Condition {cond} not in feature matrices — skipping")
            continue

        mat_info = condition_matrices[cond]
        F_c = mat_info["matrix"]
        feature_names = mat_info["feature_names"]

        logger.info(f"\nFitting GGM for {cond} — shape {F_c.shape}")

        try:
            result = estimator.fit(
                F_c,
                feature_names=feature_names,
                condition=cond,
            )
            ggm_results[cond] = result

            # Log key metrics
            theta = result["precision_matrix"]
            n_edges = result.get("n_stable_edges", np.count_nonzero(
                np.triu(theta, k=1)
            ))
            cross_density = result.get("cross_modal_edge_density", np.nan)
            lambda_val = result.get("lambda_optimal", np.nan)

            logger.info(
                f"  {cond}: λ={lambda_val:.4f}, "
                f"n_stable_edges={n_edges}, "
                f"cross_modal_density={cross_density:.3f}"
            )

        except Exception as e:
            logger.error(f"GGM fitting failed for {cond}: {e}")
            raise

    save_pickle(ggm_results, str(ggm_cache))
    logger.info(f"GGM results saved → {ggm_cache}")
    return ggm_results


def run_topology_analysis(
    ggm_results: Dict,
    condition_matrices: Dict,
    cfg: dict,
    output_dir: Path,
) -> Dict:
    """
    Compute network topology metrics and primary hypothesis test.
    Primary hypothesis: cross-modal edge density non-monotonic
    (load_9 > load_13 by significant margin).
    """
    analyzer = TopologyAnalyzer(cfg)
    topology_results = {}

    for cond, ggm_res in ggm_results.items():
        theta = ggm_res["precision_matrix"]
        feature_names = ggm_res["feature_names"]

        metrics = analyzer.compute_metrics(
            theta,
            feature_names=feature_names,
            condition=cond,
        )
        topology_results[cond] = metrics
        logger.info(
            f"{CONDITION_LABELS.get(cond, cond)}: "
            f"cross_modal_density={metrics['cross_modal_edge_density']:.3f}, "
            f"clustering={metrics.get('mean_clustering', np.nan):.3f}, "
            f"n_total_edges={metrics['n_total_edges']}"
        )

    # Save topology metrics
    topology_summary = {
        cond: {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in metrics.items()
            if not isinstance(v, np.ndarray)
        }
        for cond, metrics in topology_results.items()
    }
    save_json(topology_summary, str(output_dir / "topology_metrics.json"))
    logger.info(f"Topology metrics saved → {output_dir / 'topology_metrics.json'}")

    return topology_results


def run_permutation_tests(
    ggm_results: Dict,
    condition_matrices: Dict,
    n_permutations: int,
    output_dir: Path,
) -> Dict:
    """
    Primary test: cross-modal edge density in 9-digit vs 13-digit.
    Null hypothesis: density(load_9) == density(load_13).
    """
    tester = PermutationTester(n_permutations=n_permutations)
    perm_results = {}

    # Critical test: 9-digit vs 13-digit
    if "load_9" in ggm_results and "load_13" in ggm_results:
        logger.info("Running primary permutation test: 9-digit vs 13-digit cross-modal density")
        F_9 = condition_matrices["load_9"]["matrix"]
        F_13 = condition_matrices["load_13"]["matrix"]
        theta_9 = ggm_results["load_9"]["precision_matrix"]
        theta_13 = ggm_results["load_13"]["precision_matrix"]
        feature_names = ggm_results["load_9"]["feature_names"]

        result = tester.test_edge_density_difference(
            F_9, F_13, theta_9, theta_13,
            feature_names=feature_names,
            alternative="greater",  # H1: density(9-digit) > density(13-digit)
        )
        perm_results["load_9_vs_load_13"] = result
        logger.info(
            f"Primary test (9d > 13d): "
            f"observed_diff={result['observed_statistic']:.4f}, "
            f"p={result['p_value']:.4f}, "
            f"significant={result['significant']}"
        )

    # Secondary test: 5-digit vs 9-digit (should increase)
    if "load_5" in ggm_results and "load_9" in ggm_results:
        logger.info("Secondary test: 5-digit vs 9-digit")
        F_5 = condition_matrices["load_5"]["matrix"]
        F_9 = condition_matrices["load_9"]["matrix"]
        theta_5 = ggm_results["load_5"]["precision_matrix"]
        theta_9 = ggm_results["load_9"]["precision_matrix"]
        feature_names = ggm_results["load_5"]["feature_names"]

        result = tester.test_edge_density_difference(
            F_5, F_9, theta_5, theta_9,
            feature_names=feature_names,
            alternative="less",  # H1: density(5d) < density(9d)
        )
        perm_results["load_5_vs_load_9"] = result
        logger.info(
            f"Secondary test (9d > 5d): "
            f"observed_diff={result['observed_statistic']:.4f}, "
            f"p={result['p_value']:.4f}"
        )

    # Per-edge BH-FDR corrected tests
    if "load_9" in ggm_results and "load_13" in ggm_results:
        logger.info("Running per-edge FDR-corrected significance tests (9d vs 13d)")
        F_9 = condition_matrices["load_9"]["matrix"]
        F_13 = condition_matrices["load_13"]["matrix"]
        feature_names = ggm_results["load_9"]["feature_names"]

        edge_results = tester.test_edges_fdr(
            F_9, F_13,
            ggm_results["load_9"]["precision_matrix"],
            ggm_results["load_13"]["precision_matrix"],
            feature_names=feature_names,
        )
        perm_results["edge_fdr_9_vs_13"] = edge_results
        n_sig = int(edge_results.get("n_significant_edges", 0))
        logger.info(f"FDR-significant edges (9d vs 13d): {n_sig}")

    # Serializable summary
    perm_summary = {}
    for k, v in perm_results.items():
        if isinstance(v, dict):
            perm_summary[k] = {
                kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()
                if not isinstance(vv, np.ndarray)
            }
    save_json(perm_summary, str(output_dir / "permutation_test_results.json"))
    logger.info("Permutation test results saved")

    return perm_results


def run_bootstrap_analysis(
    ggm_results: Dict,
    topology_results: Dict,
    output_dir: Path,
) -> Dict:
    """Bootstrap confidence intervals for edge weights and topology metrics."""
    bootstrapper = BootstrapAnalyzer(n_bootstrap=500)
    boot_results = {}

    for cond, ggm_res in ggm_results.items():
        theta = ggm_res["precision_matrix"]
        feature_names = ggm_res["feature_names"]
        try:
            ci = bootstrapper.precision_matrix_ci(
                ggm_res["data_used"],
                ggm_res.get("lambda_optimal", 0.1),
                alpha=0.05,
            )
            boot_results[cond] = {"edge_weight_ci": ci}
            logger.info(f"{cond}: bootstrap CIs computed")
        except Exception as e:
            logger.warning(f"{cond}: bootstrap failed — {e}")

    return boot_results


def generate_figures(
    ggm_results: Dict,
    topology_results: Dict,
    perm_results: Dict,
    condition_matrices: Dict,
    output_dir: Path,
):
    """Generate Figure 1 (precision networks) and Figure 2 (edge density curve)."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 4-panel precision matrix network graphs
    logger.info("Generating Figure 1: Precision matrix network graphs...")
    try:
        fig1 = plot_precision_networks(
            {cond: res["precision_matrix"] for cond, res in ggm_results.items()},
            feature_names=next(iter(ggm_results.values()))["feature_names"],
            condition_labels=CONDITION_LABELS,
            topology_results=topology_results,
            output_path=str(fig_dir / "fig1_precision_networks.png"),
        )
        plt.close(fig1)
        logger.info("Figure 1 saved")
    except Exception as e:
        logger.error(f"Figure 1 failed: {e}")

    # Figure 2: Cross-modal edge density vs condition
    logger.info("Generating Figure 2: Edge density curve...")
    try:
        densities = {
            cond: topology_results[cond]["cross_modal_edge_density"]
            for cond in CONDITIONS
            if cond in topology_results
        }
        p_val = perm_results.get("load_9_vs_load_13", {}).get("p_value", None)

        fig2 = plot_edge_density_curve(
            densities,
            condition_labels=CONDITION_LABELS,
            p_value_annotation=p_val,
            output_path=str(fig_dir / "fig2_edge_density_curve.png"),
        )
        plt.close(fig2)
        logger.info("Figure 2 saved")
    except Exception as e:
        logger.error(f"Figure 2 failed: {e}")


def print_primary_result(topology_results: Dict, perm_results: Dict):
    """Print primary scientific result to console."""
    print("\n" + "=" * 65)
    print("PRIMARY RESULT: Cross-Modal Edge Density by Condition")
    print("=" * 65)
    for cond in CONDITIONS:
        if cond in topology_results:
            d = topology_results[cond]["cross_modal_edge_density"]
            label = CONDITION_LABELS.get(cond, cond)
            print(f"  {label:<25}: {d:.4f}")
    print()

    # Primary hypothesis
    test_key = "load_9_vs_load_13"
    if test_key in perm_results:
        r = perm_results[test_key]
        sig = "*** SIGNIFICANT ***" if r.get("significant", False) else "NOT significant"
        print(f"Primary test (density_9d > density_13d):")
        print(f"  Observed difference : {r.get('observed_statistic', '?'):.4f}")
        print(f"  p-value             : {r.get('p_value', '?'):.4f}")
        print(f"  Result              : {sig}")
        print()

    print("Prediction: 5-digit < 9-digit > 13-digit (non-monotonic)")
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "ggm"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(cfg["paths"]["output_root"]) / "features"

    setup_logging(
        log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "03_fit_ggm.log")
    )

    logger.info("=" * 60)
    logger.info("GGM Fitting & Topology Analysis — Script 03")
    logger.info("=" * 60)
    logger.info(f"Lambda method: {args.lambda_method}")
    logger.info(f"Stability subsamples: {args.n_subsamples}")
    logger.info(f"Permutations: {args.n_permutations}")

    # Load feature matrices
    condition_matrices = load_condition_matrices(feat_dir)
    if condition_matrices is None:
        sys.exit(1)

    # Fit GGMs
    ggm_results = fit_condition_ggms(
        condition_matrices, cfg, args, out_dir, force=args.force
    )

    if not ggm_results:
        logger.error("No GGMs fitted. Exiting.")
        sys.exit(1)

    # Topology analysis
    topology_results = run_topology_analysis(
        ggm_results, condition_matrices, cfg, out_dir
    )

    # Permutation tests
    perm_results = run_permutation_tests(
        ggm_results, condition_matrices, args.n_permutations, out_dir
    )

    # Bootstrap CIs
    boot_results = run_bootstrap_analysis(ggm_results, topology_results, out_dir)
    if boot_results:
        save_pickle(boot_results, str(out_dir / "bootstrap_edge_cis.pkl"))

    # Figures
    generate_figures(ggm_results, topology_results, perm_results, condition_matrices, out_dir)

    # Print primary result
    print_primary_result(topology_results, perm_results)

    logger.info("GGM analysis complete.")


if __name__ == "__main__":
    main()