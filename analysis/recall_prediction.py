"""
analysis/recall_prediction.py
==============================

Ridge regression for recall accuracy prediction.
Leave-one-subject-out (LOSO) cross-validation.

Compares four feature sets:
  B3: per-modality only (concatenated — no coupling)
  Novel: coupling features only
  Full: per-modality + coupling
  Resting: zero-shot (resting coupling only)

Primary claim: coupling features improve recall prediction,
especially for 13-digit (overload) condition.

Statistical test: Wilcoxon signed-rank on per-subject LOSO R² differences.

Dependencies: numpy, scipy, scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LOSOFold:
    """Result of one LOSO fold (one subject as test)."""
    subject_id: str
    y_true: np.ndarray
    y_pred: np.ndarray
    r2: float
    rmse: float
    n_test: int


@dataclass
class PredictionResult:
    """Full LOSO prediction result for one feature set."""
    feature_set_name: str
    target: str                           # "recall_accuracy" or "condition_label"

    folds: List[LOSOFold]

    # Aggregated metrics
    mean_r2: float
    std_r2: float
    median_r2: float
    per_subject_r2: Dict[str, float]

    mean_rmse: float

    # Per-condition breakdown (especially important for 13-digit overload)
    r2_by_condition: Optional[Dict[int, float]] = None


@dataclass
class FeatureSetComparison:
    """Comparison between coupling and baseline feature sets."""
    feature_set_a: str
    feature_set_b: str
    wilcoxon_statistic: float
    wilcoxon_p: float
    mean_r2_difference: float            # r2_b - r2_a (positive = b better)
    effect_significant: bool             # p < 0.05


@dataclass
class PredictionReport:
    """Full recall prediction report."""
    results: Dict[str, PredictionResult]  # feature_set_name → result
    comparisons: List[FeatureSetComparison]
    best_feature_set: str
    condition_13_improvement: float       # coupling vs marginal at 13-digit


# ── LOSO cross-validation ──────────────────────────────────────────────────────

def loso_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    feature_set_name: str,
    target_name: str = "recall_accuracy",
    ridge_alphas: Optional[List[float]] = None,
    condition_ids: Optional[np.ndarray] = None,
) -> PredictionResult:
    """
    Leave-one-subject-out ridge regression.

    For each fold: train on all subjects except one, test on held-out subject.
    Ridge alpha selected by inner CV on training set.

    Parameters
    ----------
    X : ndarray (n_trials, n_features)
    y : ndarray (n_trials,)
        Target: recall accuracy (continuous) or condition label (classification).
    subject_ids : ndarray (n_trials,)
    feature_set_name : str
    target_name : str
    ridge_alphas : list of float or None
        Default: [0.001, 0.01, 0.1, 1, 10, 100]
    condition_ids : ndarray (n_trials,) or None
        For per-condition R² breakdown.

    Returns
    -------
    PredictionResult
    """
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    logger.info(
        f"LOSO ridge: feature_set={feature_set_name}, "
        f"n_subjects={n_subjects}, n_trials={X.shape[0]}, n_features={X.shape[1]}"
    )

    folds = []
    y_all_true = []
    y_all_pred = []
    subj_all = []
    cond_all = []

    for subj in unique_subjects:
        test_mask  = subject_ids == subj
        train_mask = ~test_mask

        if train_mask.sum() < 10 or test_mask.sum() < 2:
            logger.warning(f"Skipping {subj}: insufficient trials")
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Pipeline: scale → ridge with inner CV
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ridge_alphas, cv=min(5, train_mask.sum()))),
        ])

        # Handle near-constant features
        var_train = X_train.var(axis=0)
        valid_feats = var_train > 1e-10
        if valid_feats.sum() < 2:
            logger.warning(f"{subj}: too few valid features, predicting mean")
            y_pred = np.full(len(y_test), y_train.mean())
        else:
            model.fit(X_train[:, valid_feats], y_train)
            y_pred = model.predict(X_test[:, valid_feats])

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        folds.append(LOSOFold(
            subject_id=str(subj),
            y_true=y_test,
            y_pred=y_pred,
            r2=r2,
            rmse=rmse,
            n_test=int(test_mask.sum()),
        ))

        y_all_true.extend(y_test.tolist())
        y_all_pred.extend(y_pred.tolist())
        subj_all.extend([str(subj)] * int(test_mask.sum()))
        if condition_ids is not None:
            cond_all.extend(condition_ids[test_mask].tolist())

    if not folds:
        logger.error(f"No valid LOSO folds for {feature_set_name}")
        return _empty_result(feature_set_name, target_name)

    # Aggregate
    r2_vals = np.array([f.r2 for f in folds])
    rmse_vals = np.array([f.rmse for f in folds])
    per_subject_r2 = {f.subject_id: f.r2 for f in folds}

    # Per-condition R²
    r2_by_condition = None
    if condition_ids is not None and len(cond_all) > 0:
        cond_arr = np.array(cond_all)
        y_true_arr = np.array(y_all_true)
        y_pred_arr = np.array(y_all_pred)
        r2_by_condition = {}
        for cond in np.unique(cond_arr):
            mask = cond_arr == cond
            if mask.sum() >= 2:
                r2_by_condition[int(cond)] = float(r2_score(y_true_arr[mask], y_pred_arr[mask]))

    result = PredictionResult(
        feature_set_name=feature_set_name,
        target=target_name,
        folds=folds,
        mean_r2=float(np.mean(r2_vals)),
        std_r2=float(np.std(r2_vals)),
        median_r2=float(np.median(r2_vals)),
        per_subject_r2=per_subject_r2,
        mean_rmse=float(np.mean(rmse_vals)),
        r2_by_condition=r2_by_condition,
    )

    logger.info(
        f"  → mean_R²={result.mean_r2:.4f} ± {result.std_r2:.4f}, "
        f"median_R²={result.median_r2:.4f}"
    )

    return result


# ── Feature set comparison ─────────────────────────────────────────────────────

def compare_feature_sets(
    result_a: PredictionResult,
    result_b: PredictionResult,
) -> FeatureSetComparison:
    """
    Wilcoxon signed-rank test comparing per-subject R² between two feature sets.

    Tests whether coupling features significantly improve over baseline.
    Per-subject comparison (paired test) accounts for subject variability.
    """
    # Align subjects
    subj_a = set(result_a.per_subject_r2.keys())
    subj_b = set(result_b.per_subject_r2.keys())
    common = sorted(subj_a & subj_b)

    if len(common) < 5:
        logger.warning(f"Too few common subjects ({len(common)}) for Wilcoxon test")
        return FeatureSetComparison(
            feature_set_a=result_a.feature_set_name,
            feature_set_b=result_b.feature_set_name,
            wilcoxon_statistic=np.nan,
            wilcoxon_p=1.0,
            mean_r2_difference=0.0,
            effect_significant=False,
        )

    r2_a = np.array([result_a.per_subject_r2[s] for s in common])
    r2_b = np.array([result_b.per_subject_r2[s] for s in common])
    diff = r2_b - r2_a

    try:
        stat, p = wilcoxon(diff, alternative="greater")
    except ValueError:
        stat, p = np.nan, 1.0

    mean_diff = float(np.mean(diff))

    comp = FeatureSetComparison(
        feature_set_a=result_a.feature_set_name,
        feature_set_b=result_b.feature_set_name,
        wilcoxon_statistic=float(stat) if not np.isnan(stat) else np.nan,
        wilcoxon_p=float(p),
        mean_r2_difference=mean_diff,
        effect_significant=p < 0.05,
    )

    logger.info(
        f"Comparison {result_a.feature_set_name} vs {result_b.feature_set_name}: "
        f"ΔR²={mean_diff:+.4f}, p={p:.4f}, significant={comp.effect_significant}"
    )

    return comp


# ── Full prediction pipeline ───────────────────────────────────────────────────

def run_recall_prediction(
    feature_sets: Dict[str, Tuple[np.ndarray, List[str]]],
    y: np.ndarray,
    subject_ids: np.ndarray,
    target_name: str = "recall_accuracy",
    ridge_alphas: Optional[List[float]] = None,
    condition_ids: Optional[np.ndarray] = None,
    baseline_name: str = "per_modality_only",
    novel_name: str = "coupling_only",
) -> PredictionReport:
    """
    Run full recall prediction pipeline over multiple feature sets.

    Parameters
    ----------
    feature_sets : dict {name → (X_array, feature_names)}
    y : ndarray (n_trials,)
    subject_ids : ndarray (n_trials,)
    target_name : str
    ridge_alphas : list of float or None
    condition_ids : ndarray or None
    baseline_name : str
        Name of baseline feature set for comparison.
    novel_name : str
        Name of coupling feature set.

    Returns
    -------
    PredictionReport
    """
    results = {}

    for fs_name, (X_fs, _) in feature_sets.items():
        if X_fs.shape[0] == 0 or X_fs.shape[1] == 0:
            logger.warning(f"Empty feature set: {fs_name}. Skipping.")
            continue

        result = loso_ridge_regression(
            X=X_fs,
            y=y,
            subject_ids=subject_ids,
            feature_set_name=fs_name,
            target_name=target_name,
            ridge_alphas=ridge_alphas,
            condition_ids=condition_ids,
        )
        results[fs_name] = result

    # Pairwise comparisons
    comparisons = []
    if baseline_name in results and novel_name in results:
        comp = compare_feature_sets(results[baseline_name], results[novel_name])
        comparisons.append(comp)

    # Compare all pairs involving coupling
    coupling_sets = [k for k in results if "coupling" in k.lower()]
    for cs in coupling_sets:
        if cs != baseline_name and baseline_name in results:
            comp = compare_feature_sets(results[baseline_name], results[cs])
            comparisons.append(comp)

    # Best feature set
    best_fs = max(results, key=lambda k: results[k].mean_r2) if results else ""

    # Condition 13 improvement (key metric)
    cond_13_improvement = np.nan
    if (baseline_name in results and novel_name in results and
            results[baseline_name].r2_by_condition is not None and
            results[novel_name].r2_by_condition is not None):
        r2_base_13 = results[baseline_name].r2_by_condition.get(3, np.nan)
        r2_coup_13 = results[novel_name].r2_by_condition.get(3, np.nan)
        cond_13_improvement = r2_coup_13 - r2_base_13

    logger.info(
        f"\n=== Prediction Report ===\n"
        f"Best feature set: {best_fs}\n"
        f"Condition-13 R² improvement (coupling vs marginal): {cond_13_improvement:+.4f}\n"
        + "\n".join([
            f"  {k}: mean_R²={v.mean_r2:.4f}" for k, v in results.items()
        ])
    )

    return PredictionReport(
        results=results,
        comparisons=comparisons,
        best_feature_set=best_fs,
        condition_13_improvement=float(cond_13_improvement) if not np.isnan(cond_13_improvement) else 0.0,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _empty_result(name: str, target: str) -> PredictionResult:
    return PredictionResult(
        feature_set_name=name, target=target,
        folds=[], mean_r2=np.nan, std_r2=np.nan,
        median_r2=np.nan, per_subject_r2={}, mean_rmse=np.nan,
    )


def combine_feature_sets(
    X_a: np.ndarray,
    X_b: np.ndarray,
    names_a: List[str],
    names_b: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate two feature matrices. For building 'per_modality+coupling' set."""
    assert X_a.shape[0] == X_b.shape[0], "Mismatched trial counts"
    return np.hstack([X_a, X_b]), names_a + names_b