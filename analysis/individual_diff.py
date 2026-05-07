"""
analysis/individual_diff.py
============================

Individual difference analysis: WM capacity vs coupling strength.

Tests:
  1. Resting coupling strength → WM capacity (behavioral span)
  2. 9-digit coupling strength → WM capacity (tight coordination at peak)
  3. Coupling decrement 9→13 digit → WM capacity (larger drop = earlier overload)
  4. Resting coupling → task-state coupling sensitivity (personalization test)

Finding 4 prediction:
  Subjects with stronger resting EEG-ECG coupling show larger task-induced coupling changes.

All models: ridge regression + Pearson/Spearman correlation + effect size.
Strictly interpretable: linear relationships tested and visualized.

Dependencies: numpy, scipy, scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CorrelationResult:
    """Bivariate correlation between one predictor and WM capacity."""
    predictor_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n: int

    @property
    def significant(self) -> bool:
        return self.pearson_p < 0.05 or self.spearman_p < 0.05


@dataclass
class RegressionResult:
    """Multiple regression result."""
    model_name: str
    predictors: List[str]
    r2: float                    # R² on held-out (LOO)
    r2_train: float              # training R² (for overfitting check)
    n: int
    best_alpha: float

    # Predictor importance: absolute coefficients (ridge)
    coef_magnitudes: Dict[str, float]


@dataclass
class IndividualDiffReport:
    """Full individual difference analysis report."""
    correlations: List[CorrelationResult]
    regression_models: Dict[str, RegressionResult]

    # Key finding: does 9-digit coupling add variance beyond marginal features?
    coupling_adds_variance: bool
    coupling_r2_gain: float

    # Resting-state prediction
    resting_coupling_correlates: bool
    resting_correlations: List[CorrelationResult]


# ── Bivariate correlations ─────────────────────────────────────────────────────

def correlate_with_wm_capacity(
    predictor_values: np.ndarray,
    wm_capacity: np.ndarray,
    predictor_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> List[CorrelationResult]:
    """
    Compute Pearson + Spearman correlations between each predictor and WM capacity.

    Parameters
    ----------
    predictor_values : ndarray (n_subjects, n_predictors)
    wm_capacity : ndarray (n_subjects,)
    predictor_names : list of str or None

    Returns
    -------
    list of CorrelationResult (sorted by |Pearson r|, descending)
    """
    n_subjects, n_preds = predictor_values.shape

    if predictor_names is None:
        predictor_names = [f"predictor_{i}" for i in range(n_preds)]

    # Remove subjects with NaN in either array
    valid = ~(np.isnan(wm_capacity) | np.any(np.isnan(predictor_values), axis=1))
    if valid.sum() < 5:
        logger.warning(f"Only {valid.sum()} valid subjects for correlation analysis")
        return []

    X_v = predictor_values[valid]
    y_v = wm_capacity[valid]
    n_valid = int(valid.sum())

    results = []
    for j, name in enumerate(predictor_names):
        x = X_v[:, j]
        if x.std() < 1e-8:
            continue

        r, p_pearson = pearsonr(x, y_v)
        rho, p_spearman = spearmanr(x, y_v)

        results.append(CorrelationResult(
            predictor_name=name,
            pearson_r=float(r),
            pearson_p=float(p_pearson),
            spearman_rho=float(rho),
            spearman_p=float(p_spearman),
            n=n_valid,
        ))

    results.sort(key=lambda x: abs(x.pearson_r), reverse=True)

    n_sig = sum(1 for r in results if r.significant)
    logger.info(
        f"Correlations with WM capacity: {n_sig}/{len(results)} significant predictors"
    )

    return results


# ── Ridge regression for WM capacity ──────────────────────────────────────────

def loo_ridge_wm_regression(
    X: np.ndarray,
    y: np.ndarray,
    predictor_names: List[str],
    model_name: str,
    alphas: Optional[List[float]] = None,
) -> RegressionResult:
    """
    Leave-one-out ridge regression predicting WM capacity.

    Parameters
    ----------
    X : ndarray (n_subjects, n_features)
    y : ndarray (n_subjects,)
        WM capacity (behavioral span or composite score).
    predictor_names : list of str
    model_name : str

    Returns
    -------
    RegressionResult
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Remove NaN rows
    valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    X_v = X[valid]
    y_v = y[valid]
    n = int(valid.sum())

    if n < 5:
        logger.warning(f"Insufficient subjects for LOO regression: n={n}")
        return RegressionResult(
            model_name=model_name, predictors=predictor_names,
            r2=np.nan, r2_train=np.nan, n=n, best_alpha=np.nan,
            coef_magnitudes={},
        )

    logger.info(f"LOO ridge regression: {model_name}, n={n}, n_features={X_v.shape[1]}")

    loo = LeaveOneOut()
    y_pred = np.zeros(n)

    for train_idx, test_idx in loo.split(X_v):
        X_train, X_test = X_v[train_idx], X_v[test_idx]
        y_train = y_v[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas, cv=min(5, len(train_idx)))
        model.fit(X_train_s, y_train)
        y_pred[test_idx] = model.predict(X_test_s)

    # R² on LOO predictions
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2)
    r2_loo = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Training R² (fit on full dataset)
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_v)
    model_full = RidgeCV(alphas=alphas, cv=min(5, n))
    model_full.fit(X_full_s, y_v)
    r2_train = float(model_full.score(X_full_s, y_v))
    best_alpha = float(model_full.alpha_)

    # Predictor importance
    coef = np.abs(model_full.coef_)
    coef_names = predictor_names[:len(coef)]
    coef_magnitudes = dict(zip(coef_names, coef.tolist()))
    # Sort by magnitude
    coef_magnitudes = dict(sorted(coef_magnitudes.items(), key=lambda x: x[1], reverse=True))

    logger.info(
        f"  → LOO R²={r2_loo:.4f}, train R²={r2_train:.4f}, alpha={best_alpha:.4f}"
    )

    return RegressionResult(
        model_name=model_name,
        predictors=predictor_names[:len(coef)],
        r2=r2_loo,
        r2_train=r2_train,
        n=n,
        best_alpha=best_alpha,
        coef_magnitudes=coef_magnitudes,
    )


# ── Incremental variance analysis ─────────────────────────────────────────────

def coupling_adds_variance_beyond_marginal(
    X_marginal: np.ndarray,
    X_coupling: np.ndarray,
    y: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """
    Test whether coupling features add variance beyond marginal features.

    Compare:
      Model 1: marginal features only → R²_marginal
      Model 2: marginal + coupling → R²_full
      ΔR² = R²_full - R²_marginal

    Parameters
    ----------
    X_marginal : ndarray (n, d1)
    X_coupling : ndarray (n, d2)
    y : ndarray (n,)

    Returns
    -------
    r2_marginal : float
    r2_full : float
    delta_r2 : float
    """
    result_marginal = loo_ridge_wm_regression(
        X_marginal, y,
        predictor_names=[f"m{i}" for i in range(X_marginal.shape[1])],
        model_name="marginal_only",
        alphas=alphas,
    )

    X_full = np.hstack([X_marginal, X_coupling])
    result_full = loo_ridge_wm_regression(
        X_full, y,
        predictor_names=[f"m{i}" for i in range(X_marginal.shape[1])] +
                        [f"c{i}" for i in range(X_coupling.shape[1])],
        model_name="marginal_plus_coupling",
        alphas=alphas,
    )

    delta = result_full.r2 - result_marginal.r2

    logger.info(
        f"Variance decomposition: "
        f"marginal R²={result_marginal.r2:.4f}, "
        f"full R²={result_full.r2:.4f}, "
        f"ΔR²={delta:+.4f}"
    )

    return result_marginal.r2, result_full.r2, delta


# ── Full individual difference pipeline ───────────────────────────────────────

def run_individual_diff_analysis(
    coupling_9digit: np.ndarray,           # (n_subjects, n_coupling_features)
    coupling_rest: Optional[np.ndarray],   # (n_subjects, n_coupling_features) or None
    marginal_9digit: np.ndarray,           # (n_subjects, n_marginal_features)
    coupling_drop_9_to_13: np.ndarray,     # (n_subjects,) scalar
    wm_capacity: np.ndarray,               # (n_subjects,)
    subject_ids: List[str],
    coupling_feature_names: Optional[List[str]] = None,
    marginal_feature_names: Optional[List[str]] = None,
    resting_feature_names: Optional[List[str]] = None,
) -> IndividualDiffReport:
    """
    Full individual difference analysis pipeline.

    Parameters
    ----------
    coupling_9digit : ndarray (n_subjects, n_coup)
        Task coupling features at 9-digit condition (peak coupling).
    coupling_rest : ndarray or None
        Resting-state coupling features.
    marginal_9digit : ndarray (n_subjects, n_marg)
        Marginal (single-channel) features at 9-digit.
    coupling_drop_9_to_13 : ndarray (n_subjects,)
        Coupling strength drop from 9- to 13-digit per subject.
    wm_capacity : ndarray (n_subjects,)
        Behavioral WM span or composite score.
    subject_ids : list of str
    coupling_feature_names, marginal_feature_names, resting_feature_names : list or None

    Returns
    -------
    IndividualDiffReport
    """
    n = len(subject_ids)

    if coupling_feature_names is None:
        coupling_feature_names = [f"coup_{i}" for i in range(coupling_9digit.shape[1])]
    if marginal_feature_names is None:
        marginal_feature_names = [f"marg_{i}" for i in range(marginal_9digit.shape[1])]

    # 1. Correlations: coupling features with WM capacity
    logger.info("=== Correlating coupling features with WM capacity ===")
    correlations = correlate_with_wm_capacity(
        coupling_9digit, wm_capacity, coupling_feature_names
    )

    # Add coupling drop correlation
    drop_corr = correlate_with_wm_capacity(
        coupling_drop_9_to_13.reshape(-1, 1), wm_capacity,
        predictor_names=["coupling_drop_9_to_13"]
    )
    correlations.extend(drop_corr)

    # 2. Regression: coupling at 9-digit predicts WM capacity
    logger.info("=== Ridge regression: coupling → WM capacity ===")
    reg_coupling = loo_ridge_wm_regression(
        coupling_9digit, wm_capacity,
        predictor_names=coupling_feature_names,
        model_name="coupling_9digit_only",
    )

    # 3. Regression: marginal features predict WM capacity
    reg_marginal = loo_ridge_wm_regression(
        marginal_9digit, wm_capacity,
        predictor_names=marginal_feature_names,
        model_name="marginal_9digit_only",
    )

    # 4. Does coupling add variance beyond marginal?
    r2_marg, r2_full, delta_r2 = coupling_adds_variance_beyond_marginal(
        marginal_9digit, coupling_9digit, wm_capacity
    )
    coupling_adds = delta_r2 > 0.02  # threshold: 2% R² gain

    regression_models = {
        "coupling_9digit": reg_coupling,
        "marginal_9digit": reg_marginal,
    }

    # 5. Resting state correlations and prediction
    resting_corrs = []
    resting_coupling_correlates = False

    if coupling_rest is not None:
        logger.info("=== Resting-state coupling vs WM capacity ===")
        if resting_feature_names is None:
            resting_feature_names = [f"rest_{i}" for i in range(coupling_rest.shape[1])]

        resting_corrs = correlate_with_wm_capacity(
            coupling_rest, wm_capacity, resting_feature_names
        )
        resting_coupling_correlates = any(r.significant for r in resting_corrs)

        reg_resting = loo_ridge_wm_regression(
            coupling_rest, wm_capacity,
            predictor_names=resting_feature_names,
            model_name="resting_coupling_only",
        )
        regression_models["resting_coupling"] = reg_resting

    # Summary logging
    top_corrs = sorted(correlations, key=lambda x: abs(x.pearson_r), reverse=True)[:5]
    logger.info("\n=== Top correlations with WM capacity ===")
    for cr in top_corrs:
        logger.info(
            f"  {cr.predictor_name}: r={cr.pearson_r:+.3f} (p={cr.pearson_p:.3f}), "
            f"ρ={cr.spearman_rho:+.3f} (p={cr.spearman_p:.3f})"
        )

    logger.info(
        f"\nCoupling adds variance: {coupling_adds} (ΔR²={delta_r2:+.4f})\n"
        f"Resting coupling correlates with WM: {resting_coupling_correlates}"
    )

    return IndividualDiffReport(
        correlations=correlations,
        regression_models=regression_models,
        coupling_adds_variance=coupling_adds,
        coupling_r2_gain=float(delta_r2),
        resting_coupling_correlates=resting_coupling_correlates,
        resting_correlations=resting_corrs,
    )