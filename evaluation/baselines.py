"""
evaluation/baselines.py
========================
Baseline models B1–B6 for comparison against the GGM coupling approach.

B1 — Per-modality Catch22 + XGBoost
    Replicates Papers 5 & 6. Condition classification (4-class) from
    Catch22 features extracted per modality, fused by XGBoost.

B2 — Late fusion of per-modality classifiers
    Train separate XGBoost per modality; concatenate probability predictions;
    final XGBoost meta-classifier on concatenated probabilities.

B3 — Early fusion: concatenated feature vectors, no coupling
    Critical ablation. All per-modality features concatenated, XGBoost.
    Tests whether structured coupling adds anything beyond raw concatenation.

B4 — Static full covariance matrix features (not precision matrix)
    Vectorize upper-triangle of Sigma_c (not Theta_c). Tests whether
    precision matrix structure matters vs. raw covariance.

B5 — Single cross-modal feature: EEG-pupil Pearson correlation
    Simplest coupling measure. Tests whether trivial coupling already
    explains the gain.

B6 — PCA of full multimodal feature vector
    Linear dimensionality reduction, not coupling-aware. Tests whether
    any linear projection would do as well as structured GGM.

All baselines evaluated via LOSO (leave-one-subject-out) cross-validation
on both 4-class condition classification and continuous recall accuracy regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Evaluation results for a single baseline model."""
    name: str
    description: str
    # Classification (4-class condition)
    loso_accuracy: Optional[float] = None
    loso_balanced_accuracy: Optional[float] = None
    per_subject_accuracy: Optional[np.ndarray] = field(default=None, repr=False)
    # Regression (recall accuracy)
    loso_r2: Optional[float] = None
    loso_rmse: Optional[float] = None
    per_subject_r2: Optional[np.ndarray] = field(default=None, repr=False)
    # Metadata
    n_features: int = 0
    n_subjects: int = 0
    n_trials: int = 0


@dataclass
class BaselineSuite:
    """All B1-B6 baseline results."""
    results: Dict[str, BaselineResult]
    task: str   # 'classification' | 'regression' | 'both'

    def summary_table(self) -> str:
        lines = [
            f"{'Baseline':<35} {'Acc':>8} {'BalAcc':>8} {'R²':>8} {'RMSE':>8}",
            "-" * 70,
        ]
        for name, r in self.results.items():
            acc = f"{r.loso_accuracy:.4f}" if r.loso_accuracy is not None else "  —  "
            bacc = f"{r.loso_balanced_accuracy:.4f}" if r.loso_balanced_accuracy is not None else "  —  "
            r2 = f"{r.loso_r2:.4f}" if r.loso_r2 is not None else "  —  "
            rmse = f"{r.loso_rmse:.4f}" if r.loso_rmse is not None else "  —  "
            lines.append(f"{name:<35} {acc:>8} {bacc:>8} {r2:>8} {rmse:>8}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LOSO evaluation helpers
# ---------------------------------------------------------------------------

def _loso_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate leave-one-subject-out train/test splits.

    Returns list of (X_train, X_test, y_train, y_test) tuples.
    """
    unique_subjects = np.unique(subject_ids)
    splits = []
    for sub in unique_subjects:
        test_mask = subject_ids == sub
        train_mask = ~test_mask
        splits.append((
            X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
        ))
    return splits


def _evaluate_classifier_loso(
    X: np.ndarray,
    y_class: np.ndarray,
    subject_ids: np.ndarray,
    model_factory: callable,
    scaler: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run LOSO classification. Returns (acc, balanced_acc, per_subject_acc, y_true_all, y_pred_all).
    """
    splits = _loso_split(X, y_class, subject_ids)
    per_subject_acc = []
    y_true_all, y_pred_all = [], []

    for X_tr, X_te, y_tr, y_te in splits:
        if scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        clf = model_factory()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        per_subject_acc.append(accuracy_score(y_te, y_pred))
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    return acc, bacc, np.array(per_subject_acc), y_true_all, y_pred_all


def _evaluate_regressor_loso(
    X: np.ndarray,
    y_reg: np.ndarray,
    subject_ids: np.ndarray,
    model_factory: callable,
    scaler: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run LOSO regression. Returns (r2, rmse, per_subject_r2, y_true_all, y_pred_all).
    """
    splits = _loso_split(X, y_reg, subject_ids)
    per_subject_r2 = []
    y_true_all, y_pred_all = [], []

    for X_tr, X_te, y_tr, y_te in splits:
        if scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        reg = model_factory()
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        r2_sub = r2_score(y_te, y_pred) if len(y_te) > 1 else np.nan
        per_subject_r2.append(r2_sub)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    rmse = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    return r2, rmse, np.array(per_subject_r2), y_true_all, y_pred_all


# ---------------------------------------------------------------------------
# B1: Per-modality Catch22 + XGBoost
# ---------------------------------------------------------------------------

def baseline_b1_catch22_xgboost(
    catch22_features: Dict[str, np.ndarray],
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
    xgb_params: Optional[Dict] = None,
) -> BaselineResult:
    """
    B1: Catch22 features per modality, fused by concatenation, XGBoost classifier/regressor.

    Parameters
    ----------
    catch22_features : dict modality_name → (N_trials, 22) Catch22 features
    y_class : (N_trials,) condition labels (0-3)
    y_recall : (N_trials,) continuous recall accuracy
    subject_ids : (N_trials,) subject identifiers
    xgb_params : optional XGBoost hyperparameters
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        logger.warning("XGBoost not available. Using GradientBoostingClassifier.")
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        XGBClassifier = GradientBoostingClassifier
        XGBRegressor = GradientBoostingRegressor

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
        }

    # Concatenate all modality Catch22 features
    X = np.hstack(list(catch22_features.values()))
    n_feat = X.shape[1]

    logger.info(f"B1 (Catch22+XGB): {n_feat} features, {X.shape[0]} trials")

    def clf_factory():
        return XGBClassifier(**xgb_params, use_label_encoder=False,
                             eval_metric="mlogloss", verbosity=0)

    def reg_factory():
        p = {k: v for k, v in xgb_params.items()}
        if "eval_metric" in p:
            del p["eval_metric"]
        return XGBRegressor(**p, verbosity=0)

    acc, bacc, per_s_acc, _, _ = _evaluate_classifier_loso(
        X, y_class, subject_ids, clf_factory
    )
    r2, rmse, per_s_r2, _, _ = _evaluate_regressor_loso(
        X, y_recall, subject_ids, reg_factory
    )

    logger.info(f"  B1 → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name="B1_Catch22_XGBoost",
        description="Per-modality Catch22 features, concatenated, XGBoost",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=per_s_acc,
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=per_s_r2,
        n_features=n_feat,
        n_subjects=len(np.unique(subject_ids)),
        n_trials=X.shape[0],
    )


# ---------------------------------------------------------------------------
# B2: Late fusion of per-modality classifiers
# ---------------------------------------------------------------------------

def baseline_b2_late_fusion(
    per_modality_features: Dict[str, np.ndarray],
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
    xgb_params: Optional[Dict] = None,
) -> BaselineResult:
    """
    B2: Train separate classifier per modality; concatenate predicted probabilities;
    final XGBoost meta-classifier.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        XGBClassifier = GradientBoostingClassifier
        XGBRegressor = GradientBoostingRegressor

    if xgb_params is None:
        xgb_params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}

    modalities = list(per_modality_features.keys())
    unique_subjects = np.unique(subject_ids)
    n_classes = len(np.unique(y_class))

    y_pred_class_all, y_pred_reg_all = [], []
    y_true_class_all, y_true_reg_all = [], []
    per_s_acc, per_s_r2 = [], []

    for sub in unique_subjects:
        test_mask = subject_ids == sub
        train_mask = ~test_mask

        # Stage 1: per-modality classifiers on training set, get test probabilities
        test_probs = []
        test_preds_reg = []

        for mod in modalities:
            X_mod = per_modality_features[mod]
            X_tr = X_mod[train_mask]
            X_te = X_mod[test_mask]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            clf = XGBClassifier(**xgb_params, use_label_encoder=False,
                                eval_metric="mlogloss", verbosity=0)
            clf.fit(X_tr, y_class[train_mask])
            prob = clf.predict_proba(X_te)  # (N_test, n_classes)
            test_probs.append(prob)

            reg = XGBRegressor(**xgb_params, verbosity=0)
            reg.fit(X_tr, y_recall[train_mask])
            test_preds_reg.append(reg.predict(X_te).reshape(-1, 1))

        # Stage 2: meta-classifier on concatenated probabilities
        # For LOSO, we need train meta-features → use cross-val on training set
        # Simplified: use mean prediction as meta-combination (stacking would leak)
        meta_probs_te = np.hstack(test_probs)  # (N_test, n_classes * n_modalities)
        meta_reg_te = np.hstack(test_preds_reg)  # (N_test, n_modalities)

        # Build meta-train features from train data using inner LOSO
        train_subjects = subject_ids[train_mask]
        meta_probs_tr = np.zeros((train_mask.sum(), n_classes * len(modalities)))
        meta_reg_tr = np.zeros((train_mask.sum(), len(modalities)))

        unique_train_subs = np.unique(train_subjects)
        for inner_sub in unique_train_subs:
            inner_te_mask_local = train_subjects == inner_sub
            inner_tr_mask_local = ~inner_te_mask_local
            global_inner_te = np.where(train_mask)[0][inner_te_mask_local]

            for m_idx, mod in enumerate(modalities):
                X_mod_tr = per_modality_features[mod][train_mask]
                X_inner_tr = X_mod_tr[inner_tr_mask_local]
                X_inner_te = X_mod_tr[inner_te_mask_local]

                sc = StandardScaler()
                X_inner_tr = sc.fit_transform(X_inner_tr)
                X_inner_te = sc.transform(X_inner_te)

                clf_inner = XGBClassifier(**xgb_params, use_label_encoder=False,
                                          eval_metric="mlogloss", verbosity=0)
                clf_inner.fit(X_inner_tr, y_class[train_mask][inner_tr_mask_local])
                p = clf_inner.predict_proba(X_inner_te)
                meta_probs_tr[inner_te_mask_local, m_idx*n_classes:(m_idx+1)*n_classes] = p

                reg_inner = XGBRegressor(**xgb_params, verbosity=0)
                reg_inner.fit(X_inner_tr, y_recall[train_mask][inner_tr_mask_local])
                meta_reg_tr[inner_te_mask_local, m_idx:m_idx+1] = reg_inner.predict(X_inner_te).reshape(-1, 1)

        # Meta-classifiers
        meta_clf = XGBClassifier(**xgb_params, use_label_encoder=False,
                                 eval_metric="mlogloss", verbosity=0)
        meta_clf.fit(meta_probs_tr, y_class[train_mask])
        y_pred_cls = meta_clf.predict(meta_probs_te)

        meta_reg = XGBRegressor(**xgb_params, verbosity=0)
        meta_reg.fit(meta_reg_tr, y_recall[train_mask])
        y_pred_r = meta_reg.predict(meta_reg_te)

        y_pred_class_all.extend(y_pred_cls.tolist())
        y_true_class_all.extend(y_class[test_mask].tolist())
        y_pred_reg_all.extend(y_pred_r.tolist())
        y_true_reg_all.extend(y_recall[test_mask].tolist())

        per_s_acc.append(accuracy_score(y_class[test_mask], y_pred_cls))
        if len(y_recall[test_mask]) > 1:
            per_s_r2.append(r2_score(y_recall[test_mask], y_pred_r))

    y_true_cls = np.array(y_true_class_all)
    y_pred_cls = np.array(y_pred_class_all)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    bacc = balanced_accuracy_score(y_true_cls, y_pred_cls)

    y_true_r = np.array(y_true_reg_all)
    y_pred_r = np.array(y_pred_reg_all)
    r2 = r2_score(y_true_r, y_pred_r)
    rmse = float(np.sqrt(np.mean((y_true_r - y_pred_r) ** 2)))

    logger.info(f"  B2 (late fusion) → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name="B2_Late_Fusion",
        description="Per-modality classifiers, probability stacking, meta-XGBoost",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=np.array(per_s_acc),
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=np.array(per_s_r2) if per_s_r2 else None,
        n_features=sum(v.shape[1] for v in per_modality_features.values()),
        n_subjects=len(unique_subjects),
        n_trials=list(per_modality_features.values())[0].shape[0],
    )


# ---------------------------------------------------------------------------
# B3: Early fusion — concatenated features, XGBoost, no coupling structure
# ---------------------------------------------------------------------------

def baseline_b3_early_fusion(
    per_modality_features: Dict[str, np.ndarray],
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
    xgb_params: Optional[Dict] = None,
) -> BaselineResult:
    """
    B3: Critical ablation. Concatenate all modality feature vectors; XGBoost.
    No GGM, no coupling features. Tests whether structured coupling adds beyond
    raw feature concatenation.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        XGBClassifier = GradientBoostingClassifier
        XGBRegressor = GradientBoostingRegressor

    if xgb_params is None:
        xgb_params = {"n_estimators": 200, "max_depth": 4,
                      "learning_rate": 0.05, "subsample": 0.8, "random_state": 42}

    X = np.hstack(list(per_modality_features.values()))

    def clf_factory():
        return XGBClassifier(**xgb_params, use_label_encoder=False,
                             eval_metric="mlogloss", verbosity=0)

    def reg_factory():
        return XGBRegressor(**xgb_params, verbosity=0)

    acc, bacc, per_s_acc, _, _ = _evaluate_classifier_loso(X, y_class, subject_ids, clf_factory)
    r2, rmse, per_s_r2, _, _ = _evaluate_regressor_loso(X, y_recall, subject_ids, reg_factory)

    logger.info(f"  B3 (early fusion) → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name="B3_Early_Fusion_NoStructure",
        description="Concatenated features, XGBoost, no coupling (critical ablation)",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=per_s_acc,
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=per_s_r2,
        n_features=X.shape[1],
        n_subjects=len(np.unique(subject_ids)),
        n_trials=X.shape[0],
    )


# ---------------------------------------------------------------------------
# B4: Full covariance matrix features (not precision matrix)
# ---------------------------------------------------------------------------

def baseline_b4_covariance_features(
    feature_matrices_by_condition: Dict[str, np.ndarray],
    feature_matrix_trial: np.ndarray,
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
) -> BaselineResult:
    """
    B4: Vectorize upper-triangle of per-condition sample covariance Sigma_c.
    Tests whether raw covariance vs. precision matrix matters.

    Since covariance is condition-level (not per-trial), we assign each trial the
    covariance features of its condition. This is the same design as GGM coupling
    features but using Sigma instead of Theta.
    """
    condition_labels = np.unique(y_class)
    D = list(feature_matrices_by_condition.values())[0].shape[1]
    n_upper = D * (D + 1) // 2

    # Compute upper-triangle covariance features per condition
    cov_features_by_cond: Dict[int, np.ndarray] = {}
    for c_label, (cond_name, X_c) in zip(condition_labels,
                                          feature_matrices_by_condition.items()):
        Sigma = np.cov(X_c.T)
        upper_idx = np.triu_indices(D)
        cov_features_by_cond[c_label] = Sigma[upper_idx]

    # Assign covariance features to each trial by condition label
    X_cov = np.zeros((len(y_class), n_upper))
    for i, c_label in enumerate(y_class):
        if c_label in cov_features_by_cond:
            X_cov[i] = cov_features_by_cond[c_label]
        else:
            X_cov[i] = np.zeros(n_upper)

    def clf_factory():
        return Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier())])

    def reg_factory():
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])

    acc, bacc, per_s_acc, _, _ = _evaluate_classifier_loso(X_cov, y_class, subject_ids, clf_factory, scaler=False)
    r2, rmse, per_s_r2, _, _ = _evaluate_regressor_loso(X_cov, y_recall, subject_ids, reg_factory, scaler=False)

    logger.info(f"  B4 (covariance features) → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name="B4_Covariance_Features",
        description="Upper-triangle of Sigma_c (not precision matrix) per condition",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=per_s_acc,
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=per_s_r2,
        n_features=n_upper,
        n_subjects=len(np.unique(subject_ids)),
        n_trials=len(y_class),
    )


# ---------------------------------------------------------------------------
# B5: Single cross-modal feature — EEG-pupil Pearson correlation
# ---------------------------------------------------------------------------

def baseline_b5_eeg_pupil_correlation(
    eeg_features: np.ndarray,
    pupil_features: np.ndarray,
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
) -> BaselineResult:
    """
    B5: Simplest coupling measure — per-trial Pearson r between EEG and pupil features.
    Tests whether trivial coupling already explains GGM improvement.
    """
    N = eeg_features.shape[0]
    corr_feature = np.zeros(N)

    for i in range(N):
        if eeg_features.ndim > 1 and pupil_features.ndim > 1:
            # Cross-correlation between feature vectors
            e = eeg_features[i] - eeg_features[i].mean()
            p = pupil_features[i] - pupil_features[i].mean()
            denom = (np.linalg.norm(e) * np.linalg.norm(p))
            corr_feature[i] = float(np.dot(e, p) / denom) if denom > 1e-12 else 0.0
        else:
            corr_feature[i] = float(np.corrcoef(
                eeg_features[i:i+1].ravel(), pupil_features[i:i+1].ravel()
            )[0, 1])

    X_single = corr_feature.reshape(-1, 1)

    def clf_factory():
        return RidgeClassifier()

    def reg_factory():
        return Ridge()

    acc, bacc, per_s_acc, _, _ = _evaluate_classifier_loso(X_single, y_class, subject_ids, clf_factory)
    r2, rmse, per_s_r2, _, _ = _evaluate_regressor_loso(X_single, y_recall, subject_ids, reg_factory)

    logger.info(f"  B5 (EEG-pupil corr) → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name="B5_EEG_Pupil_Correlation",
        description="Single cross-modal feature: EEG–pupil Pearson r",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=per_s_acc,
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=per_s_r2,
        n_features=1,
        n_subjects=len(np.unique(subject_ids)),
        n_trials=N,
    )


# ---------------------------------------------------------------------------
# B6: PCA of full multimodal feature vector
# ---------------------------------------------------------------------------

def baseline_b6_pca(
    per_modality_features: Dict[str, np.ndarray],
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
    n_components: int = 20,
) -> BaselineResult:
    """
    B6: PCA of concatenated features. Linear, not coupling-aware.
    Tests whether any linear projection does as well as structured GGM.
    """
    X_concat = np.hstack(list(per_modality_features.values()))
    unique_subjects = np.unique(subject_ids)

    y_pred_cls_all, y_true_cls_all = [], []
    y_pred_reg_all, y_true_reg_all = [], []
    per_s_acc, per_s_r2 = [], []

    for sub in unique_subjects:
        test_mask = subject_ids == sub
        train_mask = ~test_mask

        X_tr, X_te = X_concat[train_mask], X_concat[test_mask]
        y_cl_tr, y_cl_te = y_class[train_mask], y_class[test_mask]
        y_r_tr, y_r_te = y_recall[train_mask], y_recall[test_mask]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        n_comp = min(n_components, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
        pca = PCA(n_components=n_comp)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)

        clf = RidgeClassifier()
        clf.fit(X_tr_pca, y_cl_tr)
        y_pred_cls = clf.predict(X_te_pca)

        reg = Ridge()
        reg.fit(X_tr_pca, y_r_tr)
        y_pred_reg = reg.predict(X_te_pca)

        y_pred_cls_all.extend(y_pred_cls.tolist())
        y_true_cls_all.extend(y_cl_te.tolist())
        y_pred_reg_all.extend(y_pred_reg.tolist())
        y_true_reg_all.extend(y_r_te.tolist())

        per_s_acc.append(accuracy_score(y_cl_te, y_pred_cls))
        if len(y_r_te) > 1:
            per_s_r2.append(r2_score(y_r_te, y_pred_reg))

    y_true_cls = np.array(y_true_cls_all)
    y_pred_cls = np.array(y_pred_cls_all)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    bacc = balanced_accuracy_score(y_true_cls, y_pred_cls)

    y_true_r = np.array(y_true_reg_all)
    y_pred_r = np.array(y_pred_reg_all)
    r2 = r2_score(y_true_r, y_pred_r)
    rmse = float(np.sqrt(np.mean((y_true_r - y_pred_r) ** 2)))

    logger.info(f"  B6 (PCA) → Acc={acc:.4f}, BalAcc={bacc:.4f}, R²={r2:.4f}")

    return BaselineResult(
        name=f"B6_PCA_{n_components}comp",
        description=f"PCA({n_components} components) + Ridge, linear no coupling structure",
        loso_accuracy=acc,
        loso_balanced_accuracy=bacc,
        per_subject_accuracy=np.array(per_s_acc),
        loso_r2=r2,
        loso_rmse=rmse,
        per_subject_r2=np.array(per_s_r2) if per_s_r2 else None,
        n_features=n_components,
        n_subjects=len(unique_subjects),
        n_trials=X_concat.shape[0],
    )


# ---------------------------------------------------------------------------
# Master baseline runner
# ---------------------------------------------------------------------------

def run_all_baselines(
    catch22_features: Dict[str, np.ndarray],
    per_modality_features: Dict[str, np.ndarray],
    feature_matrices_by_condition: Dict[str, np.ndarray],
    eeg_features: np.ndarray,
    pupil_features: np.ndarray,
    y_class: np.ndarray,
    y_recall: np.ndarray,
    subject_ids: np.ndarray,
    pca_n_components: int = 20,
    xgb_params: Optional[Dict] = None,
) -> BaselineSuite:
    """
    Run all B1–B6 baselines and return aggregated results.

    Parameters
    ----------
    catch22_features : dict modality → (N_trials, 22) Catch22 features
    per_modality_features : dict modality → (N_trials, D_m) general features
    feature_matrices_by_condition : dict condition → (N_c, D) feature matrices
    eeg_features, pupil_features : (N_trials, D_mod) modality feature arrays
    y_class : (N_trials,) 0-3 condition labels
    y_recall : (N_trials,) continuous recall accuracy
    subject_ids : (N_trials,)

    Returns
    -------
    BaselineSuite
    """
    logger.info("=== Running all B1–B6 baselines ===")
    results: Dict[str, BaselineResult] = {}

    logger.info("--- B1: Catch22 + XGBoost ---")
    results["B1"] = baseline_b1_catch22_xgboost(
        catch22_features, y_class, y_recall, subject_ids, xgb_params
    )

    logger.info("--- B2: Late fusion ---")
    results["B2"] = baseline_b2_late_fusion(
        per_modality_features, y_class, y_recall, subject_ids, xgb_params
    )

    logger.info("--- B3: Early fusion (critical ablation) ---")
    results["B3"] = baseline_b3_early_fusion(
        per_modality_features, y_class, y_recall, subject_ids, xgb_params
    )

    logger.info("--- B4: Covariance features ---")
    results["B4"] = baseline_b4_covariance_features(
        feature_matrices_by_condition,
        np.hstack(list(per_modality_features.values())),
        y_class, y_recall, subject_ids
    )

    logger.info("--- B5: EEG-pupil correlation ---")
    results["B5"] = baseline_b5_eeg_pupil_correlation(
        eeg_features, pupil_features, y_class, y_recall, subject_ids
    )

    logger.info("--- B6: PCA ---")
    results["B6"] = baseline_b6_pca(
        per_modality_features, y_class, y_recall, subject_ids, pca_n_components
    )

    suite = BaselineSuite(results=results, task="both")
    logger.info("\n=== BASELINE SUMMARY ===\n" + suite.summary_table())
    return suite