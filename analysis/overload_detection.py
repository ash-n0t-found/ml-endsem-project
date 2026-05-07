"""
evaluation/overload_detection.py
=================================
Evaluation 4: Overload detection via BOCPD on coupling trajectory.

Tests whether physiological coupling changepoints align with
behavioral performance dropoffs across subjects.

Metrics:
  - Pearson / Spearman correlation between physiological and behavioral CPs
  - Classification accuracy: coupling-based vs threshold-based marginal
  - Sensitivity / specificity for overload detection
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class OverloadDetectionEvaluator:
    """
    Evaluate coupling-based overload detection against behavioral reference.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate_coupling_vs_behavioral(
        self,
        physiological_onsets: Dict[str, Optional[int]],
        behavioral_onsets: Dict[str, Optional[int]],
    ) -> Dict:
        """
        Correlate physiological (BOCPD/LGSSM) onset with behavioral onset.

        Parameters
        ----------
        physiological_onsets : subject_id → digit position of physiological changepoint
        behavioral_onsets : subject_id → digit position of behavioral performance drop

        Returns
        -------
        dict with Pearson r, Spearman rho, MAE, bias
        """
        common_subs = [
            s for s in physiological_onsets
            if s in behavioral_onsets
            and physiological_onsets[s] is not None
            and behavioral_onsets[s] is not None
        ]

        if len(common_subs) < 5:
            logger.warning(
                f"Too few subjects for onset correlation: {len(common_subs)}"
            )
            return {"n_subjects": len(common_subs), "error": "insufficient data"}

        X_phys = np.array([physiological_onsets[s] for s in common_subs], dtype=float)
        Y_behav = np.array([behavioral_onsets[s] for s in common_subs], dtype=float)

        # Pearson correlation
        r_pearson, p_pearson = scipy_stats.pearsonr(X_phys, Y_behav)

        # Spearman correlation
        r_spearman, p_spearman = scipy_stats.spearmanr(X_phys, Y_behav)

        # Mean absolute error
        mae = float(np.mean(np.abs(X_phys - Y_behav)))

        # Bias (systematic offset)
        bias = float(np.mean(X_phys - Y_behav))

        # Within-1-digit accuracy
        within_1 = float(np.mean(np.abs(X_phys - Y_behav) <= 1.0))
        within_2 = float(np.mean(np.abs(X_phys - Y_behav) <= 2.0))

        result = {
            "n_subjects": len(common_subs),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(r_spearman),
            "spearman_p": float(p_spearman),
            "mae_digits": mae,
            "bias_digits": bias,
            "within_1_digit_accuracy": within_1,
            "within_2_digit_accuracy": within_2,
            "physiological_mean": float(X_phys.mean()),
            "behavioral_mean": float(Y_behav.mean()),
            "subject_ids": common_subs,
        }

        logger.info(
            f"Onset correlation: "
            f"Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), "
            f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.4f}), "
            f"MAE={mae:.2f} digits, "
            f"N={len(common_subs)}"
        )

        return result

    def evaluate_binary_overload_classification(
        self,
        coupling_trajectories: Dict[str, np.ndarray],
        behavioral_overload: Dict[str, bool],
        threshold_method: str = "bocpd",
    ) -> Dict:
        """
        Binary classification: overload vs non-overload at the trial level.
        Physiological detector: coupling drops below threshold at some digit.
        Behavioral reference: recall accuracy drops below 50% correct.

        Parameters
        ----------
        coupling_trajectories : subject_id → (T,) coupling strength over digits
        behavioral_overload : subject_id → True if overloaded (recall < 50%)
        """
        y_true = []
        y_pred_coupling = []
        y_score_coupling = []

        for sub_id, traj in coupling_trajectories.items():
            if sub_id not in behavioral_overload:
                continue

            y_true.append(int(behavioral_overload[sub_id]))

            # Coupling-based overload score: peak-to-final ratio
            # High ratio = strong peak then collapse = likely overload
            traj = np.array(traj)
            if len(traj) < 2:
                y_pred_coupling.append(0)
                y_score_coupling.append(0.0)
                continue

            peak_val = traj.max()
            final_val = traj[-1]
            peak_idx = traj.argmax()

            # Score: relative decline from peak
            if peak_val > 1e-10:
                decline_ratio = (peak_val - final_val) / peak_val
            else:
                decline_ratio = 0.0

            # Only score as overloaded if peak is not at the last position
            if peak_idx == len(traj) - 1:
                decline_ratio = 0.0

            y_score_coupling.append(float(decline_ratio))

        if not y_true:
            return {"error": "no common subjects"}

        y_true = np.array(y_true)
        y_score = np.array(y_score_coupling)

        # Binary prediction: threshold on decline ratio
        from sklearn.model_selection import StratifiedKFold
        thresholds = np.percentile(y_score, [25, 50, 75])
        best_f1 = -1
        best_thresh = thresholds[1]

        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        y_pred = (y_score >= best_thresh).astype(int)

        result = {
            "n_subjects": len(y_true),
            "n_overloaded": int(y_true.sum()),
            "threshold_used": float(best_thresh),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if len(np.unique(y_true)) == 2:
            try:
                result["auroc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                result["auroc"] = np.nan

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result.update({
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            })

        logger.info(
            f"Binary overload classification: "
            f"Acc={result['accuracy']:.3f}, "
            f"F1={result['f1']:.3f}, "
            f"AUROC={result.get('auroc', np.nan):.3f}"
        )

        return result

    def compare_coupling_vs_marginal_threshold(
        self,
        coupling_trajectories: Dict[str, np.ndarray],
        marginal_trajectories: Dict[str, np.ndarray],  # e.g. single-channel theta
        behavioral_overload: Dict[str, bool],
    ) -> Dict:
        """
        Compare coupling-based vs single-channel threshold-based detection.
        Primary comparison for the paper: coupling > marginal.
        """
        # Coupling classifier
        coup_result = self.evaluate_binary_overload_classification(
            coupling_trajectories, behavioral_overload, threshold_method="coupling"
        )

        # Marginal classifier (same framework)
        marg_result = self.evaluate_binary_overload_classification(
            marginal_trajectories, behavioral_overload, threshold_method="marginal"
        )

        delta_auroc = (
            coup_result.get("auroc", np.nan) - marg_result.get("auroc", np.nan)
        )
        delta_f1 = coup_result.get("f1", np.nan) - marg_result.get("f1", np.nan)

        comparison = {
            "coupling": coup_result,
            "marginal": marg_result,
            "delta_auroc": float(delta_auroc) if not np.isnan(delta_auroc) else None,
            "delta_f1": float(delta_f1) if not np.isnan(delta_f1) else None,
            "coupling_superior": bool(delta_auroc > 0) if not np.isnan(delta_auroc) else None,
        }

        logger.info(
            f"Coupling vs marginal overload detection: "
            f"ΔAUROC={delta_auroc:.3f}, ΔF1={delta_f1:.3f}, "
            f"coupling_superior={comparison['coupling_superior']}"
        )

        return comparison

    def compute_individual_detection_stats(
        self,
        physiological_onsets: Dict[str, Optional[int]],
        behavioral_onsets: Dict[str, Optional[int]],
        wm_capacity: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Per-subject detection statistics.
        Includes stratified analysis by WM capacity (high vs low).
        """
        per_subject = {}
        for sub_id in physiological_onsets:
            if sub_id not in behavioral_onsets:
                continue
            phys = physiological_onsets[sub_id]
            behav = behavioral_onsets[sub_id]

            if phys is None or behav is None:
                per_subject[sub_id] = {"detected": False, "error_digits": None}
                continue

            error = int(phys) - int(behav)
            per_subject[sub_id] = {
                "detected": True,
                "physiological_onset": int(phys),
                "behavioral_onset": int(behav),
                "error_digits": int(error),
                "abs_error_digits": abs(error),
                "early_detection": error < 0,  # physiological onset before behavioral
                "wm_capacity": float(wm_capacity[sub_id]) if wm_capacity and sub_id in wm_capacity else None,
            }

        # Stratified analysis by WM capacity
        if wm_capacity:
            wm_vals = [
                per_subject[s]["wm_capacity"]
                for s in per_subject
                if per_subject[s].get("wm_capacity") is not None
            ]
            if wm_vals:
                median_wm = np.median(wm_vals)
                high_wm_errors = [
                    per_subject[s]["abs_error_digits"]
                    for s in per_subject
                    if per_subject[s].get("wm_capacity", 0) > median_wm
                    and per_subject[s].get("abs_error_digits") is not None
                ]
                low_wm_errors = [
                    per_subject[s]["abs_error_digits"]
                    for s in per_subject
                    if per_subject[s].get("wm_capacity", 999) <= median_wm
                    and per_subject[s].get("abs_error_digits") is not None
                ]
                per_subject["_stratified"] = {
                    "high_wm_mean_abs_error": float(np.mean(high_wm_errors)) if high_wm_errors else np.nan,
                    "low_wm_mean_abs_error": float(np.mean(low_wm_errors)) if low_wm_errors else np.nan,
                    "median_wm": float(median_wm),
                }

        return per_subject

    def generate_summary(
        self,
        correlation_result: Dict,
        classification_result: Dict,
        comparison_result: Dict,
    ) -> str:
        """Generate formatted summary string for logging/printing."""
        lines = [
            "=" * 60,
            "OVERLOAD DETECTION EVALUATION SUMMARY",
            "=" * 60,
            "",
            "--- Onset Correlation (physiological vs behavioral) ---",
            f"  N subjects : {correlation_result.get('n_subjects', '?')}",
            f"  Pearson r  : {correlation_result.get('pearson_r', np.nan):.3f} "
            f"(p={correlation_result.get('pearson_p', np.nan):.4f})",
            f"  Spearman ρ : {correlation_result.get('spearman_rho', np.nan):.3f} "
            f"(p={correlation_result.get('spearman_p', np.nan):.4f})",
            f"  MAE        : {correlation_result.get('mae_digits', np.nan):.2f} digits",
            f"  Within ±1  : {correlation_result.get('within_1_digit_accuracy', np.nan):.1%}",
            "",
            "--- Binary Overload Classification ---",
            f"  Coupling AUROC : {classification_result.get('auroc', np.nan):.3f}",
            f"  Coupling F1    : {classification_result.get('f1', np.nan):.3f}",
            f"  Sensitivity    : {classification_result.get('sensitivity', np.nan):.3f}",
            f"  Specificity    : {classification_result.get('specificity', np.nan):.3f}",
            "",
            "--- Coupling vs Marginal ---",
            f"  ΔAUROC (coupling − marginal) : {comparison_result.get('delta_auroc', np.nan):.3f}",
            f"  ΔF1   (coupling − marginal)  : {comparison_result.get('delta_f1', np.nan):.3f}",
            f"  Coupling superior            : {comparison_result.get('coupling_superior', '?')}",
            "=" * 60,
        ]
        return "\n".join(lines)