"""
analysis/coupling_features.py
==============================

Extract coupling-derived features from fitted GGMs for downstream prediction.

Features derived from precision matrices Θ_c are more informative than
marginal features at overload — the key claim of this research.

Feature types:
  1. Cross-modal coupling strength (Frobenius norm of cross-block entries)
  2. Stability-weighted coupling (edges weighted by stability score)
  3. Delta coupling (task − resting baseline per subject)
  4. Temporal coupling trajectory (coupling across digit sequence)
  5. Per-modality-pair coupling vectors for individual difference analysis

Used in:
  - Recall accuracy prediction (ridge regression, LOSO)
  - Individual WM capacity regression
  - Overload detection (BOCPD on coupling trajectory)

Dependencies: numpy, models/ggm.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.ggm import GGMResult, ConditionGGMs, StabilityResult
from models.ggm import frobenius_cross_modal_strength, cross_modal_edge_density
from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Feature containers ─────────────────────────────────────────────────────────

@dataclass
class CouplingFeatureVector:
    """
    Coupling feature vector for one subject-condition observation.

    Built from precision matrix blocks.
    Used as feature input to ridge regression for recall prediction.
    """
    subject_id: str
    condition: int

    # Cross-modal Frobenius norms per pair
    pair_strengths: Dict[str, float]       # "mod_a-mod_b" → norm

    # Edge density per pair
    pair_densities: Dict[str, float]

    # Stability-weighted strengths (only reliable edges)
    pair_strengths_stable: Dict[str, float]

    # Delta from resting state (if available)
    pair_delta_strengths: Optional[Dict[str, float]] = None

    # Total coupling summary
    total_coupling: float = 0.0
    n_active_pairs: int = 0

    def to_array(self, pair_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Flatten to fixed-length array for ML.

        Parameters
        ----------
        pair_order : list of "mod_a-mod_b" keys, in consistent order

        Returns
        -------
        ndarray (n_features,)
        """
        if pair_order is None:
            pair_order = sorted(self.pair_strengths.keys())

        features = []
        for key in pair_order:
            features.append(self.pair_strengths.get(key, 0.0))
            features.append(self.pair_densities.get(key, 0.0))
            features.append(self.pair_strengths_stable.get(key, 0.0))
            if self.pair_delta_strengths is not None:
                features.append(self.pair_delta_strengths.get(key, 0.0))

        features.append(self.total_coupling)
        features.append(float(self.n_active_pairs))

        return np.array(features, dtype=float)


@dataclass
class SubjectCouplingProfile:
    """
    Per-subject coupling profile across all conditions.
    Used for individual difference analysis.
    """
    subject_id: str
    condition_features: Dict[int, CouplingFeatureVector]

    # Resting-state coupling (Θ^(s)_0)
    resting_strengths: Optional[Dict[str, float]] = None

    # Coupling trajectory: list of total_coupling values in condition order
    coupling_trajectory: Optional[np.ndarray] = None

    # Derived: coupling drop at overload
    coupling_drop_9_to_13: Optional[float] = None


# ── Extraction from a single GGM ──────────────────────────────────────────────

def extract_coupling_features(
    ggm_result: GGMResult,
    modality_pairs: List[Tuple[str, str]],
    stability_result: Optional[StabilityResult] = None,
    resting_ggm: Optional[GGMResult] = None,
    subject_id: str = "unknown",
) -> CouplingFeatureVector:
    """
    Extract coupling feature vector from one GGM.

    Parameters
    ----------
    ggm_result : GGMResult
        Condition-specific GGM.
    modality_pairs : list of (mod_a, mod_b)
    stability_result : StabilityResult or None
        If provided, weight features by edge stability scores.
    resting_ggm : GGMResult or None
        If provided, compute delta features (Θ_task - Θ_rest).
    subject_id : str

    Returns
    -------
    CouplingFeatureVector
    """
    prec = ggm_result.precision_matrix
    blocks = ggm_result.modality_blocks

    pair_strengths = {}
    pair_densities = {}
    pair_strengths_stable = {}
    pair_delta_strengths = {} if resting_ggm is not None else None

    for mod_a, mod_b in modality_pairs:
        if mod_a not in blocks or mod_b not in blocks:
            continue

        key = f"{mod_a}-{mod_b}"
        sa, ea = blocks[mod_a]
        sb, eb = blocks[mod_b]

        # Raw cross-block
        block = prec[sa:ea, sb:eb]
        strength = float(np.linalg.norm(block, "fro"))
        density = cross_modal_edge_density(prec, blocks, mod_a, mod_b)

        pair_strengths[key] = strength
        pair_densities[key] = density

        # Stability-weighted: zero out edges below pi_threshold
        if stability_result is not None:
            stable_mask = stability_result.stable_edges[sa:ea, sb:eb]
            block_stable = block * stable_mask
            pair_strengths_stable[key] = float(np.linalg.norm(block_stable, "fro"))
        else:
            pair_strengths_stable[key] = strength

        # Delta from resting state
        if resting_ggm is not None and pair_delta_strengths is not None:
            rest_prec = resting_ggm.precision_matrix
            rest_block = rest_prec[sa:ea, sb:eb]
            delta_block = block - rest_block
            pair_delta_strengths[key] = float(np.linalg.norm(delta_block, "fro"))

    total_coupling = float(np.mean(list(pair_strengths.values()))) if pair_strengths else 0.0
    n_active = sum(1 for v in pair_densities.values() if v > 0.0)

    return CouplingFeatureVector(
        subject_id=subject_id,
        condition=ggm_result.condition,
        pair_strengths=pair_strengths,
        pair_densities=pair_densities,
        pair_strengths_stable=pair_strengths_stable,
        pair_delta_strengths=pair_delta_strengths,
        total_coupling=total_coupling,
        n_active_pairs=n_active,
    )


# ── Per-subject coupling profiles ─────────────────────────────────────────────

def build_subject_coupling_profiles(
    condition_ggms: ConditionGGMs,
    modality_pairs: List[Tuple[str, str]],
    subject_ids: List[str],
    resting_ggms: Optional[Dict[str, GGMResult]] = None,
    condition_order: Optional[List[int]] = None,
) -> Dict[str, SubjectCouplingProfile]:
    """
    Build per-subject coupling profiles across conditions.

    Note: condition-level GGMs are population-level (all subjects pooled).
    Subject-specific variation is captured via:
      1. Per-subject resting GGM (delta features)
      2. Per-subject coupling features computed from subject-specific subsets

    Parameters
    ----------
    condition_ggms : ConditionGGMs
        Population-level condition GGMs.
    modality_pairs : list of (mod_a, mod_b)
    subject_ids : list of str
    resting_ggms : dict {subject_id → resting GGMResult} or None
    condition_order : list of condition ints in ascending load order

    Returns
    -------
    dict {subject_id → SubjectCouplingProfile}
    """
    if condition_order is None:
        condition_order = sorted(condition_ggms.condition_ggms.keys())

    profiles = {}

    for subj_id in subject_ids:
        condition_features = {}
        resting_ggm = resting_ggms.get(subj_id) if resting_ggms else None

        for cond in condition_order:
            if cond not in condition_ggms.condition_ggms:
                continue

            ggm = condition_ggms.condition_ggms[cond]
            stability = condition_ggms.stability_results.get(cond)

            fvec = extract_coupling_features(
                ggm_result=ggm,
                modality_pairs=modality_pairs,
                stability_result=stability,
                resting_ggm=resting_ggm,
                subject_id=subj_id,
            )
            condition_features[cond] = fvec

        # Coupling trajectory across conditions
        traj = np.array([
            condition_features[c].total_coupling
            for c in condition_order
            if c in condition_features
        ])

        # Coupling drop from 9-digit to 13-digit (key individual feature)
        coupling_drop = None
        cond_9 = _find_condition(condition_order, 2)   # label 2 = 9-digit
        cond_13 = _find_condition(condition_order, 3)  # label 3 = 13-digit
        if (cond_9 is not None and cond_13 is not None and
                cond_9 in condition_features and cond_13 in condition_features):
            coupling_drop = (
                condition_features[cond_9].total_coupling -
                condition_features[cond_13].total_coupling
            )

        # Resting coupling strengths
        resting_strengths = None
        if resting_ggm is not None:
            resting_coupling_vec = extract_coupling_features(
                ggm_result=resting_ggm,
                modality_pairs=modality_pairs,
                subject_id=subj_id,
            )
            resting_strengths = resting_coupling_vec.pair_strengths

        profiles[subj_id] = SubjectCouplingProfile(
            subject_id=subj_id,
            condition_features=condition_features,
            resting_strengths=resting_strengths,
            coupling_trajectory=traj,
            coupling_drop_9_to_13=coupling_drop,
        )

    return profiles


def _find_condition(condition_order: List[int], label: int) -> Optional[int]:
    """Find condition with given label in ordered list."""
    if label in condition_order:
        return label
    return None


# ── Feature matrix assembly ────────────────────────────────────────────────────

def assemble_coupling_feature_matrix(
    profiles: Dict[str, SubjectCouplingProfile],
    modality_pairs: List[Tuple[str, str]],
    condition_order: Optional[List[int]] = None,
    include_delta: bool = True,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    """
    Assemble coupling features into matrix for ridge regression.

    One row per (subject, condition) observation.

    Parameters
    ----------
    profiles : dict {subject_id → SubjectCouplingProfile}
    modality_pairs : list of (mod_a, mod_b)
    condition_order : list of condition ints
    include_delta : bool
        Include delta-from-resting features.

    Returns
    -------
    X_coupling : ndarray (n_obs, n_features)
    feature_names : list of str
    subject_ids : list of str  (n_obs,)
    condition_ids : ndarray (n_obs,)
    """
    pair_keys = [f"{a}-{b}" for a, b in modality_pairs]

    rows = []
    sub_list = []
    cond_list = []

    # Determine feature names from first valid entry
    feat_names = None

    for subj_id, profile in profiles.items():
        conds = sorted(profile.condition_features.keys()) if condition_order is None else condition_order

        for cond in conds:
            if cond not in profile.condition_features:
                continue

            fvec = profile.condition_features[cond]

            if include_delta and fvec.pair_delta_strengths is None:
                include_delta_this = False
            else:
                include_delta_this = include_delta

            arr = fvec.to_array(pair_order=pair_keys)
            rows.append(arr)
            sub_list.append(subj_id)
            cond_list.append(cond)

    if not rows:
        return np.zeros((0, 0)), [], [], np.zeros(0, dtype=int)

    X = np.vstack(rows)

    # Generate feature names
    feat_names = []
    for key in pair_keys:
        feat_names.extend([
            f"coupling_strength_{key}",
            f"coupling_density_{key}",
            f"coupling_stable_{key}",
        ])
        if include_delta:
            feat_names.append(f"coupling_delta_{key}")
    feat_names.extend(["total_coupling", "n_active_pairs"])

    # Truncate feature names to match X width
    feat_names = feat_names[:X.shape[1]]

    return X, feat_names, sub_list, np.array(cond_list, dtype=int)


# ── Resting-state coupling features ───────────────────────────────────────────

def extract_resting_coupling_features(
    resting_ggms: Dict[str, GGMResult],
    modality_pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract resting-state coupling feature matrix (one row per subject).

    Used for zero-shot personalization:
    Subjects with higher resting coupling show larger task-induced changes.

    Returns
    -------
    X_rest : ndarray (n_subjects, n_features)
    feature_names : list of str
    subject_ids : list of str
    """
    rows = []
    subject_ids = []

    for subj_id, ggm in resting_ggms.items():
        fvec = extract_coupling_features(
            ggm_result=ggm,
            modality_pairs=modality_pairs,
            subject_id=subj_id,
        )
        arr = fvec.to_array(pair_order=[f"{a}-{b}" for a, b in modality_pairs])
        rows.append(arr)
        subject_ids.append(subj_id)

    if not rows:
        return np.zeros((0, 0)), [], []

    X = np.vstack(rows)
    pair_keys = [f"{a}-{b}" for a, b in modality_pairs]
    feat_names = []
    for key in pair_keys:
        feat_names.extend([
            f"rest_strength_{key}",
            f"rest_density_{key}",
            f"rest_stable_{key}",
        ])
    feat_names.extend(["rest_total_coupling", "rest_n_active_pairs"])
    feat_names = feat_names[:X.shape[1]]

    return X, feat_names, subject_ids