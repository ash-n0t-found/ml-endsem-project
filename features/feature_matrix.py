"""
features/feature_matrix.py
===========================
Assembles per-modality feature dicts into a unified feature matrix
suitable for GGM estimation and downstream prediction.

Responsibilities:
  1. Collect per-trial features from all modalities
  2. Assign modality block indices (for cross-block precision analysis)
  3. Apply per-subject z-score normalization
  4. Reduce to target dimensionality (PCA per modality block)
  5. Handle missing features gracefully
  6. Save/load feature matrix with metadata

Output structure:
  FeatureMatrix dataclass containing:
    - X          : ndarray (n_trials, n_features)  — normalized features
    - feature_names : list of str
    - modality_blocks : dict {modality: (start_idx, end_idx)}
    - subject_ids : ndarray (n_trials,)
    - condition_ids : ndarray (n_trials,)
    - trial_ids   : ndarray (n_trials,)
    - recall_accuracy : ndarray (n_trials,)  — behavioral target

Dependencies: numpy, scipy, scikit-learn
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.io_utils import save_cache, load_cache, setup_logger
from utils.stats import zscore_per_subject, unit_variance_per_condition

logger = setup_logger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrialRecord:
    """Container for one trial's multimodal features + metadata."""
    subject_id: str
    trial_idx: int
    condition: int              # 0=control,1=5-dig,2=9-dig,3=13-dig
    recall_accuracy: float      # fraction correct (0-1), NaN if no recall
    eeg_features: Dict[str, float] = field(default_factory=dict)
    ecg_features: Dict[str, float] = field(default_factory=dict)
    ppg_features: Dict[str, float] = field(default_factory=dict)
    pupil_features: Dict[str, float] = field(default_factory=dict)
    hep_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureMatrix:
    """Unified feature matrix with modality block metadata."""
    X: np.ndarray                          # (n_trials, n_features)
    feature_names: List[str]
    modality_blocks: Dict[str, Tuple[int, int]]   # modality → (start, end)
    subject_ids: np.ndarray                # (n_trials,)  str
    condition_ids: np.ndarray              # (n_trials,)  int
    trial_ids: np.ndarray                  # (n_trials,)  int
    recall_accuracy: np.ndarray            # (n_trials,)  float

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def get_modality_block(self, modality: str) -> np.ndarray:
        """Return feature sub-matrix for a single modality."""
        s, e = self.modality_blocks[modality]
        return self.X[:, s:e]

    def get_cross_modal_block(self, mod_a: str, mod_b: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_a, X_b) sub-matrices for a modality pair."""
        return self.get_modality_block(mod_a), self.get_modality_block(mod_b)

    def condition_mask(self, condition: int) -> np.ndarray:
        """Boolean mask for trials of a given condition."""
        return self.condition_ids == condition

    def subject_mask(self, subject_id: str) -> np.ndarray:
        """Boolean mask for trials of a given subject."""
        return self.subject_ids == subject_id

    def save(self, path: Path) -> None:
        """Save to pickle."""
        save_cache(self, path)
        logger.info(f"FeatureMatrix saved: {path} | shape={self.X.shape}")

    @classmethod
    def load(cls, path: Path) -> "FeatureMatrix":
        """Load from pickle."""
        fm = load_cache(path)
        logger.info(f"FeatureMatrix loaded: {path} | shape={fm.X.shape}")
        return fm


# ── Feature name extraction ───────────────────────────────────────────────────

# Features to EXCLUDE from GGM (quality flags, not physiological signals)
_EXCLUDE_KEYS = {
    "pct_valid", "hep_n_peaks", "hep_hrv_n",
}

def _filter_features(feat_dict: Dict[str, float]) -> Dict[str, float]:
    """Remove quality-flag features, keep physiological ones."""
    return {k: v for k, v in feat_dict.items() if k not in _EXCLUDE_KEYS}


# ── Core assembly function ────────────────────────────────────────────────────

MODALITY_ORDER = ["eeg", "ecg", "ppg", "pupil", "hep"]


def assemble_feature_matrix(
    trial_records: List[TrialRecord],
    normalize: bool = True,
    pca_per_modality: Optional[int] = None,
    max_total_features: int = 60,
    min_valid_fraction: float = 0.8,
) -> FeatureMatrix:
    """
    Assemble list of TrialRecord → FeatureMatrix.

    Parameters
    ----------
    trial_records : list of TrialRecord
    normalize : bool
        If True, apply per-subject z-score normalization.
    pca_per_modality : int, optional
        If set, reduce each modality block to this many PCs.
        Used for PID analysis (typically 3).
    max_total_features : int
        Cap on total features (applied after PCA if pca_per_modality set).
    min_valid_fraction : float
        Features with NaN rate > (1 - min_valid_fraction) are dropped.

    Returns
    -------
    FeatureMatrix
    """
    if not trial_records:
        raise ValueError("trial_records is empty")

    n = len(trial_records)

    # Collect feature names per modality
    modality_feat_names: Dict[str, List[str]] = {m: [] for m in MODALITY_ORDER}
    for m in MODALITY_ORDER:
        sample_dict = _get_modality_dict(trial_records[0], m)
        modality_feat_names[m] = sorted(_filter_features(sample_dict).keys())

    # Build raw feature matrix block by block
    blocks: Dict[str, np.ndarray] = {}
    for m in MODALITY_ORDER:
        feat_names = modality_feat_names[m]
        if not feat_names:
            continue
        block = np.full((n, len(feat_names)), np.nan)
        for i, rec in enumerate(trial_records):
            d = _filter_features(_get_modality_dict(rec, m))
            for j, fname in enumerate(feat_names):
                block[i, j] = d.get(fname, np.nan)
        blocks[m] = block

    # Drop features with too many NaNs
    cleaned_blocks: Dict[str, np.ndarray] = {}
    cleaned_names: Dict[str, List[str]] = {}
    for m, block in blocks.items():
        valid_frac = (~np.isnan(block)).mean(axis=0)
        keep = valid_frac >= min_valid_fraction
        if keep.sum() == 0:
            logger.warning(f"Modality '{m}': no features pass validity threshold. Skipping.")
            continue
        cleaned_blocks[m] = block[:, keep]
        cleaned_names[m] = [modality_feat_names[m][j] for j in np.where(keep)[0]]
        # Impute remaining NaNs with column median
        cleaned_blocks[m] = _impute_median(cleaned_blocks[m])

    if not cleaned_blocks:
        raise RuntimeError("No valid features after quality filtering.")

    # Metadata arrays
    subject_ids = np.array([r.subject_id for r in trial_records])
    condition_ids = np.array([r.condition for r in trial_records], dtype=int)
    trial_ids = np.array([r.trial_idx for r in trial_records], dtype=int)
    recall_accuracy = np.array([r.recall_accuracy for r in trial_records], dtype=float)

    # Normalize
    if normalize:
        for m in cleaned_blocks:
            cleaned_blocks[m] = zscore_per_subject(cleaned_blocks[m], subject_ids)

    # Optional PCA reduction per modality
    if pca_per_modality is not None and pca_per_modality > 0:
        reduced_blocks: Dict[str, np.ndarray] = {}
        reduced_names: Dict[str, List[str]] = {}
        for m, block in cleaned_blocks.items():
            n_comp = min(pca_per_modality, block.shape[1], block.shape[0] - 1)
            if n_comp < 1:
                reduced_blocks[m] = block
                reduced_names[m] = cleaned_names[m]
                continue
            pca = PCA(n_components=n_comp)
            reduced_blocks[m] = pca.fit_transform(block)
            reduced_names[m] = [f"{m}_pc{k+1}" for k in range(n_comp)]
            logger.debug(
                f"PCA {m}: {block.shape[1]}→{n_comp} components, "
                f"var explained={pca.explained_variance_ratio_.sum():.3f}"
            )
        cleaned_blocks = reduced_blocks
        cleaned_names = reduced_names

    # Concatenate modality blocks and track block boundaries
    feature_arrays = []
    feature_names = []
    modality_blocks: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    for m in MODALITY_ORDER:
        if m not in cleaned_blocks:
            continue
        block = cleaned_blocks[m]
        names = cleaned_names[m]
        modality_blocks[m] = (cursor, cursor + block.shape[1])
        feature_arrays.append(block)
        feature_names.extend(names)
        cursor += block.shape[1]

    X = np.hstack(feature_arrays)  # (n_trials, n_total_features)

    logger.info(
        f"Feature matrix assembled: shape={X.shape}, "
        f"modalities={list(modality_blocks.keys())}, "
        f"n_subjects={len(np.unique(subject_ids))}"
    )

    return FeatureMatrix(
        X=X,
        feature_names=feature_names,
        modality_blocks=modality_blocks,
        subject_ids=subject_ids,
        condition_ids=condition_ids,
        trial_ids=trial_ids,
        recall_accuracy=recall_accuracy,
    )


# ── Condition-stratified matrices (for GGM per condition) ─────────────────────

def split_by_condition(
    fm: FeatureMatrix,
) -> Dict[int, FeatureMatrix]:
    """
    Split feature matrix into per-condition sub-matrices.
    Also applies unit-variance scaling per condition (for GGM fitting).

    Returns
    -------
    dict : condition_id → FeatureMatrix
    """
    conditions = np.unique(fm.condition_ids)
    result = {}
    for cond in conditions:
        mask = fm.condition_mask(cond)
        X_cond = fm.X[mask].copy()
        # Unit-variance per condition (before GGM: prevent scale artifacts)
        sd = X_cond.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-10, 1.0, sd)
        X_cond = X_cond / sd

        result[int(cond)] = FeatureMatrix(
            X=X_cond,
            feature_names=fm.feature_names,
            modality_blocks=fm.modality_blocks,
            subject_ids=fm.subject_ids[mask],
            condition_ids=fm.condition_ids[mask],
            trial_ids=fm.trial_ids[mask],
            recall_accuracy=fm.recall_accuracy[mask],
        )
        logger.info(f"Condition {cond}: {mask.sum()} trials")

    return result


# ── Subject-level resting matrix ──────────────────────────────────────────────

def build_resting_feature_matrix(
    resting_features: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build feature matrix from per-subject resting-state features.

    Parameters
    ----------
    resting_features : {subject_id: {feature_name: value}}

    Returns
    -------
    X_rest : ndarray, shape (n_subjects, n_features)
    feature_names : list of str
    subject_ids : list of str
    """
    subject_ids = sorted(resting_features.keys())
    all_keys = sorted(
        {k for d in resting_features.values() for k in d.keys()
         if k not in _EXCLUDE_KEYS}
    )

    X_rest = np.full((len(subject_ids), len(all_keys)), np.nan)
    for i, subj in enumerate(subject_ids):
        for j, k in enumerate(all_keys):
            X_rest[i, j] = resting_features[subj].get(k, np.nan)

    # Impute
    X_rest = _impute_median(X_rest)
    return X_rest, all_keys, subject_ids


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_modality_dict(rec: TrialRecord, modality: str) -> Dict[str, float]:
    """Retrieve feature dict for a modality from a TrialRecord."""
    return getattr(rec, f"{modality}_features", {})


def _impute_median(X: np.ndarray) -> np.ndarray:
    """Replace NaNs with column median."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any() and (~nan_mask).any():
            X[nan_mask, j] = np.nanmedian(col)
    return X


# ── Feature selection utilities ───────────────────────────────────────────────

def select_physiological_features(
    fm: FeatureMatrix,
    max_per_modality: int = 15,
) -> FeatureMatrix:
    """
    Reduce feature matrix to most physiologically motivated features
    (max_per_modality per modality), selected by variance.
    Used to keep D=40-60 for primary GGM analysis.

    Parameters
    ----------
    fm : FeatureMatrix
    max_per_modality : int

    Returns
    -------
    FeatureMatrix with reduced feature set
    """
    keep_indices = []
    new_modality_blocks: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    for modality, (s, e) in fm.modality_blocks.items():
        block = fm.X[:, s:e]
        n_feat = block.shape[1]

        if n_feat <= max_per_modality:
            selected = list(range(n_feat))
        else:
            # Select by variance (most informative features)
            variances = np.nanvar(block, axis=0)
            selected = np.argsort(variances)[::-1][:max_per_modality].tolist()

        global_indices = [s + j for j in selected]
        keep_indices.extend(global_indices)
        new_modality_blocks[modality] = (cursor, cursor + len(selected))
        cursor += len(selected)

    X_reduced = fm.X[:, keep_indices]
    feat_names = [fm.feature_names[i] for i in keep_indices]

    logger.info(f"Feature selection: {fm.n_features} → {X_reduced.shape[1]} features")

    return FeatureMatrix(
        X=X_reduced,
        feature_names=feat_names,
        modality_blocks=new_modality_blocks,
        subject_ids=fm.subject_ids,
        condition_ids=fm.condition_ids,
        trial_ids=fm.trial_ids,
        recall_accuracy=fm.recall_accuracy,
    )