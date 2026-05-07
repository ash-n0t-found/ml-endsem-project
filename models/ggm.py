"""
models/ggm.py
=============

Sparse Gaussian Graphical Model estimation via GraphicalLasso.

Implements:
  1. Condition-specific GGM fitting (GraphicalLassoCV)
  2. Stability selection (Meinshausen-Bühlmann 2010) for reliable edge detection
  3. Resting-state per-subject GGM
  4. Cross-modal block extraction from precision matrix

All methods return interpretable precision matrices — no black-box outputs.

Dependencies: scikit-learn, numpy, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.preprocessing import StandardScaler

from utils.io_utils import setup_logger, save_cache, load_cache
from utils.stats import unit_variance_per_condition

logger = setup_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GGMResult:
    """Result of fitting a single GGM to one condition."""
    condition: int
    precision_matrix: np.ndarray          # (D, D) — Theta_c
    covariance_matrix: np.ndarray         # (D, D) — Sigma_c
    alpha: float                          # regularization strength used
    n_trials: int
    feature_names: List[str]
    modality_blocks: Dict[str, Tuple[int, int]]

    @property
    def n_features(self) -> int:
        return self.precision_matrix.shape[0]

    def cross_block_precision(self, mod_a: str, mod_b: str) -> np.ndarray:
        """Return off-diagonal block Theta[mod_a, mod_b]."""
        sa, ea = self.modality_blocks[mod_a]
        sb, eb = self.modality_blocks[mod_b]
        return self.precision_matrix[sa:ea, sb:eb]

    def diagonal_block_precision(self, mod: str) -> np.ndarray:
        """Return within-modality diagonal block Theta[mod, mod]."""
        s, e = self.modality_blocks[mod]
        return self.precision_matrix[s:e, s:e]


@dataclass
class StabilityResult:
    """Edge stability scores from Meinshausen-Bühlmann subsampling."""
    stability_scores: np.ndarray          # (D, D) — proportion of subsamples with edge
    stable_edges: np.ndarray              # (D, D) bool — edges above pi_threshold
    feature_names: List[str]
    modality_blocks: Dict[str, Tuple[int, int]]
    pi_threshold: float
    n_subsamples: int
    alpha_used: float


@dataclass
class ConditionGGMs:
    """Collection of GGMs for all conditions + resting state."""
    condition_ggms: Dict[int, GGMResult]           # condition label → GGMResult
    stability_results: Dict[int, StabilityResult]  # condition label → stability
    resting_ggms: Optional[Dict[str, GGMResult]] = None  # subject_id → resting GGM
    feature_names: List[str] = field(default_factory=list)
    modality_blocks: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        save_cache(self, path)
        logger.info(f"ConditionGGMs saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "ConditionGGMs":
        obj = load_cache(path)
        logger.info(f"ConditionGGMs loaded: {path}")
        return obj


# ── GGM fitting ────────────────────────────────────────────────────────────────

class ConditionGGMFitter:
    """
    Fit condition-specific Gaussian Graphical Models.

    For each condition, pools all trials across subjects,
    applies per-condition unit-variance scaling, then fits
    GraphicalLassoCV to recover precision matrix.

    Parameters
    ----------
    n_cv_folds : int
        Cross-validation folds for alpha selection.
    max_iter : int
        Max iterations for GraphicalLasso.
    n_alphas : int
        Number of alpha values in CV grid.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        n_cv_folds: int = 5,
        max_iter: int = 1000,
        n_alphas: int = 10,
        tol: float = 1e-4,
    ):
        self.n_cv_folds = n_cv_folds
        self.max_iter = max_iter
        self.n_alphas = n_alphas
        self.tol = tol

    def fit_condition(
        self,
        X_cond: np.ndarray,
        condition: int,
        feature_names: List[str],
        modality_blocks: Dict[str, Tuple[int, int]],
    ) -> GGMResult:
        """
        Fit GGM for a single condition.

        Parameters
        ----------
        X_cond : ndarray, shape (n_trials_cond, n_features)
            Feature matrix for this condition (already per-subject z-scored).
        condition : int
            Condition label.
        feature_names : list of str
        modality_blocks : dict

        Returns
        -------
        GGMResult
        """
        n_trials, n_features = X_cond.shape
        logger.info(f"Fitting GGM: condition={condition}, n_trials={n_trials}, n_features={n_features}")

        # Additional unit-variance scaling per feature (prevents coupling artifacts
        # from variance differences across modalities)
        scaler = StandardScaler(with_mean=False, with_std=True)
        X_scaled = scaler.fit_transform(X_cond)

        # Check for near-constant features
        variances = np.var(X_scaled, axis=0)
        bad_feats = np.where(variances < 1e-8)[0]
        if len(bad_feats) > 0:
            logger.warning(f"Condition {condition}: {len(bad_feats)} near-constant features — adding jitter")
            X_scaled[:, bad_feats] += np.random.RandomState(42).randn(n_trials, len(bad_feats)) * 1e-6

        # Fit GraphicalLassoCV
        try:
            glcv = GraphicalLassoCV(
                cv=min(self.n_cv_folds, n_trials),
                n_alphas=self.n_alphas,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glcv.fit(X_scaled)

            precision = glcv.precision_
            covariance = glcv.covariance_
            alpha_used = glcv.alpha_

        except Exception as e:
            logger.warning(f"GraphicalLassoCV failed (condition={condition}): {e}. Falling back to fixed alpha=0.1")
            gl = GraphicalLasso(alpha=0.1, max_iter=self.max_iter, tol=self.tol)
            gl.fit(X_scaled)
            precision = gl.precision_
            covariance = gl.covariance_
            alpha_used = 0.1

        logger.info(
            f"Condition {condition}: alpha={alpha_used:.4f}, "
            f"n_nonzero_offdiag={_count_nonzero_offdiag(precision)}"
        )

        return GGMResult(
            condition=condition,
            precision_matrix=precision,
            covariance_matrix=covariance,
            alpha=alpha_used,
            n_trials=n_trials,
            feature_names=feature_names,
            modality_blocks=modality_blocks,
        )

    def fit_all_conditions(
        self,
        X: np.ndarray,
        condition_ids: np.ndarray,
        feature_names: List[str],
        modality_blocks: Dict[str, Tuple[int, int]],
    ) -> Dict[int, GGMResult]:
        """
        Fit one GGM per condition.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features)
            Per-subject z-scored feature matrix.
        condition_ids : ndarray, shape (n_trials,)
        feature_names : list of str
        modality_blocks : dict

        Returns
        -------
        dict {condition_int → GGMResult}
        """
        results = {}
        for cond in np.unique(condition_ids):
            mask = condition_ids == cond
            X_cond = X[mask]
            results[int(cond)] = self.fit_condition(
                X_cond, int(cond), feature_names, modality_blocks
            )
        return results


# ── Stability selection ────────────────────────────────────────────────────────

class StabilitySelector:
    """
    Meinshausen-Bühlmann (2010) stability selection for GGM edges.

    Repeatedly subsample 50% of trials, fit GraphicalLasso at fixed alpha,
    record which edges appear. Edge stability = fraction of subsamples
    where edge is non-zero.

    Edges with stability > pi_threshold are declared "stable."

    Parameters
    ----------
    alpha : float
        GraphicalLasso regularization (use CV-selected alpha from ConditionGGMFitter).
    n_subsamples : int
        Number of subsampling iterations.
    subsample_size : float
        Fraction of samples per subsample.
    pi_threshold : float
        Stability threshold for edge inclusion (Meinshausen-Bühlmann: 0.6).
    random_state : int
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_subsamples: int = 100,
        subsample_size: float = 0.5,
        pi_threshold: float = 0.6,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.pi_threshold = pi_threshold
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        modality_blocks: Dict[str, Tuple[int, int]],
    ) -> StabilityResult:
        """
        Run stability selection on feature matrix X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features)
            Already scaled feature matrix for one condition.

        Returns
        -------
        StabilityResult
        """
        n_trials, n_features = X.shape
        n_sub = max(2, int(self.subsample_size * n_trials))

        edge_counts = np.zeros((n_features, n_features))

        logger.info(f"Stability selection: {self.n_subsamples} subsamples, alpha={self.alpha:.4f}")

        for i in range(self.n_subsamples):
            idx = self.rng.choice(n_trials, size=n_sub, replace=False)
            X_sub = X[idx]

            # Unit scale within subsample
            std = X_sub.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            X_sub = X_sub / std

            try:
                gl = GraphicalLasso(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=1e-3,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gl.fit(X_sub)
                # Edge present if |precision_ij| > 0 (GLASSO makes some exactly 0)
                edge_counts += (np.abs(gl.precision_) > 1e-10).astype(float)
            except Exception:
                # Skip failed subsample
                continue

        stability_scores = edge_counts / self.n_subsamples
        stable_edges = stability_scores >= self.pi_threshold
        # Remove diagonal
        np.fill_diagonal(stable_edges, False)

        n_stable = stable_edges.sum() // 2  # symmetric
        logger.info(f"Stability selection: {n_stable} stable edges (pi>={self.pi_threshold})")

        return StabilityResult(
            stability_scores=stability_scores,
            stable_edges=stable_edges,
            feature_names=feature_names,
            modality_blocks=modality_blocks,
            pi_threshold=self.pi_threshold,
            n_subsamples=self.n_subsamples,
            alpha_used=self.alpha,
        )


# ── Per-subject resting GGM ────────────────────────────────────────────────────

def fit_resting_ggm(
    X_rest: np.ndarray,
    subject_id: str,
    feature_names: List[str],
    modality_blocks: Dict[str, Tuple[int, int]],
    alpha: Optional[float] = None,
    n_cv_folds: int = 5,
) -> GGMResult:
    """
    Fit per-subject resting-state GGM.

    Used to compute ΔΘ_c^(s) = Θ_c - Θ_rest^(s),
    isolating load-induced coupling from trait-level coupling.

    Parameters
    ----------
    X_rest : ndarray, shape (n_rest_trials, n_features)
        Resting state feature vectors (e.g., sliding window features
        from 4-min resting recording).
    subject_id : str
    feature_names, modality_blocks : as in GGMResult
    alpha : float or None
        If None, use CV to select.

    Returns
    -------
    GGMResult with condition=-1 (resting)
    """
    n_trials, n_features = X_rest.shape
    logger.info(f"Fitting resting GGM for {subject_id}: n_windows={n_trials}, n_features={n_features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rest)

    if alpha is None:
        try:
            glcv = GraphicalLassoCV(
                cv=min(n_cv_folds, n_trials),
                n_alphas=8,
                max_iter=500,
                tol=1e-4,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glcv.fit(X_scaled)
            precision = glcv.precision_
            covariance = glcv.covariance_
            alpha_used = glcv.alpha_
        except Exception as e:
            logger.warning(f"Resting GGM CV failed for {subject_id}: {e}. Using alpha=0.1")
            gl = GraphicalLasso(alpha=0.1, max_iter=500)
            gl.fit(X_scaled)
            precision = gl.precision_
            covariance = gl.covariance_
            alpha_used = 0.1
    else:
        gl = GraphicalLasso(alpha=alpha, max_iter=500)
        gl.fit(X_scaled)
        precision = gl.precision_
        covariance = gl.covariance_
        alpha_used = alpha

    return GGMResult(
        condition=-1,  # resting sentinel
        precision_matrix=precision,
        covariance_matrix=covariance,
        alpha=alpha_used,
        n_trials=n_trials,
        feature_names=feature_names,
        modality_blocks=modality_blocks,
    )


# ── Delta precision ────────────────────────────────────────────────────────────

def compute_delta_precision(
    task_ggm: GGMResult,
    resting_ggm: GGMResult,
) -> np.ndarray:
    """
    ΔΘ_c^(s) = Θ_task_c − Θ_rest^(s)

    Isolates load-induced coupling changes from stable trait-level structure.
    Defined in strategy Section 6.4.

    Parameters
    ----------
    task_ggm : GGMResult
        Task-condition GGM.
    resting_ggm : GGMResult
        Subject resting-state GGM (condition=-1).

    Returns
    -------
    delta : ndarray, shape (D, D)
    """
    assert task_ggm.n_features == resting_ggm.n_features, \
        "Task and resting GGMs must have same feature dimension"
    return task_ggm.precision_matrix - resting_ggm.precision_matrix


# ── Utility functions ──────────────────────────────────────────────────────────

def _count_nonzero_offdiag(precision: np.ndarray) -> int:
    """Count non-zero off-diagonal entries (upper triangle only)."""
    D = precision.shape[0]
    mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    return int((np.abs(precision[mask]) > 1e-10).sum())


def cross_modal_edge_density(
    precision: np.ndarray,
    modality_blocks: Dict[str, Tuple[int, int]],
    mod_a: str,
    mod_b: str,
) -> float:
    """
    Fraction of non-zero entries in off-diagonal block Theta[mod_a, mod_b].

    Primary topological measure: increases control→9-digit, decreases at 13-digit
    under the coupling hypothesis.

    Parameters
    ----------
    precision : ndarray, shape (D, D)
    modality_blocks : dict
    mod_a, mod_b : str

    Returns
    -------
    density : float in [0, 1]
    """
    sa, ea = modality_blocks[mod_a]
    sb, eb = modality_blocks[mod_b]
    block = precision[sa:ea, sb:eb]
    n_entries = block.size
    if n_entries == 0:
        return 0.0
    n_nonzero = (np.abs(block) > 1e-10).sum()
    return float(n_nonzero) / n_entries


def frobenius_cross_modal_strength(
    precision: np.ndarray,
    modality_blocks: Dict[str, Tuple[int, int]],
    mod_a: str,
    mod_b: str,
) -> float:
    """
    Frobenius norm of cross-modal block — scalar coupling strength measure.

    Used as a trial-level coupling feature for recall prediction.
    """
    sa, ea = modality_blocks[mod_a]
    sb, eb = modality_blocks[mod_b]
    block = precision[sa:ea, sb:eb]
    return float(np.linalg.norm(block, "fro"))


def network_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """
    ||Theta_a - Theta_b||_F — network distance between two precision matrices.

    Prediction: distance(9-digit, 13-digit) > distance(5-digit, 9-digit).
    """
    return float(np.linalg.norm(theta_a - theta_b, "fro"))


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full_ggm_pipeline(
    X: np.ndarray,
    condition_ids: np.ndarray,
    feature_names: List[str],
    modality_blocks: Dict[str, Tuple[int, int]],
    ggm_config: dict,
) -> ConditionGGMs:
    """
    Full GGM pipeline: fit condition GGMs + stability selection.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
        Per-subject z-scored feature matrix.
    condition_ids : ndarray
    feature_names : list of str
    modality_blocks : dict
    ggm_config : dict
        From config.yaml['ggm'].

    Returns
    -------
    ConditionGGMs
    """
    fitter = ConditionGGMFitter(
        n_cv_folds=ggm_config.get("graphical_lasso", {}).get("cv", 5),
        max_iter=ggm_config.get("graphical_lasso", {}).get("max_iter", 1000),
        n_alphas=ggm_config.get("graphical_lasso", {}).get("alphas", 10),
        tol=ggm_config.get("graphical_lasso", {}).get("tol", 1e-4),
    )

    ss_cfg = ggm_config.get("stability_selection", {})
    n_subsamples = ss_cfg.get("n_subsamples", 100)
    subsample_size = ss_cfg.get("subsample_size", 0.5)
    pi_threshold = ss_cfg.get("pi_threshold", 0.6)

    # Fit condition GGMs
    logger.info("=== Fitting condition-specific GGMs ===")
    condition_ggms = fitter.fit_all_conditions(X, condition_ids, feature_names, modality_blocks)

    # Stability selection per condition
    logger.info("=== Running stability selection ===")
    stability_results = {}
    for cond, ggm_result in condition_ggms.items():
        mask = condition_ids == cond
        X_cond = X[mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cond)

        selector = StabilitySelector(
            alpha=ggm_result.alpha,
            n_subsamples=n_subsamples,
            subsample_size=subsample_size,
            pi_threshold=pi_threshold,
        )
        stability_results[cond] = selector.fit(X_scaled, feature_names, modality_blocks)

    return ConditionGGMs(
        condition_ggms=condition_ggms,
        stability_results=stability_results,
        feature_names=feature_names,
        modality_blocks=modality_blocks,
    )