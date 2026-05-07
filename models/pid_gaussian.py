"""
models/pid_gaussian.py
=======================

Partial Information Decomposition (PID) for Gaussian variables.

For Gaussian X1, X2, Y the PID quantities (redundancy, unique, synergy)
have exact closed-form expressions via covariance algebra.
No nonparametric entropy estimation required — sample-efficient and exact.

Implements Williams & Beer (2010) framework with Gaussian-specific
solutions from Barrett (2015) and Bertschinger et al. (2014).

Key prediction (from research strategy):
  Low load (5-digit):   high redundancy (EEG-pupil both LC-NE driven), low synergy
  Medium load (9-digit): synergy peaks (complementary systems recruited)
  Overload (13-digit):   synergy collapses (modalities decouple)

Dependencies: numpy, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import det, inv, block_diag

from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Information-theoretic primitives (Gaussian) ───────────────────────────────

def gaussian_entropy(cov: np.ndarray) -> float:
    """
    Differential entropy of multivariate Gaussian: h(X) = 0.5 * log|2πe Σ|.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Covariance matrix.

    Returns
    -------
    h : float  (nats)
    """
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        warnings.warn("Non-positive definite covariance in entropy computation", UserWarning)
        return np.nan
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + logdet)


def gaussian_mutual_information(
    cov_joint: np.ndarray,
    dim_x: int,
    dim_y: int,
) -> float:
    """
    Mutual information I(X; Y) for joint Gaussian with block covariance.

    I(X; Y) = h(X) + h(Y) - h(X, Y)

    Parameters
    ----------
    cov_joint : ndarray, shape (dim_x + dim_y, dim_x + dim_y)
        Joint covariance [[Σ_XX, Σ_XY], [Σ_YX, Σ_YY]].
    dim_x, dim_y : int
        Dimensions of X and Y blocks.

    Returns
    -------
    mi : float (nats)
    """
    cov_x = cov_joint[:dim_x, :dim_x]
    cov_y = cov_joint[dim_x:, dim_x:]
    h_x = gaussian_entropy(cov_x)
    h_y = gaussian_entropy(cov_y)
    h_xy = gaussian_entropy(cov_joint)
    if np.any(np.isnan([h_x, h_y, h_xy])):
        return np.nan
    return max(0.0, h_x + h_y - h_xy)


def gaussian_conditional_mi(
    cov_xyz: np.ndarray,
    dim_x: int,
    dim_y: int,
    dim_z: int,
) -> float:
    """
    Conditional mutual information I(X; Y | Z) for joint Gaussian.

    I(X; Y | Z) = h(X | Z) + h(Y | Z) - h(X, Y | Z)
                = I(X, Y, Z) decomposed via Schur complement.

    Parameters
    ----------
    cov_xyz : ndarray, shape (dx+dy+dz, dx+dy+dz)
        Joint covariance. Layout: [X | Y | Z].
    dim_x, dim_y, dim_z : int

    Returns
    -------
    cmi : float (nats)
    """
    dx, dy, dz = dim_x, dim_y, dim_z

    # Submatrices
    Σ_xz = cov_xyz[:dx, dx + dy:]
    Σ_yz = cov_xyz[dx:dx + dy, dx + dy:]
    Σ_z  = cov_xyz[dx + dy:, dx + dy:]
    Σ_xyz_joint = cov_xyz[:dx + dy, :dx + dy]
    Σ_xz_joint = cov_xyz[np.ix_(list(range(dx)) + list(range(dx + dy, dx + dy + dz)),
                                 list(range(dx)) + list(range(dx + dy, dx + dy + dz)))]
    Σ_yz_joint = cov_xyz[dx:, dx:]

    try:
        Σ_z_inv = np.linalg.inv(Σ_z)
    except np.linalg.LinAlgError:
        return np.nan

    # Conditional covariance of [X, Y] | Z via Schur complement
    Σ_xy_z = Σ_xyz_joint - np.block([
        [Σ_xz @ Σ_z_inv @ Σ_xz.T,  Σ_xz @ Σ_z_inv @ Σ_yz.T],
        [Σ_yz @ Σ_z_inv @ Σ_xz.T,  Σ_yz @ Σ_z_inv @ Σ_yz.T],
    ])

    # Conditional covariance of X | Z
    Σ_x_z = cov_xyz[:dx, :dx] - Σ_xz @ Σ_z_inv @ Σ_xz.T

    # Conditional covariance of Y | Z
    Σ_y_z = cov_xyz[dx:dx + dy, dx:dx + dy] - Σ_yz @ Σ_z_inv @ Σ_yz.T

    h_xy_z = gaussian_entropy(Σ_xy_z)
    h_x_z = gaussian_entropy(Σ_x_z)
    h_y_z = gaussian_entropy(Σ_y_z)

    if np.any(np.isnan([h_xy_z, h_x_z, h_y_z])):
        return np.nan

    return max(0.0, h_x_z + h_y_z - h_xy_z)


# ── Minimum Mutual Information (MMI) redundancy ───────────────────────────────

def mmi_redundancy_gaussian(
    cov_x1y: np.ndarray,
    cov_x2y: np.ndarray,
    dim_x1: int,
    dim_x2: int,
    dim_y: int,
) -> float:
    """
    Minimum Mutual Information (MMI) redundancy for Gaussian variables.

    Rmin(X1, X2; Y) = min(I(X1; Y), I(X2; Y))

    This is the simplest, most robust measure of shared information.
    Under the LC-NE hypothesis: redundancy between EEG and pupil should
    be high at low load (both driven by same LC-NE state).

    Parameters
    ----------
    cov_x1y : ndarray, shape (dim_x1 + dim_y, dim_x1 + dim_y)
        Joint covariance of X1 and Y.
    cov_x2y : ndarray, shape (dim_x2 + dim_y, dim_x2 + dim_y)
        Joint covariance of X2 and Y.

    Returns
    -------
    redundancy : float (nats)
    """
    mi_x1y = gaussian_mutual_information(cov_x1y, dim_x1, dim_y)
    mi_x2y = gaussian_mutual_information(cov_x2y, dim_x2, dim_y)
    return min(mi_x1y, mi_x2y)


# ── Full PID decomposition ─────────────────────────────────────────────────────

@dataclass
class PIDResult:
    """PID decomposition for one modality pair under one condition."""
    modality_a: str
    modality_b: str
    condition: int

    mutual_info_a: float       # I(X_a; Y)
    mutual_info_b: float       # I(X_b; Y)
    mutual_info_ab: float      # I(X_a, X_b; Y)

    redundancy: float          # Rmin(X_a, X_b; Y) — MMI approximation
    unique_a: float            # Unique info from X_a
    unique_b: float            # Unique info from X_b
    synergy: float             # Information only from joint (X_a, X_b)

    # Sanity check: should be ≈ mutual_info_ab
    pid_total: float

    def to_dict(self) -> dict:
        return {
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "condition": self.condition,
            "mi_a": self.mutual_info_a,
            "mi_b": self.mutual_info_b,
            "mi_ab": self.mutual_info_ab,
            "redundancy": self.redundancy,
            "unique_a": self.unique_a,
            "unique_b": self.unique_b,
            "synergy": self.synergy,
            "pid_total": self.pid_total,
        }


def compute_pid_gaussian(
    X_a: np.ndarray,
    X_b: np.ndarray,
    Y: np.ndarray,
    modality_a: str,
    modality_b: str,
    condition: int,
    n_pcs: int = 3,
) -> PIDResult:
    """
    Compute Gaussian PID for modality pair (X_a, X_b) predicting Y.

    Reduces each modality to n_pcs principal components for:
    - Improved Gaussianity (CLT)
    - Sample-adequate covariance estimation
    - Reduced dimensionality for reliable precision matrix inversion

    Parameters
    ----------
    X_a : ndarray, shape (n_trials, d_a)
    X_b : ndarray, shape (n_trials, d_b)
    Y   : ndarray, shape (n_trials,) or (n_trials, d_y)
        Target variable (condition label or recall accuracy).
    modality_a, modality_b : str
    condition : int
    n_pcs : int
        Number of PCs per modality for PID.

    Returns
    -------
    PIDResult
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_trials = X_a.shape[0]

    if n_trials < 10:
        logger.warning(f"Too few trials ({n_trials}) for PID. Returning NaN result.")
        return _nan_pid_result(modality_a, modality_b, condition)

    # Reduce to n_pcs per modality
    def reduce(X: np.ndarray, k: int) -> np.ndarray:
        k = min(k, X.shape[1], X.shape[0] - 1)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        pca = PCA(n_components=k, random_state=42)
        return pca.fit_transform(X_s)

    Xa_r = reduce(X_a, n_pcs)  # (n, n_pcs)
    Xb_r = reduce(X_b, n_pcs)  # (n, n_pcs)

    # Y: ensure 2D
    if Y.ndim == 1:
        Y_r = Y.reshape(-1, 1)
    else:
        Y_r = Y

    da, db, dy = Xa_r.shape[1], Xb_r.shape[1], Y_r.shape[1]

    # Joint arrays
    Xab = np.hstack([Xa_r, Xb_r])    # (n, da+db)
    Xay = np.hstack([Xa_r, Y_r])     # (n, da+dy)
    Xby = np.hstack([Xb_r, Y_r])     # (n, db+dy)
    Xaby = np.hstack([Xa_r, Xb_r, Y_r])  # (n, da+db+dy)

    # Empirical covariances with shrinkage for stability
    cov_ay  = _empirical_cov(Xay)
    cov_by  = _empirical_cov(Xby)
    cov_aby = _empirical_cov(Xaby)

    # Mutual informations
    mi_a  = gaussian_mutual_information(cov_ay, da, dy)
    mi_b  = gaussian_mutual_information(cov_by, db, dy)
    mi_ab = gaussian_mutual_information(cov_aby, da + db, dy)

    # Redundancy via MMI
    redundancy = mmi_redundancy_gaussian(cov_ay, cov_by, da, db, dy)

    # Unique and synergy
    unique_a = max(0.0, mi_a - redundancy)
    unique_b = max(0.0, mi_b - redundancy)
    synergy  = max(0.0, mi_ab - mi_a - mi_b + redundancy)

    pid_total = redundancy + unique_a + unique_b + synergy

    logger.debug(
        f"PID({modality_a},{modality_b}|cond={condition}): "
        f"R={redundancy:.4f}, U_a={unique_a:.4f}, U_b={unique_b:.4f}, S={synergy:.4f}"
    )

    return PIDResult(
        modality_a=modality_a,
        modality_b=modality_b,
        condition=condition,
        mutual_info_a=float(mi_a),
        mutual_info_b=float(mi_b),
        mutual_info_ab=float(mi_ab),
        redundancy=float(redundancy),
        unique_a=float(unique_a),
        unique_b=float(unique_b),
        synergy=float(synergy),
        pid_total=float(pid_total),
    )


def _empirical_cov(X: np.ndarray, shrinkage: float = 1e-4) -> np.ndarray:
    """
    Empirical covariance with Tikhonov regularization for numerical stability.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    shrinkage : float
        Ridge term added to diagonal.

    Returns
    -------
    cov : ndarray, shape (d, d)
    """
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += shrinkage * np.eye(cov.shape[0])
    return cov


def _nan_pid_result(mod_a: str, mod_b: str, condition: int) -> PIDResult:
    return PIDResult(
        modality_a=mod_a, modality_b=mod_b, condition=condition,
        mutual_info_a=np.nan, mutual_info_b=np.nan, mutual_info_ab=np.nan,
        redundancy=np.nan, unique_a=np.nan, unique_b=np.nan,
        synergy=np.nan, pid_total=np.nan,
    )


# ── Full PID analysis across conditions and pairs ─────────────────────────────

def run_pid_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    condition_ids: np.ndarray,
    modality_blocks: Dict[str, Tuple[int, int]],
    modality_pairs: List[Tuple[str, str]],
    n_pcs: int = 3,
) -> Dict[Tuple[str, str, int], PIDResult]:
    """
    Compute PID for all modality pairs under all conditions.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    Y : ndarray, shape (n_trials,) — target (recall accuracy or condition label)
    condition_ids : ndarray, shape (n_trials,)
    modality_blocks : dict {modality → (start, end)}
    modality_pairs : list of (mod_a, mod_b)
    n_pcs : int

    Returns
    -------
    results : dict {(mod_a, mod_b, condition) → PIDResult}
    """
    results = {}

    for cond in np.unique(condition_ids):
        mask = condition_ids == cond
        X_c = X[mask]
        Y_c = Y[mask]

        for mod_a, mod_b in modality_pairs:
            sa, ea = modality_blocks[mod_a]
            sb, eb = modality_blocks[mod_b]
            X_a = X_c[:, sa:ea]
            X_b = X_c[:, sb:eb]

            result = compute_pid_gaussian(
                X_a, X_b, Y_c,
                modality_a=mod_a,
                modality_b=mod_b,
                condition=int(cond),
                n_pcs=n_pcs,
            )
            results[(mod_a, mod_b, int(cond))] = result

    return results


# ── Bootstrap validation ───────────────────────────────────────────────────────

def bootstrap_pid(
    X_a: np.ndarray,
    X_b: np.ndarray,
    Y: np.ndarray,
    modality_a: str,
    modality_b: str,
    condition: int,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    n_pcs: int = 3,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for PID quantities.

    Tests non-monotonic synergy hypothesis:
    synergy(9-digit) > synergy(13-digit)

    Returns
    -------
    dict {quantity → (ci_lower, ci_upper)}
    """
    rng = np.random.RandomState(random_state)
    n_trials = X_a.shape[0]

    quantities = ["redundancy", "unique_a", "unique_b", "synergy"]
    bootstrap_vals = {q: [] for q in quantities}

    for _ in range(n_bootstrap):
        idx = rng.choice(n_trials, size=n_trials, replace=True)
        result = compute_pid_gaussian(
            X_a[idx], X_b[idx], Y[idx],
            modality_a=modality_a, modality_b=modality_b,
            condition=condition, n_pcs=n_pcs,
        )
        for q in quantities:
            bootstrap_vals[q].append(getattr(result, q))

    alpha = 1.0 - ci_level
    cis = {}
    for q in quantities:
        vals = np.array(bootstrap_vals[q])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            cis[q] = (
                float(np.percentile(vals, 100 * alpha / 2)),
                float(np.percentile(vals, 100 * (1 - alpha / 2))),
            )
        else:
            cis[q] = (np.nan, np.nan)

    return cis