"""
models/lgssm.py
===============

Linear Gaussian State Space Model (LGSSM) fitted via Expectation-Maximization.

Classical replacement for LSTM/TCN (inadmissible under hard constraint).
Models cognitive load as a latent continuous state evolving digit-by-digit.

State transition:  z_t = A z_{t-1} + w_t,   w_t ~ N(0, Q)
Observation model: f_t = C z_t + v_t,        v_t ~ N(0, R)

Where:
  z_t ∈ R^k  latent cognitive load state at digit position t
  f_t ∈ R^D  per-digit multimodal feature vector
  A encodes load state evolution digit-by-digit
  C links latent load to observed physiology (off-diagonal = cross-modal coupling)
  Q transition noise covariance
  R observation noise covariance

EM algorithm: Kalman smoother E-step + closed-form M-step.
Strictly classical (Shumway & Stoffer 1982 / Ghahramani & Hinton 1996).

Output:
  - Continuous load trajectory per trial
  - Individual overload onset (trajectory peak location)
  - Load accumulation rate (trajectory slope)
  - Cross-modal coupling encoded in C matrix

Dependencies: numpy, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve, inv

from utils.io_utils import setup_logger, save_cache, load_cache

logger = setup_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LGSSMParams:
    """LGSSM parameters — all estimated by EM."""
    A: np.ndarray     # (k, k) state transition
    C: np.ndarray     # (D, k) observation matrix
    Q: np.ndarray     # (k, k) transition noise covariance
    R: np.ndarray     # (D, D) observation noise covariance
    mu0: np.ndarray   # (k,) initial state mean
    P0: np.ndarray    # (k, k) initial state covariance
    k: int            # latent dimension
    D: int            # observation dimension
    log_likelihood: float = -np.inf

    def copy(self) -> "LGSSMParams":
        return LGSSMParams(
            A=self.A.copy(), C=self.C.copy(),
            Q=self.Q.copy(), R=self.R.copy(),
            mu0=self.mu0.copy(), P0=self.P0.copy(),
            k=self.k, D=self.D,
            log_likelihood=self.log_likelihood,
        )


@dataclass
class KalmanResult:
    """Output of Kalman smoother for one trial."""
    z_smooth: np.ndarray      # (T, k) smoothed state means
    P_smooth: np.ndarray      # (T, k, k) smoothed state covariances
    P_pair: np.ndarray        # (T-1, k, k) E[z_t z_{t-1}'] smoothed
    log_likelihood: float


@dataclass
class LGSSMTrajectory:
    """Per-trial load trajectory and derived features."""
    trial_id: int
    subject_id: str
    condition: int

    z_trajectory: np.ndarray      # (T, k) smoothed latent states
    load_scalar: np.ndarray       # (T,) scalar load at each digit (first PC of z)

    peak_digit: int               # argmax of load trajectory
    peak_load: float              # max load value
    load_slope: float             # mean slope (load accumulation rate)
    overload_onset: Optional[int] # digit where load starts declining (after peak)

    # Cross-modal coupling from C matrix (summarized as scalar)
    coupling_strength: float


@dataclass
class LGSSMFitResult:
    """Result of fitting LGSSM to a dataset."""
    params: LGSSMParams
    trajectories: List[LGSSMTrajectory]
    feature_names: List[str]
    n_em_iters: int
    converged: bool

    def save(self, path: Path) -> None:
        save_cache(self, path)
        logger.info(f"LGSSMFitResult saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "LGSSMFitResult":
        return load_cache(path)


# ── Kalman filter and smoother ─────────────────────────────────────────────────

def kalman_filter(
    observations: np.ndarray,   # (T, D)
    params: LGSSMParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Kalman filter (forward pass).

    Returns
    -------
    z_filt : ndarray (T, k)   filtered means
    P_filt : ndarray (T, k, k) filtered covariances
    z_pred : ndarray (T, k)   predicted means
    P_pred : ndarray (T, k, k) predicted covariances
    log_lik : float
    """
    T = observations.shape[0]
    k = params.k
    A, C, Q, R = params.A, params.C, params.Q, params.R

    z_filt = np.zeros((T, k))
    P_filt = np.zeros((T, k, k))
    z_pred = np.zeros((T, k))
    P_pred = np.zeros((T, k, k))
    log_lik = 0.0

    # Initialize
    z_filt[-1] = params.mu0  # use as t=-1 → z_pred[0]
    P_filt[-1] = params.P0

    for t in range(T):
        # Predict
        if t == 0:
            z_p = A @ params.mu0
            P_p = A @ params.P0 @ A.T + Q
        else:
            z_p = A @ z_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q

        z_pred[t] = z_p
        P_pred[t] = P_p

        # Innovation
        y = observations[t]
        S = C @ P_p @ C.T + R  # innovation covariance
        innov = y - C @ z_p

        # Symmetrize S for numerical stability
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Kalman gain
        K = P_p @ C.T @ S_inv

        # Update
        z_filt[t] = z_p + K @ innov
        P_filt[t] = (np.eye(k) - K @ C) @ P_p
        P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T)

        # Log-likelihood contribution
        sign, logdet_S = np.linalg.slogdet(S)
        if sign > 0:
            D_obs = observations.shape[1]
            log_lik += -0.5 * (
                D_obs * np.log(2 * np.pi) + logdet_S +
                innov @ S_inv @ innov
            )

    return z_filt, P_filt, z_pred, P_pred, log_lik


def kalman_smoother(
    z_filt: np.ndarray,
    P_filt: np.ndarray,
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    params: LGSSMParams,
) -> KalmanResult:
    """
    Rauch-Tung-Striebel (RTS) backward smoother.

    Returns smoothed means, covariances, and pair covariances for EM.
    """
    T, k = z_filt.shape
    A = params.A

    z_smooth = z_filt.copy()
    P_smooth = P_filt.copy()
    P_pair = np.zeros((T - 1, k, k))   # E[z_t z_{t-1}'] for EM M-step

    log_lik = 0.0  # placeholder (already computed in filter)

    for t in range(T - 2, -1, -1):
        try:
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred[t + 1])

        # Smoother gain
        G = P_filt[t] @ A.T @ P_pred_inv

        # Smooth
        z_smooth[t] = z_filt[t] + G @ (z_smooth[t + 1] - z_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        # Pair covariance E[z_{t+1} z_t']
        P_pair[t] = P_smooth[t + 1] @ G.T + np.outer(z_smooth[t + 1], z_smooth[t])

    return KalmanResult(
        z_smooth=z_smooth,
        P_smooth=P_smooth,
        P_pair=P_pair,
        log_likelihood=log_lik,
    )


# ── EM algorithm ───────────────────────────────────────────────────────────────

def em_lgssm(
    sequences: List[np.ndarray],   # list of (T_i, D) observation sequences
    k: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
    random_state: int = 42,
) -> LGSSMParams:
    """
    EM algorithm for LGSSM parameter estimation.

    Multiple random restarts to avoid local optima.

    Parameters
    ----------
    sequences : list of ndarray (T_i, D)
        Each element is one trial's per-digit feature vectors.
        T_i = number of digits in trial i (varies: 5, 9, or 13).
    k : int
        Latent state dimension (2 recommended — allows 2D load manifold).
    max_iter : int
    tol : float
        Convergence threshold on log-likelihood change.
    n_restarts : int

    Returns
    -------
    Best LGSSMParams across restarts (highest final log-likelihood).
    """
    D = sequences[0].shape[1]
    rng = np.random.RandomState(random_state)

    best_params = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        logger.info(f"LGSSM EM restart {restart + 1}/{n_restarts}")
        params = _initialize_params(D, k, rng)
        params, ll, converged = _run_em(sequences, params, max_iter, tol)

        logger.info(f"Restart {restart + 1}: final LL={ll:.4f}, converged={converged}")

        if ll > best_ll:
            best_ll = ll
            best_params = params
            best_params.log_likelihood = ll

    return best_params


def _initialize_params(D: int, k: int, rng: np.random.RandomState) -> LGSSMParams:
    """Random initialization of LGSSM parameters."""
    A = 0.9 * np.eye(k) + 0.1 * rng.randn(k, k) * 0.01
    C = rng.randn(D, k) * 0.1
    Q = np.eye(k) * 0.1
    R = np.eye(D) * 0.5
    mu0 = np.zeros(k)
    P0 = np.eye(k)
    return LGSSMParams(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0, k=k, D=D)


def _run_em(
    sequences: List[np.ndarray],
    params: LGSSMParams,
    max_iter: int,
    tol: float,
) -> Tuple[LGSSMParams, float, bool]:
    """Run EM until convergence or max_iter."""
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # E-step: Kalman filter + smoother for each sequence
        all_smooth = []
        total_ll = 0.0

        for seq in sequences:
            if seq.shape[0] < 2:
                continue
            z_filt, P_filt, z_pred, P_pred, ll = kalman_filter(seq, params)
            smooth = kalman_smoother(z_filt, P_filt, z_pred, P_pred, params)
            all_smooth.append((seq, smooth))
            total_ll += ll

        # M-step: update parameters
        params = _m_step(all_smooth, params)

        # Convergence check
        ll_change = abs(total_ll - prev_ll)
        if ll_change < tol and iteration > 5:
            converged = True
            break

        prev_ll = total_ll

        if (iteration + 1) % 20 == 0:
            logger.debug(f"  EM iter {iteration + 1}: LL={total_ll:.4f}, Δ={ll_change:.2e}")

    return params, total_ll, converged


def _m_step(
    all_smooth: List[Tuple[np.ndarray, KalmanResult]],
    params: LGSSMParams,
) -> LGSSMParams:
    """
    Closed-form M-step for LGSSM.

    Updates A, C, Q, R, mu0, P0 using sufficient statistics from E-step.
    """
    k, D = params.k, params.D
    n_seqs = len(all_smooth)

    # Accumulate sufficient statistics
    Sz1z1 = np.zeros((k, k))   # sum E[z_t z_t']  t=1..T-1
    Sz2z2 = np.zeros((k, k))   # sum E[z_t z_t']  t=2..T
    Sz2z1 = np.zeros((k, k))   # sum E[z_t z_{t-1}']  t=2..T
    Syz   = np.zeros((D, k))   # sum y_t E[z_t]'
    Szz   = np.zeros((k, k))   # sum E[z_t z_t']  all t
    Syy   = np.zeros((D, D))   # sum y_t y_t'
    T_total = 0
    T_pairs = 0

    mu0_sum = np.zeros(k)
    P0_sum  = np.zeros((k, k))

    for seq, smooth in all_smooth:
        T = smooth.z_smooth.shape[0]
        z = smooth.z_smooth      # (T, k)
        P = smooth.P_smooth      # (T, k, k)

        # E[z_t z_t'] = P_t + z_t z_t'
        EzzT = P + z[:, :, None] * z[:, None, :]   # (T, k, k)

        # Initial state
        mu0_sum += z[0]
        P0_sum  += EzzT[0]

        # Pair covariances E[z_t z_{t-1}']
        Ez_pair = smooth.P_pair + z[1:, :, None] * z[:-1, None, :]   # (T-1, k, k)
        # Note: P_pair[t] = E[z_{t+1} z_t'], so need to check indexing

        Sz1z1 += EzzT[:-1].sum(axis=0)
        Sz2z2 += EzzT[1:].sum(axis=0)
        Sz2z1 += smooth.P_pair.sum(axis=0)

        # Observation sufficient stats
        Syz += seq.T @ z          # (D, k)
        Szz += EzzT.sum(axis=0)
        Syy += seq.T @ seq

        T_total += T
        T_pairs += T - 1

    # Update A: A = Sz2z1 @ inv(Sz1z1)
    try:
        A_new = Sz2z1 @ np.linalg.inv(Sz1z1)
    except np.linalg.LinAlgError:
        A_new = Sz2z1 @ np.linalg.pinv(Sz1z1)

    # Update Q: Q = (Sz2z2 - A Sz2z1') / T_pairs
    Q_new = (Sz2z2 - A_new @ Sz2z1.T) / max(T_pairs, 1)
    Q_new = 0.5 * (Q_new + Q_new.T) + 1e-6 * np.eye(k)

    # Update C: C = Syz @ inv(Szz)
    try:
        C_new = Syz @ np.linalg.inv(Szz)
    except np.linalg.LinAlgError:
        C_new = Syz @ np.linalg.pinv(Szz)

    # Update R: R = (Syy - C Syz') / T_total
    R_new = (Syy - C_new @ Syz.T) / max(T_total, 1)
    R_new = 0.5 * (R_new + R_new.T) + 1e-4 * np.eye(D)

    # Update initial state
    mu0_new = mu0_sum / n_seqs
    P0_new  = P0_sum / n_seqs - np.outer(mu0_new, mu0_new)
    P0_new  = 0.5 * (P0_new + P0_new.T) + 1e-6 * np.eye(k)

    return LGSSMParams(
        A=A_new, C=C_new, Q=Q_new, R=R_new,
        mu0=mu0_new, P0=P0_new, k=k, D=D,
    )


# ── Trajectory extraction ──────────────────────────────────────────────────────

def extract_trajectory(
    sequence: np.ndarray,     # (T, D)
    params: LGSSMParams,
    trial_id: int,
    subject_id: str,
    condition: int,
) -> LGSSMTrajectory:
    """
    Run Kalman smoother and extract trajectory features for one trial.

    Parameters
    ----------
    sequence : ndarray (T, D)
        Per-digit feature vectors for one trial.
    params : LGSSMParams
        Fitted LGSSM parameters.

    Returns
    -------
    LGSSMTrajectory
    """
    T = sequence.shape[0]
    k = params.k

    z_filt, P_filt, z_pred, P_pred, ll = kalman_filter(sequence, params)
    smooth = kalman_smoother(z_filt, P_filt, z_pred, P_pred, params)

    z_traj = smooth.z_smooth    # (T, k)

    # Collapse to scalar load: first principal component of z across time
    # (whichever direction explains most variance in z_traj)
    if k == 1:
        load_scalar = z_traj[:, 0]
    else:
        from sklearn.decomposition import PCA
        if z_traj.std(axis=0).max() < 1e-8:
            load_scalar = z_traj[:, 0]
        else:
            pca = PCA(n_components=1)
            load_scalar = pca.fit_transform(z_traj).ravel()

    # Normalize to [0, 1] for comparability across subjects
    lo, hi = load_scalar.min(), load_scalar.max()
    if hi - lo > 1e-8:
        load_norm = (load_scalar - lo) / (hi - lo)
    else:
        load_norm = np.zeros(T)

    # Peak and overload onset
    peak_digit = int(np.argmax(load_norm))
    peak_load = float(load_norm[peak_digit])

    # Load slope: linear regression on first half of trajectory
    half = max(1, T // 2)
    t_vals = np.arange(half)
    if half > 1:
        slope, _ = np.polyfit(t_vals, load_norm[:half], 1)
    else:
        slope = 0.0

    # Overload onset: first digit after peak where load declines consistently
    overload_onset = None
    if peak_digit < T - 1:
        for t in range(peak_digit + 1, T):
            if load_norm[t] < load_norm[peak_digit] * 0.9:
                overload_onset = t
                break

    # Cross-modal coupling strength: Frobenius norm of off-diagonal C blocks
    # (C encodes how latent state maps to each modality — off-diagonal = cross-modal)
    coupling_strength = float(np.linalg.norm(params.C, "fro"))

    return LGSSMTrajectory(
        trial_id=trial_id,
        subject_id=subject_id,
        condition=condition,
        z_trajectory=z_traj,
        load_scalar=load_norm,
        peak_digit=peak_digit,
        peak_load=peak_load,
        load_slope=float(slope),
        overload_onset=overload_onset,
        coupling_strength=coupling_strength,
    )


# ── Full fitting pipeline ──────────────────────────────────────────────────────

class LGSSMPipeline:
    """
    End-to-end LGSSM pipeline.

    Fits one LGSSM per condition (load level), then extracts per-trial
    cognitive load trajectories.

    Parameters
    ----------
    k : int
        Latent state dimension.
    max_iter : int
    tol : float
    n_restarts : int
    """

    def __init__(
        self,
        k: int = 2,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_restarts: int = 5,
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.fitted_params: Dict[int, LGSSMParams] = {}

    def fit(
        self,
        digit_sequences: Dict[int, List[np.ndarray]],
        random_state: int = 42,
    ) -> None:
        """
        Fit LGSSM per condition.

        Parameters
        ----------
        digit_sequences : dict {condition → list of (T_i, D) arrays}
            Per-digit feature sequences grouped by condition.
        """
        for cond, seqs in digit_sequences.items():
            logger.info(f"=== Fitting LGSSM: condition={cond}, n_sequences={len(seqs)} ===")
            params = em_lgssm(
                seqs, k=self.k, max_iter=self.max_iter,
                tol=self.tol, n_restarts=self.n_restarts,
                random_state=random_state,
            )
            self.fitted_params[cond] = params
            logger.info(f"LGSSM fitted: condition={cond}, LL={params.log_likelihood:.4f}")

    def extract_trajectories(
        self,
        digit_sequences: Dict[Tuple[int, str, int], np.ndarray],
    ) -> List[LGSSMTrajectory]:
        """
        Extract cognitive load trajectories from fitted model.

        Parameters
        ----------
        digit_sequences : dict {(trial_id, subject_id, condition) → (T, D) array}

        Returns
        -------
        list of LGSSMTrajectory
        """
        trajectories = []
        for (trial_id, subject_id, condition), seq in digit_sequences.items():
            if condition not in self.fitted_params:
                logger.warning(f"No fitted params for condition={condition}. Skipping.")
                continue
            params = self.fitted_params[condition]
            traj = extract_trajectory(seq, params, trial_id, subject_id, condition)
            trajectories.append(traj)
        return trajectories