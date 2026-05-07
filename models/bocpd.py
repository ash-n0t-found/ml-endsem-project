"""
models/bocpd.py
===============

Bayesian Online Changepoint Detection (BOCPD) for individual overload threshold.

Applied to coupling strength trajectory across conditions/digits.
Detected changepoint estimates individual WM overload onset.

Implements Adams & MacKay (2007) BOCPD with Gaussian-unknown-mean hazard model.
Strictly classical probabilistic method.

Primary use:
  1. Per-digit coupling strength trajectory from LGSSM → detect overload onset
  2. Across-condition coupling trajectory → detect transition point
  3. Validate detected physiological changepoint against behavioral performance dropoff

Dependencies: numpy, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import gammaln

from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BOCPDResult:
    """Result of BOCPD applied to one time series."""
    run_length_probs: np.ndarray      # (T, T) R[t, r] = P(run_length=r at t)
    changepoint_probs: np.ndarray     # (T,) posterior prob of changepoint at t
    detected_changepoints: List[int]   # time indices where changepoint > threshold
    most_likely_cp: Optional[int]      # most probable single changepoint location
    subject_id: str
    series_name: str


@dataclass
class OverloadChangepoint:
    """Detected overload onset from coupling trajectory."""
    subject_id: str
    physiological_cp: Optional[int]    # detected from coupling trajectory
    behavioral_cp: Optional[int]       # from behavioral performance dropoff
    cp_agreement: Optional[float]      # |phys_cp - behav_cp| (lower = better)
    coupling_at_cp: Optional[float]    # coupling strength at detected changepoint
    coupling_decline: Optional[float]  # coupling strength drop at changepoint


# ── Gaussian hazard model ──────────────────────────────────────────────────────

class GaussianUnknownMean:
    """
    Gaussian likelihood with unknown mean (conjugate Normal-Gamma prior).

    For each run length r, maintains sufficient statistics to compute
    predictive probability P(x_t | x_{t-r:t-1}).

    Uses Normal-Gamma conjugate:
      Prior: μ ~ N(μ0, 1/(κ0 λ)), λ ~ Gamma(α0, β0)
      Posterior: updated analytically per observation.
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def predictive_prob(
        self,
        x: float,
        run_sums: np.ndarray,
        run_sq_sums: np.ndarray,
        run_lengths: np.ndarray,
    ) -> np.ndarray:
        """
        Compute P(x_t | run statistics) for all current run lengths.

        Returns Student-t predictive probability for each run.

        Parameters
        ----------
        x : float
            New observation.
        run_sums : ndarray (n_runs,)
            Sum of observations in each run.
        run_sq_sums : ndarray (n_runs,)
            Sum of squared observations in each run.
        run_lengths : ndarray (n_runs,)
            Length of each run.

        Returns
        -------
        probs : ndarray (n_runs,)
        """
        n = run_lengths

        # Posterior hyperparameters
        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + n / 2.0
        mu_n = (self.kappa0 * self.mu0 + run_sums) / kappa_n
        beta_n = (
            self.beta0
            + 0.5 * run_sq_sums
            + 0.5 * self.kappa0 * n * (run_sums / n - self.mu0) ** 2 / kappa_n
        )
        beta_n = np.where(n == 0, self.beta0, beta_n)

        # Student-t predictive distribution
        df = 2 * alpha_n
        scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
        scale = np.maximum(scale, 1e-8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_probs = stats.t.logpdf(x, df=df, loc=mu_n, scale=scale)

        return np.exp(log_probs)


# ── BOCPD core ─────────────────────────────────────────────────────────────────

def bocpd(
    data: np.ndarray,
    hazard_rate: float = 0.1,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    threshold: float = 0.5,
    subject_id: str = "",
    series_name: str = "",
) -> BOCPDResult:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Parameters
    ----------
    data : ndarray (T,)
        Time series to detect changepoints in.
        For coupling analysis: coupling strength across digits/conditions.
    hazard_rate : float
        Prior probability of changepoint at each step (H in paper).
        0.1 → expect ~1 changepoint per 10 observations.
    mu0, kappa0, alpha0, beta0 : float
        Normal-Gamma prior hyperparameters.
    threshold : float
        Posterior probability threshold for changepoint declaration.

    Returns
    -------
    BOCPDResult
    """
    T = len(data)
    if T < 3:
        logger.warning(f"BOCPD: series too short (T={T}). Returning no changepoints.")
        return BOCPDResult(
            run_length_probs=np.zeros((T, T)),
            changepoint_probs=np.zeros(T),
            detected_changepoints=[],
            most_likely_cp=None,
            subject_id=subject_id,
            series_name=series_name,
        )

    # Normalize data to improve prior specification
    data_std = data.std()
    if data_std > 1e-8:
        data_norm = (data - data.mean()) / data_std
    else:
        data_norm = data.copy()

    hazard_model = GaussianUnknownMean(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)

    # Run length distribution P(r_t | x_{1:t})
    # R[t, r] = P(run_length = r at time t)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics per run length
    run_sums    = np.zeros(T + 1)
    run_sq_sums = np.zeros(T + 1)
    run_lengths = np.arange(T + 1, dtype=float)

    changepoint_probs = np.zeros(T)

    for t in range(T):
        x = data_norm[t]

        # Current runs have lengths 0..t
        r_vals = np.arange(t + 1, dtype=float)

        # Predictive probability for each run
        pred_probs = hazard_model.predictive_prob(
            x, run_sums[:t + 1], run_sq_sums[:t + 1], r_vals
        )

        # Update run length distribution
        R_t = R[t, :t + 1]

        # Growth probabilities: run continues
        growth = R_t * pred_probs * (1 - hazard_rate)
        # Changepoint probability: run resets
        cp_mass = np.sum(R_t * pred_probs * hazard_rate)

        # Assign to t+1
        R[t + 1, 1:t + 2] = growth
        R[t + 1, 0] = cp_mass

        # Normalize
        norm = R[t + 1, :t + 2].sum()
        if norm > 0:
            R[t + 1, :t + 2] /= norm

        # Update sufficient statistics (shift by 1 for new run lengths)
        run_sums[1:t + 2] = run_sums[:t + 1] + x
        run_sq_sums[1:t + 2] = run_sq_sums[:t + 1] + x ** 2
        # run_sums[0] stays 0 (new run)

        # Changepoint probability at time t = P(r_t = 0)
        changepoint_probs[t] = R[t + 1, 0]

    # Detected changepoints
    detected_cps = [t for t in range(T) if changepoint_probs[t] >= threshold]

    # Most likely changepoint (highest probability after initial period)
    if T > 2:
        cp_probs_interior = changepoint_probs[1:-1]  # exclude endpoints
        if cp_probs_interior.max() >= threshold * 0.5:
            most_likely_cp = int(np.argmax(cp_probs_interior)) + 1
        else:
            most_likely_cp = None
    else:
        most_likely_cp = None

    return BOCPDResult(
        run_length_probs=R[:T, :T],
        changepoint_probs=changepoint_probs,
        detected_changepoints=detected_cps,
        most_likely_cp=most_likely_cp,
        subject_id=subject_id,
        series_name=series_name,
    )


# ── Coupling trajectory changepoint ───────────────────────────────────────────

def detect_coupling_overload(
    coupling_trajectory: np.ndarray,   # (T,) coupling strength over digits/conditions
    behavioral_performance: Optional[np.ndarray] = None,  # (T,) accuracy trajectory
    subject_id: str = "",
    hazard_rate: float = 0.1,
    threshold: float = 0.5,
) -> OverloadChangepoint:
    """
    Detect overload onset from coupling strength trajectory.

    Physiological changepoint: BOCPD on coupling_trajectory.
    Behavioral changepoint: first point of sustained performance decline.

    Validates: physiological CP should precede or coincide with behavioral CP.

    Parameters
    ----------
    coupling_trajectory : ndarray (T,)
        Coupling strength at each time point (digit or condition step).
    behavioral_performance : ndarray (T,) or None
        Recall accuracy at each step. If provided, compute behavioral CP.
    subject_id : str
    hazard_rate, threshold : BOCPD hyperparameters.

    Returns
    -------
    OverloadChangepoint
    """
    T = len(coupling_trajectory)

    # Physiological changepoint
    result = bocpd(
        coupling_trajectory,
        hazard_rate=hazard_rate,
        threshold=threshold,
        subject_id=subject_id,
        series_name="coupling_strength",
    )

    phys_cp = result.most_likely_cp

    # Behavioral changepoint: first sustained decline in performance
    behav_cp = None
    if behavioral_performance is not None and len(behavioral_performance) >= 3:
        behav_cp = _behavioral_changepoint(behavioral_performance)

    # Agreement
    cp_agreement = None
    if phys_cp is not None and behav_cp is not None:
        cp_agreement = float(abs(phys_cp - behav_cp))

    # Coupling at detected CP
    coupling_at_cp = None
    coupling_decline = None
    if phys_cp is not None and phys_cp < T:
        coupling_at_cp = float(coupling_trajectory[phys_cp])
        if phys_cp > 0:
            coupling_decline = float(coupling_trajectory[phys_cp - 1] - coupling_trajectory[phys_cp])

    return OverloadChangepoint(
        subject_id=subject_id,
        physiological_cp=phys_cp,
        behavioral_cp=behav_cp,
        cp_agreement=cp_agreement,
        coupling_at_cp=coupling_at_cp,
        coupling_decline=coupling_decline,
    )


def _behavioral_changepoint(performance: np.ndarray) -> Optional[int]:
    """
    Simple behavioral changepoint: first sustained decline.

    Uses a sliding window: changepoint when performance drops > 1 SD
    below mean of preceding window.
    """
    T = len(performance)
    if T < 4:
        return None

    window = max(2, T // 4)
    baseline_mean = performance[:window].mean()
    baseline_std = performance[:window].std()

    threshold_val = baseline_mean - baseline_std

    for t in range(window, T):
        if performance[t] < threshold_val:
            return t

    return None


# ── Group-level changepoint analysis ──────────────────────────────────────────

def detect_group_overload(
    coupling_trajectories: Dict[str, np.ndarray],
    behavioral_trajectories: Optional[Dict[str, np.ndarray]] = None,
    hazard_rate: float = 0.1,
    threshold: float = 0.5,
) -> Tuple[List[OverloadChangepoint], Dict]:
    """
    Detect overload changepoints for all subjects.

    Parameters
    ----------
    coupling_trajectories : dict {subject_id → (T,) trajectory}
    behavioral_trajectories : dict {subject_id → (T,) accuracy} or None

    Returns
    -------
    results : list of OverloadChangepoint (one per subject)
    summary : dict with group-level statistics
    """
    results = []

    for subj_id, traj in coupling_trajectories.items():
        behav = None
        if behavioral_trajectories is not None:
            behav = behavioral_trajectories.get(subj_id)

        cp = detect_coupling_overload(
            coupling_trajectory=traj,
            behavioral_performance=behav,
            subject_id=subj_id,
            hazard_rate=hazard_rate,
            threshold=threshold,
        )
        results.append(cp)

    # Group summary
    phys_cps = [r.physiological_cp for r in results if r.physiological_cp is not None]
    agreements = [r.cp_agreement for r in results if r.cp_agreement is not None]

    summary = {
        "n_subjects": len(results),
        "n_detected_phys": len(phys_cps),
        "mean_phys_cp": float(np.mean(phys_cps)) if phys_cps else np.nan,
        "std_phys_cp": float(np.std(phys_cps)) if phys_cps else np.nan,
        "mean_cp_agreement": float(np.mean(agreements)) if agreements else np.nan,
        "detection_rate": len(phys_cps) / len(results) if results else 0.0,
    }

    logger.info(
        f"Group BOCPD: detected={len(phys_cps)}/{len(results)} subjects, "
        f"mean_cp={summary['mean_phys_cp']:.2f}, "
        f"mean_agreement={summary['mean_cp_agreement']:.2f}"
    )

    return results, summary


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_changepoints(
    overload_results: List[OverloadChangepoint],
    wm_capacity: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Validate detected changepoints against behavioral WM capacity.

    Tests: subjects with lower WM capacity should have earlier overload onset.

    Parameters
    ----------
    overload_results : list of OverloadChangepoint
    wm_capacity : dict {subject_id → WM capacity score} or None

    Returns
    -------
    validation_stats : dict
    """
    from scipy.stats import spearmanr, pearsonr

    phys_cps = [(r.subject_id, r.physiological_cp)
                for r in overload_results if r.physiological_cp is not None]

    if len(phys_cps) < 5:
        return {"error": "Too few detected changepoints for validation"}

    subj_ids = [x[0] for x in phys_cps]
    cp_vals  = np.array([x[1] for x in phys_cps])

    stats_out = {
        "n_subjects": len(phys_cps),
        "mean_cp": float(cp_vals.mean()),
        "std_cp": float(cp_vals.std()),
    }

    if wm_capacity is not None:
        cap_vals = np.array([wm_capacity.get(s, np.nan) for s in subj_ids])
        valid = ~np.isnan(cap_vals)

        if valid.sum() >= 5:
            rho, p_spearman = spearmanr(cp_vals[valid], cap_vals[valid])
            r, p_pearson = pearsonr(cp_vals[valid], cap_vals[valid])

            stats_out.update({
                "spearman_rho": float(rho),
                "spearman_p": float(p_spearman),
                "pearson_r": float(r),
                "pearson_p": float(p_pearson),
                "n_valid": int(valid.sum()),
            })

            logger.info(
                f"CP vs WM capacity: Spearman ρ={rho:.3f} (p={p_spearman:.4f}), "
                f"Pearson r={r:.3f} (p={p_pearson:.4f})"
            )

    return stats_out