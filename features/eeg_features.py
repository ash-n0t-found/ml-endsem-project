"""
features/eeg_features.py
========================
Per-trial EEG feature extraction.

Features extracted:
- Frontal midline theta power (Fz, FCz, Cz; 4-8 Hz)
- Occipital-parietal alpha power (8-12 Hz)
- Individual Alpha Frequency (IAF) from resting EEG
- CSP spatial filter components (4-6 components)
- Sample entropy per electrode cluster
- ERP: P300 amplitude (Pz, 300-600 ms), N200 (FC, 200-350 ms)
- Per-trial spectral estimates (Welch)

All functions operate on MNE Epochs or numpy arrays.
No deep learning. No black-box transforms.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Band power ────────────────────────────────────────────────────────────────

def band_power_welch(
    signal: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    nperseg: int = 256,
) -> float:
    """
    Estimate band power via Welch's method.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Single-channel time series.
    sfreq : float
        Sampling frequency (Hz).
    fmin, fmax : float
        Frequency band limits (Hz).
    nperseg : int
        Welch segment length.

    Returns
    -------
    power : float
        Mean PSD in [fmin, fmax] (µV²/Hz).
    """
    freqs, psd = welch(signal, fs=sfreq, nperseg=min(nperseg, len(signal)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(psd[mask]))


def roi_band_power(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    roi_channels: List[str],
    fmin: float,
    fmax: float,
    nperseg: int = 256,
) -> float:
    """
    Mean band power across ROI channels.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float
    ch_names : list of str
        Channel names corresponding to data rows.
    roi_channels : list of str
        Subset of channels to average.
    fmin, fmax : float

    Returns
    -------
    power : float
    """
    ch_indices = [ch_names.index(c) for c in roi_channels if c in ch_names]
    if not ch_indices:
        logger.warning(f"No ROI channels found. Requested: {roi_channels}")
        return np.nan
    powers = [band_power_welch(data[i], sfreq, fmin, fmax, nperseg) for i in ch_indices]
    return float(np.nanmean(powers))


# ── Individual Alpha Frequency (IAF) ──────────────────────────────────────────

def compute_iaf(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    posterior_channels: Optional[List[str]] = None,
    search_range: Tuple[float, float] = (6.0, 14.0),
    nperseg: int = 512,
) -> float:
    """
    Estimate Individual Alpha Frequency (IAF) from resting-state EEG.

    IAF = frequency of peak PSD in alpha search range, averaged across
    posterior channels. Established WM capacity predictor.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Resting-state EEG (eyes-closed preferred).
    sfreq : float
    ch_names : list of str
    posterior_channels : list of str, optional
        Channels to use for IAF estimation. Defaults to O1, Oz, O2, P3, Pz, P4.
    search_range : tuple
        (fmin, fmax) Hz for alpha peak search.

    Returns
    -------
    iaf : float
        IAF in Hz. Returns nan if no clear peak found.
    """
    if posterior_channels is None:
        posterior_channels = ["O1", "Oz", "O2", "P3", "Pz", "P4"]

    ch_indices = [ch_names.index(c) for c in posterior_channels if c in ch_names]
    if not ch_indices:
        logger.warning("IAF: no posterior channels found.")
        return np.nan

    # Average PSD across posterior channels
    psds = []
    freqs = None
    for i in ch_indices:
        f, p = welch(data[i], fs=sfreq, nperseg=min(nperseg, len(data[i])))
        psds.append(p)
        if freqs is None:
            freqs = f

    mean_psd = np.mean(psds, axis=0)
    mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    if mask.sum() == 0:
        return np.nan

    peak_idx = np.argmax(mean_psd[mask])
    iaf = float(freqs[mask][peak_idx])
    return iaf


# ── Sample entropy ────────────────────────────────────────────────────────────

def sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample entropy (SampEn) of a 1-D time series.

    Measures signal irregularity / complexity.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
    m : int
        Template length (embedding dimension).
    r_factor : float
        Tolerance as fraction of signal std.

    Returns
    -------
    sampen : float
        Sample entropy. Returns nan if computation fails.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < (m + 2) * 10:
        return np.nan

    r = r_factor * np.std(signal, ddof=1)
    if r < 1e-10:
        return np.nan

    def _count_templates(template_len: int) -> int:
        count = 0
        for i in range(n - template_len):
            template = signal[i : i + template_len]
            # Vectorized Chebyshev distance
            diff = np.abs(
                signal[: n - template_len] - template[0]
            )
            for k in range(1, template_len):
                diff = np.maximum(diff, np.abs(
                    signal[k : n - template_len + k] - template[k]
                ))
            matches = np.sum(diff < r) - 1  # exclude self
            count += max(matches, 0)
        return count

    B = _count_templates(m)
    A = _count_templates(m + 1)

    if B == 0 or A == 0:
        return np.nan

    return float(-np.log(A / B))


def cluster_sample_entropy(
    data: np.ndarray,
    ch_names: List[str],
    clusters: Dict[str, List[str]],
    m: int = 2,
    r_factor: float = 0.2,
) -> Dict[str, float]:
    """
    Compute sample entropy for each electrode cluster (mean across channels).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    ch_names : list of str
    clusters : dict
        {'frontal': ['Fp1', 'Fz', ...], 'parietal': [...], ...}

    Returns
    -------
    dict : cluster_name -> mean SampEn
    """
    result = {}
    for cluster_name, cluster_chs in clusters.items():
        indices = [ch_names.index(c) for c in cluster_chs if c in ch_names]
        if not indices:
            result[cluster_name] = np.nan
            continue
        entropies = [sample_entropy(data[i], m=m, r_factor=r_factor) for i in indices]
        result[cluster_name] = float(np.nanmean(entropies))
    return result


# ── ERP features ──────────────────────────────────────────────────────────────

def erp_amplitude(
    epoch_data: np.ndarray,
    times: np.ndarray,
    ch_names: List[str],
    channels: List[str],
    tmin: float,
    tmax: float,
) -> float:
    """
    Mean amplitude in ERP time window across specified channels.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
        Single epoch (already baseline-corrected).
    times : ndarray, shape (n_times,)
        Time vector (seconds).
    ch_names : list of str
    channels : list of str
        Channels to average.
    tmin, tmax : float
        Time window (seconds).

    Returns
    -------
    amplitude : float
        Mean amplitude in µV.
    """
    ch_indices = [ch_names.index(c) for c in channels if c in ch_names]
    if not ch_indices:
        return np.nan

    time_mask = (times >= tmin) & (times <= tmax)
    if time_mask.sum() == 0:
        return np.nan

    roi_data = epoch_data[ch_indices][:, time_mask]  # (n_ch, n_times)
    return float(np.mean(roi_data))


# ── CSP features ──────────────────────────────────────────────────────────────

def fit_csp(
    epochs_by_class: Dict[int, np.ndarray],
    n_components: int = 6,
    reg: str = "ledoit_wolf",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Common Spatial Pattern (CSP) filter.

    For multi-class (>2), uses one-vs-rest CSP and concatenates filters.

    Parameters
    ----------
    epochs_by_class : dict
        class_label -> ndarray, shape (n_trials, n_channels, n_times)
    n_components : int
        Components per class pair.
    reg : str
        Covariance regularization: 'ledoit_wolf' or 'oas'.

    Returns
    -------
    W : ndarray, shape (n_total_components, n_channels)
        Spatial filters (rows).
    class_labels : ndarray
        Class label for each filter component.
    """
    from sklearn.covariance import LedoitWolf, OAS

    classes = sorted(epochs_by_class.keys())
    n_classes = len(classes)

    all_filters = []
    all_labels = []

    for i, cls_a in enumerate(classes):
        # One-vs-rest
        X_a = epochs_by_class[cls_a]  # (n_a, n_ch, n_t)
        X_rest = np.concatenate(
            [epochs_by_class[c] for c in classes if c != cls_a], axis=0
        )

        # Compute covariance matrices
        def _cov(X: np.ndarray) -> np.ndarray:
            """Mean covariance across trials."""
            covs = []
            for trial in X:
                if reg == "ledoit_wolf":
                    cov = LedoitWolf().fit(trial.T).covariance_
                else:
                    cov = OAS().fit(trial.T).covariance_
                covs.append(cov)
            return np.mean(covs, axis=0)

        C_a = _cov(X_a)
        C_rest = _cov(X_rest)

        # Solve generalized eigenvalue problem: C_a w = λ (C_a + C_rest) w
        C_sum = C_a + C_rest
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.solve(C_sum, C_a)
            )
        except np.linalg.LinAlgError:
            logger.warning(f"CSP: singular matrix for class {cls_a}. Skipping.")
            continue

        # Sort by eigenvalue (extremes = most discriminative)
        order = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, order]

        # Take first and last n_components//2 filters
        n_each = max(1, n_components // 2)
        W_cls = np.vstack([
            eigenvectors[:, :n_each].T,
            eigenvectors[:, -n_each:].T,
        ])
        all_filters.append(W_cls)
        all_labels.extend([cls_a] * len(W_cls))

    if not all_filters:
        return np.zeros((1, 1)), np.array([0])

    return np.vstack(all_filters), np.array(all_labels)


def apply_csp_features(
    epoch_data: np.ndarray,
    W: np.ndarray,
    sfreq: float,
    fmin: float = 4.0,
    fmax: float = 8.0,
) -> np.ndarray:
    """
    Apply CSP filters and compute log-band-power per component.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
        Single epoch.
    W : ndarray, shape (n_components, n_channels)
        CSP spatial filters.
    sfreq : float
    fmin, fmax : float
        Band for power computation after spatial filtering.

    Returns
    -------
    features : ndarray, shape (n_components,)
        Log-band-power per CSP component.
    """
    # Apply spatial filters: (n_components, n_times)
    filtered = W @ epoch_data

    features = np.array([
        np.log(max(band_power_welch(filtered[k], sfreq, fmin, fmax), 1e-12))
        for k in range(W.shape[0])
    ])
    return features


# ── Full per-trial EEG feature extraction ─────────────────────────────────────

def extract_eeg_features_trial(
    epoch_data: np.ndarray,
    times: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    cfg: Config,
    csp_W: Optional[np.ndarray] = None,
    iaf: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract all EEG features for a single trial epoch.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
        Preprocessed, baseline-corrected epoch.
    times : ndarray, shape (n_times,)
    ch_names : list of str
    sfreq : float
    cfg : Config
    csp_W : ndarray, optional
        Pre-fitted CSP filters. If None, CSP features skipped.
    iaf : float, optional
        Subject IAF (from resting state). Stored as feature if provided.

    Returns
    -------
    features : dict
        Feature name -> value.
    """
    features = {}

    # Frontal midline theta power
    theta_roi = cfg.eeg.theta_roi if hasattr(cfg.eeg, "theta_roi") else ["Fz", "FCz", "Cz"]
    features["theta_frontal"] = roi_band_power(
        epoch_data, sfreq, ch_names, theta_roi,
        fmin=4.0, fmax=8.0
    )

    # Posterior alpha power
    alpha_roi = cfg.eeg.alpha_roi if hasattr(cfg.eeg, "alpha_roi") else ["P3", "Pz", "P4", "O1", "Oz", "O2"]
    features["alpha_posterior"] = roi_band_power(
        epoch_data, sfreq, ch_names, alpha_roi,
        fmin=8.0, fmax=12.0
    )

    # Theta/alpha ratio (frontal engagement marker)
    if features["theta_frontal"] > 0 and features["alpha_posterior"] > 0:
        features["theta_alpha_ratio"] = features["theta_frontal"] / features["alpha_posterior"]
    else:
        features["theta_alpha_ratio"] = np.nan

    # Beta power (frontal) — secondary engagement marker
    features["beta_frontal"] = roi_band_power(
        epoch_data, sfreq, ch_names, theta_roi,
        fmin=12.0, fmax=30.0
    )

    # IAF (subject-level, stored per trial for convenience)
    if iaf is not None:
        features["iaf"] = float(iaf)

    # P300 amplitude
    p300_cfg = cfg.eeg.erp.p300 if hasattr(cfg.eeg, "erp") else None
    if p300_cfg is not None:
        p300_chs = p300_cfg.channels if hasattr(p300_cfg, "channels") else ["Pz", "P3", "P4"]
        features["p300_amplitude"] = erp_amplitude(
            epoch_data, times, ch_names, p300_chs,
            tmin=0.300, tmax=0.600
        )

    # N200 amplitude
    n200_cfg = cfg.eeg.erp.n200 if hasattr(cfg.eeg, "erp") else None
    if n200_cfg is not None:
        n200_chs = n200_cfg.channels if hasattr(n200_cfg, "channels") else ["FC1", "FCz", "FC2"]
        features["n200_amplitude"] = erp_amplitude(
            epoch_data, times, ch_names, n200_chs,
            tmin=0.200, tmax=0.350
        )

    # Sample entropy — frontal cluster
    frontal_chs = ["Fp1", "Fp2", "F3", "Fz", "F4", "FC1", "FCz", "FC2"]
    frontal_indices = [ch_names.index(c) for c in frontal_chs if c in ch_names]
    if frontal_indices:
        entropies = [sample_entropy(epoch_data[i]) for i in frontal_indices[:4]]
        features["sampen_frontal"] = float(np.nanmean(entropies))
    else:
        features["sampen_frontal"] = np.nan

    # CSP log-band-power features
    if csp_W is not None:
        csp_feats = apply_csp_features(epoch_data, csp_W, sfreq, fmin=4.0, fmax=8.0)
        for k, val in enumerate(csp_feats):
            features[f"csp_{k}"] = float(val)

    return features


def extract_eeg_features_all_trials(
    epochs_data: np.ndarray,
    times: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    cfg: Config,
    csp_W: Optional[np.ndarray] = None,
    iaf: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract EEG features for all trials.

    Parameters
    ----------
    epochs_data : ndarray, shape (n_trials, n_channels, n_times)
    times : ndarray
    ch_names : list of str
    sfreq : float
    cfg : Config
    csp_W : ndarray, optional
    iaf : float, optional

    Returns
    -------
    feature_matrix : ndarray, shape (n_trials, n_features)
    feature_names : list of str
    """
    n_trials = epochs_data.shape[0]
    all_features = []

    for i in range(n_trials):
        feats = extract_eeg_features_trial(
            epochs_data[i], times, ch_names, sfreq, cfg,
            csp_W=csp_W, iaf=iaf
        )
        all_features.append(feats)

    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([
        [f[k] for k in feature_names] for f in all_features
    ])

    if verbose:
        logger.info(
            f"EEG features: {n_trials} trials × {len(feature_names)} features"
        )

    return feature_matrix, feature_names