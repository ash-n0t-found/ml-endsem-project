"""
features/hep_features.py
========================
Heartbeat-Evoked Potential (HEP) extraction — the primary cross-modal
EEG × ECG feature.

HEP = trial-averaged EEG epoch locked to each R-peak within a trial.
Provides a direct measure of cardiac–cortical coupling via interoceptive
signaling (baroreceptor → cortex pathway).

Key references:
  Schandry & Montoya (1996); Park et al. (2014); Candia-Rivera et al. (2021)

Pipeline:
  1. Detect R-peaks within trial time window (using pre-computed R-peak series)
  2. For each R-peak: extract EEG window [tmin, tmax] post R-peak
  3. Average across R-peaks → trial HEP
  4. Extract amplitude in analysis window (200–600 ms) at frontal channels
  5. Compute HEP amplitude (mean) and HEP SNR

Dependencies: numpy, scipy, mne (optional for ERP utilities)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ── Main extraction function ──────────────────────────────────────────────────

def extract_hep_features(
    eeg_data: np.ndarray,
    r_peaks_samples: np.ndarray,
    eeg_sfreq: float,
    trial_start_sample: int,
    trial_end_sample: int,
    channel_names: List[str],
    hep_tmin: float = -0.1,
    hep_tmax: float = 0.6,
    analysis_tmin: float = 0.2,
    analysis_tmax: float = 0.6,
    channels_of_interest: Optional[List[str]] = None,
    baseline_correct: bool = True,
    min_r_peaks: int = 3,
) -> Dict[str, float]:
    """
    Extract HEP features from a single trial.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_total_samples)
        Full continuous EEG (already preprocessed, rereferenced).
    r_peaks_samples : ndarray, shape (n_r_peaks,)
        Sample indices of R-peaks in the SAME reference frame as eeg_data.
    eeg_sfreq : float
        EEG sampling frequency (Hz).
    trial_start_sample : int
        Sample index of trial start in eeg_data.
    trial_end_sample : int
        Sample index of trial end in eeg_data.
    channel_names : list of str
        Channel names corresponding to eeg_data rows.
    hep_tmin : float
        Epoch start relative to R-peak (s). Typically -0.1.
    hep_tmax : float
        Epoch end relative to R-peak (s). Typically 0.6.
    analysis_tmin : float
        Start of HEP amplitude measurement window (s).
    analysis_tmax : float
        End of HEP amplitude measurement window (s).
    channels_of_interest : list of str, optional
        Channels to use for HEP amplitude. Default: ['Fz', 'Cz', 'FCz'].
    baseline_correct : bool
        If True, subtract mean of [hep_tmin, 0] from each epoch.
    min_r_peaks : int
        Minimum number of valid R-peaks for reliable HEP estimate.

    Returns
    -------
    dict : feature_name → float
        Keys:
          hep_amplitude_<ch>  : mean HEP in analysis window per channel
          hep_amplitude_mean  : average across channels of interest
          hep_snr             : signal-to-noise ratio of HEP
          hep_n_peaks         : number of R-peaks averaged
    """
    if channels_of_interest is None:
        channels_of_interest = ["Fz", "Cz", "FCz"]

    features: Dict[str, float] = {}

    # Sample offsets for HEP epoch
    n_pre = int(abs(hep_tmin) * eeg_sfreq)   # samples before R-peak
    n_post = int(hep_tmax * eeg_sfreq)        # samples after R-peak
    epoch_len = n_pre + n_post
    times = np.linspace(hep_tmin, hep_tmax, epoch_len)

    # Select R-peaks inside trial window (with margin for epoch extraction)
    margin = n_pre + 10
    valid_mask = (
        (r_peaks_samples >= trial_start_sample + margin) &
        (r_peaks_samples <= trial_end_sample - n_post - 10)
    )
    trial_r_peaks = r_peaks_samples[valid_mask]

    n_peaks = len(trial_r_peaks)
    features["hep_n_peaks"] = float(n_peaks)

    if n_peaks < min_r_peaks:
        # Too few R-peaks — unreliable HEP
        for ch in channels_of_interest:
            features[f"hep_amplitude_{ch}"] = np.nan
        features["hep_amplitude_mean"] = np.nan
        features["hep_snr"] = np.nan
        return features

    # Resolve channel indices
    ch_indices = _resolve_channel_indices(channel_names, channels_of_interest)

    if not ch_indices:
        warnings.warn(
            f"None of channels_of_interest {channels_of_interest} found in "
            f"channel_names. Returning NaN HEP features.",
            UserWarning,
        )
        features["hep_amplitude_mean"] = np.nan
        features["hep_snr"] = np.nan
        return features

    # Extract and stack epochs: shape (n_peaks, n_channels, epoch_len)
    epochs = _extract_epochs(
        eeg_data=eeg_data,
        r_peaks=trial_r_peaks,
        n_pre=n_pre,
        epoch_len=epoch_len,
        ch_indices=list(ch_indices.values()),
    )

    if epochs is None or len(epochs) == 0:
        for ch in channels_of_interest:
            features[f"hep_amplitude_{ch}"] = np.nan
        features["hep_amplitude_mean"] = np.nan
        features["hep_snr"] = np.nan
        return features

    # Baseline correction: subtract mean in [hep_tmin, 0]
    if baseline_correct:
        baseline_mask = times <= 0
        baseline_mean = epochs[:, :, baseline_mask].mean(axis=2, keepdims=True)
        epochs = epochs - baseline_mean

    # Average across R-peaks → HEP: shape (n_channels, epoch_len)
    hep = epochs.mean(axis=0)

    # Analysis window mask
    analysis_mask = (times >= analysis_tmin) & (times <= analysis_tmax)
    if analysis_mask.sum() == 0:
        analysis_mask = np.ones(epoch_len, dtype=bool)

    # Compute amplitude per channel of interest
    amplitudes = []
    for ch_name, ch_local_idx in _local_indices(ch_indices).items():
        amp = float(hep[ch_local_idx, analysis_mask].mean())
        features[f"hep_amplitude_{ch_name}"] = amp
        amplitudes.append(amp)

    features["hep_amplitude_mean"] = float(np.mean(amplitudes)) if amplitudes else np.nan

    # SNR: mean HEP amplitude / std across single-trial epoch amplitudes
    if len(epochs) > 1:
        single_trial_amps = epochs[:, :, analysis_mask].mean(axis=(1, 2))
        signal = np.abs(np.mean(single_trial_amps))
        noise = np.std(single_trial_amps, ddof=1)
        features["hep_snr"] = float(signal / noise) if noise > 1e-10 else np.nan
    else:
        features["hep_snr"] = np.nan

    return features


# ── HEP correlation with cardiac coupling ─────────────────────────────────────

def compute_hep_hrv_correlation(
    hep_amplitudes: np.ndarray,
    hrv_rmssd: np.ndarray,
) -> Dict[str, float]:
    """
    Correlate trial-level HEP amplitude with HRV RMSSD across trials.
    Tests whether cardiac–cortical coupling tracks parasympathetic tone.

    Parameters
    ----------
    hep_amplitudes : ndarray, shape (n_trials,)
    hrv_rmssd : ndarray, shape (n_trials,)

    Returns
    -------
    dict : r, p_value, n_valid
    """
    valid = ~(np.isnan(hep_amplitudes) | np.isnan(hrv_rmssd))
    n_valid = valid.sum()
    if n_valid < 5:
        return {"hep_hrv_r": np.nan, "hep_hrv_p": np.nan, "hep_hrv_n": float(n_valid)}

    r, p = sp_stats.pearsonr(hep_amplitudes[valid], hrv_rmssd[valid])
    return {"hep_hrv_r": float(r), "hep_hrv_p": float(p), "hep_hrv_n": float(n_valid)}


# ── Resting-state HEP (per subject baseline) ──────────────────────────────────

def extract_resting_hep(
    eeg_data: np.ndarray,
    r_peaks_samples: np.ndarray,
    eeg_sfreq: float,
    channel_names: List[str],
    hep_tmin: float = -0.1,
    hep_tmax: float = 0.6,
    analysis_tmin: float = 0.2,
    analysis_tmax: float = 0.6,
    channels_of_interest: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Resting-state HEP amplitude per subject.
    Used as per-subject baseline coupling index.

    Parameters same as extract_hep_features, but operates on full
    resting recording rather than a single trial.

    Returns
    -------
    dict : feature_name → float
        Keys: resting_hep_amplitude_<ch>, resting_hep_amplitude_mean
    """
    if channels_of_interest is None:
        channels_of_interest = ["Fz", "Cz", "FCz"]

    # Use full recording as "trial"
    result = extract_hep_features(
        eeg_data=eeg_data,
        r_peaks_samples=r_peaks_samples,
        eeg_sfreq=eeg_sfreq,
        trial_start_sample=0,
        trial_end_sample=eeg_data.shape[1],
        channel_names=channel_names,
        hep_tmin=hep_tmin,
        hep_tmax=hep_tmax,
        analysis_tmin=analysis_tmin,
        analysis_tmax=analysis_tmax,
        channels_of_interest=channels_of_interest,
        baseline_correct=True,
        min_r_peaks=5,
    )

    # Rename keys to resting_ prefix
    resting = {}
    for k, v in result.items():
        resting[f"resting_{k}"] = v
    return resting


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_channel_indices(
    channel_names: List[str],
    channels_of_interest: List[str],
) -> Dict[str, int]:
    """Return {ch_name: index_in_eeg_data} for available channels."""
    ch_map = {ch: i for i, ch in enumerate(channel_names)}
    result = {}
    for ch in channels_of_interest:
        if ch in ch_map:
            result[ch] = ch_map[ch]
        else:
            warnings.warn(f"Channel '{ch}' not found in EEG data.", UserWarning)
    return result


def _local_indices(ch_indices: Dict[str, int]) -> Dict[str, int]:
    """
    Map {ch_name: global_index} → {ch_name: local_index} (0,1,2,...).
    Local index = position in the extracted epoch array.
    """
    return {ch: i for i, ch in enumerate(ch_indices.keys())}


def _extract_epochs(
    eeg_data: np.ndarray,
    r_peaks: np.ndarray,
    n_pre: int,
    epoch_len: int,
    ch_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Extract EEG epochs around R-peaks.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
    r_peaks : ndarray, shape (n_peaks,)
    n_pre : int  — samples before R-peak
    epoch_len : int — total epoch length in samples
    ch_indices : list of int — channel indices to extract

    Returns
    -------
    epochs : ndarray, shape (n_valid_peaks, n_ch, epoch_len) or None
    """
    n_total = eeg_data.shape[1]
    epochs = []
    for rp in r_peaks:
        i0 = int(rp) - n_pre
        i1 = i0 + epoch_len
        if i0 < 0 or i1 > n_total:
            continue
        epoch = eeg_data[np.ix_(ch_indices, np.arange(i0, i1))]  # (n_ch, epoch_len)
        epochs.append(epoch)

    if not epochs:
        return None
    return np.stack(epochs, axis=0)  # (n_peaks, n_ch, epoch_len)