"""
data/synchronizer.py
====================
Cross-modal temporal synchronization for ds003838.

Aligns pupillometry, ECG/PPG timestamps to EEG sample indices.
Handles different sampling rates and clock drift.

Key operations:
- Resample pupil to EEG sfreq (or common grid)
- Align trial onset markers across modalities
- Validate synchronization quality
- Extract per-trial multimodal windows
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from math import gcd

from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Synchronized trial container ──────────────────────────────────────────────

@dataclass
class SynchronizedTrial:
    """
    Single synchronized trial across all modalities.

    All arrays are aligned to the same time grid (EEG sfreq).
    """
    trial_idx: int
    condition: str                       # 'control' / 'load_5' / 'load_9' / 'load_13'
    n_digits: int
    subject_id: str

    # Time grid (seconds, relative to trial onset)
    times: np.ndarray = field(default_factory=lambda: np.array([]))

    # EEG: shape (n_channels, n_samples)
    eeg: Optional[np.ndarray] = None

    # ECG: shape (n_samples,)
    ecg: Optional[np.ndarray] = None

    # PPG: shape (n_samples,)
    ppg: Optional[np.ndarray] = None

    # Pupil: shape (n_samples,) — resampled to EEG sfreq
    pupil: Optional[np.ndarray] = None

    # Behavioral
    recall_accuracy: float = np.nan
    recall_sequence: Optional[List[int]] = None

    # Quality flags
    eeg_valid: bool = True
    ecg_valid: bool = True
    ppg_valid: bool = True
    pupil_valid: bool = True
    pupil_valid_fraction: float = 1.0   # fraction of non-blink samples

    @property
    def is_complete(self) -> bool:
        """All four modalities valid."""
        return self.eeg_valid and self.ecg_valid and self.ppg_valid and self.pupil_valid

    @property
    def label(self) -> int:
        """Integer condition label (0=control, 1=5-digit, 2=9-digit, 3=13-digit)."""
        return {'control': 0, 'load_5': 1, 'load_9': 2, 'load_13': 3}.get(self.condition, -1)


# ── Resampling utilities ──────────────────────────────────────────────────────

def resample_signal(
    signal: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """
    Resample 1-D signal from orig_sfreq to target_sfreq.

    Uses polyphase filter (resample_poly) for integer ratios;
    falls back to linear interpolation for arbitrary ratios.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
    orig_sfreq : float
    target_sfreq : float

    Returns
    -------
    resampled : ndarray
    """
    if abs(orig_sfreq - target_sfreq) < 0.01:
        return signal  # no resampling needed

    # Compute integer ratio via GCD
    orig_int = int(round(orig_sfreq))
    tgt_int = int(round(target_sfreq))
    g = gcd(orig_int, tgt_int)
    up = tgt_int // g
    down = orig_int // g

    if up <= 100 and down <= 100:
        try:
            return resample_poly(signal, up, down).astype(signal.dtype)
        except Exception:
            pass

    # Fallback: linear interpolation
    n_orig = len(signal)
    t_orig = np.arange(n_orig) / orig_sfreq
    t_new = np.arange(0, t_orig[-1], 1.0 / target_sfreq)
    interp_fn = interp1d(t_orig, signal, kind='linear', bounds_error=False,
                         fill_value=(signal[0], signal[-1]))
    return interp_fn(t_new).astype(signal.dtype)


def resample_signal_2d(
    signal: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """
    Resample 2-D signal (n_channels, n_samples) row by row.

    Parameters
    ----------
    signal : ndarray, shape (n_channels, n_samples)

    Returns
    -------
    resampled : ndarray, shape (n_channels, n_new_samples)
    """
    rows = [resample_signal(signal[i], orig_sfreq, target_sfreq) for i in range(signal.shape[0])]
    # Trim to minimum length (rounding artifacts)
    min_len = min(len(r) for r in rows)
    return np.stack([r[:min_len] for r in rows], axis=0)


# ── Pupil blink interpolation ─────────────────────────────────────────────────

def interpolate_blinks(
    pupil: np.ndarray,
    sfreq: float,
    threshold_sd: float = 3.0,
    min_valid_fraction: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Detect and linearly interpolate blinks in pupil signal.

    Blink detection: samples where |pupil| > threshold_sd * std(pupil)
    OR pupil == 0 (missing data sentinel).

    Parameters
    ----------
    pupil : ndarray, shape (n_samples,)
    sfreq : float
    threshold_sd : float
        SD threshold for outlier detection.
    min_valid_fraction : float
        Minimum fraction of valid (non-blink) samples required.

    Returns
    -------
    pupil_clean : ndarray
        Blink-interpolated pupil signal.
    blink_mask : bool ndarray
        True where blink was detected.
    valid_fraction : float
        Fraction of samples that were valid.
    """
    pupil_clean = pupil.copy().astype(float)

    # Blink = zero/negative (missing) OR extreme outlier
    blink_mask = (pupil_clean <= 0)
    if pupil_clean[~blink_mask].size > 5:
        med = np.median(pupil_clean[~blink_mask])
        sd = pupil_clean[~blink_mask].std()
        if sd > 1e-10:
            blink_mask |= np.abs(pupil_clean - med) > threshold_sd * sd

    valid_fraction = 1.0 - blink_mask.mean()

    if valid_fraction < min_valid_fraction:
        return pupil_clean, blink_mask, valid_fraction

    # Linear interpolation across blink segments
    valid_indices = np.where(~blink_mask)[0]
    if len(valid_indices) < 2:
        return pupil_clean, blink_mask, valid_fraction

    interp_fn = interp1d(
        valid_indices,
        pupil_clean[valid_indices],
        kind='linear',
        bounds_error=False,
        fill_value=(pupil_clean[valid_indices[0]], pupil_clean[valid_indices[-1]])
    )
    all_indices = np.arange(len(pupil_clean))
    pupil_clean[blink_mask] = interp_fn(all_indices[blink_mask])

    return pupil_clean, blink_mask, valid_fraction


# ── Trial extractor ───────────────────────────────────────────────────────────

class TrialSynchronizer:
    """
    Extract and synchronize per-trial multimodal windows.

    Uses EEG event markers as the primary timeline.
    Pupil is resampled to EEG sfreq. ECG/PPG channels extracted from EEG raw.
    """

    # ECG/PPG channel name patterns in ds003838
    ECG_PATTERNS = ['ECG', 'ecg', 'EKG', 'ekg']
    PPG_PATTERNS = ['PPG', 'ppg', 'PLETH', 'pleth', 'Pleth']

    def __init__(self, cfg=None):
        from utils.config_loader import load_config
        self.cfg = cfg or load_config()

    def extract_trials(
        self,
        eeg_raw,                         # mne.io.BaseRaw
        pupil_df: pd.DataFrame,
        behavioral_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame],
        subject_id: str,
    ) -> List[SynchronizedTrial]:
        """
        Extract synchronized per-trial data for one subject.

        Parameters
        ----------
        eeg_raw : mne.io.BaseRaw
            Loaded, not-yet-preprocessed EEG (includes ECG/PPG channels).
        pupil_df : pd.DataFrame
            Raw pupil data with timestamp column.
        behavioral_df : pd.DataFrame
            Trial-level behavioral data.
        events_df : pd.DataFrame or None
            BIDS events TSV.
        subject_id : str

        Returns
        -------
        trials : list of SynchronizedTrial
        """
        import mne

        sfreq = eeg_raw.info['sfreq']
        ch_names = eeg_raw.ch_names

        # Identify ECG / PPG channel indices
        ecg_idx = self._find_channel_idx(ch_names, self.ECG_PATTERNS)
        ppg_idx = self._find_channel_idx(ch_names, self.PPG_PATTERNS)

        if ecg_idx is None:
            logger.warning(f"{subject_id}: No ECG channel found in {ch_names[:5]}...")
        if ppg_idx is None:
            logger.warning(f"{subject_id}: No PPG channel found")

        # Get MNE events from annotations or events file
        events, event_id = self._get_mne_events(eeg_raw, events_df)
        if events is None or len(events) == 0:
            logger.error(f"{subject_id}: No events found — cannot extract trials")
            return []

        # Parse pupil timestamps
        pupil_times, pupil_signal = self._parse_pupil_df(pupil_df)
        pupil_sfreq = self.cfg.pupil.expected_sfreq

        # Get raw data array (all channels)
        raw_data = eeg_raw.get_data()  # shape: (n_channels, n_samples)

        # EEG channel mask (exclude ECG/PPG/misc)
        eeg_channel_mask = self._get_eeg_channel_mask(ch_names, ecg_idx, ppg_idx)

        trials = []
        trial_events = self._parse_trial_events(events, event_id)

        for i, trial_info in enumerate(trial_events):
            onset_sample = trial_info['onset_sample']
            condition = trial_info['condition']
            n_digits = trial_info['n_digits']

            # Compute epoch window in samples
            tmin = self.cfg.eeg.epochs.encoding_tmin
            tmax = self.cfg.eeg.epochs.encoding_tmax
            start = int(onset_sample + tmin * sfreq)
            end = int(onset_sample + tmax * sfreq)

            if start < 0 or end > raw_data.shape[1]:
                logger.debug(f"Trial {i}: boundary out of range, skipping")
                continue

            times = np.arange(end - start) / sfreq + tmin

            # Extract EEG
            eeg_epoch = raw_data[eeg_channel_mask, start:end]

            # Extract ECG
            ecg_epoch = None
            if ecg_idx is not None:
                ecg_epoch = raw_data[ecg_idx, start:end]

            # Extract PPG
            ppg_epoch = None
            if ppg_idx is not None:
                ppg_epoch = raw_data[ppg_idx, start:end]

            # Extract and resample pupil
            pupil_epoch = None
            pupil_valid = False
            pupil_valid_frac = 0.0
            if pupil_times is not None and pupil_signal is not None:
                t_start = start / sfreq
                t_end = end / sfreq
                pupil_epoch, pupil_valid, pupil_valid_frac = self._extract_pupil_epoch(
                    pupil_times, pupil_signal, t_start, t_end,
                    target_sfreq=sfreq, target_n=end - start,
                    pupil_sfreq=pupil_sfreq,
                )

            # Get behavioral data for this trial
            recall_acc = self._get_recall_accuracy(behavioral_df, i, condition)

            trial = SynchronizedTrial(
                trial_idx=i,
                condition=condition,
                n_digits=n_digits,
                subject_id=subject_id,
                times=times,
                eeg=eeg_epoch,
                ecg=ecg_epoch,
                ppg=ppg_epoch,
                pupil=pupil_epoch,
                recall_accuracy=recall_acc,
                eeg_valid=eeg_epoch is not None,
                ecg_valid=ecg_epoch is not None,
                ppg_valid=ppg_epoch is not None,
                pupil_valid=pupil_valid,
                pupil_valid_fraction=pupil_valid_frac,
            )
            trials.append(trial)

        logger.info(
            f"{subject_id}: Extracted {len(trials)} trials "
            f"({sum(t.is_complete for t in trials)} complete)"
        )
        return trials

    def _find_channel_idx(self, ch_names: List[str], patterns: List[str]) -> Optional[int]:
        """Find first channel matching any pattern."""
        for i, name in enumerate(ch_names):
            for pat in patterns:
                if pat.lower() in name.lower():
                    return i
        return None

    def _get_eeg_channel_mask(
        self,
        ch_names: List[str],
        ecg_idx: Optional[int],
        ppg_idx: Optional[int],
    ) -> np.ndarray:
        """Boolean mask for EEG channels (exclude ECG/PPG/status)."""
        exclude = set()
        if ecg_idx is not None:
            exclude.add(ecg_idx)
        if ppg_idx is not None:
            exclude.add(ppg_idx)
        # Also exclude common non-EEG channel names
        non_eeg = ['Status', 'TRIG', 'Trig', 'STI', 'stim']
        for i, name in enumerate(ch_names):
            for pat in non_eeg:
                if pat.lower() in name.lower():
                    exclude.add(i)
        mask = np.array([i not in exclude for i in range(len(ch_names))])
        return mask

    def _get_mne_events(self, eeg_raw, events_df):
        """Extract MNE events array from raw or events_df."""
        import mne
        try:
            events, event_id = mne.events_from_annotations(eeg_raw, verbose=False)
            if len(events) > 0:
                return events, event_id
        except Exception:
            pass

        # Fallback: construct from events_df
        if events_df is not None and len(events_df) > 0:
            sfreq = eeg_raw.info['sfreq']
            events_list = []
            event_id = {}
            for _, row in events_df.iterrows():
                onset_sample = int(float(row.get('onset', 0)) * sfreq)
                trial_type = str(row.get('trial_type', 'unknown'))
                if trial_type not in event_id:
                    event_id[trial_type] = len(event_id) + 1
                events_list.append([onset_sample, 0, event_id[trial_type]])
            if events_list:
                return np.array(events_list), event_id

        return None, {}

    def _parse_trial_events(self, events: np.ndarray, event_id: dict) -> List[dict]:
        """
        Parse condition labels from MNE events.

        ds003838 event codes: 5 = 5-digit, 9 = 9-digit, 13 = 13-digit,
        control may be coded as 1 or annotated as 'listen'.
        Adapt based on actual event_id mapping.
        """
        # Build reverse map: code → label
        code_to_condition = {}
        for name, code in event_id.items():
            name_lower = name.lower()
            if '13' in name_lower or name_lower in ['13', 'memory_13', 'load_13']:
                code_to_condition[code] = ('load_13', 13)
            elif '9' in name_lower or name_lower in ['9', 'memory_9', 'load_9']:
                code_to_condition[code] = ('load_9', 9)
            elif '5' in name_lower or name_lower in ['5', 'memory_5', 'load_5']:
                code_to_condition[code] = ('load_5', 5)
            elif 'control' in name_lower or 'listen' in name_lower or 'rest' in name_lower:
                code_to_condition[code] = ('control', 0)

        # Direct integer code mapping (fallback)
        for code in [5, 9, 13, 1]:
            if code not in code_to_condition:
                if code == 13:
                    code_to_condition[code] = ('load_13', 13)
                elif code == 9:
                    code_to_condition[code] = ('load_9', 9)
                elif code == 5:
                    code_to_condition[code] = ('load_5', 5)
                elif code == 1:
                    code_to_condition[code] = ('control', 0)

        trial_events = []
        for event in events:
            onset_sample, _, code = event
            if code in code_to_condition:
                condition, n_digits = code_to_condition[code]
                trial_events.append({
                    'onset_sample': onset_sample,
                    'condition': condition,
                    'n_digits': n_digits,
                })
        return trial_events

    def _parse_pupil_df(
        self,
        pupil_df: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract timestamp and pupil diameter arrays from DataFrame.

        Tries common column name variants for ds003838.
        """
        if pupil_df is None or len(pupil_df) == 0:
            return None, None

        # Find timestamp column
        time_col = None
        for candidate in ['timestamp', 'time', 'onset', 't', 'sample']:
            if candidate in pupil_df.columns:
                time_col = candidate
                break

        # Find pupil diameter column
        pupil_col = None
        for candidate in ['pupil_left', 'pupil_right', 'diameter', 'pupil',
                          'left_pupil_diameter', 'right_pupil_diameter']:
            if candidate in pupil_df.columns:
                pupil_col = candidate
                break

        if time_col is None or pupil_col is None:
            logger.warning(f"Pupil columns not found. Available: {list(pupil_df.columns)}")
            return None, None

        times = pupil_df[time_col].values.astype(float)
        signal = pupil_df[pupil_col].values.astype(float)

        # Normalize timestamps to start at 0
        times = times - times[0]

        return times, signal

    def _extract_pupil_epoch(
        self,
        pupil_times: np.ndarray,
        pupil_signal: np.ndarray,
        t_start: float,
        t_end: float,
        target_sfreq: float,
        target_n: int,
        pupil_sfreq: float,
        blink_threshold_sd: float = 3.0,
        min_valid_fraction: float = 0.7,
    ) -> Tuple[Optional[np.ndarray], bool, float]:
        """
        Extract and resample pupil epoch aligned to [t_start, t_end].

        Returns (pupil_resampled, is_valid, valid_fraction).
        """
        # Find samples within window
        mask = (pupil_times >= t_start) & (pupil_times <= t_end)
        if mask.sum() < 10:
            return None, False, 0.0

        epoch_signal = pupil_signal[mask]

        # Interpolate blinks
        epoch_clean, _, valid_frac = interpolate_blinks(
            epoch_signal, pupil_sfreq,
            threshold_sd=blink_threshold_sd,
            min_valid_fraction=min_valid_fraction,
        )

        if valid_frac < min_valid_fraction:
            return epoch_clean, False, valid_frac

        # Resample to target_n samples (EEG epoch length)
        if len(epoch_clean) != target_n:
            epoch_clean = resample_signal(epoch_clean, pupil_sfreq, target_sfreq)
            # Trim/pad to exact length
            if len(epoch_clean) > target_n:
                epoch_clean = epoch_clean[:target_n]
            elif len(epoch_clean) < target_n:
                epoch_clean = np.pad(epoch_clean, (0, target_n - len(epoch_clean)),
                                     mode='edge')

        return epoch_clean, True, valid_frac

    def _get_recall_accuracy(
        self,
        behavioral_df: Optional[pd.DataFrame],
        trial_idx: int,
        condition: str,
    ) -> float:
        """Extract per-trial recall accuracy from behavioral DataFrame."""
        if behavioral_df is None or len(behavioral_df) == 0:
            return np.nan

        # Try direct index
        if trial_idx < len(behavioral_df):
            row = behavioral_df.iloc[trial_idx]
            for col in ['accuracy', 'recall_accuracy', 'correct', 'score', 'proportion_correct']:
                if col in behavioral_df.columns:
                    val = row.get(col, np.nan)
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass

        return np.nan


# ── Quality report ────────────────────────────────────────────────────────────

def compute_synchronization_report(trials: List[SynchronizedTrial]) -> dict:
    """
    Compute synchronization quality metrics across all trials.

    Returns dict with counts, valid fractions, per-condition breakdown.
    """
    if not trials:
        return {'n_trials': 0}

    conditions = ['control', 'load_5', 'load_9', 'load_13']
    report = {
        'n_trials': len(trials),
        'n_complete': sum(t.is_complete for t in trials),
        'fraction_complete': np.mean([t.is_complete for t in trials]),
        'pupil_valid_fraction_mean': np.nanmean([t.pupil_valid_fraction for t in trials]),
        'per_condition': {},
    }

    for cond in conditions:
        cond_trials = [t for t in trials if t.condition == cond]
        if not cond_trials:
            continue
        report['per_condition'][cond] = {
            'n': len(cond_trials),
            'n_complete': sum(t.is_complete for t in cond_trials),
            'mean_recall_accuracy': np.nanmean([t.recall_accuracy for t in cond_trials]),
        }

    return report