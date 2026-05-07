"""
preprocessing/eeg_prep.py
=========================
EEG preprocessing pipeline for ds003838.

Operations (in order):
1. Channel selection and type assignment
2. Bandpass filter (1-40 Hz)
3. Notch filter (50 Hz)
4. Average reference
5. ICA artifact removal (ocular + cardiac)
6. Epoch extraction (encoding / retention windows)
7. Epoch rejection (peak-to-peak amplitude)
8. Baseline correction

Inputs:  mne.io.BaseRaw (task + rest)
Outputs: mne.Epochs objects + preprocessed raw for resting-state

All steps logged. Each step individually callable for debugging.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger, timed

logger = setup_logger(__name__)


# ── Preprocessed output container ────────────────────────────────────────────

@dataclass
class EEGPreprocessed:
    """
    Output of full EEG preprocessing pipeline for one subject.

    Attributes
    ----------
    subject_id : str
    epochs_encoding : mne.Epochs
        Epochs for encoding window (digit presentation).
    epochs_retention : mne.Epochs
        Epochs for retention window.
    raw_rest_preprocessed : mne.io.BaseRaw
        Resting-state raw (filtered, referenced, ICA-cleaned).
    eeg_channel_names : list of str
        EEG channel names after channel selection.
    ica : mne.preprocessing.ICA or None
        Fitted ICA object for inspection.
    n_epochs_rejected : int
        Number of epochs rejected by peak-to-peak criterion.
    events : ndarray, shape (n_events, 3)
        MNE events array used for epoching.
    event_id : dict
        Event label → code mapping.
    """
    subject_id: str
    epochs_encoding: object = None          # mne.Epochs
    epochs_retention: object = None         # mne.Epochs
    raw_rest_preprocessed: object = None    # mne.io.BaseRaw
    eeg_channel_names: List[str] = field(default_factory=list)
    ica: object = None
    n_epochs_rejected: int = 0
    events: Optional[np.ndarray] = None
    event_id: Dict = field(default_factory=dict)
    sfreq: float = 500.0
    preprocessing_log: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.preprocessing_log.append(msg)
        logger.info(f"[{self.subject_id}] {msg}")


# ── Preprocessing pipeline ────────────────────────────────────────────────────

class EEGPreprocessor:
    """
    Full EEG preprocessing pipeline.

    Usage::

        preprocessor = EEGPreprocessor(cfg)
        result = preprocessor.preprocess(raw_task, raw_rest, subject_id)
        epochs = result.epochs_encoding
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()

    @timed(__name__)
    def preprocess(
        self,
        raw_task,                       # mne.io.BaseRaw — task recording
        raw_rest=None,                  # mne.io.BaseRaw — resting-state
        subject_id: str = "unknown",
    ) -> EEGPreprocessed:
        """
        Run full preprocessing pipeline on task (and optionally resting) EEG.

        Parameters
        ----------
        raw_task : mne.io.BaseRaw
        raw_rest : mne.io.BaseRaw, optional
        subject_id : str

        Returns
        -------
        EEGPreprocessed
        """
        import mne
        mne.set_log_level("WARNING")

        result = EEGPreprocessed(subject_id=subject_id)

        # ── Step 1: Channel selection ──
        raw_task = self._select_and_type_channels(raw_task, result)

        # ── Step 2: Bandpass filter ──
        raw_task = self._apply_bandpass(raw_task, result)

        # ── Step 3: Notch filter ──
        raw_task = self._apply_notch(raw_task, result)

        # ── Step 4: Average reference ──
        raw_task = self._apply_reference(raw_task, result)

        # ── Step 5: ICA ──
        ica = self._fit_ica(raw_task, result)
        if ica is not None:
            raw_task = self._apply_ica(raw_task, ica, result)
            result.ica = ica

        # ── Step 6: Extract events ──
        events, event_id = self._extract_events(raw_task, result)
        result.events = events
        result.event_id = event_id

        # ── Step 7: Epoch extraction ──
        result.epochs_encoding, n_rej = self._make_epochs(
            raw_task, events, event_id,
            tmin=self.cfg.eeg.epochs.encoding_tmin,
            tmax=self.cfg.eeg.epochs.encoding_tmax,
            label="encoding",
            result=result,
        )
        result.n_epochs_rejected = n_rej

        result.epochs_retention, _ = self._make_epochs(
            raw_task, events, event_id,
            tmin=self.cfg.eeg.epochs.retention_tmin,
            tmax=self.cfg.eeg.epochs.retention_tmax,
            label="retention",
            result=result,
        )

        # ── Step 8: Resting-state preprocessing ──
        if raw_rest is not None:
            result.raw_rest_preprocessed = self._preprocess_rest(raw_rest, result)

        result.eeg_channel_names = raw_task.ch_names
        result.sfreq = raw_task.info["sfreq"]

        result.log(f"Preprocessing complete. Epochs: {len(result.epochs_encoding) if result.epochs_encoding else 0}")
        return result

    # ── Channel selection ─────────────────────────────────────────────────────

    def _select_and_type_channels(self, raw, result: EEGPreprocessed):
        """
        Set channel types. ECG/PPG marked as 'ecg'/'misc' for ICA artifact detection.
        EEG channels kept as-is.
        """
        import mne

        # Map physiological channels
        ch_type_mapping = {}
        for name in raw.ch_names:
            name_lower = name.lower()
            if any(p in name_lower for p in ['ecg', 'ekg']):
                ch_type_mapping[name] = 'ecg'
            elif any(p in name_lower for p in ['ppg', 'pleth']):
                ch_type_mapping[name] = 'misc'
            elif any(p in name_lower for p in ['status', 'stim', 'trig']):
                ch_type_mapping[name] = 'stim'

        if ch_type_mapping:
            raw.set_channel_types(ch_type_mapping)
            result.log(f"Channel types set: {ch_type_mapping}")

        return raw

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _apply_bandpass(self, raw, result: EEGPreprocessed):
        """Bandpass filter: 1-40 Hz. Applied to EEG channels only."""
        cfg = self.cfg.eeg.filter
        picks = 'eeg'
        raw.filter(
            l_freq=cfg.l_freq,
            h_freq=cfg.h_freq,
            method=cfg.method,
            fir_window=cfg.fir_window,
            picks=picks,
            n_jobs=1,
            verbose=False,
        )
        result.log(f"Bandpass filter: {cfg.l_freq}-{cfg.h_freq} Hz")
        return raw

    def _apply_notch(self, raw, result: EEGPreprocessed):
        """Notch filter at powerline frequency (50 Hz for India)."""
        freqs = list(self.cfg.eeg.notch.freqs)
        raw.notch_filter(freqs=freqs, picks='eeg', verbose=False)
        result.log(f"Notch filter: {freqs} Hz")
        return raw

    def _apply_reference(self, raw, result: EEGPreprocessed):
        """Average reference. Standard for cognitive EEG."""
        import mne
        raw.set_eeg_reference('average', projection=False, verbose=False)
        result.log("Average reference applied")
        return raw

    # ── ICA ───────────────────────────────────────────────────────────────────

    def _fit_ica(self, raw, result: EEGPreprocessed):
        """
        Fit ICA on bandpass-filtered raw. Identify and flag:
        - Ocular artifacts: via EOG channels (Fp1/Fp2 proxies) or correlation
        - Cardiac artifacts: via ECG channel correlation
        """
        try:
            import mne
            from mne.preprocessing import ICA

            cfg = self.cfg.eeg.ica
            ica = ICA(
                n_components=cfg.n_components,
                method=cfg.method,
                random_state=cfg.random_state,
                max_iter=1000,
            )

            # Fit on EEG channels only
            ica.fit(raw, picks='eeg', verbose=False)

            # Find EOG components
            eog_chs = [c for c in cfg.eog_channels if c in raw.ch_names]
            eog_indices = []
            if eog_chs:
                try:
                    eog_indices, _ = ica.find_bads_eog(
                        raw, ch_name=eog_chs[0], verbose=False
                    )
                except Exception as e:
                    logger.debug(f"EOG ICA detection failed: {e}")

            # Find ECG components
            ecg_indices = []
            ecg_chs = [c for c in raw.ch_names if 'ecg' in c.lower() or 'ekg' in c.lower()]
            if ecg_chs:
                try:
                    ecg_indices, _ = ica.find_bads_ecg(
                        raw, ch_name=ecg_chs[0], verbose=False
                    )
                except Exception as e:
                    logger.debug(f"ECG ICA detection failed: {e}")

            ica.exclude = list(set(eog_indices + ecg_indices))
            result.log(
                f"ICA fitted. Excluding {len(ica.exclude)} components: "
                f"EOG={eog_indices}, ECG={ecg_indices}"
            )
            return ica

        except Exception as e:
            result.log(f"ICA failed: {e} — skipping ICA")
            return None

    def _apply_ica(self, raw, ica, result: EEGPreprocessed):
        """Apply ICA to remove identified artifact components."""
        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)
        result.log("ICA applied")
        return raw_clean

    # ── Events ────────────────────────────────────────────────────────────────

    def _extract_events(self, raw, result: EEGPreprocessed):
        """Extract MNE events from annotations or stim channel."""
        import mne

        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if len(events) > 0:
                result.log(f"Events from annotations: {len(events)} events, types={list(event_id.keys())}")
                return events, event_id
        except Exception:
            pass

        # Try stim channel
        try:
            stim_chs = [c for c in raw.ch_names if 'stim' in c.lower() or 'status' in c.lower()]
            if stim_chs:
                events = mne.find_events(raw, stim_channel=stim_chs[0], verbose=False)
                event_id = {str(int(e[2])): int(e[2]) for e in events}
                result.log(f"Events from stim channel '{stim_chs[0]}': {len(events)}")
                return events, event_id
        except Exception as e:
            logger.debug(f"Stim channel event extraction failed: {e}")

        result.log("WARNING: No events found — epoching will fail")
        return np.zeros((0, 3), dtype=int), {}

    # ── Epoching ──────────────────────────────────────────────────────────────

    def _make_epochs(
        self,
        raw,
        events: np.ndarray,
        event_id: dict,
        tmin: float,
        tmax: float,
        label: str,
        result: EEGPreprocessed,
    ) -> Tuple[object, int]:
        """
        Create MNE Epochs from events.

        Returns (epochs, n_rejected).
        """
        import mne

        if events is None or len(events) == 0:
            result.log(f"No events for {label} epochs")
            return None, 0

        # Filter event_id to valid condition codes
        valid_event_id = self._get_condition_event_id(event_id)
        if not valid_event_id:
            valid_event_id = event_id

        reject_thresh = self.cfg.eeg.epochs.reject_peak_to_peak
        baseline = tuple(self.cfg.eeg.epochs.baseline)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=valid_event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    reject={"eeg": reject_thresh},
                    picks="eeg",
                    preload=True,
                    verbose=False,
                    on_missing="warn",
                )
                n_total = len(epochs.events)
                epochs.drop_bad(verbose=False)
                n_kept = len(epochs)
                n_rejected = n_total - n_kept
                result.log(
                    f"{label} epochs: {n_kept}/{n_total} kept "
                    f"({n_rejected} rejected, tmin={tmin}, tmax={tmax})"
                )
                return epochs, n_rejected
            except Exception as e:
                result.log(f"Epoch extraction failed for {label}: {e}")
                return None, 0

    def _get_condition_event_id(self, event_id: dict) -> dict:
        """Map event_id keys to condition labels for MNE Epochs."""
        cfg_events = self.cfg.paradigm.event_ids
        valid = {}

        for name, code in event_id.items():
            name_lower = name.lower()
            if any(x in name_lower for x in ['13', 'load_13', 'memory_13']):
                valid[name] = code
            elif any(x in name_lower for x in ['9', 'load_9', 'memory_9']):
                valid[name] = code
            elif any(x in name_lower for x in ['5', 'load_5', 'memory_5']):
                valid[name] = code
            elif any(x in name_lower for x in ['control', 'listen', 'baseline']):
                valid[name] = code

        return valid

    # ── Resting-state ─────────────────────────────────────────────────────────

    def _preprocess_rest(self, raw_rest, result: EEGPreprocessed):
        """
        Preprocess resting-state EEG: filter + reference only.
        ICA not applied (rest used for spectral features only).
        """
        raw_rest = raw_rest.copy()
        self._select_and_type_channels(raw_rest, result)
        self._apply_bandpass(raw_rest, result)
        self._apply_notch(raw_rest, result)
        self._apply_reference(raw_rest, result)
        result.log("Resting-state EEG preprocessed")
        return raw_rest


# ── Batch preprocessing ───────────────────────────────────────────────────────

def preprocess_subjects_batch(
    loaded_data: dict,
    cfg: Optional[Config] = None,
    load_rest: bool = True,
) -> Dict[str, EEGPreprocessed]:
    """
    Preprocess EEG for multiple subjects.

    Parameters
    ----------
    loaded_data : dict
        subject_id → LoadedSubjectData

    Returns
    -------
    dict : subject_id → EEGPreprocessed
    """
    preprocessor = EEGPreprocessor(cfg)
    results = {}

    for sid, data in loaded_data.items():
        if data.eeg_raw is None:
            logger.warning(f"{sid}: no EEG data, skipping")
            continue
        try:
            result = preprocessor.preprocess(
                raw_task=data.eeg_raw,
                raw_rest=data.eeg_rest_raw if load_rest else None,
                subject_id=sid,
            )
            results[sid] = result
        except Exception as e:
            logger.error(f"{sid}: EEG preprocessing failed — {e}")

    logger.info(f"EEG preprocessing complete: {len(results)}/{len(loaded_data)} subjects")
    return results


if __name__ == "__main__":
    from data.loader import load_subjects_batch
    cfg = load_config()
    loaded = load_subjects_batch(cfg.dev_subjects[:2], cfg=cfg)
    results = preprocess_subjects_batch(loaded, cfg=cfg)
    for sid, r in results.items():
        print(f"{sid}: epochs={len(r.epochs_encoding) if r.epochs_encoding else 0}, "
              f"log_lines={len(r.preprocessing_log)}")