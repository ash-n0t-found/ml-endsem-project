"""
data/loader.py
==============
BIDS-aware data loader for ds003838.
Loads EEG (.fif/.vhdr/.set), ECG, PPG, pupillometry (.tsv), and
behavioral data (.tsv) for a given subject.

Handles:
- BIDS directory traversal
- Multi-modality file discovery
- Raw MNE loading (EEG)
- TSV loading (pupil, behavioral)
- Resting-state vs task file discrimination

All heavy computation (preprocessing, epoching) is deferred to
preprocessing/ modules. This module only loads raw data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SubjectRawData:
    """
    Container for all raw modality data for one subject.

    Attributes
    ----------
    subject_id : str
    eeg_task_path : Path or None
        Path to task EEG file (.vhdr / .set / .fif)
    eeg_rest_path : Path or None
        Path to resting-state EEG file
    ecg_ppg_path : Path or None
        Path to ECG/PPG file (may be embedded in EEG or separate .tsv)
    pupil_task_path : Path or None
        Path to task pupillometry .tsv
    pupil_rest_path : Path or None
        Path to resting pupillometry .tsv
    behavioral_path : Path or None
        Path to behavioral results .tsv
    events_path : Path or None
        Path to BIDS events .tsv (task onsets)
    sidecar_paths : dict
        JSON sidecar paths per modality
    """
    subject_id: str
    eeg_task_path: Optional[Path] = None
    eeg_rest_path: Optional[Path] = None
    ecg_ppg_path: Optional[Path] = None
    pupil_task_path: Optional[Path] = None
    pupil_rest_path: Optional[Path] = None
    behavioral_path: Optional[Path] = None
    events_path: Optional[Path] = None
    sidecar_paths: Dict[str, Path] = field(default_factory=dict)

    def modalities_available(self) -> List[str]:
        """Return list of available modality names."""
        available = []
        if self.eeg_task_path is not None:
            available.append("eeg")
        if self.ecg_ppg_path is not None or self.eeg_task_path is not None:
            # ECG/PPG often embedded in EEG file
            available.append("ecg_ppg")
        if self.pupil_task_path is not None:
            available.append("pupil")
        if self.behavioral_path is not None:
            available.append("behavioral")
        return available

    def is_complete(self, required: List[str] = None) -> bool:
        """Check all required modalities are present."""
        if required is None:
            required = ["eeg", "pupil", "behavioral"]
        return all(m in self.modalities_available() for m in required)


@dataclass
class LoadedSubjectData:
    """
    Container for loaded (in-memory) raw data for one subject.
    EEG stored as MNE Raw object; others as DataFrames or arrays.
    """
    subject_id: str
    eeg_raw: object = None           # mne.io.BaseRaw
    eeg_rest_raw: object = None      # mne.io.BaseRaw
    pupil_df: Optional[pd.DataFrame] = None
    pupil_rest_df: Optional[pd.DataFrame] = None
    behavioral_df: Optional[pd.DataFrame] = None
    events_df: Optional[pd.DataFrame] = None
    channel_names: List[str] = field(default_factory=list)
    sfreq_eeg: float = 0.0
    sfreq_pupil: float = 0.0


# ── BIDS directory traversal ──────────────────────────────────────────────────

class BIDSDatasetInspector:
    """
    Inspect BIDS directory structure for ds003838.
    Discovers file paths without loading data.
    """

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

    def find_subject_files(self, subject_id: str) -> SubjectRawData:
        """
        Discover all modality files for subject_id.

        ds003838 BIDS layout:
        sub-XXX/
          eeg/
            sub-XXX_task-memory_eeg.vhdr (or .set or .fif)
            sub-XXX_task-rest_eeg.vhdr
            sub-XXX_task-memory_events.tsv
          beh/
            sub-XXX_task-memory_beh.tsv
          eyetrack/  (or eeg/ with _et suffix)
            sub-XXX_task-memory_eyetrack.tsv
        """
        subj_dir = self.dataset_root / subject_id
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

        raw_data = SubjectRawData(subject_id=subject_id)

        # ── EEG files ──
        raw_data.eeg_task_path = self._find_eeg_file(subj_dir, task="memory")
        raw_data.eeg_rest_path = self._find_eeg_file(subj_dir, task="rest")

        # ── Events ──
        raw_data.events_path = self._find_tsv(subj_dir, suffix="events", task="memory")

        # ── Pupillometry ──
        raw_data.pupil_task_path = self._find_pupil_file(subj_dir, task="memory")
        raw_data.pupil_rest_path = self._find_pupil_file(subj_dir, task="rest")

        # ── Behavioral ──
        raw_data.behavioral_path = self._find_behavioral_file(subj_dir)

        # ── ECG/PPG — embedded in EEG or separate ──
        # In ds003838, ECG and PPG are channels within the EEG .vhdr file
        # Mark as available if EEG file exists
        if raw_data.eeg_task_path is not None:
            raw_data.ecg_ppg_path = raw_data.eeg_task_path  # embedded

        logger.debug(
            f"{subject_id}: EEG={raw_data.eeg_task_path is not None}, "
            f"REST={raw_data.eeg_rest_path is not None}, "
            f"PUPIL={raw_data.pupil_task_path is not None}, "
            f"BEH={raw_data.behavioral_path is not None}"
        )
        return raw_data

    def _find_eeg_file(self, subj_dir: Path, task: str) -> Optional[Path]:
        """Search for EEG file with given task label."""
        eeg_dir = subj_dir / "eeg"
        if not eeg_dir.exists():
            # Some BIDS datasets flatten into subject root
            eeg_dir = subj_dir

        for ext in [".vhdr", ".set", ".fif", ".edf"]:
            # Try standard BIDS naming
            pattern = f"*task-{task}*eeg{ext}"
            matches = list(eeg_dir.glob(pattern))
            if matches:
                return matches[0]
            # Fallback: any file with task name and extension
            pattern = f"*{task}*{ext}"
            matches = list(eeg_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _find_pupil_file(self, subj_dir: Path, task: str) -> Optional[Path]:
        """Search for pupillometry TSV."""
        for search_dir in [subj_dir / "eyetrack", subj_dir / "eeg", subj_dir]:
            if not search_dir.exists():
                continue
            for pattern in [
                f"*task-{task}*eyetrack*.tsv",
                f"*task-{task}*pupil*.tsv",
                f"*{task}*pupil*.tsv",
                f"*{task}*eye*.tsv",
            ]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def _find_behavioral_file(self, subj_dir: Path) -> Optional[Path]:
        """Search for behavioral TSV."""
        for search_dir in [subj_dir / "beh", subj_dir / "eeg", subj_dir]:
            if not search_dir.exists():
                continue
            for pattern in ["*beh*.tsv", "*behavior*.tsv", "*recall*.tsv"]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def _find_tsv(self, subj_dir: Path, suffix: str, task: str) -> Optional[Path]:
        """Generic TSV finder by BIDS suffix."""
        for search_dir in [subj_dir / "eeg", subj_dir]:
            if not search_dir.exists():
                continue
            pattern = f"*task-{task}*{suffix}*.tsv"
            matches = list(search_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def inspect_all_subjects(self, subject_ids: List[str]) -> Dict[str, SubjectRawData]:
        """Inspect all subjects; return dict keyed by subject_id."""
        results = {}
        for sid in subject_ids:
            try:
                results[sid] = self.find_subject_files(sid)
            except FileNotFoundError as e:
                logger.warning(f"Subject {sid} skipped: {e}")
        return results


# ── Data loader ───────────────────────────────────────────────────────────────

class SubjectDataLoader:
    """
    Load raw data into memory for a single subject.
    Uses MNE for EEG; pandas for TSV files.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()

    def load_subject(
        self,
        raw_data: SubjectRawData,
        load_rest: bool = True,
        preload_eeg: bool = True,
    ) -> LoadedSubjectData:
        """
        Load all available modalities for subject.

        Parameters
        ----------
        raw_data : SubjectRawData
        load_rest : bool
            Whether to load resting-state EEG.
        preload_eeg : bool
            Whether to preload EEG into memory (required for most processing).

        Returns
        -------
        LoadedSubjectData
        """
        loaded = LoadedSubjectData(subject_id=raw_data.subject_id)

        # ── EEG task ──
        if raw_data.eeg_task_path is not None:
            logger.info(f"{raw_data.subject_id}: Loading task EEG from {raw_data.eeg_task_path.name}")
            loaded.eeg_raw = self._load_eeg(raw_data.eeg_task_path, preload=preload_eeg)
            if loaded.eeg_raw is not None:
                loaded.sfreq_eeg = loaded.eeg_raw.info["sfreq"]
                loaded.channel_names = loaded.eeg_raw.ch_names

        # ── EEG resting ──
        if load_rest and raw_data.eeg_rest_path is not None:
            logger.info(f"{raw_data.subject_id}: Loading rest EEG from {raw_data.eeg_rest_path.name}")
            loaded.eeg_rest_raw = self._load_eeg(raw_data.eeg_rest_path, preload=preload_eeg)

        # ── Pupillometry ──
        if raw_data.pupil_task_path is not None:
            logger.info(f"{raw_data.subject_id}: Loading pupil data")
            loaded.pupil_df = self._load_pupil(raw_data.pupil_task_path)
            if loaded.pupil_df is not None:
                loaded.sfreq_pupil = self._estimate_pupil_sfreq(loaded.pupil_df)

        if load_rest and raw_data.pupil_rest_path is not None:
            loaded.pupil_rest_df = self._load_pupil(raw_data.pupil_rest_path)

        # ── Behavioral ──
        if raw_data.behavioral_path is not None:
            logger.info(f"{raw_data.subject_id}: Loading behavioral data")
            loaded.behavioral_df = self._load_behavioral(raw_data.behavioral_path)

        # ── Events ──
        if raw_data.events_path is not None:
            loaded.events_df = self._load_events_tsv(raw_data.events_path)

        return loaded

    def _load_eeg(self, path: Path, preload: bool = True) -> object:
        """Load EEG file using MNE. Returns mne.io.BaseRaw or None."""
        try:
            import mne
            mne.set_log_level("WARNING")
            ext = path.suffix.lower()
            if ext == ".vhdr":
                raw = mne.io.read_raw_brainvision(str(path), preload=preload, verbose=False)
            elif ext == ".set":
                raw = mne.io.read_raw_eeglab(str(path), preload=preload, verbose=False)
            elif ext in [".fif", ".fif.gz"]:
                raw = mne.io.read_raw_fif(str(path), preload=preload, verbose=False)
            elif ext == ".edf":
                raw = mne.io.read_raw_edf(str(path), preload=preload, verbose=False)
            else:
                logger.error(f"Unknown EEG format: {ext}")
                return None
            logger.debug(
                f"EEG loaded: {path.name} | sfreq={raw.info['sfreq']}Hz | "
                f"n_ch={len(raw.ch_names)} | duration={raw.times[-1]:.1f}s"
            )
            return raw
        except ImportError:
            logger.error("MNE not installed. Run: pip install mne")
            return None
        except Exception as e:
            logger.error(f"EEG load failed for {path}: {e}")
            return None

    def _load_pupil(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Load pupillometry TSV.
        Expected columns: timestamp, pupil_left or pupil_right, confidence.
        Actual column names vary by dataset — inspect and remap.
        """
        try:
            df = pd.read_csv(path, sep="\t", low_memory=False)
            logger.debug(f"Pupil TSV loaded: {path.name} | shape={df.shape} | cols={list(df.columns[:8])}")
            df = self._normalize_pupil_columns(df)
            return df
        except Exception as e:
            logger.error(f"Pupil load failed for {path}: {e}")
            return None

    def _normalize_pupil_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap pupil DataFrame columns to standard names.
        Standard: timestamp_s, pupil_diameter, confidence, is_blink
        """
        col_map = {}

        # Timestamp
        for c in ["timestamp", "time", "onset", "t"]:
            matches = [col for col in df.columns if c in col.lower()]
            if matches:
                col_map[matches[0]] = "timestamp_s"
                break

        # Pupil diameter — prefer right eye, fallback left
        for c in ["pupil_right", "right_pupil", "diameter_right", "pupil_r",
                  "pupil_left", "left_pupil", "diameter_left", "pupil_l",
                  "pupil", "diameter"]:
            matches = [col for col in df.columns if c.lower() in col.lower()]
            if matches:
                col_map[matches[0]] = "pupil_diameter"
                break

        # Confidence / validity
        for c in ["confidence", "validity", "valid"]:
            matches = [col for col in df.columns if c.lower() in col.lower()]
            if matches:
                col_map[matches[0]] = "confidence"
                break

        df = df.rename(columns=col_map)

        # Add is_blink column from confidence or from NaN
        if "confidence" in df.columns:
            df["is_blink"] = (df["confidence"] < 0.5).astype(bool)
        elif "pupil_diameter" in df.columns:
            df["is_blink"] = df["pupil_diameter"].isna()

        return df

    def _load_behavioral(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Load behavioral TSV with recall accuracy per trial.
        Expected columns: trial, condition, n_digits, recall_accuracy, ...
        """
        try:
            df = pd.read_csv(path, sep="\t", low_memory=False)
            logger.debug(f"Behavioral loaded: {path.name} | shape={df.shape}")
            df = self._normalize_behavioral_columns(df)
            return df
        except Exception as e:
            logger.error(f"Behavioral load failed for {path}: {e}")
            return None

    def _normalize_behavioral_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize behavioral column names."""
        # Map common variants to standard names
        renames = {}

        for c in df.columns:
            cl = c.lower()
            if any(x in cl for x in ["accuracy", "correct", "score", "recall"]):
                if "recall_accuracy" not in df.columns:
                    renames[c] = "recall_accuracy"
            elif any(x in cl for x in ["condition", "load", "n_digit", "ndigit"]):
                if "condition" not in df.columns:
                    renames[c] = "condition"
            elif any(x in cl for x in ["trial", "trial_n", "trial_num"]):
                if "trial_idx" not in df.columns:
                    renames[c] = "trial_idx"

        return df.rename(columns=renames)

    def _load_events_tsv(self, path: Path) -> Optional[pd.DataFrame]:
        """Load BIDS events TSV: onset, duration, trial_type columns."""
        try:
            df = pd.read_csv(path, sep="\t")
            logger.debug(f"Events loaded: {path.name} | n_events={len(df)}")
            return df
        except Exception as e:
            logger.error(f"Events load failed for {path}: {e}")
            return None

    def _estimate_pupil_sfreq(self, df: pd.DataFrame) -> float:
        """Estimate pupil sampling frequency from timestamp column."""
        if "timestamp_s" not in df.columns or len(df) < 10:
            return self.cfg.pupil.expected_sfreq

        timestamps = df["timestamp_s"].dropna().values[:100]
        if len(timestamps) < 2:
            return self.cfg.pupil.expected_sfreq

        diffs = np.diff(timestamps)
        median_dt = np.median(diffs[diffs > 0])
        if median_dt > 0:
            return float(1.0 / median_dt)
        return self.cfg.pupil.expected_sfreq


# ── Convenience batch loader ───────────────────────────────────────────────────

def load_subjects_batch(
    subject_ids: List[str],
    cfg: Optional[Config] = None,
    load_rest: bool = True,
    required_modalities: Optional[List[str]] = None,
) -> Dict[str, LoadedSubjectData]:
    """
    Load multiple subjects, skip those with missing required modalities.

    Parameters
    ----------
    subject_ids : list of str
    cfg : Config, optional
    load_rest : bool
    required_modalities : list of str, optional
        Default: ['eeg', 'pupil', 'behavioral']

    Returns
    -------
    dict : subject_id → LoadedSubjectData
    """
    if required_modalities is None:
        required_modalities = ["eeg", "pupil", "behavioral"]

    cfg = cfg or load_config()
    inspector = BIDSDatasetInspector(cfg.dataset_root)
    loader_ = SubjectDataLoader(cfg)

    loaded = {}
    for sid in subject_ids:
        try:
            raw_paths = inspector.find_subject_files(sid)
            if not raw_paths.is_complete(required_modalities):
                missing = [m for m in required_modalities if m not in raw_paths.modalities_available()]
                logger.warning(f"{sid}: skipped — missing modalities: {missing}")
                continue
            data = loader_.load_subject(raw_paths, load_rest=load_rest)
            loaded[sid] = data
            logger.info(f"{sid}: loaded successfully")
        except Exception as e:
            logger.error(f"{sid}: load failed — {e}")

    logger.info(f"Batch load complete: {len(loaded)}/{len(subject_ids)} subjects loaded")
    return loaded


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    inspector = BIDSDatasetInspector(cfg.dataset_root)

    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Dev subjects: {cfg.dev_subjects}")
    print()

    for sid in cfg.dev_subjects[:3]:
        try:
            raw = inspector.find_subject_files(sid)
            print(f"{sid}:")
            print(f"  EEG task:  {raw.eeg_task_path}")
            print(f"  EEG rest:  {raw.eeg_rest_path}")
            print(f"  Pupil:     {raw.pupil_task_path}")
            print(f"  Behavioral:{raw.behavioral_path}")
            print(f"  Complete:  {raw.is_complete()}")
        except FileNotFoundError as e:
            print(f"{sid}: {e}")
        print()