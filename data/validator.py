"""
data/validator.py
=================
Quality validation for loaded raw data.

Checks:
- Channel presence (EEG ROI channels, ECG, PPG)
- Sampling frequency consistency
- Pupil data completeness (blink rate, valid samples)
- Behavioral data completeness (trial counts, condition balance)
- Temporal alignment feasibility
- Event marker presence and counts

Produces a per-subject validation report and quality flags.
"""

from __future__ import annotations

from unittest import result
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.loader import LoadedSubjectData
from utils.config_loader import Config, load_config
from utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ── Validation result container ───────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Per-subject validation outcome.

    Attributes
    ----------
    subject_id : str
    passed : bool
        Overall pass/fail (all critical checks passed).
    warnings : list of str
        Non-critical issues (data usable but imperfect).
    errors : list of str
        Critical issues (subject should be excluded).
    quality_flags : dict
        Quantitative quality metrics.
    """
    subject_id: str
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    quality_flags: Dict = field(default_factory=dict)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"[{self.subject_id}] WARNING: {msg}")

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error(f"[{self.subject_id}] ERROR: {msg}")

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.subject_id} [{status}] | "
            f"errors={len(self.errors)} | warnings={len(self.warnings)}"
        )


# ── Validator ─────────────────────────────────────────────────────────────────

class SubjectValidator:
    """
    Validate a LoadedSubjectData object.
    Returns ValidationResult with pass/fail and quality metrics.
    """

    # Required ROI channels that must be present for EEG features
    REQUIRED_EEG_CHANNELS = [
        "Fz", "FCz", "Cz",          # frontal midline theta ROI
        "Pz", "P3", "P4",           # parietal alpha / P300
        "O1", "Oz", "O2",           # occipital alpha
        "FC1", "FC2",               # N200
    ]

    # Channels that carry ECG/PPG in ds003838 BrainVision files
    EXPECTED_PHYSIO_CHANNELS = ["ECG", "PPG"]  # exact names may vary — see below

    # Minimum valid pupil sample proportion
    MIN_PUPIL_VALID_PROP = 0.70

    # Expected trial counts per condition (rough — allow ±20%)
    EXPECTED_TRIALS_PER_CONDITION = {
        "control": 27,
        "load_5": 36,
        "load_9": 36,
        "load_13": 36,
    }

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()

    def validate(self, data: LoadedSubjectData) -> ValidationResult:
        """
        Run all validation checks on loaded subject data.

        Parameters
        ----------
        data : LoadedSubjectData

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(subject_id=data.subject_id)

        # ── EEG checks ──
        if data.eeg_raw is not None:
            self._check_eeg_channels(data, result)
            self._check_eeg_sfreq(data, result)
            self._check_eeg_duration(data, result)
            self._check_physio_channels(data, result)
        else:
            result.add_error("EEG task data not loaded")

        # ── Resting EEG ──
        if data.eeg_rest_raw is None:
            result.add_warning("Resting-state EEG not available — resting GGM will be skipped")

        # ── Pupil checks ──
        if data.pupil_df is not None:
            self._check_pupil_quality(data, result)
        else:
            result.add_error("Pupil data not loaded")

        # ── Behavioral checks ──
        if data.behavioral_df is not None:
            self._check_behavioral(data, result)
        else:
            result.add_error("Behavioral data not loaded")

        # ── Events checks ──
        if data.events_df is not None:
            self._check_events(data, result)
        else:
            result.add_warning("Events TSV not found — will extract events from EEG stim channel")

        # ── Cross-modality alignment check ──
        self._check_temporal_alignment(data, result)

        logger.info(result.summary())
        return result

    # ── EEG checks ────────────────────────────────────────────────────────────

    def _check_eeg_channels(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Verify required ROI channels present."""
        ch_names_upper = [c.upper() for c in data.channel_names]
        missing = []
        for ch in self.REQUIRED_EEG_CHANNELS:
            if ch.upper() not in ch_names_upper:
                missing.append(ch)
        if missing:
            result.add_warning(f"Missing EEG ROI channels: {missing} — feature extraction may degrade")
        result.quality_flags["n_eeg_channels"] = len(data.channel_names)
        result.quality_flags["missing_roi_channels"] = missing

    def _check_eeg_sfreq(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Verify EEG sampling frequency matches expected."""
        expected = self.cfg.eeg.expected_sfreq
        actual = data.sfreq_eeg
        if abs(actual - expected) > 1.0:
            result.add_warning(
                f"EEG sfreq={actual}Hz differs from expected={expected}Hz — "
                "update config or resample"
            )
        result.quality_flags["sfreq_eeg"] = actual

    def _check_eeg_duration(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Verify EEG recording is reasonably long."""
        if data.eeg_raw is None:
            return
        duration_s = data.eeg_raw.times[-1]
        result.quality_flags["eeg_duration_s"] = duration_s
        if duration_s < 300:  # less than 5 minutes is suspicious
            result.add_warning(f"Short EEG recording: {duration_s:.0f}s (expected > 300s)")

    def _check_physio_channels(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Check ECG and PPG channels are present in EEG file."""
        if data.eeg_raw is None:
            return
        ch_names_upper = [c.upper() for c in data.channel_names]
        found_physio = []
        for ch in self.EXPECTED_PHYSIO_CHANNELS:
            # Fuzzy match — ECG might be named "ECG", "EKG", etc.
            variants = [ch, ch.replace("ECG", "EKG")]
            if any(v.upper() in ch_names_upper or
                   any(v.upper() in c.upper() for c in ch_names_upper)
                   for v in variants):
                found_physio.append(ch)

        result.quality_flags["physio_channels_found"] = found_physio
        missing_physio = [c for c in self.EXPECTED_PHYSIO_CHANNELS if c not in found_physio]
        if missing_physio:
            result.add_warning(
                f"Physio channels not found in EEG file: {missing_physio} — "
                "ECG/PPG features may be unavailable"
            )

    # ── Pupil checks ──────────────────────────────────────────────────────────

    def _check_pupil_quality(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Check pupil data completeness and blink rate."""
        df = data.pupil_df
        n_samples = len(df)
        result.quality_flags["n_pupil_samples"] = n_samples

        if n_samples < 100:
            result.add_error(f"Pupil data too short: {n_samples} samples")
            return

        # Blink / invalid sample rate
        if "is_blink" in df.columns:
            blink_rate = df["is_blink"].mean()
            valid_prop = 1.0 - blink_rate
            result.quality_flags["pupil_valid_proportion"] = valid_prop
            result.quality_flags["pupil_blink_rate"] = blink_rate

            if valid_prop < self.MIN_PUPIL_VALID_PROP:
                result.add_error(
                    f"Pupil valid proportion={valid_prop:.2f} < threshold={self.MIN_PUPIL_VALID_PROP}"
                )
            elif valid_prop < 0.80:
                result.add_warning(f"Pupil valid proportion={valid_prop:.2f} (marginal)")

        # Check sampling frequency
        if data.sfreq_pupil > 0:
            expected_pupil = self.cfg.pupil.expected_sfreq
            if abs(data.sfreq_pupil - expected_pupil) > 5.0:
                result.add_warning(
                    f"Pupil sfreq={data.sfreq_pupil:.1f}Hz differs from expected={expected_pupil}Hz"
                )
            result.quality_flags["sfreq_pupil"] = data.sfreq_pupil

        # Check for diameter column
        if "pupil_diameter" not in df.columns:
            result.add_error("Pupil diameter column not found after normalization")

    # ── Behavioral checks ─────────────────────────────────────────────────────

    def _check_behavioral(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Check behavioral data: trial counts, recall accuracy range, condition balance."""
        df = data.behavioral_df
        result.quality_flags["n_behavioral_rows"] = len(df)

        # Check recall accuracy column
        if "recall_accuracy" not in df.columns:
            result.add_warning(
                "recall_accuracy column not found — "
                f"available columns: {list(df.columns)}"
            )
            return

        # Extract recall_accuracy as a clean 1-D Series.
        # Guard against duplicate columns (e.g. from malformed TSVs) which
        # cause df["recall_accuracy"] to return a DataFrame instead of a Series.
        raw_col = df["recall_accuracy"]
        if isinstance(raw_col, pd.DataFrame):
            # Duplicate column present — take first occurrence and warn
            result.add_warning(
                "Duplicate 'recall_accuracy' columns detected in behavioral data — "
                "using first occurrence"
            )
            raw_col = raw_col.iloc[:, 0]

        acc = pd.to_numeric(raw_col, errors="coerce").dropna()

        result.quality_flags["mean_recall_accuracy"] = float(acc.mean()) if len(acc) > 0 else float("nan")
        result.quality_flags["recall_accuracy_range"] = (
            (float(acc.min()), float(acc.max())) if len(acc) > 0 else (float("nan"), float("nan"))
        )

        # Sanity: accuracy should be in [0, 1] or [0, 100]
        if len(acc) > 0 and acc.max() > 1.5:
            result.add_warning("Recall accuracy appears to be percentage (>1.0) — normalizing by /100")

        # Check condition column
        if "condition" in df.columns:
            condition_counts = df["condition"].value_counts()
            result.quality_flags["trial_counts_per_condition"] = condition_counts.to_dict()
            logger.debug(f"{data.subject_id}: condition trial counts = {condition_counts.to_dict()}")
        else:
            result.add_warning("Condition column not found in behavioral data")

        # Check for any trials with NaN accuracy
        n_nan = raw_col.isna().sum()
        if n_nan > 0:
            result.add_warning(f"{n_nan} trials have NaN recall accuracy")

    # ── Events checks ─────────────────────────────────────────────────────────

    def _check_events(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """Check events TSV for required columns and trial structure."""
        df = data.events_df
        required_cols = ["onset", "duration", "trial_type"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            result.add_warning(f"Events TSV missing columns: {missing_cols}")
            return

        n_events = len(df)
        result.quality_flags["n_events"] = n_events
        event_types = df["trial_type"].value_counts().to_dict() if "trial_type" in df.columns else {}
        result.quality_flags["event_types"] = event_types
        logger.debug(f"{data.subject_id}: event types = {event_types}")

    # ── Temporal alignment check ──────────────────────────────────────────────

    def _check_temporal_alignment(self, data: LoadedSubjectData, result: ValidationResult) -> None:
        """
        Check feasibility of cross-modality temporal alignment.
        Verifies EEG and pupil recordings overlap in time.
        """
        if data.eeg_raw is None or data.pupil_df is None:
            return

        eeg_duration = data.eeg_raw.times[-1]

        # Pupil duration from timestamps
        if "timestamp_s" in data.pupil_df.columns:
            pupil_ts = data.pupil_df["timestamp_s"].dropna()
            pupil_duration = float(pupil_ts.max() - pupil_ts.min()) if len(pupil_ts) > 1 else 0
            result.quality_flags["pupil_duration_s"] = pupil_duration

            # Check for reasonable overlap (not identical timing systems expected)
            duration_ratio = pupil_duration / max(eeg_duration, 1)
            result.quality_flags["eeg_pupil_duration_ratio"] = duration_ratio

            if duration_ratio < 0.5:
                result.add_warning(
                    f"Pupil duration ({pupil_duration:.0f}s) << EEG duration ({eeg_duration:.0f}s) — "
                    "alignment may be problematic"
                )


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_subjects_batch(
    loaded_data: Dict[str, LoadedSubjectData],
    cfg: Optional[Config] = None,
) -> Tuple[Dict[str, ValidationResult], List[str], List[str]]:
    """
    Validate all loaded subjects.

    Parameters
    ----------
    loaded_data : dict
        subject_id → LoadedSubjectData
    cfg : Config, optional

    Returns
    -------
    results : dict
        subject_id → ValidationResult
    passed_ids : list of str
        Subject IDs that passed all critical checks.
    failed_ids : list of str
        Subject IDs that failed at least one critical check.
    """
    validator = SubjectValidator(cfg)
    results = {}
    passed_ids = []
    failed_ids = []

    for sid, data in loaded_data.items():
        result = validator.validate(data)
        results[sid] = result
        if result.passed:
            passed_ids.append(sid)
        else:
            failed_ids.append(sid)

    logger.info(
        f"Validation complete: {len(passed_ids)} passed, {len(failed_ids)} failed"
    )
    if failed_ids:
        logger.warning(f"Failed subjects: {failed_ids}")

    return results, passed_ids, failed_ids


def print_validation_report(results: Dict[str, ValidationResult]) -> None:
    """Print formatted validation report to stdout."""
    print("\n" + "=" * 70)
    print("SUBJECT VALIDATION REPORT")
    print("=" * 70)

    passed = [r for r in results.values() if r.passed]
    failed = [r for r in results.values() if not r.passed]

    print(f"Total: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")
    print()

    for result in results.values():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} {result.subject_id}")
        for e in result.errors:
            print(f"       ERROR: {e}")
        for w in result.warnings:
            print(f"       WARN:  {w}")

    print("=" * 70)


if __name__ == "__main__":
    from data.loader import load_subjects_batch

    cfg = load_config()
    loaded = load_subjects_batch(cfg.dev_subjects[:3], cfg=cfg)
    results, passed, failed = validate_subjects_batch(loaded, cfg=cfg)
    print_validation_report(results)