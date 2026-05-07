"""
scripts/00_verify_data.py
=========================
Dataset integrity verification for ds003838 — Phase 0 of the multimodal
physiological coupling framework.

Verifies the development subject set (default: 10 subjects) against the
current repo architecture before any preprocessing or modeling proceeds.

What this script validates
--------------------------
1. BIDS directory structure — all expected modality files discoverable
2. Modality file completeness — EEG task, resting EEG, pupil, behavioral
3. Raw data loadability — EEG opens without error; TSVs parse correctly
4. EEG quality — channel count, ROI channels present, sampling freq, duration
5. Physio channels — ECG / PPG embedded in EEG file
6. Pupil quality — valid sample proportion above threshold, diameter column
7. Behavioral completeness — recall_accuracy column, condition trial counts
8. Events — BIDS events TSV present and parseable
9. Temporal alignment feasibility — EEG/pupil duration ratio check
10. Synchronization report — per-trial completeness across modalities

Outputs (under <output_root>/verification/)
--------------------------------------------
- 00_verification_details.json   — per-subject full ValidationResult dicts
- 00_sync_reports.json           — per-subject synchronization quality dicts
- 00_data_verification_summary.csv — wide-format QC table (one row per subject)
- 00_verification_overview.png   — QC heatmap + trial count bar chart (--plot)
- 00_verify_data.log             — full run log

Usage
-----
    # Dev subset (default: 10 subjects from config)
    python scripts/00_verify_data.py

    # Specific subjects
    python scripts/00_verify_data.py --subjects sub-032 sub-033 sub-034

    # Full clean subset (slow — loads all 53 subjects)
    python scripts/00_verify_data.py --all

    # With QC plots
    python scripts/00_verify_data.py --plot

    # Override output directory
    python scripts/00_verify_data.py --output-dir /tmp/verify_out

Notes
-----
- Runs on the real dataset at cfg.dataset_root — requires files to be
  present locally.
- MNE is imported lazily; if not installed the EEG load steps are skipped
  with an error flag (other checks still run).
- Sub-032 is the Paper 4 reference subject (BrainBeats validation anchor);
  it must pass all checks for downstream replication to be credible.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from utils.io_utils import setup_logger, save_json, ensure_dir
from utils.config_loader import load_config, Config
from data.loader import (
    BIDSDatasetInspector,
    SubjectDataLoader,
    SubjectRawData,
    LoadedSubjectData,
    load_subjects_batch,
)
from data.validator import (
    SubjectValidator,
    ValidationResult,
    validate_subjects_batch,
    print_validation_report,
)
from data.synchronizer import TrialSynchronizer, compute_synchronization_report

# ── Module logger (configured after output dir is known) ─────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 0 — ds003838 dataset integrity verification.\n"
            "Uses the current repo architecture: BIDSDatasetInspector, "
            "SubjectDataLoader, SubjectValidator, TrialSynchronizer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUB_ID",
        help="Subject IDs to verify. Overrides --all and config dev subset.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Verify all 53 clean subjects (slow — loads every file).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Override output directory (default: <output_root>/verification).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate QC overview plot (requires matplotlib).",
    )
    p.add_argument(
        "--no-load",
        action="store_true",
        help=(
            "Inspection-only mode: discover file paths but do NOT load data. "
            "Fast check for BIDS structure without reading large EEG files."
        ),
    )
    p.add_argument(
        "--sync-check",
        action="store_true",
        help=(
            "Run TrialSynchronizer on each subject to verify trial extraction. "
            "Slower — requires full EEG preload."
        ),
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Inspection-only path report (fast, no data loading)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_subject_paths(
    subject_id: str,
    inspector: BIDSDatasetInspector,
) -> dict:
    """
    Discover file paths for one subject without loading any data.
    Returns a dict summarising what was found.
    """
    try:
        raw_data: SubjectRawData = inspector.find_subject_files(subject_id)
    except FileNotFoundError as e:
        return {
            "subject": subject_id,
            "directory_exists": False,
            "error": str(e),
            "eeg_task": None,
            "eeg_rest": None,
            "pupil_task": None,
            "behavioral": None,
            "events": None,
            "ecg_ppg_embedded": False,
            "modalities_available": [],
            "is_complete": False,
        }

    return {
        "subject": subject_id,
        "directory_exists": True,
        "error": None,
        "eeg_task": str(raw_data.eeg_task_path) if raw_data.eeg_task_path else None,
        "eeg_rest": str(raw_data.eeg_rest_path) if raw_data.eeg_rest_path else None,
        "pupil_task": str(raw_data.pupil_task_path) if raw_data.pupil_task_path else None,
        "behavioral": str(raw_data.behavioral_path) if raw_data.behavioral_path else None,
        "events": str(raw_data.events_path) if raw_data.events_path else None,
        "ecg_ppg_embedded": (raw_data.ecg_ppg_path is not None),
        "modalities_available": raw_data.modalities_available(),
        "is_complete": raw_data.is_complete(),
    }


def run_inspection_pass(
    subjects: List[str],
    inspector: BIDSDatasetInspector,
) -> List[dict]:
    """Run path-only inspection for all subjects. Returns list of path dicts."""
    logger.info(f"Running inspection-only pass on {len(subjects)} subjects")
    results = []
    for sid in subjects:
        r = inspect_subject_paths(sid, inspector)
        status = "OK" if r["is_complete"] else ("DIR_MISSING" if not r["directory_exists"] else "INCOMPLETE")
        logger.info(
            f"{sid}: {status} | "
            f"eeg={'Y' if r['eeg_task'] else 'N'} "
            f"rest={'Y' if r['eeg_rest'] else 'N'} "
            f"pupil={'Y' if r['pupil_task'] else 'N'} "
            f"beh={'Y' if r['behavioral'] else 'N'}"
        )
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Synchronization check
# ─────────────────────────────────────────────────────────────────────────────

def run_sync_check(
    subject_id: str,
    loaded: LoadedSubjectData,
    synchronizer: TrialSynchronizer,
) -> dict:
    """
    Extract trials via TrialSynchronizer and compute synchronization report.
    Returns the compute_synchronization_report dict, extended with subject_id.

    If critical data is missing (no EEG raw, no pupil, no behavioral), returns
    an error report without attempting extraction.
    """
    if loaded.eeg_raw is None:
        return {"subject": subject_id, "error": "EEG raw not available", "n_trials": 0}
    if loaded.pupil_df is None:
        return {"subject": subject_id, "error": "Pupil DataFrame not available", "n_trials": 0}
    if loaded.behavioral_df is None:
        return {"subject": subject_id, "error": "Behavioral DataFrame not available", "n_trials": 0}

    try:
        trials = synchronizer.extract_trials(
            eeg_raw=loaded.eeg_raw,
            pupil_df=loaded.pupil_df,
            behavioral_df=loaded.behavioral_df,
            events_df=loaded.events_df,
            subject_id=subject_id,
        )
        report = compute_synchronization_report(trials)
        report["subject"] = subject_id
        report["error"] = None
        logger.info(
            f"{subject_id}: sync report — "
            f"n_trials={report['n_trials']} | "
            f"complete={report['n_complete']} | "
            f"frac={report['fraction_complete']:.2f}"
        )
        return report
    except Exception as e:
        logger.error(f"{subject_id}: synchronization check failed — {e}", exc_info=True)
        return {"subject": subject_id, "error": str(e), "n_trials": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Summary table construction
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_df(
    path_reports: List[dict],
    validation_results: Optional[Dict[str, ValidationResult]],
    sync_reports: Optional[Dict[str, dict]],
) -> pd.DataFrame:
    """
    Assemble a wide-format QC summary table — one row per subject.

    Columns:
    - From path inspection: file presence flags
    - From validation: EEG/pupil/behavioral quality flags
    - From sync reports: trial counts and completeness fractions
    """
    rows = []
    for pr in path_reports:
        sid = pr["subject"]
        row: dict = {
            "subject": sid,
            # Path inspection
            "dir_exists": pr.get("directory_exists", False),
            "eeg_task_found": pr.get("eeg_task") is not None,
            "eeg_rest_found": pr.get("eeg_rest") is not None,
            "pupil_found": pr.get("pupil_task") is not None,
            "behavioral_found": pr.get("behavioral") is not None,
            "events_found": pr.get("events") is not None,
            "ecg_ppg_embedded": pr.get("ecg_ppg_embedded", False),
            "bids_complete": pr.get("is_complete", False),
        }

        # Validation results
        if validation_results and sid in validation_results:
            vr: ValidationResult = validation_results[sid]
            qf = vr.quality_flags
            row.update({
                "validation_passed": vr.passed,
                "n_errors": len(vr.errors),
                "n_warnings": len(vr.warnings),
                "n_eeg_channels": qf.get("n_eeg_channels", np.nan),
                "sfreq_eeg": qf.get("sfreq_eeg", np.nan),
                "eeg_duration_s": qf.get("eeg_duration_s", np.nan),
                "pupil_valid_prop": qf.get("pupil_valid_proportion", np.nan),
                "pupil_blink_rate": qf.get("pupil_blink_rate", np.nan),
                "n_pupil_samples": qf.get("n_pupil_samples", np.nan),
                "mean_recall_accuracy": qf.get("mean_recall_accuracy", np.nan),
                "n_behavioral_rows": qf.get("n_behavioral_rows", np.nan),
                "n_events": qf.get("n_events", np.nan),
                "physio_found": str(qf.get("physio_channels_found", [])),
                "eeg_pupil_duration_ratio": qf.get("eeg_pupil_duration_ratio", np.nan),
            })
        else:
            row.update({
                "validation_passed": False,
                "n_errors": np.nan,
                "n_warnings": np.nan,
                "n_eeg_channels": np.nan,
                "sfreq_eeg": np.nan,
                "eeg_duration_s": np.nan,
                "pupil_valid_prop": np.nan,
                "pupil_blink_rate": np.nan,
                "n_pupil_samples": np.nan,
                "mean_recall_accuracy": np.nan,
                "n_behavioral_rows": np.nan,
                "n_events": np.nan,
                "physio_found": "[]",
                "eeg_pupil_duration_ratio": np.nan,
            })

        # Sync reports
        if sync_reports and sid in sync_reports:
            sr = sync_reports[sid]
            row.update({
                "sync_n_trials": sr.get("n_trials", 0),
                "sync_n_complete": sr.get("n_complete", 0),
                "sync_fraction_complete": sr.get("fraction_complete", np.nan),
                "sync_pupil_valid_mean": sr.get("pupil_valid_fraction_mean", np.nan),
                "sync_n_control": sr.get("per_condition", {}).get("control", {}).get("n", 0),
                "sync_n_load5": sr.get("per_condition", {}).get("load_5", {}).get("n", 0),
                "sync_n_load9": sr.get("per_condition", {}).get("load_9", {}).get("n", 0),
                "sync_n_load13": sr.get("per_condition", {}).get("load_13", {}).get("n", 0),
                "sync_recall_control": sr.get("per_condition", {}).get("control", {}).get("mean_recall_accuracy", np.nan),
                "sync_recall_load5": sr.get("per_condition", {}).get("load_5", {}).get("mean_recall_accuracy", np.nan),
                "sync_recall_load9": sr.get("per_condition", {}).get("load_9", {}).get("mean_recall_accuracy", np.nan),
                "sync_recall_load13": sr.get("per_condition", {}).get("load_13", {}).get("mean_recall_accuracy", np.nan),
                "sync_error": str(sr.get("error") or ""),
            })
        else:
            row.update({
                "sync_n_trials": np.nan,
                "sync_n_complete": np.nan,
                "sync_fraction_complete": np.nan,
                "sync_pupil_valid_mean": np.nan,
                "sync_n_control": np.nan,
                "sync_n_load5": np.nan,
                "sync_n_load9": np.nan,
                "sync_n_load13": np.nan,
                "sync_recall_control": np.nan,
                "sync_recall_load5": np.nan,
                "sync_recall_load9": np.nan,
                "sync_recall_load13": np.nan,
                "sync_error": "",
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_verification_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """
    QC overview figure — two panels:
    Left:  Heatmap of pass/fail per modality/check per subject.
    Right: Synchronised trial counts per condition per subject.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    # Boolean columns for heatmap (only columns that exist in df)
    check_cols_candidates = [
        ("eeg_task_found", "EEG task"),
        ("eeg_rest_found", "EEG rest"),
        ("pupil_found", "Pupil"),
        ("behavioral_found", "Behavioral"),
        ("ecg_ppg_embedded", "ECG/PPG"),
        ("validation_passed", "Valid"),
    ]
    check_cols = [(c, lbl) for c, lbl in check_cols_candidates if c in df.columns]
    cols = [c for c, _ in check_cols]
    labels = [lbl for _, lbl in check_cols]

    n_subs = len(df)
    sub_labels = df["subject"].str.replace("sub-", "", regex=False).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_subs * 0.55), 6))

    # ── Left: QC heatmap ──
    ax = axes[0]
    if cols:
        grid = df[cols].fillna(False).astype(int).values  # (n_subs, n_checks)
        im = ax.imshow(grid.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_subs))
        ax.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title("Modality & Validation Checks\n(green=pass, red=fail)", fontsize=11, fontweight="bold")
        # Annotate cells
        for xi in range(n_subs):
            for yi in range(len(labels)):
                val = grid[xi, yi]
                ax.text(xi, yi, "✓" if val else "✗", ha="center", va="center",
                        fontsize=7, color="white" if val else "black")
    else:
        ax.text(0.5, 0.5, "No check columns available", ha="center", va="center")

    # ── Right: Trial counts per condition ──
    ax2 = axes[1]
    trial_cols = {
        "sync_n_control": ("#90be6d", "Control"),
        "sync_n_load5": ("#4dac26", "5-digit"),
        "sync_n_load9": ("#f1a340", "9-digit"),
        "sync_n_load13": ("#d6604d", "13-digit"),
    }
    available_trial_cols = [(c, col, lbl) for c, (col, lbl) in
                             zip(range(len(trial_cols)), trial_cols.items())
                             if lbl in [v[1] for v in trial_cols.values()]]

    x = np.arange(n_subs)
    n_conds = sum(1 for c in trial_cols if c in df.columns)
    if n_conds > 0:
        width = 0.8 / n_conds
        offset = -0.4 + width / 2
        for col_name, (color, cond_label) in trial_cols.items():
            if col_name not in df.columns:
                continue
            vals = pd.to_numeric(df[col_name], errors="coerce").fillna(0).values
            ax2.bar(x + offset, vals, width, label=cond_label, color=color, alpha=0.85)
            offset += width
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("N synchronized trials", fontsize=10)
    ax2.set_title("Synchronized Trial Counts per Condition", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"ds003838 Phase-0 Verification — {n_subs} subjects",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "00_verification_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"QC plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_terminal_summary(
    df: pd.DataFrame,
    path_reports: List[dict],
    validation_results: Optional[Dict[str, ValidationResult]],
    sync_reports: Optional[Dict[str, dict]],
) -> None:
    """Print a concise, human-readable summary to stdout."""
    n = len(df)
    print("\n" + "=" * 68)
    print("  ds003838 — PHASE 0 DATA VERIFICATION SUMMARY")
    print("=" * 68)
    print(f"  Subjects checked : {n}")

    if "bids_complete" in df.columns:
        n_bids = int(df["bids_complete"].sum())
        print(f"  BIDS complete    : {n_bids}/{n}")

    if "validation_passed" in df.columns:
        n_val = int(df["validation_passed"].fillna(False).sum())
        print(f"  Validation pass  : {n_val}/{n}")

    if "sync_n_trials" in df.columns and df["sync_n_trials"].notna().any():
        mean_trials = df["sync_n_trials"].mean()
        mean_complete = df["sync_n_complete"].mean()
        print(f"  Mean sync trials : {mean_trials:.1f} (complete: {mean_complete:.1f})")

    print()
    print(f"  {'Subject':<12} {'BIDS':>5} {'Valid':>6} {'EEG':>5} {'Pupil':>6} "
          f"{'Beh':>4} {'Trials':>7} {'Cmplt':>6}")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        sid = str(row["subject"]).replace("sub-", "")
        bids = "Y" if row.get("bids_complete") else "N"
        val = "Y" if row.get("validation_passed") else ("N" if row.get("n_errors", 0) else "?")
        eeg = "Y" if row.get("eeg_task_found") else "N"
        pup = f"{row.get('pupil_valid_prop', float('nan')):.2f}" if not pd.isna(row.get("pupil_valid_prop", float('nan'))) else " N/A"
        beh = "Y" if row.get("behavioral_found") else "N"
        trials = int(row.get("sync_n_trials", 0)) if not pd.isna(row.get("sync_n_trials", float('nan'))) else 0
        cmplt = int(row.get("sync_n_complete", 0)) if not pd.isna(row.get("sync_n_complete", float('nan'))) else 0
        print(f"  {sid:<12} {bids:>5} {val:>6} {eeg:>5} {pup:>6} {beh:>4} {trials:>7} {cmplt:>6}")

    print()
    # Per-subject errors
    if validation_results:
        failed = [sid for sid, vr in validation_results.items() if not vr.passed]
        if failed:
            print(f"  Subjects with critical errors: {failed}")
            for sid in failed:
                vr = validation_results[sid]
                for e in vr.errors:
                    print(f"    [{sid}] ERROR: {e}")
        else:
            print("  No subjects with critical errors.")

    # Sub-032 anchor check (Paper 4 reference subject)
    if "sub-032" in df["subject"].values:
        anchor = df[df["subject"] == "sub-032"].iloc[0]
        anchor_ok = bool(anchor.get("bids_complete", False)) and bool(anchor.get("validation_passed", False))
        anchor_status = "PASS ✓" if anchor_ok else "FAIL ✗"
        print(f"\n  sub-032 (Paper 4 anchor): {anchor_status}")
    print("=" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validation_result_to_dict(vr: ValidationResult) -> dict:
    """Convert ValidationResult dataclass to JSON-serializable dict."""
    return {
        "subject": vr.subject_id,
        "passed": vr.passed,
        "errors": vr.errors,
        "warnings": vr.warnings,
        "quality_flags": {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in vr.quality_flags.items()
        },
    }


def _sync_report_json_safe(report: dict) -> dict:
    """Make sync report JSON-serializable (convert numpy scalars)."""
    def _safe(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: _safe(vv) for kk, vv in v.items()}
        return v
    return {k: _safe(v) for k, v in report.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:  # returns exit code
    args = parse_args()

    # ── Config ──
    cfg: Config = load_config(args.config)

    # ── Output directory ──
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["output_root"]) / "verification"
    ensure_dir(out_dir)

    # ── Logging (file + console) ──
    setup_logger(
        name="",           # root logger
        level=cfg.get("logging", {}).get("level", "INFO") if hasattr(cfg.get("logging", {}), "get") else "INFO",
        log_file=out_dir / "00_verify_data.log",
    )
    # Re-grab module logger after root is configured
    global logger
    logger = logging.getLogger(__name__)

    logger.info("=" * 68)
    logger.info("ds003838 — Phase 0 Data Verification")
    logger.info("=" * 68)
    logger.info(f"Config  : {args.config}")
    logger.info(f"Dataset : {cfg['paths']['dataset_root']}")
    logger.info(f"Output  : {out_dir}")

    # ── Subject selection ──
    if args.subjects:
        subjects = args.subjects
        logger.info(f"Subjects: user-specified ({len(subjects)})")
    elif args.all:
        subjects = cfg["subjects"]["all_clean"]
        logger.info(f"Subjects: all_clean ({len(subjects)})")
    else:
        subjects = cfg["subjects"]["development"]
        logger.info(f"Subjects: development subset ({len(subjects)})")

    logger.info(f"Subject list: {subjects}")

    # ── BIDS Inspector ──
    inspector = BIDSDatasetInspector(cfg["paths"]["dataset_root"])

    # ── Phase 1: Path inspection (always runs) ──
    logger.info("\n--- Phase 1: BIDS path inspection ---")
    path_reports = run_inspection_pass(subjects, inspector)

    # Save path reports
    save_json(path_reports, str(out_dir / "00_path_inspection.json"))
    logger.info(f"Path inspection → {out_dir / '00_path_inspection.json'}")

    # ── Early exit if inspection-only mode ──
    if args.no_load:
        logger.info("--no-load specified: skipping data loading and validation.")
        df = build_summary_df(path_reports, None, None)
        csv_path = out_dir / "00_data_verification_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV → {csv_path}")
        print_terminal_summary(df, path_reports, None, None)
        n_incomplete = int((~df["bids_complete"]).sum())
        return 1 if n_incomplete > 0 else 0

    # ── Phase 2: Load subjects ──
    logger.info("\n--- Phase 2: Subject data loading ---")
    loader = SubjectDataLoader(cfg)

    # Determine which subjects have complete BIDS paths before loading
    # (avoid loading subjects we know are broken)
    loadable = [r["subject"] for r in path_reports if r.get("is_complete", False)]
    unloadable = [r["subject"] for r in path_reports if not r.get("is_complete", False)]
    if unloadable:
        logger.warning(f"Skipping load for incomplete subjects: {unloadable}")

    loaded_data: Dict[str, LoadedSubjectData] = {}
    for sid in loadable:
        try:
            raw_paths = inspector.find_subject_files(sid)
            data = loader.load_subject(raw_paths, load_rest=False, preload_eeg=False)
            loaded_data[sid] = data
            logger.info(f"{sid}: loaded successfully")
        except Exception as e:
            logger.error(f"{sid}: load failed — {e}", exc_info=True)

    logger.info(f"Loaded {len(loaded_data)}/{len(loadable)} subjects")

    # ── Phase 3: Validate loaded data ──
    logger.info("\n--- Phase 3: Data validation ---")
    validation_results: Optional[Dict[str, ValidationResult]] = None
    passed_ids: List[str] = []
    failed_ids: List[str] = []

    if loaded_data:
        validation_results, passed_ids, failed_ids = validate_subjects_batch(
            loaded_data, cfg=cfg
        )
        print_validation_report(validation_results)

        # Save validation details
        val_dicts = [_validation_result_to_dict(vr) for vr in validation_results.values()]
        save_json(val_dicts, str(out_dir / "00_validation_details.json"))
        logger.info(f"Validation details → {out_dir / '00_validation_details.json'}")
    else:
        logger.warning("No subjects loaded — validation skipped")

    # ── Phase 4: Synchronization check (optional) ──
    sync_reports: Optional[Dict[str, dict]] = None

    if args.sync_check and loaded_data:
        logger.info("\n--- Phase 4: Synchronization check ---")
        synchronizer = TrialSynchronizer(cfg)
        sync_reports = {}

        # Only sync-check subjects that passed validation
        sync_candidates = passed_ids if passed_ids else list(loaded_data.keys())
        logger.info(f"Running sync check on {len(sync_candidates)} subjects")

        for sid in sync_candidates:
            if sid in loaded_data:
                report = run_sync_check(sid, loaded_data[sid], synchronizer)
                sync_reports[sid] = report

        # Save sync reports
        sync_reports_safe = {k: _sync_report_json_safe(v) for k, v in sync_reports.items()}
        save_json(list(sync_reports_safe.values()), str(out_dir / "00_sync_reports.json"))
        logger.info(f"Sync reports → {out_dir / '00_sync_reports.json'}")
    elif args.sync_check:
        logger.warning("--sync-check requested but no subjects were loaded")

    # ── Phase 5: Build summary table ──
    logger.info("\n--- Phase 5: Summary table ---")
    df = build_summary_df(path_reports, validation_results, sync_reports)
    csv_path = out_dir / "00_data_verification_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV → {csv_path}")

    # ── Phase 6: Plots ──
    if args.plot:
        logger.info("\n--- Phase 6: QC visualization ---")
        plot_verification_summary(df, out_dir)

    # ── Terminal summary ──
    print_terminal_summary(df, path_reports, validation_results, sync_reports)

    # ── Exit code ──
    n_critical_errors = int(df["n_errors"].fillna(0).gt(0).sum()) if "n_errors" in df.columns else 0
    n_bids_incomplete = int((~df["bids_complete"].fillna(False)).sum())

    if n_critical_errors > 0:
        logger.warning(
            f"{n_critical_errors} subjects have critical errors — "
            "review before proceeding to preprocessing."
        )
        return 1

    if n_bids_incomplete > 0:
        logger.warning(
            f"{n_bids_incomplete} subjects have incomplete BIDS structure — "
            "may affect downstream completeness."
        )
        return 1

    logger.info("All subjects passed verification. Safe to proceed to Phase 1.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())