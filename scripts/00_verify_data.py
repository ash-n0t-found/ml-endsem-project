"""
Script 00: Data Verification
============================
Verify dataset integrity across all modalities for the development subject set.
Replicates known physiological patterns (theta↑, alpha↓, PWA↓ with load)
to confirm preprocessing is correct before scaling.

Run: python scripts/00_verify_data.py [--subjects sub-032 sub-033 ...] [--all]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_json, load_json
from utils.config_loader import load_config
from data.loader import BIDSDatasetInspector
from data.validator import SubjectValidator
from data.synchronizer import TrialSynchronizer

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Verify ds003838 dataset integrity")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject IDs to verify (default: dev subset from config)")
    p.add_argument("--all", action="store_true",
                   help="Verify all clean subjects (slow)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    p.add_argument("--plot", action="store_true",
                   help="Generate sanity-check plots")
    return p.parse_args()


def verify_subject(subject_id: str, loader: BIDSLoader,
                   validator: ModalityValidator,
                   synchronizer: ModalitySynchronizer,
                   cfg: dict) -> dict:
    """
    Per-subject verification pipeline.
    Returns dict with pass/fail flags and diagnostics.
    """
    result = {
        "subject": subject_id,
        "modalities_present": {},
        "n_trials": {},
        "sync_ok": False,
        "event_alignment_ok": False,
        "known_pattern_ok": False,
        "errors": [],
        "warnings": [],
    }

    # ---- 1. Load raw data --------------------------------------------------
    try:
        raw_data = loader.load_subject(subject_id)
    except Exception as e:
        result["errors"].append(f"Load failed: {e}")
        logger.error(f"{subject_id}: load failed — {e}")
        return result

    # ---- 2. Validate modality presence -------------------------------------
    modality_checks = validator.check_modalities(raw_data)
    result["modalities_present"] = modality_checks
    missing = [m for m, ok in modality_checks.items() if not ok]
    if missing:
        result["errors"].append(f"Missing modalities: {missing}")
        logger.warning(f"{subject_id}: missing {missing}")

    # ---- 3. Check trial counts per condition -------------------------------
    if "events" in raw_data:
        events_df = raw_data["events"]
        for cond_name, cond_cfg in cfg["paradigm"]["conditions"].items():
            if cond_name == "control":
                continue
            n_dig = cond_cfg["n_digits"]
            # Count trials matching this condition
            mask = events_df.get("trial_type", pd.Series(dtype=str)) == str(n_dig)
            n = int(mask.sum())
            result["n_trials"][cond_name] = n
            if n == 0:
                result["warnings"].append(f"No trials found for {cond_name}")
                logger.warning(f"{subject_id}: 0 trials for {cond_name}")

    # ---- 4. Synchronization check ------------------------------------------
    try:
        sync_result = synchronizer.check_alignment(raw_data)
        result["sync_ok"] = sync_result["aligned"]
        if not sync_result["aligned"]:
            result["warnings"].append(
                f"Sync offset: {sync_result.get('max_offset_ms', 'unknown')} ms"
            )
    except Exception as e:
        result["warnings"].append(f"Sync check error: {e}")

    # ---- 5. Event alignment check ------------------------------------------
    try:
        align_ok = validator.check_event_alignment(raw_data)
        result["event_alignment_ok"] = align_ok
        if not align_ok:
            result["warnings"].append("Event markers may be misaligned across modalities")
    except Exception as e:
        result["warnings"].append(f"Event alignment check error: {e}")

    # ---- 6. Known-pattern sanity check -------------------------------------
    # Theta power should increase from 5→9→13 digits
    # (basic sanity: if EEG present and events decodable)
    try:
        pattern_ok = validator.check_known_patterns(raw_data, cfg)
        result["known_pattern_ok"] = pattern_ok
        if not pattern_ok:
            result["warnings"].append(
                "Known pattern check failed: theta/alpha trends unexpected"
            )
    except Exception as e:
        result["warnings"].append(f"Known pattern check skipped: {e}")

    # ---- Summary -----------------------------------------------------------
    n_errors = len(result["errors"])
    n_warnings = len(result["warnings"])
    status = "PASS" if n_errors == 0 else "FAIL"
    logger.info(
        f"{subject_id}: {status} | "
        f"errors={n_errors}, warnings={n_warnings} | "
        f"sync={result['sync_ok']} | "
        f"pattern={result['known_pattern_ok']}"
    )
    return result


def summarize_results(all_results: list, output_dir: Path) -> pd.DataFrame:
    """Build and save summary table."""
    rows = []
    for r in all_results:
        rows.append({
            "subject": r["subject"],
            "eeg_ok": r["modalities_present"].get("eeg", False),
            "ecg_ok": r["modalities_present"].get("ecg", False),
            "ppg_ok": r["modalities_present"].get("ppg", False),
            "pupil_ok": r["modalities_present"].get("pupil", False),
            "sync_ok": r["sync_ok"],
            "events_ok": r["event_alignment_ok"],
            "pattern_ok": r["known_pattern_ok"],
            "n_errors": len(r["errors"]),
            "n_warnings": len(r["warnings"]),
            "trials_5": r["n_trials"].get("load_5", 0),
            "trials_9": r["n_trials"].get("load_9", 0),
            "trials_13": r["n_trials"].get("load_13", 0),
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "00_data_verification_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Verification summary → {csv_path}")
    return df


def plot_verification_summary(df: pd.DataFrame, output_dir: Path):
    """Bar chart of pass/fail per check per subject."""
    checks = ["eeg_ok", "ecg_ok", "ppg_ok", "pupil_ok",
              "sync_ok", "events_ok", "pattern_ok"]
    n_subs = len(df)
    check_labels = ["EEG", "ECG", "PPG", "Pupil", "Sync", "Events", "Pattern"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: heatmap-style grid
    ax = axes[0]
    grid = df[checks].astype(int).values
    im = ax.imshow(grid.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_subs))
    ax.set_xticklabels(df["subject"].str.replace("sub-", ""), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(check_labels)))
    ax.set_yticklabels(check_labels, fontsize=9)
    ax.set_title("Modality & QC Checks (green=pass)", fontsize=11, fontweight="bold")

    # Right: trial count bar chart
    ax2 = axes[1]
    x = np.arange(n_subs)
    w = 0.25
    ax2.bar(x - w, df["trials_5"], w, label="5-digit", color="#4dac26")
    ax2.bar(x, df["trials_9"], w, label="9-digit", color="#f1a340")
    ax2.bar(x + w, df["trials_13"], w, label="13-digit", color="#d6604d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["subject"].str.replace("sub-", ""), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("N trials")
    ax2.set_title("Trial Counts per Condition", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "00_verification_overview.png", dpi=150, bbox_inches="tight")
    logger.info(f"Verification plot → {output_dir / '00_verification_overview.png'}")
    plt.close(fig)


def print_summary(df: pd.DataFrame):
    """Print human-readable pass/fail summary to terminal."""
    checks = ["eeg_ok", "ecg_ok", "ppg_ok", "pupil_ok",
              "sync_ok", "events_ok", "pattern_ok"]
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 60)
    n_total = len(df)
    print(f"Subjects checked: {n_total}")
    print(f"All checks pass:  {int((df[checks].all(axis=1)).sum())}/{n_total}")
    print()
    for chk, lbl in zip(checks, ["EEG", "ECG", "PPG", "Pupil", "Sync", "Events", "Pattern"]):
        n_pass = int(df[chk].sum())
        print(f"  {lbl:<10}: {n_pass}/{n_total} pass")
    print()
    failed = df[df["n_errors"] > 0]["subject"].tolist()
    if failed:
        print(f"Subjects with errors: {failed}")
    else:
        print("No subjects with critical errors.")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Output directory
    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(out_dir / "00_verify_data.log"))

    logger.info("=" * 60)
    logger.info("ds003838 Data Verification — Script 00")
    logger.info("=" * 60)

    # Subject selection
    if args.all:
        subjects = cfg["subjects"]["all_clean"]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["subjects"]["development"]

    logger.info(f"Verifying {len(subjects)} subjects: {subjects}")

    # Initialize pipeline objects
    loader = BIDSDatasetInspector(cfg)
    validator = SubjectValidator(cfg)
    synchronizer = TrialSynchronizer(cfg)

    # Run verification per subject
    all_results = []
    for sub_id in subjects:
        logger.info(f"\n--- {sub_id} ---")
        result = verify_subject(sub_id, loader, validator, synchronizer, cfg)
        all_results.append(result)

    # Save per-subject JSON
    save_json(all_results, str(out_dir / "00_verification_details.json"))

    # Summary table
    df = summarize_results(all_results, out_dir)

    # Plots
    if args.plot:
        plot_verification_summary(df, out_dir)

    # Terminal summary
    print_summary(df)

    # Exit with error code if any subject has critical errors
    n_failed = int((df["n_errors"] > 0).sum())
    if n_failed > 0:
        logger.warning(f"{n_failed} subjects failed verification — review before proceeding")
        sys.exit(1)
    else:
        logger.info("All subjects passed verification. Safe to proceed.")
        sys.exit(0)


if __name__ == "__main__":
    main()