"""
download_ds006848.py
--------------------
Download and validate OpenNeuro dataset ds006848 (AlphaDirection1).
Run: python download_ds006848.py

Target layout:
  dataset/ds006848/
    participants.tsv
    sub-001/eeg/*.vhdr  *.eeg  *.vmrk  *_events.tsv
    ...
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


path = "dataset/ds006848"
print("Exists:", os.path.exists(path))
# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ds006848")

# ── Constants ─────────────────────────────────────────────────
DATASET_ID    = "ds006848"
S3_BUCKET     = "openneuro.org"
TARGET_DIR    = Path("dataset") / DATASET_ID
MIN_EEG_BYTES = 1 * 1024 * 1024   # 1 MB — real EEG binaries are 50-300 MB


# ============================================================
# 1. Dependency management
# ============================================================

def _pip_install(package: str, import_name: str) -> bool:
    """Install package if not importable. Returns True if available."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        log.info(f"Installing {package} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True,
        )
        if result.returncode == 0:
            log.info(f"  {package} installed.")
            return True
        log.warning(f"  {package} install failed: {result.stderr.decode()[:200]}")
        return False


def ensure_dependencies() -> dict:
    """
    Install required packages, return availability flags.
    Script continues even if optional packages fail.
    """
    flags = {
        "openneuro": _pip_install("openneuro-py", "openneuro"),
        "boto3":     _pip_install("boto3",         "boto3"),
        "tqdm":      _pip_install("tqdm",           "tqdm"),
        "mne":       _pip_install("mne",            "mne"),
    }
    log.info(f"Dependencies: {flags}")
    return flags


# ============================================================
# 2. Download — Method A: openneuro-py
# ============================================================

def download_openneuro_py(dataset_id: str, target_dir: Path) -> bool:
    """
    Download via openneuro-py (official OpenNeuro S3 client).
    Skips files where local content already matches remote.
    Returns True on success.
    """
    try:
        import openneuro
    except ImportError:
        log.warning("openneuro-py not available.")
        return False

    log.info(f"openneuro-py {openneuro.__version__} — starting download ...")
    try:
        openneuro.download(
            dataset=dataset_id,
            target_dir=str(target_dir),
            include=None,
            exclude=None,
            verify_hash=False,
        )
        log.info("openneuro-py download complete.")
        return True
    except Exception as exc:
        log.warning(f"openneuro-py failed: {exc}")
        return False


# ============================================================
# 3. Download — Method B: boto3 anonymous S3 (fallback)
# ============================================================

def download_boto3_s3(dataset_id: str, target_dir: Path) -> bool:
    """
    Direct anonymous S3 download from s3://openneuro.org/<dataset_id>/.
    Resumes: skips files where local size == S3 ContentLength.
    Deletes partial files so next run re-fetches cleanly.
    Returns True if zero failures.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        log.error("boto3 not available — cannot download.")
        return False

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    prefix = f"{dataset_id}/"
    log.info(f"boto3 S3 — listing s3://{S3_BUCKET}/{prefix} ...")

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )

    paginator   = s3.get_paginator("list_objects_v2")
    all_objects = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        all_objects.extend(page.get("Contents", []))

    if not all_objects:
        log.error("No objects returned from S3. Check dataset ID and bucket.")
        return False

    log.info(f"  Objects to process: {len(all_objects)}")

    n_downloaded = 0
    n_skipped    = 0
    n_failed     = 0

    iterator = (
        _tqdm(all_objects, desc="Downloading", unit="file")
        if _tqdm else all_objects
    )

    for obj in iterator:
        key         = obj["Key"]
        remote_size = obj["Size"]

        # Strip dataset prefix to get relative path
        rel   = Path(key[len(prefix):])
        local = target_dir / rel
        local.parent.mkdir(parents=True, exist_ok=True)

        # S3 "directory" markers — zero bytes
        if remote_size == 0:
            local.touch()
            n_skipped += 1
            continue

        # Resume: exact size match = complete file
        if local.exists() and local.stat().st_size == remote_size:
            n_skipped += 1
            continue

        try:
            s3.download_file(S3_BUCKET, key, str(local))
            n_downloaded += 1
            if not _tqdm:
                mb = remote_size / 1e6
                log.info(f"  ✓ {rel}  ({mb:.1f} MB)")
        except Exception as exc:
            log.error(f"  ✗ {rel}: {exc}")
            if local.exists():
                local.unlink()   # remove partial file
            n_failed += 1

    log.info(
        f"boto3 done — downloaded={n_downloaded}  "
        f"skipped={n_skipped}  failed={n_failed}"
    )
    return n_failed == 0


# ============================================================
# 4. Validation
# ============================================================

def validate_dataset(target_dir: Path) -> dict:
    """
    For every sub-XXX directory:
      - Check .vhdr, .eeg, .vmrk present
      - Check .eeg size > MIN_EEG_BYTES
    Returns summary dict.
    """
    subject_dirs = sorted([
        d for d in target_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    log.info(f"\n{'='*55}")
    log.info(f"VALIDATION — {len(subject_dirs)} subjects")
    log.info(f"{'='*55}")

    results = {
        "n_subjects":  len(subject_dirs),
        "valid":       [],
        "small_eeg":   [],
        "missing":     [],
    }

    for subj_dir in subject_dirs:
        sid     = subj_dir.name
        eeg_dir = subj_dir / "eeg"

        if not eeg_dir.is_dir():
            log.warning(f"  ✗ {sid}  no eeg/ directory")
            results["missing"].append(sid)
            continue

        vhdr_files = sorted(eeg_dir.glob("*.vhdr"))
        eeg_files  = sorted(eeg_dir.glob("*.eeg"))
        vmrk_files = sorted(eeg_dir.glob("*.vmrk"))

        missing_types = []
        if not vhdr_files:
            missing_types.append(".vhdr")
        if not eeg_files:
            missing_types.append(".eeg")
        if not vmrk_files:
            missing_types.append(".vmrk")

        if missing_types:
            log.warning(f"  ✗ {sid}  missing: {missing_types}")
            results["missing"].append(sid)
            continue

        eeg_size = eeg_files[0].stat().st_size
        eeg_mb   = eeg_size / 1e6

        if eeg_size < MIN_EEG_BYTES:
            log.warning(
                f"  ✗ {sid}  .eeg too small: {eeg_mb:.3f} MB  ← incomplete/corrupt"
            )
            results["small_eeg"].append(sid)
        else:
            log.info(f"  ✓ {sid}  .eeg = {eeg_mb:.1f} MB")
            results["valid"].append(sid)

    log.info(
        f"\n  Valid        : {len(results['valid'])}/{len(subject_dirs)}\n"
        f"  Small/corrupt: {len(results['small_eeg'])}\n"
        f"  Missing files: {len(results['missing'])}"
    )
    return results


# ============================================================
# 5. MNE sanity check
# ============================================================

def mne_sanity_check(target_dir: Path) -> bool:
    """
    Load first subject with a valid .vhdr using MNE (preload=False).
    Print channel names and sampling rate.
    Returns True if load succeeds.
    """
    try:
        import mne
        import warnings
        warnings.filterwarnings("ignore")
        mne.set_log_level("WARNING")
    except ImportError:
        log.warning("mne not installed — skipping sanity check.")
        return False

    subject_dirs = sorted([
        d for d in target_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    for subj_dir in subject_dirs:
        eeg_dir    = subj_dir / "eeg"
        vhdr_files = sorted(eeg_dir.glob("*.vhdr")) if eeg_dir.is_dir() else []
        eeg_files  = sorted(eeg_dir.glob("*.eeg"))  if eeg_dir.is_dir() else []

        if not vhdr_files or not eeg_files:
            continue
        if eeg_files[0].stat().st_size < MIN_EEG_BYTES:
            continue

        sid = subj_dir.name
        log.info(f"\n{'='*55}")
        log.info(f"MNE SANITY CHECK — {sid}")
        log.info(f"{'='*55}")

        try:
            raw = mne.io.read_raw_brainvision(
                vhdr_fname=str(vhdr_files[0]),
                preload=False,
                verbose=False,
            )
            log.info(f"  ✓ Loaded successfully")
            log.info(f"  Channels   : {len(raw.ch_names)}")
            log.info(f"  Sfreq      : {raw.info['sfreq']} Hz")
            log.info(f"  Duration   : {raw.times[-1]:.1f} s")
            log.info(f"  First 8 ch : {raw.ch_names[:8]}")
            del raw
            return True
        except Exception as exc:
            log.error(f"  ✗ MNE load failed for {sid}: {exc}")
            continue

    log.error("MNE sanity check: no valid subject found.")
    return False


# ============================================================
# 6. Main
# ============================================================

def main():
    log.info("="*55)
    log.info(f"ds006848 — Download & Validate")
    log.info(f"Target : {TARGET_DIR.resolve()}")
    log.info("="*55)

    # ── Dependencies ─────────────────────────────────────────
    flags = ensure_dependencies()

    # ── Create target directory ───────────────────────────────
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download ─────────────────────────────────────────────
    success = False

    if flags["openneuro"]:
        log.info("\n--- Method A: openneuro-py ---")
        success = download_openneuro_py(DATASET_ID, TARGET_DIR)

    if not success:
        if flags["boto3"]:
            log.info("\n--- Method B: boto3 anonymous S3 ---")
            success = download_boto3_s3(DATASET_ID, TARGET_DIR)
        else:
            log.error("boto3 unavailable — cannot fall back.")

    if not success:
        log.error(
            "\nAll download methods failed.\n"
            "Check internet connection and that dataset ID is correct.\n"
            f"Dataset ID: {DATASET_ID}"
        )
        sys.exit(1)

    # ── Validation ───────────────────────────────────────────
    results = validate_dataset(TARGET_DIR)

    n_bad = len(results["small_eeg"]) + len(results["missing"])
    if n_bad > 0:
        log.warning(
            f"\n{n_bad} subjects incomplete. "
            "Re-run this script to resume — boto3 will skip complete files."
        )

    # ── MNE sanity check ─────────────────────────────────────
    mne_ok = mne_sanity_check(TARGET_DIR)

    # ── Final verdict ─────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("FINAL RESULT")
    log.info(f"{'='*55}")
    log.info(f"  Valid subjects : {len(results['valid'])}/{results['n_subjects']}")
    log.info(f"  MNE check      : {'✓ PASS' if mne_ok else '✗ FAIL'}")

    if results["valid"] and mne_ok:
        log.info("\n  ✓ Dataset ready. Proceed to Section 3 of the pipeline.")
    else:
        log.warning(
            "\n  Dataset not fully ready. "
            "Re-run this script until all subjects pass validation."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()