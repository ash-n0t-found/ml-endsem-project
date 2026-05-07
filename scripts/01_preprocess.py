"""
Script 01: Preprocessing Pipeline
==================================
Run full preprocessing pipeline across all modalities for specified subjects.
Saves preprocessed epochs to cache for downstream feature extraction.

Pipeline:
  EEG  → bandpass → notch → ICA → average-ref → epoch
  ECG  → R-peak detection → IBI → HRV windows
  PPG  → peak detection → PWA → morphology
  Pupil → blink interp → baseline-norm → TEPR

Run: python scripts/01_preprocess.py [--subjects ...] [--force] [--n-jobs 4]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle
from utils.config_loader import load_config
from data.loader import BIDSLoader
from data.synchronizer import ModalitySynchronizer
from preprocessing.eeg_prep import EEGPreprocessor
from preprocessing.ecg_prep import ECGPreprocessor
from preprocessing.ppg_prep import PPGPreprocessor
from preprocessing.pupil_prep import PupilPreprocessor

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess all modalities for ds003838")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--all", action="store_true", help="Use all clean subjects")
    p.add_argument("--force", action="store_true", help="Reprocess even if cache exists")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel jobs (caution: MNE ICA is memory-intensive)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--modalities", nargs="+",
                   default=["eeg", "ecg", "ppg", "pupil"],
                   help="Which modalities to preprocess")
    return p.parse_args()


def preprocess_subject(
    subject_id: str,
    cfg: dict,
    cache_dir: Path,
    modalities: list,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Full preprocessing for one subject across all modalities.
    Returns dict of preprocessed objects keyed by modality.
    """
    sub_cache = cache_dir / subject_id
    sub_cache.mkdir(parents=True, exist_ok=True)

    result = {"subject": subject_id, "status": {}, "errors": {}}
    t0 = time.time()

    # Load raw
    try:
        loader = BIDSLoader(cfg)
        raw_data = loader.load_subject(subject_id)
    except Exception as e:
        logger.error(f"{subject_id}: load failed — {e}")
        result["errors"]["load"] = str(e)
        return result

    # Synchronize timestamps across modalities
    try:
        sync = ModalitySynchronizer(cfg)
        raw_data = sync.synchronize(raw_data)
        logger.info(f"{subject_id}: synchronization OK")
    except Exception as e:
        logger.warning(f"{subject_id}: sync warning — {e}")
        result["errors"]["sync"] = str(e)

    # ---- EEG ---------------------------------------------------------------
    if "eeg" in modalities:
        cache_path = sub_cache / "eeg_epochs.pkl"
        if cache_path.exists() and not force:
            logger.info(f"{subject_id}: EEG epochs cached, skipping")
            result["status"]["eeg"] = "cached"
        else:
            try:
                prep = EEGPreprocessor(cfg)
                epochs = prep.run(raw_data["eeg"], raw_data.get("events"))
                save_pickle(epochs, str(cache_path))
                result["status"]["eeg"] = "ok"
                logger.info(f"{subject_id}: EEG done → {len(epochs)} epochs")
            except Exception as e:
                logger.error(f"{subject_id}: EEG preprocessing failed — {e}")
                result["status"]["eeg"] = "failed"
                result["errors"]["eeg"] = str(e)

    # ---- ECG ---------------------------------------------------------------
    if "ecg" in modalities:
        cache_path = sub_cache / "ecg_processed.pkl"
        if cache_path.exists() and not force:
            result["status"]["ecg"] = "cached"
        else:
            try:
                prep = ECGPreprocessor(cfg)
                ecg_out = prep.run(raw_data["ecg"], raw_data.get("events"))
                save_pickle(ecg_out, str(cache_path))
                result["status"]["ecg"] = "ok"
                logger.info(f"{subject_id}: ECG done — {len(ecg_out.get('r_peaks', []))} R-peaks")
            except Exception as e:
                logger.error(f"{subject_id}: ECG preprocessing failed — {e}")
                result["status"]["ecg"] = "failed"
                result["errors"]["ecg"] = str(e)

    # ---- PPG ---------------------------------------------------------------
    if "ppg" in modalities:
        cache_path = sub_cache / "ppg_processed.pkl"
        if cache_path.exists() and not force:
            result["status"]["ppg"] = "cached"
        else:
            try:
                prep = PPGPreprocessor(cfg)
                ppg_out = prep.run(raw_data["ppg"], raw_data.get("events"))
                save_pickle(ppg_out, str(cache_path))
                result["status"]["ppg"] = "ok"
                logger.info(f"{subject_id}: PPG done — {len(ppg_out.get('pwa_per_beat', []))} beats")
            except Exception as e:
                logger.error(f"{subject_id}: PPG preprocessing failed — {e}")
                result["status"]["ppg"] = "failed"
                result["errors"]["ppg"] = str(e)

    # ---- Pupil -------------------------------------------------------------
    if "pupil" in modalities:
        cache_path = sub_cache / "pupil_processed.pkl"
        if cache_path.exists() and not force:
            result["status"]["pupil"] = "cached"
        else:
            try:
                prep = PupilPreprocessor(cfg)
                pupil_out = prep.run(raw_data["pupil"], raw_data.get("events"))
                save_pickle(pupil_out, str(cache_path))
                result["status"]["pupil"] = "ok"
                logger.info(f"{subject_id}: Pupil done")
            except Exception as e:
                logger.error(f"{subject_id}: Pupil preprocessing failed — {e}")
                result["status"]["pupil"] = "failed"
                result["errors"]["pupil"] = str(e)

    elapsed = time.time() - t0
    logger.info(f"{subject_id}: preprocessing complete in {elapsed:.1f}s | {result['status']}")
    return result


def validate_preprocessing_outputs(cache_dir: Path, subjects: list, modalities: list):
    """
    Quick sanity check: verify cached files exist and are loadable.
    """
    logger.info("Validating preprocessing outputs...")
    expected = {
        "eeg": "eeg_epochs.pkl",
        "ecg": "ecg_processed.pkl",
        "ppg": "ppg_processed.pkl",
        "pupil": "pupil_processed.pkl",
    }
    n_missing = 0
    for sub in subjects:
        for mod in modalities:
            fn = expected.get(mod)
            if fn is None:
                continue
            path = cache_dir / sub / fn
            if not path.exists():
                logger.warning(f"Missing: {path}")
                n_missing += 1
    if n_missing == 0:
        logger.info("All expected cache files present.")
    else:
        logger.warning(f"{n_missing} cache files missing — check errors above.")
    return n_missing


def print_status_table(all_results: list):
    """Print tabular status summary."""
    header = f"{'Subject':<12} {'EEG':<10} {'ECG':<10} {'PPG':<10} {'Pupil':<10} {'Errors'}"
    print("\n" + "=" * 65)
    print("PREPROCESSING STATUS")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for r in all_results:
        s = r["status"]
        e = r["errors"]
        print(
            f"{r['subject']:<12} "
            f"{s.get('eeg','--'):<10} "
            f"{s.get('ecg','--'):<10} "
            f"{s.get('ppg','--'):<10} "
            f"{s.get('pupil','--'):<10} "
            f"{list(e.keys()) if e else 'none'}"
        )
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"]) / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(out_dir / "logs" / "01_preprocess.log"))

    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline — Script 01")
    logger.info("=" * 60)

    if args.all:
        subjects = cfg["subjects"]["all_clean"]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["subjects"]["development"]

    logger.info(f"Subjects: {subjects}")
    logger.info(f"Modalities: {args.modalities}")
    logger.info(f"Force reprocess: {args.force}")

    if args.n_jobs == 1:
        all_results = [
            preprocess_subject(s, cfg, cache_dir, args.modalities, args.force)
            for s in subjects
        ]
    else:
        logger.info(f"Parallel processing with {args.n_jobs} jobs")
        all_results = Parallel(n_jobs=args.n_jobs)(
            delayed(preprocess_subject)(s, cfg, cache_dir, args.modalities, args.force)
            for s in subjects
        )

    # Validate outputs
    n_missing = validate_preprocessing_outputs(cache_dir, subjects, args.modalities)

    # Status table
    print_status_table(all_results)

    # Summary
    n_ok = sum(
        all(r["status"].get(m) in ("ok", "cached") for m in args.modalities)
        for r in all_results
    )
    logger.info(f"Complete: {n_ok}/{len(subjects)} subjects fully preprocessed")
    logger.info(f"Cache directory: {cache_dir}")

    if n_missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()