"""
Script 02: Feature Extraction
==============================
Extract per-trial, per-subject multimodal feature matrices from preprocessed data.
Builds condition-stratified feature matrices F_c ∈ R^{N_c x D} for GGM fitting.

Feature sets extracted:
  EEG  : frontal theta, occipital alpha, IAF, CSP components, sample entropy, ERP
  ECG  : HRV (time/frequency), HEP amplitude at Fz/Cz
  PPG  : PWA, slope, complexity, IBI
  Pupil: mean diameter, TEPR, dilation rate, DFA
  Cross: HEP (EEG-ECG joint), resting-state per-subject features

Run: python scripts/02_extract_features.py [--subjects ...] [--force]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import setup_logging, save_pickle, load_pickle, save_json
from utils.config_loader import load_config
from utils.normalization import SubjectNormalizer
from features.eeg_features import EEGFeatureExtractor
from features.ecg_features import ECGFeatureExtractor
from features.ppg_features import PPGFeatureExtractor
from features.pupil_features import PupilFeatureExtractor
from features.hep_features import HEPExtractor
from features.feature_matrix import FeatureMatrixBuilder

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Extract multimodal features for ds003838")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="Reextract even if cached")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resting-only", action="store_true",
                   help="Extract only resting-state features")
    return p.parse_args()


def load_preprocessed(subject_id: str, cache_dir: Path) -> Optional[Dict]:
    """Load all preprocessed modality outputs for one subject."""
    sub_cache = cache_dir / subject_id
    data = {}
    files = {
        "eeg_epochs": "eeg_epochs.pkl",
        "ecg": "ecg_processed.pkl",
        "ppg": "ppg_processed.pkl",
        "pupil": "pupil_processed.pkl",
    }
    for key, fname in files.items():
        path = sub_cache / fname
        if path.exists():
            try:
                data[key] = load_pickle(str(path))
            except Exception as e:
                logger.warning(f"{subject_id}: failed to load {key} — {e}")
        else:
            logger.warning(f"{subject_id}: {fname} not found — run 01_preprocess.py first")
    return data if data else None


def extract_subject_features(
    subject_id: str,
    cfg: dict,
    cache_dir: Path,
    feat_cache_dir: Path,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Extract all features for one subject.
    Returns dict: condition → feature_df (rows=trials, cols=features)
    """
    feat_cache = feat_cache_dir / f"{subject_id}_features.pkl"
    if feat_cache.exists() and not force:
        logger.info(f"{subject_id}: features cached, loading")
        return load_pickle(str(feat_cache))

    t0 = time.time()
    logger.info(f"{subject_id}: extracting features...")

    prep_data = load_preprocessed(subject_id, cache_dir)
    if prep_data is None:
        logger.error(f"{subject_id}: no preprocessed data found")
        return None

    subject_features = {"subject": subject_id, "conditions": {}, "resting": {}}

    # ---- Per-modality extractors -------------------------------------------
    eeg_ext = EEGFeatureExtractor(cfg)
    ecg_ext = ECGFeatureExtractor(cfg)
    ppg_ext = PPGFeatureExtractor(cfg)
    pupil_ext = PupilFeatureExtractor(cfg)
    hep_ext = HEPExtractor(cfg)

    # ---- Resting-state features (computed once per subject) ----------------
    try:
        rest_eeg = prep_data.get("eeg_epochs")
        rest_ecg = prep_data.get("ecg")
        rest_ppg = prep_data.get("ppg")
        rest_pupil = prep_data.get("pupil")

        resting_feats = {}

        if rest_eeg is not None:
            resting_feats.update(eeg_ext.extract_resting(rest_eeg))

        if rest_ecg is not None:
            resting_feats.update(ecg_ext.extract_resting(rest_ecg))

        if rest_ppg is not None:
            resting_feats.update(ppg_ext.extract_resting(rest_ppg))

        if rest_pupil is not None:
            resting_feats.update(pupil_ext.extract_resting(rest_pupil))

        # Resting EEG-ECG coherence (cross-modal coupling baseline)
        if rest_eeg is not None and rest_ecg is not None:
            resting_feats.update(
                hep_ext.extract_resting_coherence(rest_eeg, rest_ecg)
            )

        subject_features["resting"] = resting_feats
        logger.info(f"{subject_id}: resting features — {len(resting_feats)} features")
    except Exception as e:
        logger.warning(f"{subject_id}: resting feature extraction failed — {e}")

    # ---- Per-trial features for each condition -----------------------------
    conditions = [c for c in cfg["paradigm"]["conditions"] if c != "control"]

    for cond in conditions:
        try:
            cond_trials = {}

            # EEG features
            if "eeg_epochs" in prep_data:
                eeg_df = eeg_ext.extract_per_trial(
                    prep_data["eeg_epochs"], condition=cond
                )
                cond_trials["eeg"] = eeg_df

            # ECG/HRV features
            if "ecg" in prep_data:
                ecg_df = ecg_ext.extract_per_trial(
                    prep_data["ecg"], condition=cond
                )
                cond_trials["ecg"] = ecg_df

            # PPG features
            if "ppg" in prep_data:
                ppg_df = ppg_ext.extract_per_trial(
                    prep_data["ppg"], condition=cond
                )
                cond_trials["ppg"] = ppg_df

            # Pupil features
            if "pupil" in prep_data:
                pupil_df = pupil_ext.extract_per_trial(
                    prep_data["pupil"], condition=cond
                )
                cond_trials["pupil"] = pupil_df

            # HEP: cross-modal EEG-ECG (requires both)
            if "eeg_epochs" in prep_data and "ecg" in prep_data:
                hep_df = hep_ext.extract_per_trial(
                    prep_data["eeg_epochs"],
                    prep_data["ecg"],
                    condition=cond
                )
                cond_trials["hep"] = hep_df

            subject_features["conditions"][cond] = cond_trials
            n_trials = next(
                (len(v) for v in cond_trials.values() if hasattr(v, "__len__")),
                "?"
            )
            logger.info(f"{subject_id}/{cond}: {n_trials} trials extracted")

        except Exception as e:
            logger.error(f"{subject_id}/{cond}: feature extraction failed — {e}")
            subject_features["conditions"][cond] = {}

    elapsed = time.time() - t0
    logger.info(f"{subject_id}: feature extraction complete in {elapsed:.1f}s")

    save_pickle(subject_features, str(feat_cache))
    return subject_features


def build_condition_matrices(
    all_subject_features: Dict[str, Dict],
    cfg: dict,
    normalizer: SubjectNormalizer,
    output_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Pool all subjects × trials per condition → F_c ∈ R^{N_c × D}.
    Apply per-subject z-score normalization before pooling.
    """
    logger.info("Building condition-stratified feature matrices...")

    builder = FeatureMatrixBuilder(cfg)
    condition_matrices = {}

    conditions = [c for c in cfg["paradigm"]["conditions"] if c != "control"]

    for cond in conditions:
        logger.info(f"Pooling condition: {cond}")
        try:
            F_c, feature_names, trial_index = builder.build_condition_matrix(
                all_subject_features,
                condition=cond,
                normalizer=normalizer,
            )
            condition_matrices[cond] = {
                "matrix": F_c,
                "feature_names": feature_names,
                "trial_index": trial_index,  # (subject, trial_idx) per row
                "shape": F_c.shape,
            }
            logger.info(
                f"{cond}: F_c shape = {F_c.shape} "
                f"({F_c.shape[0]} trials × {F_c.shape[1]} features)"
            )
        except Exception as e:
            logger.error(f"Failed to build matrix for {cond}: {e}")

    # Save matrices
    mat_path = output_dir / "condition_feature_matrices.pkl"
    save_pickle(condition_matrices, str(mat_path))
    logger.info(f"Condition matrices saved → {mat_path}")

    # Save feature names (for interpretability)
    if condition_matrices:
        example_cond = next(iter(condition_matrices))
        feat_names = condition_matrices[example_cond]["feature_names"]
        save_json(
            {"feature_names": feat_names, "n_features": len(feat_names)},
            str(output_dir / "feature_names.json")
        )
        logger.info(f"Feature names saved — {len(feat_names)} total features")

    return condition_matrices


def save_feature_summary(
    all_subject_features: Dict[str, Dict],
    condition_matrices: Dict[str, Dict],
    output_dir: Path,
):
    """Save feature extraction summary stats."""
    summary = {
        "n_subjects": len(all_subject_features),
        "subjects": list(all_subject_features.keys()),
        "conditions": {},
    }
    for cond, mat_info in condition_matrices.items():
        summary["conditions"][cond] = {
            "n_trials": int(mat_info["shape"][0]),
            "n_features": int(mat_info["shape"][1]),
        }

    # Per-subject resting feature counts
    resting_counts = {}
    for sub_id, feats in all_subject_features.items():
        resting_counts[sub_id] = len(feats.get("resting", {}))
    summary["resting_feature_counts"] = resting_counts

    save_json(summary, str(output_dir / "02_feature_extraction_summary.json"))
    logger.info(f"Feature summary saved → {output_dir / '02_feature_extraction_summary.json'}")

    # Print table
    print("\n" + "=" * 55)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"Subjects processed: {summary['n_subjects']}")
    print()
    for cond, info in summary["conditions"].items():
        print(f"  {cond:<12}: {info['n_trials']:>4} trials × {info['n_features']:>3} features")
    print("=" * 55 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir or cfg["paths"]["output_root"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg["paths"]["cache_dir"]) / "preprocessed"
    feat_cache_dir = Path(cfg["paths"]["cache_dir"]) / "features"
    feat_cache_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(Path(cfg["paths"]["output_root"]) / "logs" / "02_extract_features.log"))

    logger.info("=" * 60)
    logger.info("Feature Extraction Pipeline — Script 02")
    logger.info("=" * 60)

    if args.all:
        subjects = cfg["subjects"]["all_clean"]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["subjects"]["development"]

    logger.info(f"Subjects: {subjects}")

    # ---- Extract per-subject features --------------------------------------
    all_subject_features = {}
    for sub_id in subjects:
        result = extract_subject_features(
            sub_id, cfg, cache_dir, feat_cache_dir, force=args.force
        )
        if result is not None:
            all_subject_features[sub_id] = result
        else:
            logger.warning(f"{sub_id}: skipped — no data")

    if not all_subject_features:
        logger.error("No features extracted. Run 01_preprocess.py first.")
        sys.exit(1)

    if args.resting_only:
        logger.info("Resting-only mode — skipping condition matrices")
        # Save resting features
        resting_out = {s: f["resting"] for s, f in all_subject_features.items()}
        save_pickle(resting_out, str(out_dir / "resting_features.pkl"))
        logger.info(f"Resting features saved → {out_dir / 'resting_features.pkl'}")
        return

    # ---- Build condition-stratified matrices --------------------------------
    normalizer = SubjectNormalizer(method="zscore")
    condition_matrices = build_condition_matrices(
        all_subject_features, cfg, normalizer, out_dir
    )

    # ---- Summary -----------------------------------------------------------
    save_feature_summary(all_subject_features, condition_matrices, out_dir)

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()