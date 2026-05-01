# ============================================================
# RS-HESR | Section 5 --- Signal Preprocessing
# Input: processed/config.json
#        processed/loading_registry.json
# Output: processed/sub-XXX_preprocessed.h5 (per subject)
#         processed/preprocessing_qc.json
# ============================================================

import os, json, warnings, gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import h5py
import mne
from scipy.signal import butter, filtfilt, iirnotch

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")


# ── Load validated artifacts from earlier sections ────────────
def load_config(root="processed"):
    with open(Path(root) / "config.json") as f:
        return json.load(f)

def save_config(cfg):
    with open(Path(cfg["processed_root"]) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()
REG_PATH = Path(CONFIG["processed_root"]) / "loading_registry.json"

with open(REG_PATH) as f:
    REGISTRY = json.load(f)

PROCESSED = Path(CONFIG["processed_root"])
SUBJECTS = CONFIG["full_subjects"]  # EEG + ECG + PPG confirmed

print(f"Preprocessing {len(SUBJECTS)} subjects")
print(f"ECG canonical channel : {CONFIG['ecg_canonical_ch']}")
print(f"PPG canonical channel : {CONFIG['ppg_canonical_ch']}")
print(f"Cohort sfreq          : {CONFIG['cohort_sfreq']} Hz\n")


# ============================================================
# CONDITION DISCOVERY --- auto-map event labels to conditions
# ============================================================

REST_KEYWORDS = {"rest", "baseline", "resting", "fixation", "eyes_closed", "eyes_opened"}
RESPONSE_KEYWORDS = {"response", "recall", "button", "resp", "answer", "digits_retrieval"}
ENCODING_KEYWORDS = {"encoding", "stimulus", "stim", "digit", "present", "boundary", "start_cartoon", "end_cartoon", "experiment_end"}

# Default condition mapping (exact-match fallback)
COND_INT = CONFIG.get("condition_to_int", {
    "simultaneous": 0,
    "slow": 1,
    "fast": 2,
    "fast_delay": 3,
})


def discover_conditions_from_events(subjects, registry, max_subjects=5):
    """
    Scan event files to discover actual trial_type labels and build
    a mapping to canonical condition names.
    """
    all_labels = set()
    label_col_candidates = ["trial_type", "condition", "description",
                            "event_type", "stim_type", "value", "event_name"]

    for subj_id in subjects[:max_subjects]:
        reg = registry.get(subj_id, {})
        ev_path = reg.get("events_path")
        if not ev_path or not Path(ev_path).exists():
            continue
        try:
            df = pd.read_csv(ev_path, sep="	")
            df.columns = [c.lower().strip() for c in df.columns]
            label_col = next((c for c in label_col_candidates
                              if c in df.columns), None)
            if label_col:
                labs = df[label_col].dropna().astype(str).str.lower().str.strip()
                all_labels.update(labs.unique())
        except Exception as e:
            print(f"  [discover] Could not read {subj_id} events: {e}")
            continue

    if not all_labels:
        return {}, set()

    # Build mapping from actual label -> canonical condition
    cond_map = {}
    known_patterns = {
        "simultaneous": ["simultaneous", "simul", "sim", "all_at_once", "encoding_set_simultaneous"],
        "slow": ["slow", "slow_seq", "slow_sequential", "seq_slow", "retention_slow"],
        "fast": ["fast", "fast_seq", "fast_sequential", "seq_fast", "retention_fast"],
        "fast_delay": ["fast_delay", "fastdelay", "fast+delay", "delay_fast", "fd", 
                       "fast_del", "retention_fastdelay", "retention_fast_delay"],
    }

    for lab in sorted(all_labels):
        mapped = None
        # Rest detection (exact or boundary-safe)
        if any(lab == r or lab.startswith(r + "_") or lab.endswith("_" + r)
               for r in REST_KEYWORDS):
            mapped = "rest"
        else:
            for canon, patterns in known_patterns.items():
                if any(p in lab for p in patterns):
                    mapped = canon
                    break
        if mapped:
            cond_map[lab] = mapped

    return cond_map, all_labels


def classify_condition(condition_str):
    """
    Classify a condition string into (canonical_name, cond_int, is_rest).
    Uses fuzzy matching for robustness.
    """
    cs = condition_str.lower().strip()

    # Rest
    if any(cs == r or cs.startswith(r + "_") or cs.endswith("_" + r)
           for r in REST_KEYWORDS):
        return "rest", -1, True

    # Fuzzy match to task conditions
    if "simul" in cs or "all_at_once" in cs:
        return "simultaneous", 0, False
    if ("fast" in cs and ("delay" in cs or "del" in cs)) or "fastdelay" in cs or "fast_delay" in cs:
        return "fast_delay", 3, False
    if cs == "slow" or ("slow" in cs and "fast" not in cs):
        return "slow", 1, False
    if cs == "fast" or ("fast" in cs and "delay" not in cs and "slow" not in cs):
        return "fast", 2, False

    # Exact fallback
    ci = COND_INT.get(cs, -1)
    if ci >= 0:
        for k, v in COND_INT.items():
            if v == ci:
                return k, ci, False

    return cs, -1, False


# Discover conditions before main loop
COND_MAP_PATH = PROCESSED / "event_label_map.json"
COND_MAP = {}

if COND_MAP_PATH.exists():
    with open(COND_MAP_PATH) as f:
        COND_MAP = json.load(f)
    print(f"Loaded existing event_label_map.json ({len(COND_MAP)} entries)")
else:
    print("No event_label_map.json found --- discovering from events...")
    COND_MAP, discovered_labels = discover_conditions_from_events(SUBJECTS, REGISTRY)
    if COND_MAP:
        with open(COND_MAP_PATH, "w") as f:
            json.dump(COND_MAP, f, indent=2)
        print(f"Discovered {len(discovered_labels)} unique labels, mapped {len(COND_MAP)}")
        print(f"  Labels found: {sorted(discovered_labels)}")
        print(f"  Mapping: {COND_MAP}")
    else:
        print("  WARNING: Could not discover any event labels. Will rely on fuzzy matching.")


# ============================================================
# EVENTS FILE RESOLUTION
# Subjects with both rest and verbalwm have multiple events files.
# We need verbalwm for task epochs and rest for rest epochs.
# ============================================================

def resolve_events_files(subj_dir: Path):
    """
    Find the appropriate events files for a subject.
    Returns (task_events_path, rest_events_path) or (None, None).
    """
    eeg_dir = subj_dir / "eeg"
    if not eeg_dir.is_dir():
        return None, None

    events_files = sorted(eeg_dir.glob("*_events.tsv"))
    if not events_files:
        return None, None

    task_ev = None
    rest_ev = None

    for ef in events_files:
        name_lower = ef.name.lower()
        if "verbalwm" in name_lower or "task-verbalwm" in name_lower:
            task_ev = str(ef)
        elif "rest" in name_lower or "task-rest" in name_lower:
            rest_ev = str(ef)

    # Fallback: if only one file, use it for both
    if task_ev is None and len(events_files) == 1:
        task_ev = str(events_files[0])
    if rest_ev is None and len(events_files) == 1:
        rest_ev = str(events_files[0])

    # Fallback: if task not found but rest found, and there's another file, use the other
    if task_ev is None and rest_ev is not None:
        for ef in events_files:
            if str(ef) != rest_ev:
                task_ev = str(ef)
                break

    # Fallback: if rest not found but task found, and there's another file, use the other  
    if rest_ev is None and task_ev is not None:
        for ef in events_files:
            if str(ef) != task_ev:
                rest_ev = str(ef)
                break

    return task_ev, rest_ev


def load_events_df(ev_path):
    """Load events TSV and normalize column names."""
    if not ev_path or not Path(ev_path).exists():
        return None
    try:
        df = pd.read_csv(ev_path, sep="	")
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as e:
        return None


def find_label_column(df):
    """Find the column containing event labels/trial types."""
    if df is None or df.empty:
        return None
    candidates = ["trial_type", "condition", "description", "event_type", 
                  "stim_type", "value", "event_name", "label"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_onset_column(df):
    """Find the onset column (seconds or samples)."""
    if df is None or df.empty:
        return None
    candidates = ["onset", "onset_s", "onset_sec", "onset (s)", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# HELPERS --- zero-phase filters (all signals, all uses)
# filtfilt guarantees zero group delay -> temporal alignment preserved
# ============================================================

def bp_filter(x, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    pad = min(len(x) - 1, 3 * max(len(b), len(a)))
    return filtfilt(b, a, x, padlen=pad)

def notch(x, freq, fs, Q=30):
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return filtfilt(b, a, x)


# ============================================================
# EEG PREPROCESSING
# Operates on MNE Raw object; returns modified Raw + QC dict.
# ECG/PPG channels are typed correctly before ICA so they are
# excluded from the decomposition --- their signals stay intact.
# ============================================================

def preprocess_eeg(raw, ecg_ch, ppg_ch, fs):
    qc = {
        "ica_n_excluded": 0,
        "ica_skipped": False,
        "ica_skip_reason": None,
        "ica_posterior_check_failed": False,
    }

    # Retype auxiliary channels FIRST --- before any ICA or picks
    type_map = {}
    if ecg_ch and ecg_ch in raw.ch_names:
        type_map[ecg_ch] = "ecg"
    if ppg_ch and ppg_ch in raw.ch_names:
        type_map[ppg_ch] = "misc"
    if type_map:
        raw.set_channel_types(type_map)

    # Bandpass EEG only --- 1 Hz lower bound avoids epoch ringing
    raw.filter(
        l_freq=1.0, h_freq=40.0,
        method="iir", iir_params=dict(order=4, ftype="butter"),
        picks="eeg", verbose=False,
    )
    raw.notch_filter(freqs=[50.0, 100.0], picks="eeg", verbose=False)

    # Average reference --- makes spectral estimates cross-subject comparable
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)

    # ICA --- conservative cardiac + ocular artifact removal
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    min_samples = int(20 * fs)

    if raw.n_times < min_samples:
        qc["ica_skipped"] = True
        qc["ica_skip_reason"] = f"Recording too short ({raw.n_times/fs:.1f}s)"
        return raw, qc

    if not (ecg_ch and ecg_ch in raw.ch_names):
        qc["ica_skipped"] = True
        qc["ica_skip_reason"] = "No ECG channel for cardiac IC identification"
        return raw, qc

    try:
        n_comp = min(20, n_eeg - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_comp, random_state=42,
            max_iter=800, verbose=False,
        )
        ica.fit(raw, picks="eeg", verbose=False)

        # Ocular artifacts --- EOG if present, else frontal heuristic
        eog_idx = []
        eog_chs = [c for c in raw.ch_names
                   if any(k in c.lower() for k in ["eog","veo","heo"])]
        if eog_chs:
            eog_idx, _ = ica.find_bads_eog(
                raw, ch_name=eog_chs[0], threshold=3.0, verbose=False)

        # Cardiac artifacts --- ECG correlation, threshold 0.4 (conservative)
        ecg_idx, _ = ica.find_bads_ecg(
            raw, ch_name=ecg_ch,
            method="correlation", threshold=0.4, verbose=False,
        )

        # Topography gate for cardiac ICs:
        posterior = {"O1","O2","Oz","P7","P8","P3","P4","Pz","PO3","PO4"}
        frontal = {"Fz","FCz","Cz","F3","F4","FC3","FC4","AFz","AF3","AF4"}
        eeg_names = [raw.ch_names[i]
                     for i in mne.pick_types(raw.info, eeg=True)]

        post_idx = [i for i, c in enumerate(eeg_names) if c in posterior]
        front_idx = [i for i, c in enumerate(eeg_names) if c in frontal]

        safe_cardiac = []
        for ic in ecg_idx:
            topo = ica.get_components()[:, ic]
            pp = np.mean(np.abs(topo[post_idx])) if post_idx else 0.0
            fp = np.mean(np.abs(topo[front_idx])) if front_idx else 0.0
            if pp > 1.5 * fp:
                safe_cardiac.append(ic)
            else:
                qc["ica_posterior_check_failed"] = True

        to_exclude = list(set(eog_idx) | set(safe_cardiac))
        if to_exclude:
            ica.exclude = to_exclude
            raw = ica.apply(raw, verbose=False)
            qc["ica_n_excluded"] = len(to_exclude)
        else:
            qc["ica_skipped"] = True
            qc["ica_skip_reason"] = (
                "No safe components to exclude "
                "(cardiac ICs are frontal-dominant --- HEP signal preserved)"
            )

    except Exception as e:
        qc["ica_skipped"] = True
        qc["ica_skip_reason"] = f"ICA error: {e}"

    return raw, qc


# ============================================================
# EPOCH EXTRACTION
# Retention windows only --- structurally identical across all 4
# WM conditions (6s silence post-encoding). This is the period
# where cognitive load differences are stable and HEP analysis
# is valid (no encoding-timing confound across conditions).
#
# Rest segments extracted with same time window for normalization.
# ============================================================

def _map_condition(label: str, cond_map: dict) -> str:
    """Map raw event label -> condition name using config event map."""
    lab = str(label).lower().strip()
    return cond_map.get(lab, lab)


def extract_epochs_from_df(events_df, raw_data, ch_names, eeg_idx, ecg_idx, ppg_idx,
                           fs, tmin, tmax, cond_map, epoch_type="task"):
    """
    Extract fixed-length epochs from a events DataFrame.

    epoch_type: "task" or "rest" --- controls which conditions are accepted.
    """
    n_total = raw_data.shape[1]
    pre_samp = int(tmin * fs)
    post_samp = int(tmax * fs)

    if events_df is None or events_df.empty:
        return []

    label_col = find_label_column(events_df)
    onset_col = find_onset_column(events_df)

    if onset_col is None:
        return []

    epochs = []
    for _, row in events_df.iterrows():
        # Get onset
        onset_val = float(row[onset_col])
        onset_samp = int(onset_val * fs) if onset_val < 1e4 else int(onset_val)

        # Get condition label
        if label_col is not None:
            raw_label = str(row[label_col])
            condition = _map_condition(raw_label, cond_map)
            condition, cond_int, is_rest = classify_condition(condition)
        else:
            # No label column --- skip unless we can infer from context
            continue

        # Filter by epoch type
        if epoch_type == "task" and is_rest:
            continue
        if epoch_type == "rest" and not is_rest:
            continue

        # Skip pure response/encoding markers for task epochs
        if epoch_type == "task" and cond_int < 0 and not is_rest:
            skip_keywords = RESPONSE_KEYWORDS | ENCODING_KEYWORDS
            if any(k in condition for k in skip_keywords):
                continue
            # If still unknown after filtering, skip
            if cond_int < 0:
                continue

        start = onset_samp + pre_samp
        end = onset_samp + post_samp

        if start < 0 or end > n_total:
            continue

        eeg_seg = raw_data[eeg_idx, start:end].astype(np.float32)
        ecg_seg = (raw_data[ecg_idx[0], start:end].astype(np.float32)
                   if ecg_idx else None)
        ppg_seg = (raw_data[ppg_idx[0], start:end].astype(np.float32)
                   if ppg_idx else None)

        epochs.append({
            "condition": condition,
            "label_int": cond_int,
            "is_rest": is_rest,
            "onset_samp": onset_samp,
            "eeg": eeg_seg,
            "ecg": ecg_seg,
            "ppg": ppg_seg,
        })

    return epochs


# ============================================================
# SAVE TO HDF5
# One file per subject; rest and task epochs stored separately.
# Structure allows Section 6 to load rest vs task independently.
# ============================================================

def save_subject_h5(subject_id, task_epochs, rest_epochs,
                    eeg_ch_names, fs, out_dir):
    path = Path(out_dir) / f"{subject_id}_preprocessed.h5"
    with h5py.File(path, "w") as hf:
        # Metadata
        m = hf.create_group("metadata")
        m.attrs["subject_id"] = subject_id
        m.attrs["fs"] = fs
        m.attrs["eeg_channels"] = json.dumps(eeg_ch_names)
        m.attrs["n_task"] = len(task_epochs)
        m.attrs["n_rest"] = len(rest_epochs)

        # Task epochs
        tg = hf.create_group("task")
        for i, ep in enumerate(task_epochs):
            g = tg.create_group(f"{i:04d}")
            g.create_dataset("eeg", data=ep["eeg"],
                             dtype="float32", compression="gzip")
            for sig in ("ecg","ppg"):
                arr = ep[sig] if ep[sig] is not None \
                    else np.zeros(ep["eeg"].shape[1], np.float32)
                g.create_dataset(sig, data=arr.astype(np.float32),
                                 compression="gzip")
            g.attrs["condition"] = ep["condition"]
            g.attrs["label_int"] = ep["label_int"]
            g.attrs["onset_samp"] = ep["onset_samp"]

        # Rest epochs
        rg = hf.create_group("rest")
        for i, ep in enumerate(rest_epochs):
            g = rg.create_group(f"{i:04d}")
            g.create_dataset("eeg", data=ep["eeg"],
                             dtype="float32", compression="gzip")
            for sig in ("ecg","ppg"):
                arr = ep[sig] if ep[sig] is not None \
                    else np.zeros(ep["eeg"].shape[1], np.float32)
                g.create_dataset(sig, data=arr.astype(np.float32),
                                 compression="gzip")
            g.attrs["condition"] = ep["condition"]
            g.attrs["onset_samp"] = ep["onset_samp"]

    return str(path)


# ============================================================
# MAIN PREPROCESSING LOOP
# One subject at a time --- RAM released between subjects via gc.
# ============================================================

print("="*55)
print("PREPROCESSING")
print("="*55)

qc_all = {}
preprocessed_paths = {}
tmin = float(CONFIG["retention_tmin"])
tmax = float(CONFIG["retention_tmax"])

for subj_id in SUBJECTS:
    reg = REGISTRY[subj_id]
    vhdr = reg["vhdr_path"]

    # Resolve events files (verbalwm for task, rest for rest)
    subj_dir = Path(vhdr).parent.parent
    task_ev_path, rest_ev_path = resolve_events_files(subj_dir)

    # Fallback to registry path if resolution fails
    if task_ev_path is None and reg.get("events_path"):
        task_ev_path = reg["events_path"]

    ecg_ch = (reg["ecg_channels"][0]
              if reg["ecg_channels"]
              else CONFIG.get("ecg_canonical_ch"))
    ppg_ch = (reg["ppg_channels"][0]
              if reg["ppg_channels"]
              else CONFIG.get("ppg_canonical_ch"))

    subj_qc = {
        "status": "unknown",
        "skip_reason": None,
        "n_task": 0,
        "n_rest": 0,
        "condition_counts": {},
        "ica": {},
        "flags": [],
        "task_events_file": task_ev_path,
        "rest_events_file": rest_ev_path,
    }

    try:
        # Load raw (lazy)
        raw = mne.io.read_raw_brainvision(
            vhdr_fname=vhdr, preload=False, verbose=False)

        # Resample to cohort target if needed
        fs = raw.info["sfreq"]
        target_fs = float(CONFIG["cohort_sfreq"] or fs)
        if fs != target_fs:
            raw.load_data(verbose=False)
            raw.resample(target_fs, verbose=False)
            fs = raw.info["sfreq"]

        # Preprocess EEG
        raw.load_data(verbose=False)
        raw, ica_qc = preprocess_eeg(raw, ecg_ch, ppg_ch, fs)
        subj_qc["ica"] = ica_qc

        # Get channel indices after ICA
        eeg_idx = mne.pick_types(raw.info, eeg=True, exclude=[]).tolist()
        ecg_idx = ([raw.ch_names.index(ecg_ch)]
                   if ecg_ch and ecg_ch in raw.ch_names else [])
        ppg_idx = ([raw.ch_names.index(ppg_ch)]
                   if ppg_ch and ppg_ch in raw.ch_names else [])
        eeg_ch_names = [raw.ch_names[i] for i in eeg_idx]

        # Pull full data array once
        data = raw.get_data()
        del raw

        # Load events dataframes
        task_events_df = load_events_df(task_ev_path)
        rest_events_df = load_events_df(rest_ev_path)

        # Debug info
        if task_events_df is not None:
            subj_qc["flags"].append(f"task_events_cols: {list(task_events_df.columns)}")
            subj_qc["flags"].append(f"task_events_rows: {len(task_events_df)}")
        if rest_events_df is not None:
            subj_qc["flags"].append(f"rest_events_cols: {list(rest_events_df.columns)}")
            subj_qc["flags"].append(f"rest_events_rows: {len(rest_events_df)}")

        # Extract task epochs from verbalwm events
        task_epochs = extract_epochs_from_df(
            task_events_df, data, eeg_ch_names,
            eeg_idx, ecg_idx, ppg_idx,
            fs, tmin, tmax, COND_MAP, epoch_type="task"
        )

        # Extract rest epochs from rest events (or verbalwm if rest file missing)
        rest_epochs = extract_epochs_from_df(
            rest_events_df, data, eeg_ch_names,
            eeg_idx, ecg_idx, ppg_idx,
            fs, tmin, tmax, COND_MAP, epoch_type="rest"
        )

        # Fallback: if no rest epochs from rest file, try to get rest from task file
        if not rest_epochs and task_events_df is not None:
            rest_epochs = extract_epochs_from_df(
                task_events_df, data, eeg_ch_names,
                eeg_idx, ecg_idx, ppg_idx,
                fs, tmin, tmax, COND_MAP, epoch_type="rest"
            )

        del data

        if not task_epochs:
            # Debug: show what labels were found
            if task_events_df is not None and not task_events_df.empty:
                label_col = find_label_column(task_events_df)
                if label_col:
                    labs = task_events_df[label_col].dropna().astype(str).unique()
                else:
                    labs = task_events_df.iloc[:, -1].astype(str).unique()
                subj_qc["flags"].append(f"no_task_epochs; found labels: {list(labs)[:10]}")
            raise ValueError("No task epochs extracted")

        if not rest_epochs:
            subj_qc["flags"].append(
                "no_rest_epochs --- RS normalization will use global mean fallback")

        subj_qc["condition_counts"] = dict(
            Counter(e["condition"] for e in task_epochs))
        subj_qc["n_task"] = len(task_epochs)
        subj_qc["n_rest"] = len(rest_epochs)

        # Save
        h5_path = save_subject_h5(
            subj_id, task_epochs, rest_epochs,
            eeg_ch_names, fs, str(PROCESSED)
        )
        preprocessed_paths[subj_id] = h5_path
        subj_qc["status"] = "success"

        ica_str = (f"ICA_excl={ica_qc['ica_n_excluded']}"
                   if not ica_qc["ica_skipped"]
                   else f"ICA_skip({ica_qc['ica_skip_reason'][:30]})")
        print(f" ✓ {subj_id:<12} "
              f"task={len(task_epochs):>3} "
              f"rest={len(rest_epochs):>3} "
              f"conds={subj_qc['condition_counts']} "
              f"{ica_str}")

    except Exception as exc:
        subj_qc["status"] = "skip"
        subj_qc["skip_reason"] = str(exc)
        print(f" ✗ {subj_id:<12} SKIP: {exc}")

    qc_all[subj_id] = subj_qc
    gc.collect()


# ============================================================
# SAVE QC + UPDATE CONFIG
# ============================================================

qc_path = PROCESSED / "preprocessing_qc.json"
with open(qc_path, "w") as f:
    json.dump(qc_all, f, indent=2, default=str)

CONFIG["preprocessed_subjects"] = preprocessed_paths
save_config(CONFIG)


# ============================================================
# SUMMARY
# ============================================================

n_ok = sum(1 for v in qc_all.values() if v["status"] == "success")
n_skip = sum(1 for v in qc_all.values() if v["status"] == "skip")
total_task = sum(v["n_task"] for v in qc_all.values() if v["status"]=="success")
total_rest = sum(v["n_rest"] for v in qc_all.values() if v["status"]=="success")

print(f"\n{'='*55}")
print(f"PREPROCESSING SUMMARY")
print(f"{'='*55}")
print(f" Success          : {n_ok}/{len(SUBJECTS)}")
print(f" Skipped          : {n_skip}")
print(f" Total task epochs: {total_task}")
print(f" Total rest epochs: {total_rest}")
print(f" QC saved         : {qc_path.name}")
print(f" Config updated   : config.json")
print(f"\nSection 5 COMPLETE")
print(f"Next: Section 6 --- Representation Construction (HEST extraction)")