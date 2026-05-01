# ============================================================
# RS-HESR | Section 6 — Representation Construction (HEST extraction)
# FIXED VERSION: Robust channel mapping + dimension consistency + validation
# 
# Input:  processed/sub-XXX_preprocessed.h5 (from Section 5)
# Output: processed/sub-XXX_hest.h5 (per subject)
#         processed/hest_qc.json
#         processed/figures/section6_hest_summary.png
#         updated config.json with hest_subjects mapping
# ============================================================

import os, json, warnings, gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
def load_config(root="processed"):
    with open(Path(root) / "config.json") as f:
        return json.load(f)

def save_config(cfg):
    with open(Path(cfg["processed_root"]) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()
PROCESSED = Path(CONFIG["processed_root"])
FIG_DIR = Path(CONFIG.get("figures", str(PROCESSED / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Subjects that successfully passed Section 5
PREPROCESSED = CONFIG.get("preprocessed_subjects", {})
SUBJECTS = sorted(PREPROCESSED.keys())

print(f"HEST extraction for {len(SUBJECTS)} subjects")
print(f"Output dir: {PROCESSED}\n")

# ── Constants ─────────────────────────────────────────────────
# CRITICAL FIX: Read actual sampling rate from config, default to 1000 Hz
FS = float(CONFIG.get("cohort_sfreq", 1000.0))

# HEST window around R-peak (ms)
HEST_PRE_MS = 400   # before R-peak
HEST_POST_MS = 400  # after R-peak
# Temporal downsampling factor (400+400ms at 1000Hz = 800 samples -> 160 after /5)
HEST_DOWNSAMPLE = 5

# CRITICAL FIX: Calculate expected times consistently
N_HEP_TIMES = int((HEST_PRE_MS + HEST_POST_MS) / 1000.0 * FS / HEST_DOWNSAMPLE)

# Frequency bands for Morlet wavelets
FREQ_BANDS = {
    "theta": (4, 7),
    "alpha_low": (8, 10),
    "alpha_high": (10, 13),
    "beta_low": (13, 18),
}

# Literature-prior cardiac-sensitive channels [P25, Candia-Rivera 2022]
TARGET_CHANNELS = ["Fz", "FCz", "Cz", "F3", "F4", "FC3", "FC4", "Pz", "CPz"]

# PPG feature names (order must match extraction)
PPG_FEATURE_NAMES = [
    "ibi_mean", "ibi_range", "ibi_slope", "ibi_r2", "ibi_samp_en", "ibi_d1_mean",
    "ppg_amp_mean", "ppg_amp_var", "ppg_rise_mean", "ppg_width_mean",
    "ppg_area_mean", "ppg_area_var",
]

print(f"Sampling rate: {FS} Hz")
print(f"HEST window: {HEST_PRE_MS}+{HEST_POST_MS}ms, downsample={HEST_DOWNSAMPLE}")
print(f"Expected HEST shape per epoch: ({len(TARGET_CHANNELS)}, {len(FREQ_BANDS)}, {N_HEP_TIMES})")
print(f"Flat feature dimension: {len(TARGET_CHANNELS) * len(FREQ_BANDS) * N_HEP_TIMES}\n")


# ============================================================
# ROBUST CHANNEL MAPPING — Fixed for ds006848 channel variability
# ============================================================

# Channel priority map for neighbor substitution when exact channel missing
# Neighbors are ordered by spatial proximity and functional relevance
CHANNEL_PRIORITY_MAP = {
    "Fz":     {"exact": ["Fz"],     "neighbors": ["F1", "F2", "AFz", "FCz"]},
    "FCz":    {"exact": ["FCz"],    "neighbors": ["FC1", "FC2", "Cz", "Fz"]},
    "Cz":     {"exact": ["Cz"],     "neighbors": ["C1", "C2", "FCz", "CPz"]},
    "F3":     {"exact": ["F3"],     "neighbors": ["F1", "F5", "AF3", "FC3"]},
    "F4":     {"exact": ["F4"],     "neighbors": ["F2", "F6", "AF4", "FC4"]},
    "FC3":    {"exact": ["FC3"],    "neighbors": ["FC1", "FC5", "C3", "F3"]},
    "FC4":    {"exact": ["FC4"],    "neighbors": ["FC2", "FC6", "C4", "F4"]},
    "Pz":     {"exact": ["Pz"],     "neighbors": ["P1", "P2", "POz", "CPz"]},
    "CPz":    {"exact": ["CPz"],    "neighbors": ["CP1", "CP2", "Pz", "Cz"]},
}


def robust_channel_map(ch_names, target_channels=TARGET_CHANNELS, priority_map=CHANNEL_PRIORITY_MAP):
    """
    Map target channels to available EEG channels with robust fallback.

    Strategy (in order):
      1. Exact match (case-sensitive)
      2. Case-insensitive match
      3. Neighbor substitution from priority map (exact then case-insensitive)
      4. Zero-filled placeholder (-1 sentinel) if truly missing

    Returns:
        ch_idx: list of indices into ch_names (length = len(target_channels))
                Index -1 means "use zero placeholder"
        ch_map_log: dict with mapping decisions for each target
        n_missing: number of channels that required substitution or placeholder
    """
    ch_names_norm = [c.strip() for c in ch_names]
    ch_lower = [c.lower() for c in ch_names_norm]

    ch_idx = []
    ch_map_log = {}
    n_missing = 0

    for target in target_channels:
        mapped = False
        log_entry = {"target": target, "mapped_to": None, "method": None}

        # 1. Exact match
        if target in ch_names_norm:
            idx = ch_names_norm.index(target)
            ch_idx.append(idx)
            log_entry["mapped_to"] = ch_names_norm[idx]
            log_entry["method"] = "exact"
            ch_map_log[target] = log_entry
            mapped = True
            continue

        # 2. Case-insensitive match
        target_lower = target.lower()
        if target_lower in ch_lower:
            idx = ch_lower.index(target_lower)
            ch_idx.append(idx)
            log_entry["mapped_to"] = ch_names_norm[idx]
            log_entry["method"] = "case_insensitive"
            ch_map_log[target] = log_entry
            mapped = True
            continue

        # 3. Neighbor substitution
        if target in priority_map:
            for neighbor in priority_map[target]["neighbors"]:
                # Exact neighbor
                if neighbor in ch_names_norm:
                    idx = ch_names_norm.index(neighbor)
                    ch_idx.append(idx)
                    log_entry["mapped_to"] = ch_names_norm[idx]
                    log_entry["method"] = f"neighbor({neighbor})"
                    ch_map_log[target] = log_entry
                    mapped = True
                    n_missing += 1
                    break
                # Case-insensitive neighbor
                neighbor_lower = neighbor.lower()
                if neighbor_lower in ch_lower:
                    idx = ch_lower.index(neighbor_lower)
                    ch_idx.append(idx)
                    log_entry["mapped_to"] = ch_names_norm[idx]
                    log_entry["method"] = f"neighbor_case({neighbor})"
                    ch_map_log[target] = log_entry
                    mapped = True
                    n_missing += 1
                    break

        if not mapped:
            # 4. Zero placeholder
            ch_idx.append(-1)
            log_entry["mapped_to"] = "ZERO_PLACEHOLDER"
            log_entry["method"] = "placeholder"
            ch_map_log[target] = log_entry
            n_missing += 1

    return ch_idx, ch_map_log, n_missing


# ============================================================
# ECG R-PEAK DETECTION
# Pan-Tompkins-like derivative + threshold on bandpassed ECG
# ============================================================

def detect_r_peaks(ecg_signal, fs):
    """
    Detect R-peaks in ECG signal.
    Returns array of sample indices.
    """
    # Bandpass 5-40 Hz to enhance QRS
    nyq = fs / 2.0
    b, a = butter(3, [5/nyq, 40/nyq], btype="band")
    ecg_filt = filtfilt(b, a, ecg_signal)

    # Differentiate, square, moving average
    diff = np.diff(ecg_filt)
    squared = diff ** 2
    window = int(0.15 * fs)  # 150ms window
    ma = np.convolve(squared, np.ones(window)/window, mode="same")

    # Find peaks with minimum 300ms separation (~200 BPM max)
    min_dist = int(0.3 * fs)
    peaks, _ = find_peaks(ma, distance=min_dist, prominence=np.std(ma)*0.5)

    # Refine: find actual max in original filtered signal near each peak
    r_peaks = []
    search = int(0.05 * fs)  # ±50ms search window
    for p in peaks:
        start = max(0, p - search)
        end = min(len(ecg_filt), p + search)
        r_peaks.append(start + np.argmax(ecg_filt[start:end]))

    return np.array(r_peaks, dtype=int)


def clean_r_peaks(r_peaks, fs):
    """Remove ectopic beats by RR interval outlier rejection."""
    if len(r_peaks) < 3:
        return r_peaks
    rr = np.diff(r_peaks) / fs * 1000.0  # ms
    median_rr = np.median(rr)
    # Keep beats where RR is within 30% of median
    valid = np.abs(rr - median_rr) < 0.3 * median_rr
    valid = np.concatenate([[True], valid])  # first peak always valid
    return r_peaks[valid]


# ============================================================
# MORLET WAVELET TIME-FREQUENCY
# Narrow wavelets (n_cycles=4) for time precision
# ============================================================

def morlet_wavelet_tf(signal, fs, fmin, fmax, n_cycles=4, n_freqs=5):
    """
    Compute time-frequency power using Morlet wavelets.
    Returns power (n_freqs, n_times).
    """
    from scipy.signal import morlet2, convolve

    freqs = np.linspace(fmin, fmax, n_freqs)
    sigma_t = n_cycles / (2 * np.pi * freqs)
    max_t = 5 * sigma_t.max()
    t = np.arange(-max_t, max_t + 1/fs, 1/fs)

    power = np.zeros((n_freqs, len(signal)))
    for i, f in enumerate(freqs):
        wavelet = morlet2(len(t), w=n_cycles, s=1.0)
        # Normalize energy
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        conv = convolve(signal, wavelet, mode="same")
        power[i] = np.abs(conv) ** 2

    return power, freqs


def extract_band_power(eeg_segment, fs, band_name):
    """
    Extract average power in a frequency band using Morlet wavelets.
    Returns (n_freqs_band, n_times) power.
    """
    fmin, fmax = FREQ_BANDS[band_name]
    n_cycles = max(4, int((fmin + fmax) / 2 * 0.5))
    power, _ = morlet_wavelet_tf(eeg_segment, fs, fmin, fmax, n_cycles=n_cycles, n_freqs=3)
    return power


# ============================================================
# HEST EXTRACTION (per trial) — FIXED with robust channel mapping
# ============================================================

def extract_hest_trial(eeg_trial, ecg_trial, fs, ch_names):
    """
    Extract Heartbeat-Evoked Spectrotemporal representation for one trial.
    FIXED: Robust channel mapping with fallback for missing channels.

    eeg_trial: (n_eeg_ch, n_times)
    ecg_trial: (n_times,)
    ch_names: list of EEG channel names

    Returns:
        hest: (n_target_ch, n_bands, n_timepoints) — averaged over beats
        n_beats: number of heartbeats used
        rr_seq: RR intervals during retention (ms)
        ch_map_log: dict of channel mapping decisions
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_trial, fs)
    r_peaks = clean_r_peaks(r_peaks, fs)

    if len(r_peaks) < 2:
        return None, 0, None, {}

    # RR sequence
    rr_seq = np.diff(r_peaks) / fs * 1000.0  # ms

    # HEST window in samples
    pre_samp = int(HEST_PRE_MS / 1000.0 * fs)
    post_samp = int(HEST_POST_MS / 1000.0 * fs)
    n_times = pre_samp + post_samp

    # === ROBUST CHANNEL MAPPING ===
    ch_idx, ch_map_log, n_missing = robust_channel_map(ch_names)

    # Log mapping decisions (only if non-exact)
    if n_missing > 0:
        missing_details = [f"{k}->{v['mapped_to']}({v['method']})" 
                          for k, v in ch_map_log.items() if v['method'] != 'exact']
        print(f"    [ChannelMap] {n_missing} non-exact mappings: {missing_details}")

    n_ch = len(TARGET_CHANNELS)  # Always 9 — guaranteed by design
    n_bands = len(FREQ_BANDS)
    hest_accum = []

    for rp in r_peaks:
        start = rp - pre_samp
        end = rp + post_samp
        if start < 0 or end > eeg_trial.shape[1]:
            continue

        # Build segment with placeholder handling
        # Result: (n_target_ch, n_bands, n_times)
        trial_tf = np.zeros((n_ch, n_bands, n_times))

        for c_idx_out, c_idx_in in enumerate(ch_idx):
            if c_idx_in == -1:
                # Zero placeholder — already zeros, skip computation
                continue

            # Extract single channel segment
            segment_ch = eeg_trial[c_idx_in, start:end]  # (n_times,)

            # Compute time-frequency power per band
            for b_idx, band_name in enumerate(FREQ_BANDS.keys()):
                power = extract_band_power(segment_ch, fs, band_name)
                trial_tf[c_idx_out, b_idx, :] = np.mean(power, axis=0)

        hest_accum.append(trial_tf)

    if len(hest_accum) == 0:
        return None, 0, rr_seq, ch_map_log

    # Average over heartbeats
    hest = np.mean(hest_accum, axis=0)  # (n_ch, n_bands, n_times)

    # Temporal downsampling (factor 5)
    if hest.shape[2] > HEST_DOWNSAMPLE:
        hest = hest[:, :, ::HEST_DOWNSAMPLE]

    return hest, len(hest_accum), rr_seq, ch_map_log


# ============================================================
# RESTING-STATE NORMALIZATION
# Per-subject: z-score task HEST by resting-state HEST statistics
# This is the cross-subject invariance mechanism
# ============================================================

def compute_rest_hest_stats(h5_path, fs):
    """
    Compute mean and std of HEST across all rest epochs for a subject.
    Returns (mean, std) or (None, None) if no rest epochs.
    """
    rest_hests = []
    ch_map_log_global = {}

    with h5py.File(h5_path, "r") as hf:
        if "rest" not in hf:
            return None, None, {}

        ch_names = json.loads(hf["metadata"].attrs["eeg_channels"])
        rest_group = hf["rest"]

        for ep_key in rest_group.keys():
            ep = rest_group[ep_key]
            eeg = ep["eeg"][:]
            ecg = ep["ecg"][:]

            hest, n_beats, _, ch_map_log = extract_hest_trial(eeg, ecg, fs, ch_names)
            if hest is not None and n_beats >= 3:
                rest_hests.append(hest)
                # Store mapping from first successful epoch
                if not ch_map_log_global:
                    ch_map_log_global = ch_map_log

    if len(rest_hests) == 0:
        return None, None, ch_map_log_global

    rest_stack = np.stack(rest_hests, axis=0)  # (n_rest_trials, ch, bands, times)
    rest_mean = np.mean(rest_stack, axis=0)
    rest_std = np.std(rest_stack, axis=0) + 1e-8  # avoid div by zero

    return rest_mean, rest_std, ch_map_log_global


def normalize_hest(hest_task, rest_mean, rest_std):
    """Z-score normalization: (task - rest_mean) / rest_std"""
    if rest_mean is None:
        return hest_task  # no normalization possible
    return (hest_task - rest_mean) / rest_std


# ============================================================
# PPG FEATURE EXTRACTION
# Short-window features valid at WM retention timescales
# ============================================================

def extract_ppg_features(ppg_signal, fs):
    """
    Extract PPG morphology and short-window IBI features.
    Returns dict of 12 features.
    """
    features = {}

    # Bandpass 0.5-8 Hz
    nyq = fs / 2.0
    b, a = butter(3, [0.5/nyq, 8/nyq], btype="band")
    ppg_filt = filtfilt(b, a, ppg_signal)

    # Peak detection
    min_dist = int(0.4 * fs)  # 400ms -> max 150 BPM
    peaks, props = find_peaks(ppg_filt, distance=min_dist,
                              prominence=0.3 * np.std(ppg_filt))

    if len(peaks) < 3:
        # Return zeros if insufficient data
        return {k: 0.0 for k in PPG_FEATURE_NAMES}

    # IBI sequence
    ibi = np.diff(peaks) / fs * 1000.0  # ms

    # --- Short-window IBI features ---
    features["ibi_mean"] = float(np.mean(ibi))
    features["ibi_range"] = float(np.max(ibi) - np.min(ibi))

    # IBI trend
    t = np.arange(len(ibi))
    slope, intercept = np.polyfit(t, ibi, 1)
    features["ibi_slope"] = float(slope)
    pred = slope * t + intercept
    ss_res = np.sum((ibi - pred) ** 2)
    ss_tot = np.sum((ibi - np.mean(ibi)) ** 2)
    features["ibi_r2"] = float(1 - ss_res / (ss_tot + 1e-8))

    # Sample entropy (simplified)
    features["ibi_samp_en"] = float(np.std(ibi) / (np.mean(ibi) + 1e-8))

    # First derivative mean
    features["ibi_d1_mean"] = float(np.mean(np.diff(ibi)))

    # --- Pulse morphology ---
    if len(peaks) >= 2:
        amplitudes = []
        rise_times = []
        widths = []
        areas = []

        for i in range(len(peaks) - 1):
            segment = ppg_filt[peaks[i]:peaks[i+1]]
            if len(segment) < 5:
                continue

            amp = np.max(segment) - np.min(segment)
            amplitudes.append(amp)

            # Rise time: time from min to max
            max_idx = np.argmax(segment)
            rise_times.append(max_idx / fs * 1000.0)  # ms

            # Width at half amplitude
            half = (np.max(segment) + np.min(segment)) / 2
            above_half = np.where(segment >= half)[0]
            if len(above_half) > 1:
                widths.append((above_half[-1] - above_half[0]) / fs * 1000.0)
            else:
                widths.append(0.0)

            # Area under pulse
            areas.append(np.trapz(segment) / fs)

        features["ppg_amp_mean"] = float(np.mean(amplitudes)) if amplitudes else 0.0
        features["ppg_amp_var"] = float(np.var(amplitudes)) if amplitudes else 0.0
        features["ppg_rise_mean"] = float(np.mean(rise_times)) if rise_times else 0.0
        features["ppg_width_mean"] = float(np.mean(widths)) if widths else 0.0
        features["ppg_area_mean"] = float(np.mean(areas)) if areas else 0.0
        features["ppg_area_var"] = float(np.var(areas)) if areas else 0.0
    else:
        for k in ["ppg_amp_mean", "ppg_amp_var", "ppg_rise_mean",
                  "ppg_width_mean", "ppg_area_mean", "ppg_area_var"]:
            features[k] = 0.0

    return features


# ============================================================
# SAVE REPRESENTATION
# CRITICAL FIX: Add validation after save
# ============================================================

def save_hest_h5(subject_id, task_representations, rest_mean, rest_std,
                 eeg_ch_names, fs, out_dir, ch_map_log=None):
    """
    Save HEST representations to HDF5.
    Each task epoch: hest_norm, rr_sequence, ppg_features, label

    CRITICAL FIX: Validates file after writing to ensure persistence.
    """
    path = Path(out_dir) / f"{subject_id}_hest.h5"

    with h5py.File(path, "w") as hf:
        # Metadata
        m = hf.create_group("metadata")
        m.attrs["subject_id"] = subject_id
        m.attrs["fs"] = fs
        m.attrs["eeg_channels"] = json.dumps(eeg_ch_names)
        m.attrs["target_channels"] = json.dumps(TARGET_CHANNELS)
        m.attrs["freq_bands"] = json.dumps(list(FREQ_BANDS.keys()))
        m.attrs["n_task_epochs"] = len(task_representations)
        m.attrs["hest_shape"] = json.dumps([len(TARGET_CHANNELS), len(FREQ_BANDS), N_HEP_TIMES])

        # Store channel mapping log for reproducibility
        if ch_map_log:
            m.attrs["channel_map"] = json.dumps(ch_map_log)

        # Rest statistics for documentation
        if rest_mean is not None:
            hf.create_dataset("rest_mean", data=rest_mean, compression="gzip")
            hf.create_dataset("rest_std", data=rest_std, compression="gzip")

        # Task epochs
        tg = hf.create_group("task")
        for i, rep in enumerate(task_representations):
            g = tg.create_group(f"{i:04d}")
            g.create_dataset("hest_norm", data=rep["hest_norm"],
                             dtype="float32", compression="gzip")
            g.create_dataset("rr_sequence", data=rep["rr_sequence"],
                             dtype="float32", compression="gzip")
            g.create_dataset("ppg_features", data=rep["ppg_features"],
                             dtype="float32", compression="gzip")
            g.attrs["condition"] = rep["condition"]
            g.attrs["label_int"] = rep["label_int"]
            g.attrs["n_beats"] = rep["n_beats"]

    # CRITICAL FIX: Validate file exists and is readable
    if not path.exists():
        raise RuntimeError(f"File was not created: {path}")

    # Verify structure
    with h5py.File(path, "r") as hf:
        if "task" not in hf:
            raise RuntimeError(f"Missing 'task' group in {path}")
        n_saved = len(hf["task"])
        if n_saved != len(task_representations):
            raise RuntimeError(f"Epoch count mismatch: saved {n_saved}, expected {len(task_representations)}")
        # Verify first epoch has hest_norm
        first_key = sorted(hf["task"].keys())[0]
        if "hest_norm" not in hf["task"][first_key]:
            raise RuntimeError(f"Missing 'hest_norm' dataset in first epoch")
        hest_shape = hf["task"][first_key]["hest_norm"].shape
        expected_shape = (len(TARGET_CHANNELS), len(FREQ_BANDS), N_HEP_TIMES)
        if hest_shape != expected_shape:
            raise RuntimeError(f"HEST shape mismatch: got {hest_shape}, expected {expected_shape}")

    return str(path)


# ============================================================
# MAIN HEST EXTRACTION LOOP
# ============================================================

print("=" * 55)
print("HEST EXTRACTION")
print("=" * 55)

qc_all = {}
hest_paths = {}
condition_counts = defaultdict(int)

for subj_id in SUBJECTS:
    h5_path = PREPROCESSED[subj_id]
    subj_qc = {
        "status": "unknown",
        "skip_reason": None,
        "n_task": 0,
        "n_rest": 0,
        "n_beats_total": 0,
        "conditions": {},
        "channel_map": {},
        "n_placeholder_ch": 0,
    }

    try:
        # First pass: compute resting-state statistics
        rest_mean, rest_std, ch_map_log_rest = compute_rest_hest_stats(h5_path, FS)

        with h5py.File(h5_path, "r") as hf:
            ch_names = json.loads(hf["metadata"].attrs["eeg_channels"])
            fs_file = hf["metadata"].attrs["fs"]

            task_group = hf["task"]
            task_reps = []
            ch_map_log_task = {}

            for ep_key in task_group.keys():
                ep = task_group[ep_key]
                eeg = ep["eeg"][:]
                ecg = ep["ecg"][:]
                ppg = ep["ppg"][:]
                condition = ep.attrs["condition"]
                label_int = ep.attrs["label_int"]

                # Extract HEST with robust channel mapping
                hest, n_beats, rr_seq, ch_map_log = extract_hest_trial(eeg, ecg, fs_file, ch_names)

                if hest is None:
                    continue

                # Store mapping log from first successful epoch
                if not ch_map_log_task:
                    ch_map_log_task = ch_map_log

                # Normalize by resting state
                hest_norm = normalize_hest(hest, rest_mean, rest_std)

                # Extract PPG features
                ppg_feats = extract_ppg_features(ppg, fs_file)
                ppg_vec = np.array([ppg_feats[k] for k in PPG_FEATURE_NAMES],
                                   dtype=np.float32)

                # Pad/truncate RR sequence to max 10 beats
                rr_padded = np.zeros(10, dtype=np.float32)
                if rr_seq is not None and len(rr_seq) > 0:
                    n_rr = min(len(rr_seq), 10)
                    rr_padded[:n_rr] = rr_seq[:n_rr]

                task_reps.append({
                    "hest_norm": hest_norm.astype(np.float32),
                    "rr_sequence": rr_padded,
                    "ppg_features": ppg_vec,
                    "condition": condition,
                    "label_int": label_int,
                    "n_beats": n_beats,
                })

                subj_qc["n_beats_total"] += n_beats
                condition_counts[condition] += 1

            subj_qc["n_task"] = len(task_reps)
            subj_qc["conditions"] = dict(Counter(r["condition"] for r in task_reps))
            subj_qc["n_rest"] = len(hf["rest"]) if "rest" in hf else 0

        if not task_reps:
            raise ValueError("No valid HEST representations extracted")

        # Count placeholder channels
        n_placeholders = sum(1 for v in ch_map_log_task.values() if v["method"] == "placeholder")
        subj_qc["n_placeholder_ch"] = n_placeholders
        subj_qc["channel_map"] = ch_map_log_task

        # Save with validation
        hest_path = save_hest_h5(
            subj_id, task_reps, rest_mean, rest_std,
            ch_names, fs_file, str(PROCESSED), ch_map_log=ch_map_log_task
        )
        hest_paths[subj_id] = hest_path
        subj_qc["status"] = "success"

        rest_info = "RS_norm" if rest_mean is not None else "NO_REST"
        placeholder_info = f" placeholders={n_placeholders}" if n_placeholders > 0 else ""
        print(f" ✓ {subj_id:<12} "
              f"task={len(task_reps):>3} "
              f"beats={subj_qc['n_beats_total']:>4} "
              f"conds={subj_qc['conditions']} "
              f"[{rest_info}]{placeholder_info}")

    except Exception as exc:
        subj_qc["status"] = "skip"
        subj_qc["skip_reason"] = str(exc)
        print(f" ✗ {subj_id:<12} SKIP: {exc}")

    qc_all[subj_id] = subj_qc
    gc.collect()


# ============================================================
# SAVE QC + UPDATE CONFIG
# CRITICAL FIX: Atomic config update with verification
# ============================================================

qc_path = PROCESSED / "hest_qc.json"
with open(qc_path, "w") as f:
    json.dump(qc_all, f, indent=2, default=str)

# CRITICAL FIX: Only update config if we have valid hest files
if hest_paths:
    CONFIG["hest_subjects"] = hest_paths
    # Also store dimension metadata for model loader
    CONFIG["hest_flat_dim"] = len(TARGET_CHANNELS) * len(FREQ_BANDS) * N_HEP_TIMES
    CONFIG["hest_shape"] = [len(TARGET_CHANNELS), len(FREQ_BANDS), N_HEP_TIMES]
    CONFIG["target_channels"] = TARGET_CHANNELS
    CONFIG["freq_bands"] = list(FREQ_BANDS.keys())
    save_config(CONFIG)
    print(f"\n✓ Config updated: {len(hest_paths)} subjects in hest_subjects")
else:
    print(f"\n⚠ WARNING: No HEST files produced. Config NOT updated.")
    raise RuntimeError("HEST extraction failed for all subjects. Check logs above.")


# ============================================================
# VISUALIZATION: HEST summary
# ============================================================

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Section 6 — HEST Representation Summary", fontsize=13, fontweight="bold")

gs = fig.add_gridspec(3, 3)

# Panel A: Success rate
ax1 = fig.add_subplot(gs[0, 0])
status = {"success": 0, "skip": 0}
for v in qc_all.values():
    status[v["status"]] += 1
colors = {"success": "#2ecc71", "skip": "#e74c3c"}
ax1.bar(status.keys(), status.values(), color=[colors[k] for k in status.keys()], edgecolor="white")
ax1.set_title("Subject Status", fontweight="bold")
ax1.set_ylabel("Count")
for i, (k, v) in enumerate(status.items()):
    ax1.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# Panel B: Condition distribution
ax2 = fig.add_subplot(gs[0, 1:])
conds = sorted(condition_counts.keys())
vals = [condition_counts[c] for c in conds]
colors_cond = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
ax2.bar(conds, vals, color=colors_cond[:len(conds)], edgecolor="white")
ax2.set_title("Task Epoch Distribution by Condition", fontweight="bold")
ax2.set_ylabel("Count")
for i, v in enumerate(vals):
    ax2.text(i, v + 5, str(v), ha="center", fontweight="bold")

# Panel C: Beats per subject
ax3 = fig.add_subplot(gs[1, 0])
sids = [s for s in SUBJECTS if qc_all[s]["status"] == "success"]
beats = [qc_all[s]["n_beats_total"] for s in sids]
ax3.barh(range(len(sids)), beats, color="#9b59b6", alpha=0.8, edgecolor="white")
ax3.set_yticks(range(len(sids)))
ax3.set_yticklabels([s.replace("sub-", "") for s in sids], fontsize=7)
ax3.set_title("Total Heartbeats Detected", fontweight="bold")
ax3.set_xlabel("Count")

# Panel D: Rest normalization availability
ax4 = fig.add_subplot(gs[1, 1])
has_rest = sum(1 for s in SUBJECTS if qc_all[s]["status"] == "success"
               and qc_all[s].get("n_rest", 0) > 0)
no_rest = sum(1 for s in SUBJECTS if qc_all[s]["status"] == "success"
              and qc_all[s].get("n_rest", 0) == 0)
ax4.pie([has_rest, no_rest], labels=["RS normalized", "No rest fallback"],
        colors=["#2ecc71", "#f39c12"], autopct="%1.0f%%", startangle=90)
ax4.set_title("Resting-State Normalization", fontweight="bold")

# Panel E: HEST shape illustration
ax5 = fig.add_subplot(gs[1, 2])
ax5.text(0.5, 0.5, f"HEST shape per epoch:\n"
         f"({len(TARGET_CHANNELS)} ch, {len(FREQ_BANDS)} bands, "
         f"{N_HEP_TIMES} time)\n\n"
         f"PPG features: {len(PPG_FEATURE_NAMES)}\n"
         f"RR sequence: 10 (padded)\n\n"
         f"Total features per epoch:\n"
         f"HEST: {len(TARGET_CHANNELS)*len(FREQ_BANDS)*N_HEP_TIMES}\n"
         f"PPG: {len(PPG_FEATURE_NAMES)}\n"
         f"RR: 10",
         ha="center", va="center", transform=ax5.transAxes,
         fontsize=9, family="monospace",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax5.set_title("Representation Structure", fontweight="bold")
ax5.axis("off")

# Panel F: Sample HEST heatmap (first successful subject)
ax6 = fig.add_subplot(gs[2, :])
sample_subj = next((s for s in SUBJECTS if qc_all[s]["status"] == "success"), None)
if sample_subj:
    with h5py.File(hest_paths[sample_subj], "r") as hf:
        hest_sample = hf["task"]["0000"]["hest_norm"][:]  # (ch, bands, time)
        # Plot first channel, all bands
        for b_idx, band in enumerate(FREQ_BANDS.keys()):
            ax6.plot(hest_sample[0, b_idx, :] + b_idx * 2, label=band, alpha=0.8)
        ax6.set_title(f"Sample HEST (normalized) — {sample_subj}, channel {TARGET_CHANNELS[0]}",
                      fontweight="bold")
        ax6.set_xlabel("Time (downsampled)")
        ax6.set_ylabel("Band power (z-score from rest)")
        ax6.legend(loc="upper right", fontsize=8)

fig_path = FIG_DIR / "section6_hest_summary.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n✓ Figure saved: {fig_path}")


# ============================================================
# SUMMARY
# ============================================================

n_ok = sum(1 for v in qc_all.values() if v["status"] == "success")
n_skip = sum(1 for v in qc_all.values() if v["status"] == "skip")
total_epochs = sum(v["n_task"] for v in qc_all.values() if v["status"] == "success")
total_beats = sum(v["n_beats_total"] for v in qc_all.values() if v["status"] == "success")
total_placeholders = sum(v.get("n_placeholder_ch", 0) for v in qc_all.values() if v["status"] == "success")

print(f"\n{'='*55}")
print(f"HEST EXTRACTION SUMMARY")
print(f"{'='*55}")
print(f" Success          : {n_ok}/{len(SUBJECTS)}")
print(f" Skipped          : {n_skip}")
print(f" Total epochs     : {total_epochs}")
print(f" Total beats      : {total_beats}")
print(f" Total placeholders: {total_placeholders} (zero-filled channels)")
print(f" Conditions       : {dict(condition_counts)}")
print(f" QC saved         : {qc_path.name}")
print(f" Config updated   : config.json")
print(f"\nSection 6 COMPLETE")
print(f"Next: Section 7 — Model Training (CP-GSTNet)")