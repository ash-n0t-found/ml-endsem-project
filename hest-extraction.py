# ============================================================
# RS-HESR | Section 6 — Representation Construction (HEST extraction)
# Input:  processed/sub-XXX_preprocessed.h5 (from Section 5)
# Output: processed/sub-XXX_hest.h5 (per subject)
#         processed/hest_qc.json
#         processed/figures/section6_hest_summary.png
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
FS = float(CONFIG.get("cohort_sfreq", 1000.0))

# HEST window around R-peak (ms)
HEST_PRE_MS = 400   # before R-peak
HEST_POST_MS = 400  # after R-peak
# Temporal downsampling factor (400+400ms at 500Hz = 400 samples -> 80)
HEST_DOWNSAMPLE = 5

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
# HEST EXTRACTION (per trial)
# ============================================================

def extract_hest_trial(eeg_trial, ecg_trial, fs, ch_names):
    """
    Extract Heartbeat-Evoked Spectrotemporal representation for one trial.
    
    eeg_trial: (n_eeg_ch, n_times)
    ecg_trial: (n_times,)
    ch_names: list of EEG channel names
    
    Returns:
        hest: (n_target_ch, n_bands, n_timepoints) — averaged over beats
        n_beats: number of heartbeats used
        rr_seq: RR intervals during retention (ms)
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_trial, fs)
    r_peaks = clean_r_peaks(r_peaks, fs)
    
    if len(r_peaks) < 2:
        return None, 0, None
    
    # RR sequence
    rr_seq = np.diff(r_peaks) / fs * 1000.0  # ms
    
    # HEST window in samples
    pre_samp = int(HEST_PRE_MS / 1000.0 * fs)
    post_samp = int(HEST_POST_MS / 1000.0 * fs)
    n_times = pre_samp + post_samp
    
    # Select target channels
    ch_idx = [ch_names.index(ch) for ch in TARGET_CHANNELS if ch in ch_names]
    if not ch_idx:
        # Fallback: use first 9 channels
        ch_idx = list(range(min(9, len(ch_names))))
    n_ch = len(ch_idx)
    
    n_bands = len(FREQ_BANDS)
    hest_accum = []
    
    for rp in r_peaks:
        start = rp - pre_samp
        end = rp + post_samp
        if start < 0 or end > eeg_trial.shape[1]:
            continue
        
        # Extract EEG around R-peak for target channels
        segment = eeg_trial[ch_idx, start:end]  # (n_ch, n_times)
        
        # Compute time-frequency power per band
        # Result: (n_ch, n_bands, n_times)
        trial_tf = np.zeros((n_ch, n_bands, n_times))
        for b_idx, band_name in enumerate(FREQ_BANDS.keys()):
            for c_idx in range(n_ch):
                power = extract_band_power(segment[c_idx], fs, band_name)
                # Average over sub-frequencies in band
                trial_tf[c_idx, b_idx, :] = np.mean(power, axis=0)
        
        hest_accum.append(trial_tf)
    
    if len(hest_accum) == 0:
        return None, 0, rr_seq
    
    # Average over heartbeats
    hest = np.mean(hest_accum, axis=0)  # (n_ch, n_bands, n_times)
    
    # Temporal downsampling (factor 5)
    if hest.shape[2] > HEST_DOWNSAMPLE:
        hest = hest[:, :, ::HEST_DOWNSAMPLE]
    
    return hest, len(hest_accum), rr_seq


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
    
    with h5py.File(h5_path, "r") as hf:
        if "rest" not in hf:
            return None, None
        
        ch_names = json.loads(hf["metadata"].attrs["eeg_channels"])
        rest_group = hf["rest"]
        
        for ep_key in rest_group.keys():
            ep = rest_group[ep_key]
            eeg = ep["eeg"][:]
            ecg = ep["ecg"][:]
            
            hest, n_beats, _ = extract_hest_trial(eeg, ecg, fs, ch_names)
            if hest is not None and n_beats >= 3:
                rest_hests.append(hest)
    
    if len(rest_hests) == 0:
        return None, None
    
    rest_stack = np.stack(rest_hests, axis=0)  # (n_rest_trials, ch, bands, times)
    rest_mean = np.mean(rest_stack, axis=0)
    rest_std = np.std(rest_stack, axis=0) + 1e-8  # avoid div by zero
    
    return rest_mean, rest_std


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
# ============================================================

def save_hest_h5(subject_id, task_representations, rest_mean, rest_std,
                 eeg_ch_names, fs, out_dir):
    """
    Save HEST representations to HDF5.
    Each task epoch: hest_norm, rr_sequence, ppg_features, label
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
    }
    
    try:
        # First pass: compute resting-state statistics
        rest_mean, rest_std = compute_rest_hest_stats(h5_path, FS)
        
        with h5py.File(h5_path, "r") as hf:
            ch_names = json.loads(hf["metadata"].attrs["eeg_channels"])
            fs_file = hf["metadata"].attrs["fs"]
            
            task_group = hf["task"]
            task_reps = []
            
            for ep_key in task_group.keys():
                ep = task_group[ep_key]
                eeg = ep["eeg"][:]
                ecg = ep["ecg"][:]
                ppg = ep["ppg"][:]
                condition = ep.attrs["condition"]
                label_int = ep.attrs["label_int"]
                
                # Extract HEST
                hest, n_beats, rr_seq = extract_hest_trial(eeg, ecg, fs_file, ch_names)
                
                if hest is None:
                    continue
                
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
        
        # Save
        hest_path = save_hest_h5(
            subj_id, task_reps, rest_mean, rest_std,
            ch_names, fs_file, str(PROCESSED)
        )
        hest_paths[subj_id] = hest_path
        subj_qc["status"] = "success"
        
        rest_info = "RS_norm" if rest_mean is not None else "NO_REST"
        print(f" ✓ {subj_id:<12} "
              f"task={len(task_reps):>3} "
              f"beats={subj_qc['n_beats_total']:>4} "
              f"conds={subj_qc['conditions']} "
              f"[{rest_info}]")
        
    except Exception as exc:
        subj_qc["status"] = "skip"
        subj_qc["skip_reason"] = str(exc)
        print(f" ✗ {subj_id:<12} SKIP: {exc}")
    
    qc_all[subj_id] = subj_qc
    gc.collect()


# ============================================================
# SAVE QC + UPDATE CONFIG
# ============================================================

qc_path = PROCESSED / "hest_qc.json"
with open(qc_path, "w") as f:
    json.dump(qc_all, f, indent=2, default=str)

CONFIG["hest_subjects"] = hest_paths
save_config(CONFIG)


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
ax5.text(0.5, 0.5, f"HEST shape per epoch:\\n"
         f"({len(TARGET_CHANNELS)} ch, {len(FREQ_BANDS)} bands, "
         f"{int((HEST_PRE_MS+HEST_POST_MS)*FS/1000/HEST_DOWNSAMPLE)} time)\\n\\n"
         f"PPG features: {len(PPG_FEATURE_NAMES)}\\n"
         f"RR sequence: 10 (padded)\\n\\n"
         f"Total features per epoch:\\n"
         f"HEST: {len(TARGET_CHANNELS)*len(FREQ_BANDS)*int((HEST_PRE_MS+HEST_POST_MS)*FS/1000/HEST_DOWNSAMPLE)}\\n"
         f"PPG: {len(PPG_FEATURE_NAMES)}\\n"
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

print(f"\n{'='*55}")
print(f"HEST EXTRACTION SUMMARY")
print(f"{'='*55}")
print(f" Success          : {n_ok}/{len(SUBJECTS)}")
print(f" Skipped          : {n_skip}")
print(f" Total epochs     : {total_epochs}")
print(f" Total beats      : {total_beats}")
print(f" Conditions       : {dict(condition_counts)}")
print(f" QC saved         : {qc_path.name}")
print(f" Config updated   : config.json")
print(f"\nSection 6 COMPLETE")
print(f"Next: Section 7 — Model Training (CP-GSTNet)")