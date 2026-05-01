# ============================================================
# RS-HESR  |  Section 3 — Dataset Inspection & Validation
# Input:   processed/config.json  (from Section 2)
# Output:  processed/loading_registry.json
#          processed/figures/section3_summary.png
#          updated config.json
# ============================================================

import json, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")


# ── Config loader (defined in Section 2; redefined here for
#    independence so Section 3 runs as a standalone cell) ──────
def load_config(root="processed"):
    with open(Path(root) / "config.json") as f:
        return json.load(f)

def save_config(cfg):
    with open(Path(cfg["processed_root"]) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG   = load_config()
DATASET  = Path(CONFIG["dataset_root"])
PROC     = Path(CONFIG["processed_root"])
FIG_DIR  = Path(CONFIG.get("figures", str(PROC / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

assert DATASET.is_dir(), f"Dataset not found: {DATASET}"
print(f"Dataset  : {DATASET}")
print(f"Processed: {PROC}\n")


# ============================================================
# 3.1  Channel-type classifier
#      Applied to raw.ch_names — names are the only reliable
#      discriminator in BrainVision (all types default to EEG).
# ============================================================

ECG_KW = {"ecg","ekg","cardiac","heart","bio1"}
PPG_KW = {"ppg","pleth","pulse","photo","bio2","bio3"}

def classify_channels(ch_names):
    eeg, ecg, ppg = [], [], []
    for ch in ch_names:
        low = ch.lower()
        if any(k in low for k in ECG_KW):
            ecg.append(ch)
        elif any(k in low for k in PPG_KW):
            ppg.append(ch)
        else:
            eeg.append(ch)
    return eeg, ecg, ppg


# ============================================================
# 3.2  Single-subject inspector
#      Uses MNE read_raw_brainvision(preload=False) — reads
#      header and channel info only; zero signal data loaded.
# ============================================================

def inspect_subject(subj_dir: Path) -> dict:
    sid = subj_dir.name
    out = {
        "subject_id":   sid,
        "status":       "unknown",
        "flags":        [],
        "vhdr_path":    None,
        "events_path":  None,
        "beh_path":     None,
        "channels":     [],
        "eeg_channels": [],
        "ecg_channels": [],
        "ppg_channels": [],
        "sfreq":        None,
        "n_channels":   None,
        "duration_s":   None,
        "n_annotations":0,
    }

    eeg_dir = subj_dir / "eeg"
    if not eeg_dir.is_dir():
        out["status"] = "skip"; out["flags"].append("no_eeg_dir")
        return out

    # Discover files dynamically — no hardcoded names
    vhdr_files   = sorted(eeg_dir.glob("*.vhdr"))
    events_files = sorted(eeg_dir.glob("*_events.tsv"))
    beh_dir      = subj_dir / "beh"
    beh_files    = sorted(beh_dir.glob("*_beh.tsv")) if beh_dir.is_dir() else []

    if not vhdr_files:
        out["status"] = "skip"; out["flags"].append("no_vhdr")
        return out

    out["vhdr_path"]   = str(vhdr_files[0])
    out["events_path"] = str(events_files[0]) if events_files else None
    out["beh_path"]    = str(beh_files[0])    if beh_files    else None

    if not events_files:
        out["flags"].append("no_events_tsv")

    # MNE load — metadata only (preload=False)
    try:
        raw = mne.io.read_raw_brainvision(
            vhdr_fname=out["vhdr_path"],
            preload=False, verbose=False,
        )
    except Exception as e:
        out["status"] = "skip"
        out["flags"].append(f"mne_error: {e}")
        return out

    eeg_chs, ecg_chs, ppg_chs = classify_channels(raw.ch_names)

    out["channels"]     = list(raw.ch_names)
    out["eeg_channels"] = eeg_chs
    out["ecg_channels"] = ecg_chs
    out["ppg_channels"] = ppg_chs
    out["sfreq"]        = raw.info["sfreq"]
    out["n_channels"]   = len(raw.ch_names)
    out["duration_s"]   = raw.times[-1] if raw.times is not None else None
    out["n_annotations"]= len(raw.annotations)

    del raw   # no signal was loaded; frees file handle

    # Status
    has_critical = "no_vhdr" in out["flags"] or "mne_error" in out["flags"]
    if has_critical:
        out["status"] = "skip"
    elif not ecg_chs:
        out["status"] = "warn"; out["flags"].append("no_ecg")
    elif not ppg_chs:
        out["status"] = "warn"; out["flags"].append("no_ppg")
    else:
        out["status"] = "ok"

    return out


# ============================================================
# 3.3  Inspect all subjects
# ============================================================

subj_dirs = sorted([
    d for d in DATASET.iterdir()
    if d.is_dir() and d.name.startswith("sub-")
])
print(f"Subject directories : {len(subj_dirs)}\n")

registry   = {}
sfreq_cnt  = defaultdict(int)
ecg_names  = defaultdict(int)
ppg_names  = defaultdict(int)

for sd in subj_dirs:
    res = inspect_subject(sd)
    registry[res["subject_id"]] = res

    sym   = {"ok":"✓","warn":"⚠","skip":"✗"}.get(res["status"],"?")
    flags = "|".join(res["flags"]) if res["flags"] else "clean"
    dur   = f"{res['duration_s']:.0f}s" if res["duration_s"] else "?"

    print(
        f"  {sym} {res['subject_id']:<12}"
        f"  {str(res['sfreq'])+'Hz':<10}"
        f"  ch={res['n_channels']:<5}"
        f"  dur={dur:<8}"
        f"  ECG={res['ecg_channels']}"
        f"  PPG={res['ppg_channels']}"
        + (f"  [{flags}]" if res["flags"] else "")
    )

    if res["sfreq"]:
        sfreq_cnt[res["sfreq"]] += 1
    for ch in res["ecg_channels"]:
        ecg_names[ch] += 1
    for ch in res["ppg_channels"]:
        ppg_names[ch] += 1

# Canonical names (most frequent across cohort)
ecg_canon = max(ecg_names, key=ecg_names.get) if ecg_names else None
ppg_canon = max(ppg_names, key=ppg_names.get) if ppg_names else None
fs_canon  = max(sfreq_cnt, key=sfreq_cnt.get) if sfreq_cnt else None


# ============================================================
# 3.4  Subject lists
# ============================================================

all_subjects  = sorted(registry.keys())
valid_eeg     = [s for s,r in registry.items() if r["status"] in ("ok","warn")]
has_ecg       = [s for s,r in registry.items() if r["ecg_channels"]]
has_ppg       = [s for s,r in registry.items() if r["ppg_channels"]]
full_subjects = [s for s,r in registry.items()
                 if r["status"] in ("ok","warn")
                 and r["ecg_channels"] and r["ppg_channels"]]
flagged       = [s for s,r in registry.items() if r["flags"]]

print(f"\n{'='*55}")
print(f"SUMMARY")
print(f"{'='*55}")
print(f"  Total subjects         : {len(all_subjects)}")
print(f"  Valid EEG (ok+warn)    : {len(valid_eeg)}")
print(f"  ECG detected           : {len(has_ecg)}")
print(f"  PPG detected           : {len(has_ppg)}")
print(f"  Full (EEG+ECG+PPG)     : {len(full_subjects)}")
print(f"  Flagged                : {len(flagged)}")
print(f"\n  ECG name variants      : {dict(ecg_names)}")
print(f"  PPG name variants      : {dict(ppg_names)}")
print(f"  Sampling rates         : {dict(sfreq_cnt)}")
print(f"  Canonical ECG ch       : {ecg_canon}")
print(f"  Canonical PPG ch       : {ppg_canon}")
print(f"  Canonical sfreq        : {fs_canon} Hz")


# ============================================================
# 3.5  Visualisation — comprehensive summary figure
# ============================================================

fig = plt.figure(figsize=(16, 12), constrained_layout=True)
fig.suptitle("Section 3 — Dataset Inspection Summary\n"
             f"Dataset: ds006848  |  {len(all_subjects)} subjects",
             fontsize=13, fontweight="bold")

gs = fig.add_gridspec(3, 3)

# ── Panel A: subject status bar ───────────────────────────────
ax_status = fig.add_subplot(gs[0, 0])
status_counts = defaultdict(int)
for r in registry.values():
    status_counts[r["status"]] += 1
colors = {"ok":"#2ecc71","warn":"#f39c12","skip":"#e74c3c"}
labels = list(status_counts.keys())
vals   = [status_counts[k] for k in labels]
ax_status.bar(labels, vals,
              color=[colors.get(l,"gray") for l in labels], edgecolor="white")
ax_status.set_title("Subject Status", fontweight="bold")
ax_status.set_ylabel("Count")
for i,(l,v) in enumerate(zip(labels,vals)):
    ax_status.text(i, v+0.1, str(v), ha="center", fontweight="bold")
ax_status.set_ylim(0, max(vals)+3)

# ── Panel B: modality availability ───────────────────────────
ax_modal = fig.add_subplot(gs[0, 1])
modal_labels = ["EEG only", "EEG+ECG", "EEG+PPG", "EEG+ECG+PPG"]
n_eeg_only = len([s for s in valid_eeg
                   if s not in has_ecg and s not in has_ppg])
n_ecg_only = len([s for s in has_ecg if s not in has_ppg])
n_ppg_only = len([s for s in has_ppg if s not in has_ecg])
modal_vals = [n_eeg_only, n_ecg_only, n_ppg_only, len(full_subjects)]
ax_modal.barh(modal_labels, modal_vals,
              color=["#3498db","#9b59b6","#1abc9c","#e67e22"],
              edgecolor="white")
ax_modal.set_title("Modality Availability", fontweight="bold")
ax_modal.set_xlabel("Count")
for i, v in enumerate(modal_vals):
    ax_modal.text(v+0.1, i, str(v), va="center", fontweight="bold")
ax_modal.set_xlim(0, max(modal_vals)+4)

# ── Panel C: sampling rate distribution ──────────────────────
ax_fs = fig.add_subplot(gs[0, 2])
fs_labels = [str(int(k))+"Hz" for k in sfreq_cnt.keys()]
fs_vals   = list(sfreq_cnt.values())
wedge_colors = ["#2ecc71","#3498db","#e74c3c","#f39c12"]
ax_fs.pie(fs_vals, labels=fs_labels,
          colors=wedge_colors[:len(fs_vals)],
          autopct="%1.0f%%", startangle=90,
          textprops={"fontsize":9})
ax_fs.set_title("Sampling Rates", fontweight="bold")

# ── Panel D: channel count per subject ───────────────────────
ax_ch = fig.add_subplot(gs[1, :2])
subj_ids = [r["subject_id"] for r in registry.values()
            if r["n_channels"] is not None]
n_eeg_ch = [len(r["eeg_channels"]) for r in registry.values()
             if r["n_channels"] is not None]
n_ecg_ch = [len(r["ecg_channels"]) for r in registry.values()
             if r["n_channels"] is not None]
n_ppg_ch = [len(r["ppg_channels"]) for r in registry.values()
             if r["n_channels"] is not None]
x = np.arange(len(subj_ids))
w = 0.6
ax_ch.bar(x, n_eeg_ch, w, label="EEG",  color="#3498db", alpha=0.85)
ax_ch.bar(x, n_ecg_ch, w, bottom=n_eeg_ch, label="ECG", color="#e74c3c", alpha=0.85)
bot2 = [a+b for a,b in zip(n_eeg_ch, n_ecg_ch)]
ax_ch.bar(x, n_ppg_ch, w, bottom=bot2, label="PPG", color="#2ecc71", alpha=0.85)
ax_ch.set_xticks(x)
ax_ch.set_xticklabels([s.replace("sub-","") for s in subj_ids],
                       rotation=45, ha="right", fontsize=7)
ax_ch.set_title("Channel Count per Subject (EEG / ECG / PPG)", fontweight="bold")
ax_ch.set_ylabel("N channels")
ax_ch.legend(loc="upper right", fontsize=8)
ax_ch.set_xlim(-0.5, len(subj_ids)-0.5)

# ── Panel E: recording duration per subject ───────────────────
ax_dur = fig.add_subplot(gs[1, 2])
durs = [r["duration_s"]/60 for r in registry.values()
        if r["duration_s"] is not None]
sids_dur = [r["subject_id"].replace("sub-","") for r in registry.values()
             if r["duration_s"] is not None]
ax_dur.barh(sids_dur, durs, color="#9b59b6", alpha=0.8, edgecolor="white")
ax_dur.set_title("Recording Duration (min)", fontweight="bold")
ax_dur.set_xlabel("Minutes")
ax_dur.tick_params(axis="y", labelsize=7)
if durs:
    ax_dur.axvline(np.mean(durs), color="black", ls="--", lw=1,
                   label=f"mean={np.mean(durs):.1f}min")
    ax_dur.legend(fontsize=8)

# ── Panel F: events per subject ───────────────────────────────
ax_ev = fig.add_subplot(gs[2, :2])
n_annots = [r["n_annotations"] for r in registry.values()]
sids_ev  = [r["subject_id"].replace("sub-","") for r in registry.values()]
bar_colors = ["#2ecc71" if r["status"]=="ok"
               else "#f39c12" if r["status"]=="warn"
               else "#e74c3c"
               for r in registry.values()]
ax_ev.bar(range(len(sids_ev)), n_annots, color=bar_colors, edgecolor="white")
ax_ev.set_xticks(range(len(sids_ev)))
ax_ev.set_xticklabels(sids_ev, rotation=45, ha="right", fontsize=7)
ax_ev.set_title("Annotation Count per Subject\n"
                "(green=ok  orange=warn  red=skip)", fontweight="bold")
ax_ev.set_ylabel("N annotations")
ax_ev.set_xlim(-0.5, len(sids_ev)-0.5)

# ── Panel G: ECG / PPG channel name frequency ─────────────────
ax_names = fig.add_subplot(gs[2, 2])
all_aux   = {**{f"ECG:{k}":v for k,v in ecg_names.items()},
             **{f"PPG:{k}":v for k,v in ppg_names.items()}}
if all_aux:
    ax_names.barh(list(all_aux.keys()), list(all_aux.values()),
                  color=["#e74c3c"]*len(ecg_names) +
                        ["#2ecc71"]*len(ppg_names),
                  edgecolor="white")
    ax_names.set_title("Aux Channel Name Variants\n(across cohort)",
                        fontweight="bold")
    ax_names.set_xlabel("Subject count")
    for i,v in enumerate(all_aux.values()):
        ax_names.text(v+0.05, i, str(v), va="center", fontsize=8)
else:
    ax_names.text(0.5, 0.5, "No ECG/PPG detected",
                   ha="center", va="center", transform=ax_names.transAxes)
    ax_names.set_title("Aux Channel Name Variants", fontweight="bold")

# Save
fig_path = FIG_DIR / "section3_summary.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n✓ Figure saved : {fig_path}")

# Display if running interactively
try:
    from IPython.display import Image, display
    display(Image(str(fig_path)))
except Exception:
    pass


# ============================================================
# 3.6  Save loading_registry.json
#      Serialisable dict only — no MNE objects
# ============================================================

reg_out = {}
for sid, r in registry.items():
    reg_out[sid] = {
        "status":       r["status"],
        "flags":        r["flags"],
        "vhdr_path":    r["vhdr_path"],
        "events_path":  r["events_path"],
        "beh_path":     r["beh_path"],
        "channels":     r["channels"],
        "eeg_channels": r["eeg_channels"],
        "ecg_channels": r["ecg_channels"],
        "ppg_channels": r["ppg_channels"],
        "sfreq":        r["sfreq"],
        "n_channels":   r["n_channels"],
        "duration_s":   r["duration_s"],
        "n_annotations":r["n_annotations"],
    }

reg_path = PROC / "loading_registry.json"
with open(reg_path, "w") as f:
    json.dump(reg_out, f, indent=2)
print(f"✓ Registry saved: {reg_path.name}")


# ============================================================
# 3.7  Update config.json
# ============================================================

CONFIG["discovered_subjects"] = all_subjects
CONFIG["valid_eeg_subjects"]  = valid_eeg
CONFIG["full_subjects"]       = full_subjects
CONFIG["ecg_canonical_ch"]    = ecg_canon
CONFIG["ppg_canonical_ch"]    = ppg_canon
CONFIG["cohort_sfreq"]        = fs_canon

save_config(CONFIG)
print(f"✓ Config updated: config.json")
print(f"  full_subjects ({len(full_subjects)}): {full_subjects}")
print(f"\nSection 3 COMPLETE")
print(f"Next: Section 4 — Data Loading")