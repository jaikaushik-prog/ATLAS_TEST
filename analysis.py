"""
prmon Anomaly Detection Pipeline — Combined Time-Series
=========================================================
Builds a single combined time-series by interleaving baseline segments with
four injected anomaly windows (subtle CPU, extreme CPU, hard memory, extreme
memory).  Three novelty-detection methods (Z-Score, LOF, One-Class SVM) are
trained on the clean baseline portion and evaluated against ground-truth
labels using precision, recall, and F1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 1. Data Loading ──────────────────────────────────────────────────────────

FEATURES = ["pss", "rss", "vmem", "utime", "stime", "nthreads"]

def load_prmon(path):
    df = pd.read_csv(path, sep=r"\s+", on_bad_lines="skip")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

baseline     = load_prmon("baseline_long.txt")
cpu_extreme  = load_prmon("anomaly_cpu_long.txt")
mem_extreme  = load_prmon("anomaly_mem_long.txt")
cpu_subtle   = load_prmon("anomaly_cpu_subtle.txt")
mem_hard     = load_prmon("anomaly_mem_hard.txt")

print(f"Loaded  baseline:     {len(baseline)} rows")
print(f"Loaded  cpu_extreme:  {len(cpu_extreme)} rows")
print(f"Loaded  cpu_subtle:   {len(cpu_subtle)} rows")
print(f"Loaded  mem_extreme:  {len(mem_extreme)} rows")
print(f"Loaded  mem_hard:     {len(mem_hard)} rows\n")

# ── 2. Build Combined Time-Series ────────────────────────────────────────────
# Layout: [B_A] [subtle_cpu] [B_B] [extreme_cpu] [B_C] [hard_mem] [B_D] [extreme_mem] [B_E]
# Labels:  0       1           0       2            0       3         0       4           0

seg_b_a      = baseline.iloc[0:60].copy()
seg_cpu_sub  = cpu_subtle.iloc[50:100].copy()        # 50 pts subtle CPU
seg_b_b      = baseline.iloc[60:140].copy()
seg_cpu_ext  = cpu_extreme.iloc[100:150].copy()      # 50 pts extreme CPU
seg_b_c      = baseline.iloc[140:240].copy()
seg_mem_hard = mem_hard.iloc[50:110].copy()           # 60 pts hard memory (16MB)
seg_b_d      = baseline.iloc[240:340].copy()
seg_mem_ext  = mem_extreme.iloc[80:140].copy()        # 60 pts extreme memory (1GB)
seg_b_e      = baseline.iloc[340:].copy()

seg_b_a["label"]      = 0
seg_cpu_sub["label"]  = 1   # subtle CPU
seg_b_b["label"]      = 0
seg_cpu_ext["label"]  = 2   # extreme CPU
seg_b_c["label"]      = 0
seg_mem_hard["label"] = 3   # hard memory (16MB)
seg_b_d["label"]      = 0
seg_mem_ext["label"]  = 4   # extreme memory (1GB)
seg_b_e["label"]      = 0

segments = [seg_b_a, seg_cpu_sub, seg_b_b, seg_cpu_ext, seg_b_c,
            seg_mem_hard, seg_b_d, seg_mem_ext, seg_b_e]
combined = pd.concat(segments, ignore_index=True)

# Create a continuous time index for the combined series
combined["t"] = np.arange(len(combined))

# Binary ground truth: 0 = normal, 1 = anomalous (any type)
combined["is_anomaly"] = (combined["label"] > 0).astype(int)

# Compute segment boundaries for shading
boundaries = {}
offset = 0
for seg, name in zip(segments, ["B_A","cpu_subtle","B_B","cpu_extreme","B_C",
                                 "mem_hard","B_D","mem_extreme","B_E"]):
    boundaries[name] = (offset, offset + len(seg))
    offset += len(seg)

cpu_sub_start, cpu_sub_end = boundaries["cpu_subtle"]
cpu_ext_start, cpu_ext_end = boundaries["cpu_extreme"]
mem_hrd_start, mem_hrd_end = boundaries["mem_hard"]
mem_ext_start, mem_ext_end = boundaries["mem_extreme"]

n_normal = (combined["label"] == 0).sum()
n_anom = (combined["label"] > 0).sum()
print(f"Combined series: {len(combined)} rows")
print(f"  Normal points:          {n_normal}")
print(f"  Subtle CPU anomaly:     {(combined['label']==1).sum()}  (t={cpu_sub_start}..{cpu_sub_end-1})")
print(f"  Extreme CPU anomaly:    {(combined['label']==2).sum()}  (t={cpu_ext_start}..{cpu_ext_end-1})")
print(f"  Hard memory anomaly:    {(combined['label']==3).sum()}  (t={mem_hrd_start}..{mem_hrd_end-1})")
print(f"  Extreme memory anomaly: {(combined['label']==4).sum()}  (t={mem_ext_start}..{mem_ext_end-1})")
print()

# ── 3. Scaling (fit on baseline-only portion) ────────────────────────────────

# Training data: only the clean baseline segments
baseline_mask = combined["label"] == 0
train_X = combined.loc[baseline_mask, FEATURES].values

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Transform the entire combined series
all_X_scaled = scaler.transform(combined[FEATURES].values)

# ── 4. Method 1 – Z-Score ────────────────────────────────────────────────────

ZSCORE_THRESH = 3.0

def zscore_detect(X, threshold=ZSCORE_THRESH):
    return np.where(np.any(np.abs(X) > threshold, axis=1), -1, 1)

combined["zscore"] = zscore_detect(all_X_scaled)

# ── 5. Method 2 – Local Outlier Factor (novelty detection) ───────────────────

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=True)
lof.fit(train_X_scaled)
combined["lof"] = lof.predict(all_X_scaled)

# ── 6. Method 3 – One-Class SVM ─────────────────────────────────────────────

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(train_X_scaled)
combined["ocsvm"] = ocsvm.predict(all_X_scaled)

# ── 7. Evaluation ────────────────────────────────────────────────────────────

y_true = combined["is_anomaly"].values  # 1 = anomaly, 0 = normal

print("=" * 72)
print(f"{'Detection Results — Combined Time-Series':^72}")
print("=" * 72)
print(f"{'Method':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}   {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 72)

methods = [("zscore", "Z-Score"), ("lof", "LOF"), ("ocsvm", "OC-SVM")]
results = {}

for col, label in methods:
    y_pred = (combined[col] == -1).astype(int).values  # 1 = flagged anomaly
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    results[col] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
    print(f"{label:<18} {p:>10.3f} {r:>10.3f} {f:>10.3f}   {tp:>5} {fp:>5} {fn:>5}")

print("=" * 72)
n_total = len(combined)
n_anom = int(y_true.sum())
print(f"Total: {n_total} points  |  Normal: {n_total - n_anom}  |  Anomalous: {n_anom}")
print()

# ── 7b. False Positive Investigation ─────────────────────────────────────────

for col, label in methods:
    fp_mask = (combined[col] == -1) & (combined["is_anomaly"] == 0)
    n_fp = fp_mask.sum()
    if n_fp > 0 and n_fp <= 20:   # only print detail for small FP counts
        print(f"── {label}: {n_fp} false positive(s) in baseline ──")
        fp_indices = combined.loc[fp_mask].index.tolist()
        for idx in fp_indices:
            row = combined.iloc[idx]
            z_vals = all_X_scaled[idx]
            # Find which features exceeded threshold
            exceeded = [(FEATURES[i], f"z={z_vals[i]:.2f}") for i in range(len(FEATURES))
                        if abs(z_vals[i]) > ZSCORE_THRESH]
            raw_vals = {f: row[f] for f in FEATURES}
            print(f"  t={int(row['t']):>3}  segment=baseline  "
                  f"pss={int(row['pss'])}  utime={int(row['utime'])}  nthreads={int(row['nthreads'])}")
            if exceeded:
                print(f"         z-score trigger: {exceeded}")
            else:
                print(f"         (flagged by {label}, not z-score)")
        print()

# ── 8. Visualization ────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-darkgrid")
COL_NORMAL   = "#2196F3"
COL_CPU_SUB  = "#FFB74D"   # light orange – subtle CPU
COL_CPU_EXT  = "#E65100"   # dark orange  – extreme CPU
COL_MEM_HRD  = "#F48FB1"   # light pink   – hard memory
COL_MEM_EXT  = "#C2185B"   # dark pink    – extreme memory
COL_FLAG     = "#D32F2F"

WINDOW_DEFS = [
    ("Subtle CPU",   cpu_sub_start, cpu_sub_end, COL_CPU_SUB),
    ("Extreme CPU",  cpu_ext_start, cpu_ext_end, COL_CPU_EXT),
    ("Hard Mem",     mem_hrd_start, mem_hrd_end, COL_MEM_HRD),
    ("Extreme Mem",  mem_ext_start, mem_ext_end, COL_MEM_EXT),
]

def shade_windows(ax, label_it=True):
    """Add semi-transparent shading for all 4 anomaly windows."""
    for wname, ws, we, wc in WINDOW_DEFS:
        kw = {"label": wname} if label_it else {}
        ax.axvspan(ws, we - 1, alpha=0.18, color=wc, **kw)

# ────────────────────────────────────────────────────────────────────────────
# Plot 1: Multi-Feature Overview (PSS log-scale + utime + nthreads)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig.suptitle("Combined Time-Series — Multi-Feature Overview (4 Anomaly Windows)", fontsize=14)

ax = axes[0]
ax.plot(combined["t"], combined["pss"], color=COL_NORMAL, lw=1, alpha=0.85)
shade_windows(ax)
ax.set_ylabel("PSS (kB)")
ax.set_yscale("log")
ax.set_title("PSS (log scale) — memory footprint")
ax.legend(loc="upper left", fontsize=7, ncol=4)

ax = axes[1]
ax.plot(combined["t"], combined["utime"], color="#7E57C2", lw=1, alpha=0.85)
shade_windows(ax, label_it=False)
ax.set_ylabel("utime (ticks)")
ax.set_title("User CPU Time (cumulative) — CPU usage indicator")

ax = axes[2]
ax.plot(combined["t"], combined["nthreads"], color="#26A69A", lw=1.2, alpha=0.85)
shade_windows(ax, label_it=False)
ax.set_ylabel("nthreads")
ax.set_title("Thread Count — concurrency indicator")
ax.set_xlabel("Sample Index")

fig.tight_layout()
fig.savefig("plot1_multifeature_overview.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 2: Detection Overlay (log-scale PSS with flagged points per method)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Anomaly Detection Results — Per Method (log-scale PSS)", fontsize=14, y=1.01)

for ax, (col, label) in zip(axes, methods):
    ax.plot(combined["t"], combined["pss"], color=COL_NORMAL, lw=0.8, alpha=0.5)
    shade_windows(ax)
    mask = combined[col] == -1
    ax.scatter(combined.loc[mask, "t"], combined.loc[mask, "pss"],
               color=COL_FLAG, s=14, zorder=5, label="Flagged anomaly", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("PSS (kB)")
    r = results[col]
    ax.set_title(f"{label}  —  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}"
                 f"  (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
    ax.legend(loc="upper left", fontsize=7, ncol=5)

axes[-1].set_xlabel("Sample Index")
fig.tight_layout()
fig.savefig("plot2_detection_overlay.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 3: Zoomed-In Anomaly Windows (2x2 grid, one per window)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Zoomed-In Anomaly Windows — PSS with Detection Flags", fontsize=14)

pad = 15
for ax, (wname, ws, we, wc) in zip(axes.flat, WINDOW_DEFS):
    seg = combined.iloc[max(0, ws - pad):min(len(combined), we + pad)]
    ax.plot(seg["t"], seg["pss"], color=COL_NORMAL, lw=1.2, alpha=0.8)
    ax.axvspan(ws, we - 1, alpha=0.18, color=wc)
    ax.axvline(ws, color=wc, ls="--", lw=1, alpha=0.7)
    ax.axvline(we - 1, color=wc, ls="--", lw=1, alpha=0.7)
    for col_name, lab, mk in [("zscore","Z-Score","o"), ("lof","LOF","s"), ("ocsvm","OC-SVM","D")]:
        m = seg[col_name] == -1
        if m.any():
            ax.scatter(seg.loc[m, "t"], seg.loc[m, "pss"], marker=mk, s=28,
                       zorder=5, alpha=0.7, label=lab)
    ax.set_ylabel("PSS (kB)")
    ax.set_title(f"{wname} (t={ws}–{we-1}, {we-ws} pts)")
    ax.legend(fontsize=7, ncol=3)

axes[1, 0].set_xlabel("Sample Index")
axes[1, 1].set_xlabel("Sample Index")
fig.tight_layout()
fig.savefig("plot3_zoomed_windows.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 4: Method Comparison Bar Chart (Precision / Recall / F1)
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
method_labels = [label for _, label in methods]
x = np.arange(len(method_labels))
w = 0.25

prec_vals = [results[col]["precision"] for col, _ in methods]
rec_vals  = [results[col]["recall"]    for col, _ in methods]
f1_vals   = [results[col]["f1"]        for col, _ in methods]

bars1 = ax.bar(x - w, prec_vals, w, label="Precision", color="#42A5F5")
bars2 = ax.bar(x,     rec_vals,  w, label="Recall",    color="#66BB6A")
bars3 = ax.bar(x + w, f1_vals,   w, label="F1-Score",  color="#FFA726")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(method_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Method Comparison — Precision / Recall / F1")
ax.legend()
fig.tight_layout()
fig.savefig("plot4_method_comparison.png", dpi=150)
plt.close(fig)

print("✓ All plots saved: plot1–plot4.")