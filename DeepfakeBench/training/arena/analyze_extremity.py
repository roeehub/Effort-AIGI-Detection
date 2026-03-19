"""
Analyze probability extremity patterns in TP vs FP windows.

Question: Do true-fake windows have more "extreme" (>0.95, >0.98) frames
compared to false-positive windows that merely pass the vote threshold?
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# === Config ===
W = 32
T = 0.30
K = 18
SEED = 42

CSV_DIR = "inference_results"
FILES = {
    "poc_phase1": f"{CSV_DIR}/r9a_run1__poc-phase-1-test.csv",
    "live_deepfake": f"{CSV_DIR}/r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    "teams_flat": f"{CSV_DIR}/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
}

# === Build windows (same logic as evaluation scripts) ===
def build_windows(source_name, df, W):
    windows = []
    if source_name == "poc_phase1":
        for vid, grp in df.groupby("video_id"):
            probs = grp.sort_values("frame_name")["prob_fake"].values
            label = grp["label"].iloc[0]
            method = grp["method"].iloc[0]
            for i in range(0, len(probs) - W + 1, W):
                chunk = probs[i:i+W]
                if len(chunk) == W:
                    windows.append((chunk, label, method, source_name))
    else:
        # Pool by label, random partition
        rng = np.random.RandomState(SEED)
        for label_val in df["label"].unique():
            sub = df[df["label"] == label_val]
            method = sub["method"].iloc[0] if "method" in sub.columns else ("real" if label_val == 0 else "fake")
            probs = sub["prob_fake"].values.copy()
            rng.shuffle(probs)
            for i in range(0, len(probs) - W + 1, W):
                chunk = probs[i:i+W]
                if len(chunk) == W:
                    windows.append((chunk, label_val, method, source_name))
    return windows

all_windows = []
for src, path in FILES.items():
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    ws = build_windows(src, df, W)
    all_windows.extend(ws)
    print(f"{src}: {len(ws)} windows")

print(f"\nTotal windows: {len(all_windows)}")

# === Classify each window and compute extremity stats ===
results = []
for probs, label, method, source in all_windows:
    votes = np.sum(probs >= T)
    is_true_fake = (label == 1)
    predicted_fake = (votes >= K)
    
    # Window classification
    if is_true_fake and predicted_fake:
        category = "TP"  # true positive (correctly detected fake)
    elif is_true_fake and not predicted_fake:
        category = "FN"  # false negative (missed fake)
    elif not is_true_fake and predicted_fake:
        category = "FP"  # false positive (real classified as fake)
    else:
        category = "TN"  # true negative (correctly classified real)
    
    suspicious = probs[probs >= T]
    
    results.append({
        "category": category,
        "method": method,
        "source": source,
        "label": label,
        "votes": int(votes),
        "n_above_T": int(len(suspicious)),
        # Extremity metrics for suspicious frames
        "mean_suspicious": float(np.mean(suspicious)) if len(suspicious) > 0 else 0,
        "median_suspicious": float(np.median(suspicious)) if len(suspicious) > 0 else 0,
        "max_prob": float(np.max(probs)),
        "n_above_90": int(np.sum(probs >= 0.90)),
        "n_above_95": int(np.sum(probs >= 0.95)),
        "n_above_98": int(np.sum(probs >= 0.98)),
        "n_above_99": int(np.sum(probs >= 0.99)),
        # Also track the clean frames in suspicious windows
        "mean_all": float(np.mean(probs)),
        "std_all": float(np.std(probs)),
        # Ratio: what fraction of suspicious frames are extreme?
        "pct_above_95_of_suspicious": float(np.sum(probs >= 0.95) / len(suspicious)) if len(suspicious) > 0 else 0,
        "pct_above_98_of_suspicious": float(np.sum(probs >= 0.98) / len(suspicious)) if len(suspicious) > 0 else 0,
    })

rdf = pd.DataFrame(results)

# === Report ===
print("\n" + "="*80)
print("EXTREMITY ANALYSIS: TP vs FP windows (W=32, T=0.30, K=18)")
print("="*80)

for cat in ["TP", "FP", "FN", "TN"]:
    sub = rdf[rdf["category"] == cat]
    if len(sub) == 0:
        continue
    print(f"\n--- {cat} ({len(sub)} windows) ---")
    cols = ["votes", "mean_suspicious", "median_suspicious", "max_prob",
            "n_above_90", "n_above_95", "n_above_98", "n_above_99",
            "pct_above_95_of_suspicious", "pct_above_98_of_suspicious"]
    print(sub[cols].describe().round(3).to_string())

# === Direct comparison: TP vs FP ===
tp = rdf[rdf["category"] == "TP"]
fp = rdf[rdf["category"] == "FP"]

print("\n" + "="*80)
print("HEAD-TO-HEAD: TP vs FP")
print("="*80)

metrics = [
    ("votes", "Avg votes"),
    ("mean_suspicious", "Mean prob of suspicious frames"),
    ("n_above_95", "Avg # frames >= 0.95"),
    ("n_above_98", "Avg # frames >= 0.98"),
    ("n_above_99", "Avg # frames >= 0.99"),
    ("pct_above_95_of_suspicious", "% of suspicious that are >= 0.95"),
    ("pct_above_98_of_suspicious", "% of suspicious that are >= 0.98"),
    ("max_prob", "Max prob in window"),
]

print(f"\n{'Metric':<45} {'TP (mean)':>10} {'FP (mean)':>10} {'Gap':>10}")
print("-"*75)
for col, label in metrics:
    tp_val = tp[col].mean()
    fp_val = fp[col].mean()
    gap = tp_val - fp_val
    print(f"{label:<45} {tp_val:>10.3f} {fp_val:>10.3f} {gap:>+10.3f}")

# === Distribution of n_above_95 for TP vs FP ===
print("\n" + "="*80)
print("DISTRIBUTION: # frames >= 0.95 per window")
print("="*80)
for cat, sub, name in [(tp, tp, "TP"), (fp, fp, "FP")]:
    print(f"\n{name} ({len(sub)} windows):")
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 33]
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        cnt = len(sub[(sub["n_above_95"] >= lo) & (sub["n_above_95"] < hi)])
        pct = 100*cnt/len(sub) if len(sub) > 0 else 0
        print(f"  {lo:>2}-{hi-1:<2} frames >= 0.95: {cnt:>4} ({pct:>5.1f}%)")

# === Distribution of n_above_98 for TP vs FP ===
print("\n" + "="*80)
print("DISTRIBUTION: # frames >= 0.98 per window")
print("="*80)
for cat, sub, name in [(tp, tp, "TP"), (fp, fp, "FP")]:
    print(f"\n{name} ({len(sub)} windows):")
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 33]
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        cnt = len(sub[(sub["n_above_98"] >= lo) & (sub["n_above_98"] < hi)])
        pct = 100*cnt/len(sub) if len(sub) > 0 else 0
        print(f"  {lo:>2}-{hi-1:<2} frames >= 0.98: {cnt:>4} ({pct:>5.1f}%)")

# === FP breakdown by method ===
print("\n" + "="*80)
print("FALSE POSITIVE WINDOWS — per-method detail")
print("="*80)
for _, row in fp.sort_values("n_above_95").iterrows():
    print(f"  src={row['source']:<15} method={row['method']:<20} votes={row['votes']:>2}  "
          f">=0.95:{row['n_above_95']:>2}  >=0.98:{row['n_above_98']:>2}  >=0.99:{row['n_above_99']:>2}  "
          f"max={row['max_prob']:.4f}  mean_susp={row['mean_suspicious']:.3f}")

# === Can we find a threshold? ===
print("\n" + "="*80)
print("POTENTIAL RULE: require min N frames >= extreme_threshold")
print("="*80)
for extreme_t in [0.90, 0.95, 0.98, 0.99]:
    col = f"n_above_{int(extreme_t*100)}"
    if col not in rdf.columns:
        continue
    fp_max = int(fp[col].max()) if len(fp) > 0 else 0
    print(f"\n  Extreme threshold = {extreme_t}")
    print(f"  FP max {col}: {fp_max}")
    for min_n in range(0, min(fp_max + 3, 20)):
        # How many TP would we lose if we require >= min_n extreme frames?
        tp_pass = len(tp[tp[col] >= min_n])
        fp_pass = len(fp[fp[col] >= min_n])
        tp_pct = 100*tp_pass/len(tp) if len(tp) > 0 else 0
        fp_cnt = fp_pass
        print(f"    require >= {min_n:>2} frames at {extreme_t}: TP kept={tp_pass}/{len(tp)} ({tp_pct:.1f}%), FP remaining={fp_cnt}/{len(fp)}")
