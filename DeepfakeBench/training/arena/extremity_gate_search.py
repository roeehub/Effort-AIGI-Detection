"""
Exhaustive grid search over extremity gate parameters.

The idea: after the base vote rule (votes >= K → FAKE), add a second condition
that requires a minimum number of "extreme" frames (prob >= E) for borderline
FAKE windows. High-confidence FAKE windows (votes >= H) skip the gate entirely.

Parameters to search:
  E  = extremity threshold (0.80 to 0.99)
  M  = min extreme frames required (1 to 20)
  H  = high-confidence vote cutoff above which gate is skipped (K to 32)

We also search over the base parameters T and K to see if the extremity gate
changes the optimal operating point.

Metric: minimize FP with minimal TPR loss. Also track Teams FP separately.
"""

import pandas as pd
import numpy as np
from itertools import product
import time

# === Config ===
W = 32
SEED = 42

CSV_DIR = "inference_results"
FILES = {
    "poc_phase1": f"{CSV_DIR}/r9a_run1__poc-phase-1-test.csv",
    "live_deepfake": f"{CSV_DIR}/r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    "teams_flat": f"{CSV_DIR}/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
}

TARGET_METHODS = {"FaceSwap", "inswap", "mobileswap", "simswap"}

# === Build windows ===
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

print("Loading data...")
all_windows = []
for src, path in FILES.items():
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    ws = build_windows(src, df, W)
    all_windows.extend(ws)
    print(f"  {src}: {len(ws)} windows")

# Precompute arrays
probs_arr = np.array([w[0] for w in all_windows])  # (N, 32)
labels_arr = np.array([w[1] for w in all_windows])
methods_arr = np.array([w[2] for w in all_windows])
sources_arr = np.array([w[3] for w in all_windows])

is_real = (labels_arr == 0)
is_fake = (labels_arr == 1)
is_teams_real = is_real & ((sources_arr == "teams_flat") | (sources_arr == "live_deepfake"))
is_target = np.array([m in TARGET_METHODS for m in methods_arr])

N = len(all_windows)
print(f"\nTotal: {N} windows, {is_fake.sum()} fake, {is_real.sum()} real, {is_teams_real.sum()} teams_real")
print(f"Target method windows: {(is_target & is_fake).sum()}")

# === Grid search ===
# Base params
T_values = np.arange(0.25, 0.55, 0.01)  # broader range to see if extremity changes optimal T
K_values = np.arange(14, 24)  # broader range

# Extremity gate params
E_values = np.arange(0.80, 1.00, 0.01)  # extremity threshold
M_values = np.arange(1, 16)              # min extreme frames
H_offsets = np.arange(0, 10)             # H = K + offset (0 = no gate skip, gate applies to all)

print(f"\nGrid: {len(T_values)} T × {len(K_values)} K × {len(E_values)} E × {len(M_values)} M × {len(H_offsets)} H_offsets")
print(f"Total combos: {len(T_values)*len(K_values)*len(E_values)*len(M_values)*len(H_offsets):,}")

# Precompute vote counts for all T values
print("\nPrecomputing vote matrices...")
t0 = time.time()

# For each T, compute votes per window: shape (len(T_values), N)
votes_matrix = {}
for ti, T in enumerate(T_values):
    votes_matrix[ti] = np.sum(probs_arr >= T, axis=1)  # (N,)

# For each E, compute extreme counts per window
extreme_matrix = {}
for ei, E in enumerate(E_values):
    extreme_matrix[ei] = np.sum(probs_arr >= E, axis=1)  # (N,)

print(f"Precompute done in {time.time()-t0:.1f}s")

# === Vectorized grid search ===
print("\nRunning grid search...")
t0 = time.time()

# Also run baseline (no extremity gate) for comparison
results = []

for ti, T in enumerate(T_values):
    votes = votes_matrix[ti]
    
    for ki, K in enumerate(K_values):
        K = int(K)
        
        # Baseline: no extremity gate
        pred_fake_base = (votes >= K)
        base_tp = np.sum(pred_fake_base & is_fake)
        base_fp = np.sum(pred_fake_base & is_real)
        base_fn = np.sum(~pred_fake_base & is_fake)
        base_tn = np.sum(~pred_fake_base & is_real)
        base_teams_fp = np.sum(pred_fake_base & is_teams_real)
        base_tgt_tp = np.sum(pred_fake_base & is_target & is_fake)
        base_tgt_total = np.sum(is_target & is_fake)
        
        base_tpr = base_tp / is_fake.sum() if is_fake.sum() > 0 else 0
        base_fpr = base_fp / is_real.sum() if is_real.sum() > 0 else 0
        base_tgt_tpr = base_tgt_tp / base_tgt_total if base_tgt_total > 0 else 0
        
        results.append({
            "T": round(T, 2), "K": K, "E": 0, "M": 0, "H": 0,
            "gate": "none",
            "tp": int(base_tp), "fp": int(base_fp), "fn": int(base_fn), "tn": int(base_tn),
            "teams_fp": int(base_teams_fp),
            "tpr": base_tpr, "fpr": base_fpr,
            "tgt_tpr": base_tgt_tpr,
            "tgt_tp": int(base_tgt_tp), "tgt_total": int(base_tgt_total),
        })
        
        # With extremity gate
        for ei, E in enumerate(E_values):
            if E <= T:
                continue  # E must be > T to make sense
            
            extremes = extreme_matrix[ei]
            
            for mi, M in enumerate(M_values):
                M = int(M)
                
                for ho in H_offsets:
                    H = K + int(ho)
                    if H > 32:
                        continue
                    
                    # Decision: FAKE if (votes >= H) OR (votes >= K AND extremes >= M)
                    # i.e., high-confidence skips gate; borderline needs extremity
                    pred_fake = (votes >= H) | ((votes >= K) & (extremes >= M))
                    
                    tp = np.sum(pred_fake & is_fake)
                    fp = np.sum(pred_fake & is_real)
                    fn = np.sum(~pred_fake & is_fake)
                    tn = np.sum(~pred_fake & is_real)
                    teams_fp = np.sum(pred_fake & is_teams_real)
                    tgt_tp = np.sum(pred_fake & is_target & is_fake)
                    
                    # Skip if identical to baseline (gate had no effect)
                    if int(tp) == int(base_tp) and int(fp) == int(base_fp):
                        continue
                    
                    tpr = tp / is_fake.sum() if is_fake.sum() > 0 else 0
                    fpr = fp / is_real.sum() if is_real.sum() > 0 else 0
                    tgt_tpr = tgt_tp / base_tgt_total if base_tgt_total > 0 else 0
                    
                    results.append({
                        "T": round(T, 2), "K": K, "E": round(E, 2),
                        "M": M, "H": H,
                        "gate": "extremity",
                        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                        "teams_fp": int(teams_fp),
                        "tpr": tpr, "fpr": fpr,
                        "tgt_tpr": tgt_tpr,
                        "tgt_tp": int(tgt_tp), "tgt_total": int(base_tgt_total),
                    })

elapsed = time.time() - t0
print(f"Grid search done: {len(results):,} results in {elapsed:.1f}s")

rdf = pd.DataFrame(results)
rdf["bal_acc"] = (rdf["tpr"] + (1 - rdf["fpr"])) / 2

# Save full results
rdf.to_csv("strategy_results/extremity_gate_full.csv", index=False)

# === Analysis (vectorized — no iterrows) ===
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

def print_rows(df, cols, fmt, max_rows=30):
    """Fast row printing using numpy arrays."""
    arrays = {c: df[c].values for c in cols}
    for i in range(min(len(df), max_rows)):
        vals = {c: arrays[c][i] for c in cols}
        print(fmt.format(**vals))

# 1. Baseline reference (our current best: T=0.30, K=18)
baseline = rdf[(rdf["T"]==0.30) & (rdf["K"]==18) & (rdf["gate"]=="none")]
if len(baseline) > 0:
    b = baseline.iloc[0]
    print(f"\nBASELINE (T=0.30, K=18, no gate):")
    print(f"  TP={b['tp']}, FP={b['fp']}, Teams_FP={b['teams_fp']}")
    print(f"  TPR={b['tpr']:.3f}, FPR={b['fpr']:.3f}, bal_acc={b['bal_acc']:.3f}")
    print(f"  Target TPR={b['tgt_tpr']:.3f}")

# 2. Zero Teams FP
zero_teams = rdf[rdf["teams_fp"] == 0].copy()
print(f"\nTotal strategies with 0 Teams FP: {len(zero_teams):,}")

# 3. PARETO FRONTIER: 0 Teams FP, best TPR at each FP level
print("\n--- PARETO FRONTIER: 0 Teams FP, best TPR at each FP level ---")
print(f"\n{'FP':>3} {'Best TPR':>8} {'TgtTPR':>7} {'BalAcc':>7} | {'gate':<10} {'T':>4} {'K':>3} {'E':>4} {'M':>3} {'H':>3}")
print("-"*80)
for fp_val in range(0, 21):
    sub = zero_teams[zero_teams["fp"] == fp_val]
    if len(sub) == 0:
        continue
    idx = sub["tpr"].idxmax()
    best = sub.loc[idx]
    print(f"{fp_val:>3} {best['tpr']:>8.4f} {best['tgt_tpr']:>7.3f} {best['bal_acc']:>7.4f} | "
          f"{best['gate']:<10} {best['T']:>4.2f} {best['K']:>3.0f} {best['E']:>4.2f} {best['M']:>3.0f} {best['H']:>3.0f}")

# 4. Best gated strategies at each FP level (0 Teams FP)
print("\n\n--- Best GATED strategies at each FP count (0 Teams FP) ---")
gated_zero = zero_teams[zero_teams["gate"] == "extremity"].copy()
print(f"\n{'T':>4} {'K':>3} {'E':>4} {'M':>3} {'H':>3} {'TP':>5} {'FP':>3} {'TPR':>6} {'FPR':>6} {'TgtTPR':>7} {'BalAcc':>7}")
print("-"*65)
for fp_target in range(0, 20):
    sub = gated_zero[gated_zero["fp"] == fp_target]
    if len(sub) == 0:
        continue
    idx = sub.sort_values(["bal_acc", "tgt_tpr"], ascending=False).index[0]
    best = sub.loc[idx]
    print(f"{best['T']:>4.2f} {best['K']:>3.0f} {best['E']:>4.2f} {best['M']:>3.0f} {best['H']:>3.0f} "
          f"{best['tp']:>5.0f} {best['fp']:>3.0f} {best['tpr']:>6.3f} {best['fpr']:>6.3f} {best['tgt_tpr']:>7.3f} {best['bal_acc']:>7.3f}")

# 5. Best gates for T=0.30, K=18 base (0 Teams FP)
print("\n\n--- Best gates for T=0.30, K=18 base (0 Teams FP) ---")
base_gated = gated_zero[(gated_zero["T"]==0.30) & (gated_zero["K"]==18)].copy()
if len(base_gated) > 0:
    # Deduplicate by (fp, tpr rounded)
    base_gated["_key"] = base_gated["fp"].astype(str) + "_" + base_gated["tpr"].round(3).astype(str)
    base_dedup = base_gated.sort_values(["fp", "tpr"], ascending=[True, False]).drop_duplicates("_key")
    print(f"\n{'E':>4} {'M':>3} {'H':>3} {'TP':>5} {'FP':>3} {'TPR':>6} {'FPR':>6} {'TgtTPR':>7} {'BalAcc':>7} {'FP_saved':>8}")
    print("-"*65)
    baseline_fp = int(b["fp"])
    for idx, row in zip(base_dedup.index[:30], base_dedup.iloc[:30].itertuples()):
        fp_saved = baseline_fp - int(row.fp)
        print(f"{row.E:>4.2f} {row.M:>3.0f} {row.H:>3.0f} "
              f"{row.tp:>5.0f} {row.fp:>3.0f} {row.tpr:>6.3f} {row.fpr:>6.3f} {row.tgt_tpr:>7.3f} {row.bal_acc:>7.3f} {fp_saved:>+8d}")

# 6. IMPROVEMENT OVER BASELINE (same or fewer FP, 0 Teams FP)
print("\n\n--- IMPROVEMENT OVER BASELINE (same or fewer FP, 0 Teams FP) ---")
baseline_fp = int(b["fp"])
baseline_tpr = float(b["tpr"])
improved = zero_teams[(zero_teams["fp"] <= baseline_fp)].copy()
improved = improved.sort_values(["tpr", "tgt_tpr", "fp"], ascending=[False, False, True])
print(f"\nBaseline: FP={baseline_fp}, TPR={baseline_tpr:.4f}")
print(f"Strategies with FP <= {baseline_fp} AND 0 Teams FP: {len(improved):,}")
if len(improved) > 0:
    improved["_key"] = improved["fp"].astype(str) + "_" + improved["tpr"].round(4).astype(str)
    top = improved.drop_duplicates("_key").head(20)
    print(f"\nTop 20 by TPR:")
    print(f"{'gate':<10} {'T':>4} {'K':>3} {'E':>4} {'M':>3} {'H':>3} {'TP':>5} {'FP':>3} {'TPR':>6} {'FPR':>6} {'TgtTPR':>7} {'BalAcc':>7}")
    print("-"*70)
    for row in top.itertuples():
        print(f"{row.gate:<10} {row.T:>4.2f} {row.K:>3.0f} {row.E:>4.2f} {row.M:>3.0f} {row.H:>3.0f} "
              f"{row.tp:>5.0f} {row.fp:>3.0f} {row.tpr:>6.4f} {row.fpr:>6.3f} {row.tgt_tpr:>7.3f} {row.bal_acc:>7.4f}")

# 7. Relaxed: allow up to 1-2 Teams FP
print("\n\n--- RELAXED: allow 1-2 Teams FP ---")
for max_teams in [1, 2]:
    relaxed = rdf[rdf["teams_fp"] <= max_teams].copy()
    relaxed = relaxed.sort_values(["fp", "tpr"], ascending=[True, False])
    relaxed_dedup = relaxed.drop_duplicates("fp").head(12)
    print(f"\n  Max {max_teams} Teams FP — best by FP:")
    print(f"  {'gate':<10} {'T':>4} {'K':>3} {'E':>4} {'M':>3} {'H':>3} {'TP':>5} {'FP':>3} {'T_FP':>4} {'TPR':>6} {'TgtTPR':>7} {'BalAcc':>7}")
    print("  " + "-"*75)
    for row in relaxed_dedup.itertuples():
        print(f"  {row.gate:<10} {row.T:>4.2f} {row.K:>3.0f} {row.E:>4.2f} {row.M:>3.0f} {row.H:>3.0f} "
              f"{row.tp:>5.0f} {row.fp:>3.0f} {row.teams_fp:>4.0f} {row.tpr:>6.3f} {row.tgt_tpr:>7.3f} {row.bal_acc:>7.3f}")

# 8. NEW: What does the gate buy across different base T/K combos?
print("\n\n--- GATE VALUE: best gated vs best ungated at same T,K (0 Teams FP) ---")
ungated = zero_teams[zero_teams["gate"] == "none"].copy()
gated = zero_teams[zero_teams["gate"] == "extremity"].copy()

# For each (T,K), find best ungated and best gated
ungated_best = ungated.groupby(["T","K"]).agg({"fp":"min","tpr":"max","tgt_tpr":"max","bal_acc":"max"}).reset_index()
ungated_best.columns = ["T","K","base_fp","base_tpr","base_tgt","base_bal"]

gated_agg = gated.loc[gated.groupby(["T","K"])["bal_acc"].idxmax()][["T","K","E","M","H","fp","tpr","tgt_tpr","bal_acc"]].copy()
gated_agg.columns = ["T","K","E","M","H","gate_fp","gate_tpr","gate_tgt","gate_bal"]

merged = ungated_best.merge(gated_agg, on=["T","K"], how="inner")
merged["fp_saved"] = merged["base_fp"] - merged["gate_fp"]
merged["tpr_cost"] = merged["base_tpr"] - merged["gate_tpr"]
merged = merged[merged["fp_saved"] > 0].sort_values("fp_saved", ascending=False)

print(f"\n{'T':>4} {'K':>3} | {'Base FP':>7} {'Gate FP':>7} {'Saved':>5} | {'Base TPR':>8} {'Gate TPR':>8} {'Cost':>6} | {'E':>4} {'M':>3} {'H':>3}")
print("-"*85)
for row in merged.head(25).itertuples():
    print(f"{row.T:>4.2f} {row.K:>3.0f} | {row.base_fp:>7.0f} {row.gate_fp:>7.0f} {row.fp_saved:>+5.0f} | "
          f"{row.base_tpr:>8.4f} {row.gate_tpr:>8.4f} {row.tpr_cost:>+6.4f} | {row.E:>4.2f} {row.M:>3.0f} {row.H:>3.0f}")

print(f"\n\nFull results saved to strategy_results/extremity_gate_full.csv")
print(f"Total rows: {len(rdf):,}")
