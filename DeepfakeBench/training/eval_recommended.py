#!/usr/bin/env python3
"""Full breakdown for recommended strategy: W=32, T=0.30, K=18"""
import numpy as np; np.random.seed(42)
import pandas as pd
from pathlib import Path
from collections import defaultdict

INF_DIR = Path("inference_results")
FILES = {
    "teams_flat":    INF_DIR / "r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
    "live_deepfake": INF_DIR / "r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    "poc_phase1":    INF_DIR / "r9a_run1__poc-phase-1-test.csv",
}

W, T, K = 32, 0.30, 18

def build_poc(df, W, src):
    wins = []
    for vid, g in df.groupby("video_id"):
        probs = g.sort_values("frame_name")["prob_fake"].values
        lbl = int(g["label"].iloc[0]); meth = g["method"].iloc[0]
        for s in range(0, len(probs) - W + 1, W):
            wins.append((probs[s:s+W], lbl, meth, src))
    return wins

def build_pool(df, W, src):
    wins = []
    for lbl in [0, 1]:
        for meth, mg in df[df["label"] == lbl].groupby("method"):
            p = mg["prob_fake"].values.copy()
            np.random.shuffle(p)
            for s in range(0, len(p) - W + 1, W):
                wins.append((p[s:s+W], lbl, meth, src))
    return wins

dfs = {n: pd.read_csv(p) for n, p in FILES.items()}

print(f"Strategy: W={W}, T={T}, K={K}  (K/W={K/W:.2f})")
print(f"Rule: count frames with prob_fake >= {T}, if count >= {K} out of {W} → FAKE")
print()

for bucket_name, bucket_list in [
    ("teams_flat",    [("teams_flat", build_pool)]),
    ("live_deepfake", [("live_deepfake", build_pool)]),
    ("poc_phase1",    [("poc_phase1", build_poc)]),
    ("ALL THREE COMBINED", [("teams_flat", build_pool), ("live_deepfake", build_pool), ("poc_phase1", build_poc)]),
]:
    windows = []
    for src, builder in bucket_list:
        windows += builder(dfs[src], W, src)

    print(f"  [{bucket_name}] ({len(windows)} windows)")
    print(f"  {'Method':<35s} {'n':>5} {'nF':>5} {'nR':>5}  {'TPR':>7}  {'TNR':>7}  {'Acc':>7}  {'FPR':>7}")
    print(f"  {'─'*90}")

    stats = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"nf":0,"nr":0})
    for probs, lbl, meth, src in windows:
        votes = int(np.sum(probs >= T))
        pred_fake = votes >= K
        s = stats[meth]
        if lbl == 1:
            s["nf"] += 1
            s["tp" if pred_fake else "fn"] += 1
        else:
            s["nr"] += 1
            s["fp" if pred_fake else "tn"] += 1

    tot = {"tp":0,"fp":0,"tn":0,"fn":0,"nf":0,"nr":0}
    for meth in sorted(stats.keys()):
        s = stats[meth]
        n = s["nf"] + s["nr"]
        tpr = f"{s['tp']/s['nf']*100:.1f}%" if s["nf"] > 0 else "    n/a"
        tnr = f"{s['tn']/s['nr']*100:.1f}%" if s["nr"] > 0 else "    n/a"
        fpr = f"{s['fp']/s['nr']*100:.1f}%" if s["nr"] > 0 else "    n/a"
        acc = (s["tp"] + s["tn"]) / n * 100 if n > 0 else 0
        print(f"  {meth:<35s} {n:5d} {s['nf']:5d} {s['nr']:5d}  {tpr:>7s}  {tnr:>7s}  {acc:6.1f}%  {fpr:>7s}")
        for k in tot:
            tot[k] += s[k]

    n = tot["nf"] + tot["nr"]
    tpr = tot["tp"] / tot["nf"] * 100 if tot["nf"] > 0 else 0
    tnr = tot["tn"] / tot["nr"] * 100 if tot["nr"] > 0 else 0
    fpr = tot["fp"] / tot["nr"] * 100 if tot["nr"] > 0 else 0
    acc = (tot["tp"] + tot["tn"]) / n * 100 if n > 0 else 0
    bal = (tpr + tnr) / 2
    print(f"  {'─'*90}")
    print(f"  {'TOTAL':<35s} {n:5d} {tot['nf']:5d} {tot['nr']:5d}  {tpr:6.1f}%  {tnr:6.1f}%  {acc:6.1f}%  {fpr:6.1f}%   bal_acc={bal:.1f}%")
    print()
