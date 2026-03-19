#!/usr/bin/env python3
"""
Teams-focused strategy search (vectorized).
Priority: (1) Teams FPR = 0%  (2) Max TPR on swap methods  (3) bonus
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

SEED = 42
np.random.seed(SEED)

INF_DIR = Path("inference_results")
FILES = {
    "teams_flat":     INF_DIR / "r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
    "live_deepfake":  INF_DIR / "r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    "poc_phase1":     INF_DIR / "r9a_run1__poc-phase-1-test.csv",
}

TARGET_METHODS = ["FaceSwap", "inswap", "mobileswap", "simswap"]
WINDOW_SIZES = [16, 24, 32]

# ── Window builders ──────────────────────────────────────────────────
def build_windows_poc(df, W, source="poc"):
    wins = []
    for vid, g in df.groupby("video_id"):
        probs = g.sort_values("frame_name")["prob_fake"].values
        label = int(g["label"].iloc[0])
        method = g["method"].iloc[0]
        if len(probs) < W:
            continue
        for s in range(0, len(probs) - W + 1, W):
            wins.append((probs[s:s+W], label, method, source))
    return wins

def build_windows_pool(df, W, source="pool"):
    wins = []
    for lbl in [0, 1]:
        sub = df[df["label"] == lbl]
        for method, mg in sub.groupby("method"):
            probs = mg["prob_fake"].values.copy()
            np.random.shuffle(probs)
            for s in range(0, len(probs) - W + 1, W):
                wins.append((probs[s:s+W], lbl, method, source))
    return wins

def build_windows_flat(df, W, source="flat"):
    return build_windows_pool(df, W, source=source)


def main():
    t0 = time.time()

    dfs = {n: pd.read_csv(p) for n, p in FILES.items()}
    for n, d in dfs.items():
        print(f"Loaded {len(d)} frames from {n}")

    teams_bucket_methods = set()
    for n in ["teams_flat", "live_deepfake"]:
        teams_bucket_methods.update(dfs[n]["method"].unique())

    print(f"Teams methods: {sorted(teams_bucket_methods)}")
    print(f"Target swap methods: {TARGET_METHODS}")

    thresholds = np.round(np.arange(0.30, 0.96, 0.01), 3)
    k_ratios   = np.round(np.arange(0.30, 0.80, 0.02), 3)

    all_results = []

    for W in WINDOW_SIZES:
        print(f"\n{'='*60}  W={W}  {'='*60}")

        # Build windows — track source bucket
        windows = []
        windows += build_windows_flat(dfs["teams_flat"], W, source="teams_flat")
        windows += build_windows_pool(dfs["live_deepfake"], W, source="live_deepfake")
        windows += build_windows_poc(dfs["poc_phase1"], W, source="poc_phase1")

        # Convert to arrays
        N = len(windows)
        prob_matrix = np.zeros((N, W))
        labels      = np.zeros(N, dtype=int)
        methods_arr = []
        sources_arr = []
        for i, (p, l, m, src) in enumerate(windows):
            prob_matrix[i] = p
            labels[i] = l
            methods_arr.append(m)
            sources_arr.append(src)
        methods_arr = np.array(methods_arr)
        sources_arr = np.array(sources_arr)

        is_real = (labels == 0)
        is_fake = (labels == 1)
        # Teams real = real windows from teams_flat or live_deepfake ONLY (not poc_phase1)
        is_teams_real = is_real & ((sources_arr == "teams_flat") | (sources_arr == "live_deepfake"))

        target_masks = {}
        for tm in TARGET_METHODS:
            target_masks[tm] = (methods_arr == tm) & is_fake

        n_teams_real = int(is_teams_real.sum())
        print(f"  Windows: {N} (fake={is_fake.sum()}, real={is_real.sum()}, teams_real={n_teams_real})")

        for T in thresholds:
            votes = (prob_matrix >= T).sum(axis=1)  # (N,)

            for kr in k_ratios:
                K = max(1, int(round(kr * W)))

                # Binary
                pred_fake = (votes >= K)
                teams_fp = int((pred_fake & is_teams_real).sum())
                teams_fpr = teams_fp / n_teams_real if n_teams_real > 0 else 0

                tprs = {}
                for tm in TARGET_METHODS:
                    mask = target_masks[tm]
                    n_m = int(mask.sum())
                    tprs[tm] = float((pred_fake & mask).sum()) / n_m if n_m > 0 else None

                valid = [v for v in tprs.values() if v is not None]
                mean_tpr = float(np.mean(valid)) if valid else 0

                tp = int((pred_fake & is_fake).sum())
                fp = int((pred_fake & is_real).sum())
                tn = int((~pred_fake & is_real).sum())
                fn = int((~pred_fake & is_fake).sum())
                tpr_all = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr_all = tn / (tn + fp) if (tn + fp) > 0 else 0

                all_results.append({
                    "W": W, "T": round(float(T), 3), "K": K,
                    "K_ratio": round(float(kr), 3), "margin": 0,
                    "teams_fp": teams_fp, "teams_real_n": n_teams_real,
                    "mean_target_tpr": mean_tpr,
                    "FaceSwap_tpr": tprs.get("FaceSwap"),
                    "inswap_tpr": tprs.get("inswap"),
                    "mobileswap_tpr": tprs.get("mobileswap"),
                    "simswap_tpr": tprs.get("simswap"),
                    "overall_tpr": tpr_all, "overall_tnr": tnr_all,
                    "overall_bal_acc": (tpr_all + tnr_all) / 2,
                    "unc_pct": 0,
                })

            # Uncertain margins
            for margin in [1, 2, 3]:
                for kr in k_ratios:
                    K = max(1, int(round(kr * W)))
                    K_hi = K + margin
                    K_lo = K - margin
                    if K_hi > W or K_lo < 1:
                        continue

                    uncertain = (votes >= K_lo) & (votes < K_hi)
                    pred_fake_u = (votes >= K_hi)
                    pred_real_u = (votes < K_lo)

                    teams_fp_u = int((pred_fake_u & is_teams_real).sum())

                    tprs = {}
                    for tm in TARGET_METHODS:
                        mask = target_masks[tm]
                        n_m = int(mask.sum())
                        tprs[tm] = float((pred_fake_u & mask).sum()) / n_m if n_m > 0 else None

                    valid = [v for v in tprs.values() if v is not None]
                    mean_tpr = float(np.mean(valid)) if valid else 0

                    tp = int((pred_fake_u & is_fake).sum())
                    fp = int((pred_fake_u & is_real).sum())
                    tn = int((pred_real_u & is_real).sum())
                    fn = int((pred_real_u & is_fake).sum())
                    unc = int(uncertain.sum())
                    tpr_all = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tnr_all = tn / (tn + fp) if (tn + fp) > 0 else 0

                    all_results.append({
                        "W": W, "T": round(float(T), 3), "K": K,
                        "K_ratio": round(float(kr), 3), "margin": margin,
                        "teams_fp": teams_fp_u, "teams_real_n": n_teams_real,
                        "mean_target_tpr": mean_tpr,
                        "FaceSwap_tpr": tprs.get("FaceSwap"),
                        "inswap_tpr": tprs.get("inswap"),
                        "mobileswap_tpr": tprs.get("mobileswap"),
                        "simswap_tpr": tprs.get("simswap"),
                        "overall_tpr": tpr_all, "overall_tnr": tnr_all,
                        "overall_bal_acc": (tpr_all + tnr_all) / 2,
                        "unc_pct": unc / N if N > 0 else 0,
                    })

        print(f"  Combos so far: {len(all_results)}")

    df_r = pd.DataFrame(all_results)
    print(f"\nTotal combos: {len(df_r)}  Time: {time.time()-t0:.1f}s")

    out_dir = Path("strategy_results")
    out_dir.mkdir(exist_ok=True)
    df_r.to_csv(out_dir / "teams_focused_all.csv", index=False)

    # ── REPORT ───────────────────────────────────────────────────────
    lines = []
    def pr(s=""):
        lines.append(s); print(s)

    pr("=" * 110)
    pr("TEAMS-FOCUSED STRATEGY SEARCH  —  Priority: (1) Teams FPR=0%  (2) Max target swap TPR  (3) bonus")
    pr(f"Target methods: {TARGET_METHODS}")
    pr(f"Total combos evaluated: {len(df_r)}")
    pr("=" * 110)

    z = df_r[df_r["teams_fp"] == 0].copy()
    pr(f"\nStrategies with ZERO Teams false positives: {len(z)} / {len(df_r)}")

    zb = z[z["margin"] == 0]
    pr(f"  ... of which binary (no uncertain zone): {len(zb)}")

    for W in WINDOW_SIZES:
        wsub = zb[zb["W"] == W].sort_values("mean_target_tpr", ascending=False)
        if len(wsub) == 0:
            pr(f"\n  W={W}: no zero-FPR binary strategies"); continue

        pr(f"\n  W={W} — Top 15 binary strategies (Teams FP=0, sorted by mean target TPR):")
        pr(f"  {'T':>5} {'K':>3} {'K/W':>5} | {'FSwap':>7} {'insw':>7} {'mob':>7} {'sim':>7} {'MeanTgt':>8} | {'AllTPR':>7} {'AllTNR':>7} {'BalAcc':>7}")
        pr(f"  {'─'*90}")

        seen = set(); count = 0
        for _, r in wsub.iterrows():
            key = (r["T"], r["K"])
            if key in seen: continue
            seen.add(key)
            fs = r["FaceSwap_tpr"]; ins = r["inswap_tpr"]
            mob = r["mobileswap_tpr"]; sim = r["simswap_tpr"]
            pr(f"  {r['T']:5.2f} {int(r['K']):3d} {r['K_ratio']:5.2f} | "
               f"{fs*100:6.1f}% {ins*100:6.1f}% {mob*100:6.1f}% {sim*100:6.1f}% {r['mean_target_tpr']*100:7.1f}% | "
               f"{r['overall_tpr']*100:6.1f}% {r['overall_tnr']*100:6.1f}% {r['overall_bal_acc']*100:6.1f}%")
            count += 1
            if count >= 15: break

    zu = z[z["margin"] > 0]
    pr(f"\n\n  Uncertain-zone strategies with zero Teams FP: {len(zu)}")
    for W in WINDOW_SIZES:
        wsub = zu[zu["W"] == W].sort_values("mean_target_tpr", ascending=False)
        if len(wsub) == 0: continue
        pr(f"\n  W={W} — Top 10 uncertain strategies (Teams FP=0):")
        pr(f"  {'T':>5} {'K':>3} {'mrg':>3} | {'FSwap':>7} {'insw':>7} {'mob':>7} {'sim':>7} {'MeanTgt':>8} | {'AllTPR':>7} {'AllTNR':>7} {'BalAcc':>7} {'unc%':>5}")
        pr(f"  {'─'*100}")
        seen = set(); count = 0
        for _, r in wsub.iterrows():
            key = (r["T"], r["K"], r["margin"])
            if key in seen: continue
            seen.add(key)
            fs = r["FaceSwap_tpr"]; ins = r["inswap_tpr"]
            mob = r["mobileswap_tpr"]; sim = r["simswap_tpr"]
            pr(f"  {r['T']:5.2f} {int(r['K']):3d} {int(r['margin']):3d} | "
               f"{fs*100:6.1f}% {ins*100:6.1f}% {mob*100:6.1f}% {sim*100:6.1f}% {r['mean_target_tpr']*100:7.1f}% | "
               f"{r['overall_tpr']*100:6.1f}% {r['overall_tnr']*100:6.1f}% {r['overall_bal_acc']*100:6.1f}% {r['unc_pct']*100:4.1f}%")
            count += 1
            if count >= 10: break

    # ── RECOMMENDED ──────────────────────────────────────────────────
    pr(f"\n{'='*110}")
    pr("RECOMMENDED STRATEGIES")
    pr(f"{'='*110}")

    for W in WINDOW_SIZES:
        wb = zb[zb["W"] == W]
        if len(wb) > 0:
            best = wb.loc[wb["mean_target_tpr"].idxmax()]
            pr(f"\n  ★ W={W} BINARY  T={best['T']:.2f}  K={int(best['K'])}  (K/W={best['K_ratio']:.2f})")
            pr(f"    Teams FP: 0  |  FaceSwap {best['FaceSwap_tpr']*100:.1f}%  inswap {best['inswap_tpr']*100:.1f}%  mobileswap {best['mobileswap_tpr']*100:.1f}%  simswap {best['simswap_tpr']*100:.1f}%")
            pr(f"    Mean target TPR: {best['mean_target_tpr']*100:.1f}%  |  Overall bal_acc: {best['overall_bal_acc']*100:.1f}%  (TPR={best['overall_tpr']*100:.1f}%, TNR={best['overall_tnr']*100:.1f}%)")

    # ── BASELINE COMPARISON ──────────────────────────────────────────
    pr(f"\n{'='*110}")
    pr("BASELINE COMPARISON  (W=16, T=0.80, K=8)")
    pr(f"{'='*110}")

    bl = df_r[(df_r["W"]==16) & (abs(df_r["T"]-0.80)<0.005) & (df_r["K"]==8) & (df_r["margin"]==0)]
    if len(bl) > 0:
        b = bl.iloc[0]
        pr(f"  Teams FP: {int(b['teams_fp'])}  |  FaceSwap {b['FaceSwap_tpr']*100:.1f}%  inswap {b['inswap_tpr']*100:.1f}%  mobileswap {b['mobileswap_tpr']*100:.1f}%  simswap {b['simswap_tpr']*100:.1f}%")
        pr(f"  Mean target TPR: {b['mean_target_tpr']*100:.1f}%  |  Overall bal_acc: {b['overall_bal_acc']*100:.1f}%  (TPR={b['overall_tpr']*100:.1f}%, TNR={b['overall_tnr']*100:.1f}%)")

    # ── PARETO FRONTIER ──────────────────────────────────────────────
    pr(f"\n{'='*110}")
    pr("PARETO FRONTIER: Teams FP count vs Mean Target TPR (binary, all W)")
    pr("  How many Teams FPs do you need to tolerate for higher swap TPR?")
    pr(f"{'='*110}")

    for W in WINDOW_SIZES:
        wb = df_r[(df_r["W"]==W) & (df_r["margin"]==0)].copy()
        pareto_pts = []
        for fp_level in sorted(wb["teams_fp"].unique()):
            sub = wb[wb["teams_fp"] == fp_level]
            best_row = sub.loc[sub["mean_target_tpr"].idxmax()]
            if not pareto_pts or best_row["mean_target_tpr"] > pareto_pts[-1]["mean_target_tpr"]:
                pareto_pts.append(best_row)

        n_tr = int(wb.iloc[0]['teams_real_n'])
        pr(f"\n  W={W} (Teams has {n_tr} real windows):")
        pr(f"  {'TeamsFP':>7} {'FPR%':>6} {'T':>5} {'K':>3} | {'FSwap':>7} {'insw':>7} {'mob':>7} {'sim':>7} {'MeanTgt':>8} | {'BalAcc':>7}")
        pr(f"  {'─'*85}")
        for r in pareto_pts[:15]:
            fpr_pct = int(r['teams_fp']) / n_tr * 100
            pr(f"  {int(r['teams_fp']):7d} {fpr_pct:5.1f}% {r['T']:5.2f} {int(r['K']):3d} | "
               f"{r['FaceSwap_tpr']*100:6.1f}% {r['inswap_tpr']*100:6.1f}% "
               f"{r['mobileswap_tpr']*100:6.1f}% {r['simswap_tpr']*100:6.1f}% "
               f"{r['mean_target_tpr']*100:7.1f}% | {r['overall_bal_acc']*100:6.1f}%")

    report_path = Path("/tmp/teams_focused_report.txt")
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}  ({len(lines)} lines)")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
