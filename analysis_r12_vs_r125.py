#!/usr/bin/env python3
"""
R12 vs R12.5 Comparison Script — Informing R13 Experiment Design
================================================================
Pulls W&B metrics from both projects, aligns by epoch/step, and
produces a comparison table + trend analysis for R13 planning.

Uses summary + scan_history (filtered to OOD rows only) for speed.
"""

import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Run registry ──────────────────────────────────────────────
R12_PROJECT = "dtect-vision/phase2r12-experiments"
R125_PROJECT = "dtect-vision/phase2r125-experiments"

RUNS = {
    # R12 long-runs (still running)
    "R12_A": {"project": R12_PROJECT, "id": "4v8av986", "desc": "Scratch, k=32, seed=737, no GRL (BASELINE for R12.5)"},
    "R12_B": {"project": R12_PROJECT, "id": "wupk4909", "desc": "Scratch, k=64, seed=42, no GRL"},
    "R12_G": {"project": R12_PROJECT, "id": "0xxqwhxg", "desc": "Scratch, k=32, seed=737, no GRL (compound aug)"},
    # R12.5 ablation (still running)
    "R125_A": {"project": R125_PROJECT, "id": "uq160clj", "desc": "+GammaUp only (p=0.15, γ∈[0.45,0.85])"},
    "R125_B": {"project": R125_PROJECT, "id": "109xganm", "desc": "+GammaUp + DirectionalShadow (shadow_p=0.10)"},
    "R125_C": {"project": R125_PROJECT, "id": "z5u13bzu", "desc": "+Max lighting envelope (wide CCT/brightness)"},
}

# ── Actual W&B metric key names (verified) ────────────────────
# Per-method AUC is -1 for single-class groups; only acc is meaningful
HISTORY_KEYS = [
    "ood/overall/auc",
    "ood/overall/eer",
    "ood/overall/acc",
    "val_primary/ood_composite",
    "val_primary/holdout_auc_for_composite",
    "val_primary/ood_auc_for_composite",
    # Per-source OOD accuracy
    "ood/method/zoom_vcd_real_real/acc",
    "ood/method/external_youtube_avspeech_real/acc",
    "ood/method/wma_failure_fake_fake/acc",
    "ood/method/teams_ood_real_real/acc",
    "ood/method/teams_ood_fake_fake/acc",
    # Holdout facedancer (logged from holdout eval)
    "val_holdout/method/facedancer/acc",
    # Alignment
    "ood/epoch",
    "train/step",
    "_step",
]

# Friendly column names
COL_MAP = {
    "ood/overall/auc": "ood_auc",
    "ood/overall/eer": "ood_eer",
    "ood/overall/acc": "ood_acc",
    "val_primary/ood_composite": "composite",
    "val_primary/holdout_auc_for_composite": "holdout_auc",
    "val_primary/ood_auc_for_composite": "ood_auc_comp",
    "ood/method/zoom_vcd_real_real/acc": "vcd_real",
    "ood/method/external_youtube_avspeech_real/acc": "yt_real",
    "ood/method/wma_failure_fake_fake/acc": "wma_fake",
    "ood/method/teams_ood_real_real/acc": "teams_real",
    "ood/method/teams_ood_fake_fake/acc": "teams_fake",
    "val_holdout/method/facedancer/acc": "facedancer",
    "ood/epoch": "epoch",
    "train/step": "step",
}


import signal

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("W&B API call timed out")

def fetch_run_data(run_info):
    """Pull summary + epoch-level OOD history for a run.
    
    Uses run.history(samples=10000) to capture sparse OOD eval rows.
    Adds a 90s timeout to avoid hanging on slow W&B responses.
    """
    api = wandb.Api(timeout=120)
    run = api.run(f"{run_info['project']}/{run_info['id']}")

    summary = dict(run.summary)

    # Use moderate sample count with timeout protection
    history_rows = []
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(90)  # 90 second timeout
        
        raw_df = run.history(samples=10000)
        
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        
        for _, row in raw_df.iterrows():
            oauc = row.get("ood/overall/auc")
            if pd.isna(oauc) or oauc is None:
                continue
            mapped = {}
            for wk, friendly in COL_MAP.items():
                val = row.get(wk)
                mapped[friendly] = val if not pd.isna(val) else None
            if mapped.get("step") is None:
                s = row.get("_step")
                mapped["step"] = s if not pd.isna(s) else None
            history_rows.append(mapped)
    except (TimeoutError, Exception) as e:
        signal.alarm(0)
        print(f"\n    (history: {e}, using summary only)")

    return summary, history_rows, run.state


def build_df(all_histories):
    """Build a tidy DataFrame from all runs' history."""
    rows = []
    for name, hist in all_histories.items():
        for h in hist:
            h["run"] = name
            rows.append(h)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["epoch"] = df["epoch"].astype(int)
        if "step" in df.columns:
            df["step"] = df["step"].apply(lambda x: int(x) if x is not None else None)
    return df


def sep(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def main():
    print("Fetching W&B data for R12 (long-run) and R12.5 runs...\n")

    summaries = {}
    all_histories = {}
    states = {}

    for name, info in RUNS.items():
        print(f"  Fetching {name} ({info['id']})...", end=" ", flush=True)
        try:
            summary, history, state = fetch_run_data(info)
            summaries[name] = summary
            all_histories[name] = history
            states[name] = state
            print(f"✓  {state}, {len(history)} OOD evals")
        except Exception as e:
            print(f"✗  {e}")

    # ── 1. Current Status ─────────────────────────────────────
    sep("1. RUN STATUS & LATEST SUMMARY")
    header = f"{'Run':8s} {'State':10s} {'Epoch':>5s} {'Step':>7s}  {'H.AUC':>7s} {'OOD AUC':>8s} {'Composite':>9s} {'VCD Real':>8s} {'FD':>6s} {'WMA':>6s} {'T.Real':>6s} {'T.Fake':>6s}"
    print(header)
    print("-" * len(header))
    for name, info in RUNS.items():
        s = summaries.get(name, {})
        st = states.get(name, "?")
        ep = s.get("ood/epoch", "?")
        step = s.get("train/step", s.get("_step", "?"))
        hauc = s.get("val_primary/holdout_auc_for_composite")
        oauc = s.get("ood/overall/auc")
        comp = s.get("val_primary/ood_composite")
        vcd = s.get("ood/method/zoom_vcd_real_real/acc")
        fd = s.get("val_holdout/method/facedancer/acc")
        wma = s.get("ood/method/wma_failure_fake_fake/acc")
        tr = s.get("ood/method/teams_ood_real_real/acc")
        tf = s.get("ood/method/teams_ood_fake_fake/acc")
        
        def fmt(v, pct=False):
            if v is None: return "   N/A"
            return f"{v:6.1%}" if pct else f"{v:8.4f}"
        
        print(f"{name:8s} {st:10s} {str(ep):>5s} {str(step):>7s}  {fmt(hauc)} {fmt(oauc)} {fmt(comp)} {fmt(vcd, True)} {fmt(fd, True)} {fmt(wma, True)} {fmt(tr, True)} {fmt(tf, True)}")
    
    # Also show best composite checkpoint from summary
    print(f"\n  Best OOD Composite Checkpoints (from W&B summary):")
    for name in RUNS:
        s = summaries.get(name, {})
        bc = s.get("best_ood_composite/metric")
        bh = s.get("best_ood_composite/holdout_auc")
        bo = s.get("best_ood_composite/ood_auc")
        bs = s.get("best_ood_composite/step")
        bp = s.get("best_ood_composite/gcs_path", "")
        if bc:
            print(f"    {name:8s}  composite={bc:.4f}  H.AUC={bh:.4f}  OOD={bo:.4f}  step={bs}  {bp}")

    # ── 2. Build DataFrame ────────────────────────────────────
    df = build_df(all_histories)
    if df.empty:
        print("\nNo OOD history data found. Exiting.")
        return

    # ── 3. Epoch-by-epoch pivot tables ────────────────────────
    sep("2. EPOCH-BY-EPOCH METRICS")
    col_order = [c for c in ["R12_A", "R12_B", "R12_G", "R125_A", "R125_B", "R125_C"] if c in df["run"].unique()]

    for metric in ["holdout_auc", "ood_auc", "composite", "vcd_real", "facedancer", "wma_fake", "teams_real", "teams_fake"]:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index="epoch", columns="run", values=metric, aggfunc="last")
        present = [c for c in col_order if c in pivot.columns]
        if not present:
            continue
        pivot = pivot[present]
        # Format percentages for accuracy columns
        if metric in ["vcd_real", "facedancer", "wma_fake", "teams_real", "teams_fake"]:
            pivot = pivot.map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        print(f"\n--- {metric} ---")
        print(pivot.to_string())

    # ── 4. Matched-epoch deltas (R12_A baseline) ──────────────
    sep("3. R12_A vs R12.5 — MATCHED EPOCH DELTAS")
    print("  (Positive = R12.5 run BETTER than R12_A baseline)\n")

    # Deduplicate: keep last entry per (run, epoch)
    df_dedup = df.drop_duplicates(subset=["run", "epoch"], keep="last")
    baseline_df = df_dedup[df_dedup["run"] == "R12_A"].set_index("epoch")
    compare_runs = ["R125_A", "R125_B", "R125_C"]
    delta_metrics = ["holdout_auc", "ood_auc", "composite", "vcd_real", "wma_fake", "teams_real", "facedancer"]

    for cname in compare_runs:
        cdf = df_dedup[df_dedup["run"] == cname].set_index("epoch")
        common_eps = sorted(set(baseline_df.index) & set(cdf.index))
        if not common_eps:
            print(f"  {cname}: No common epochs with R12_A")
            continue
        
        desc = RUNS[cname]["desc"]
        print(f"  {cname} ({desc})")
        print(f"  {'Ep':>4s}", end="")
        for m in delta_metrics:
            label = m[:14]
            print(f"  {label:>14s}", end="")
        print()

        for ep in common_eps:
            print(f"  {ep:4d}", end="")
            for m in delta_metrics:
                if m not in baseline_df.columns or m not in cdf.columns:
                    print(f"  {'N/A':>14s}", end="")
                    continue
                bv = baseline_df.at[ep, m] if ep in baseline_df.index else None
                cv = cdf.at[ep, m] if ep in cdf.index else None
                try:
                    if bv is not None and cv is not None and not (pd.isna(bv) or pd.isna(cv)):
                        d = float(cv) - float(bv)
                        sign = "+" if d >= 0 else ""
                        if m in ["vcd_real", "wma_fake", "teams_real", "facedancer"]:
                            print(f"  {sign}{d*100:13.2f}pp", end="")
                        else:
                            print(f"  {sign}{d:14.4f}", end="")
                    else:
                        print(f"  {'N/A':>14s}", end="")
                except (TypeError, ValueError):
                    print(f"  {'N/A':>14s}", end="")
            print()
        print()

    # ── 5. Best composite per run ─────────────────────────────
    sep("4. BEST OOD COMPOSITE PER RUN (from history)")
    if "composite" in df.columns:
        valid = df.dropna(subset=["composite"])
        if not valid.empty:
            best_idx = valid.groupby("run")["composite"].idxmax()
            best = valid.loc[best_idx].sort_values("composite", ascending=False)
            display_cols = ["run", "epoch", "step", "holdout_auc", "ood_auc", "composite", "vcd_real", "facedancer", "wma_fake", "teams_real"]
            available = [c for c in display_cols if c in best.columns]
            print(best[available].to_string(index=False))

    # ── 6. Convergence trends ─────────────────────────────────
    sep("5. CONVERGENCE TRENDS (direction from epoch 3 → latest)")
    for name in col_order:
        rdf = df[df["run"] == name].sort_values("epoch")
        if len(rdf) < 2:
            print(f"  {name:8s}  (only 1 eval point)")
            continue
        ep3 = rdf[rdf["epoch"] == 3]
        latest = rdf.iloc[-1]
        first = ep3.iloc[0] if not ep3.empty else rdf.iloc[0]
        first_label = "ep3" if not ep3.empty else f"ep{int(first['epoch'])}"

        pieces = [f"  {name:8s}  {first_label}→ep{int(latest['epoch'])}:"]
        for m, label in [("ood_auc", "OOD"), ("composite", "Comp"), ("holdout_auc", "H.AUC"), ("vcd_real", "VCD")]:
            if m in rdf.columns:
                v0 = first.get(m)
                v1 = latest.get(m)
                if v0 is not None and v1 is not None and not (pd.isna(v0) or pd.isna(v1)):
                    d = v1 - v0
                    arrow = "↑" if d >= 0 else "↓"
                    if m in ["vcd_real"]:
                        pieces.append(f"{label} {arrow}{d*100:+.1f}pp")
                    else:
                        pieces.append(f"{label} {arrow}{d:+.4f}")
        print("  ".join(pieces))

    # ── 7. Key questions for R13 ──────────────────────────────
    sep("6. KEY QUESTIONS ANSWERED (for R13)")

    r12a_hist = df[df["run"] == "R12_A"]
    r125a_hist = df[df["run"] == "R125_A"]
    r125b_hist = df[df["run"] == "R125_B"]
    r125c_hist = df[df["run"] == "R125_C"]

    print("\n  Q1: Does GammaUp alone (R125_A) improve VCD Real? (R12_A best was 83.9%)")
    if not r125a_hist.empty:
        best_vcd_125a = r125a_hist["vcd_real"].max()
        best_ep = r125a_hist.loc[r125a_hist["vcd_real"].idxmax(), "epoch"]
        print(f"      R125_A best VCD Real = {best_vcd_125a:.1%} at epoch {best_ep}")
    if not r12a_hist.empty:
        best_vcd_12a = r12a_hist["vcd_real"].max()
        best_ep_12a = r12a_hist.loc[r12a_hist["vcd_real"].idxmax(), "epoch"]
        print(f"      R12_A  best VCD Real = {best_vcd_12a:.1%} at epoch {best_ep_12a}")
        if not r125a_hist.empty:
            delta = best_vcd_125a - best_vcd_12a
            print(f"      → Delta: {delta*100:+.1f}pp  {'YES, improved' if delta > 0.01 else 'NO / marginal' if delta > -0.01 else 'WORSE'}")

    # helper for safe formatting
    def _f(v, pct=False):
        if v is None: return "N/A"
        return f"{v:.1%}" if pct else f"{v:.4f}"

    print("\n  Q2: Does DirectionalShadow add value on top of GammaUp?")
    if not r125a_hist.empty and not r125b_hist.empty:
        # Compare at latest common epoch
        common = sorted(set(r125a_hist["epoch"]) & set(r125b_hist["epoch"]))
        if common:
            ep = max(common)
            a_ood = r125a_hist[r125a_hist["epoch"] == ep]["ood_auc"].values[0]
            b_ood = r125b_hist[r125b_hist["epoch"] == ep]["ood_auc"].values[0]
            a_vcd = r125a_hist[r125a_hist["epoch"] == ep]["vcd_real"].values[0]
            b_vcd = r125b_hist[r125b_hist["epoch"] == ep]["vcd_real"].values[0]
            print(f"      At epoch {ep}:")
            print(f"      R125_A (GammaUp only):     OOD_AUC={_f(a_ood)}  VCD={_f(a_vcd, True)}")
            print(f"      R125_B (+Shadow):           OOD_AUC={_f(b_ood)}  VCD={_f(b_vcd, True)}")
            if a_ood and b_ood:
                print(f"      → Shadow delta: OOD {b_ood - a_ood:+.4f}   VCD {((b_vcd or 0) - (a_vcd or 0))*100:+.1f}pp")

    print("\n  Q3: Does max lighting envelope (R125_C) hit an OOD ceiling?")
    if not r125c_hist.empty:
        latest_c = r125c_hist.iloc[-1]
        c_ood = latest_c.get('ood_auc')
        c_vcd = latest_c.get('vcd_real')
        print(f"      R125_C latest: OOD_AUC={_f(c_ood)}  VCD={_f(c_vcd, True)}")
        if not r12a_hist.empty:
            match_ep = int(latest_c["epoch"])
            r12a_at = r12a_hist[r12a_hist["epoch"] == match_ep]
            if not r12a_at.empty:
                r12a_at = r12a_at.iloc[0]
                a_ood_q3 = r12a_at.get('ood_auc')
                a_vcd_q3 = r12a_at.get('vcd_real')
                print(f"      R12_A  at ep{match_ep}:  OOD_AUC={_f(a_ood_q3)}  VCD={_f(a_vcd_q3, True)}")
                if c_ood and a_ood_q3:
                    d = c_ood - a_ood_q3
                    print(f"      → Max envelope delta: OOD {d:+.4f}  VCD {((c_vcd or 0) - (a_vcd_q3 or 0))*100:+.1f}pp  {'HELPING' if d > 0.002 else 'HURTING' if d < -0.002 else 'NEUTRAL'}")

    print("\n  Q4: R12_G vs R12_A (same seed=737, compound aug vs base aug)")
    r12g_hist = df[df["run"] == "R12_G"]
    if not r12g_hist.empty and not r12a_hist.empty:
        # Compare at latest common epoch using OOD AUC (more reliable than composite in sampled history)
        g_latest = r12g_hist.iloc[-1]
        a_latest = r12a_hist.iloc[-1]
        print(f"      R12_G latest (ep{int(g_latest['epoch'])}): OOD={_f(g_latest.get('ood_auc'))}  VCD={_f(g_latest.get('vcd_real'), True)}")
        print(f"      R12_A latest (ep{int(a_latest['epoch'])}): OOD={_f(a_latest.get('ood_auc'))}  VCD={_f(a_latest.get('vcd_real'), True)}")

    # ── 8. Summary table ──────────────────────────────────────
    sep("7. OVERALL RANKING (latest epoch, all 6 runs)")
    latest = df.loc[df.groupby("run")["epoch"].idxmax()].sort_values("ood_auc", ascending=False)
    rank_cols = ["run", "epoch", "step", "holdout_auc", "ood_auc", "vcd_real", "wma_fake", "teams_real", "teams_fake"]
    available = [c for c in rank_cols if c in latest.columns]
    print(latest[available].to_string(index=False))

    # Also show from summary (best composite checkpoints)
    print(f"\n  Best Composite (from summary, not sampled history):")
    comp_data = []
    for name in RUNS:
        s = summaries.get(name, {})
        bc = s.get("best_ood_composite/metric")
        bo = s.get("best_ood_composite/ood_auc")
        bh = s.get("best_ood_composite/holdout_auc")
        bs = s.get("best_ood_composite/step")
        if bc:
            comp_data.append({"run": name, "composite": bc, "H.AUC": bh, "OOD": bo, "step": bs})
    if comp_data:
        cdf = pd.DataFrame(comp_data).sort_values("composite", ascending=False)
        print(cdf.to_string(index=False))

    print(f"\n{'='*100}")
    print("  Done. Use sections 3 & 6 to inform R13 experiment design.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
