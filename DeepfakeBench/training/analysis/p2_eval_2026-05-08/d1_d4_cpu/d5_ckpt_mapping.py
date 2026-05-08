#!/usr/bin/env python3
"""D5 — Verify canary `_step` <-> Slot D saved-ckpt mapping.

Scores 4 additional Slot D ckpts (periodic_step3000, top_n_step10500,
top_n_step14500, periodic_step8000) on the same 800-frame canary parquet,
plus reuses scores already computed for top_n_step6000 and top_n_step19000.

For each ckpt, compute the same headline stats the in-training canary logs
(score_p50_on_reals, score_p95_on_reals, lockbox_recall_at_FPR_10pct,
chronic_mean/<id>) and compare to the logged canary fire stats at each
`_step` in {3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000}.

If wandb `_step` == optimizer step, then a saved ckpt with step label `S`
should produce stats that match the canary fire at `_step=S`. If not, the
divergence quantifies how much a 1000-2000 step gap moves model state.

Output:
  outputs/d5_ckpt_mapping_summary.csv  — per-ckpt headline metrics
  outputs/d5_per_fire_match.csv        — closest ckpt for each canary fire
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import (CKPT_DIR, FRAME_CACHE, SCORES_DIR,
                           CANARY_PARQUET, load_detector, score_ckpt,
                           download_all_frames)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("d5_mapping")

THIS = Path(__file__).resolve().parent
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

NEW_CKPTS = [
    ("slotD_periodic_step3000",  CKPT_DIR / "slotD_periodic_step3000.pth"),
    ("slotD_top_n_step10500",    CKPT_DIR / "slotD_top_n_step10500.pth"),
    ("slotD_top_n_step14500",    CKPT_DIR / "slotD_top_n_step14500.pth"),
    ("slotD_periodic_step8000",  CKPT_DIR / "slotD_periodic_step8000.pth"),
]
EXISTING_CKPTS = [
    ("slotD_top_n_step6000",  CKPT_DIR / "slotD_top_n_step6000.pth", 6000),
    ("slotD_top_n_step19000", CKPT_DIR / "slotD_top_n_step19000.pth", 19000),
]
ALL_LABELLED = NEW_CKPTS + [(c[0], c[1]) for c in EXISTING_CKPTS]
STEP_LABELS = {
    "slotD_periodic_step3000": 3000, "slotD_top_n_step10500": 10500,
    "slotD_top_n_step14500": 14500, "slotD_periodic_step8000": 8000,
    "slotD_top_n_step6000": 6000, "slotD_top_n_step19000": 19000,
}

CHRONIC_IDS = ["PC_Generator__s22", "PC_Generator__s45", "Q__s6", "Roy_D",
               "bla_bla_chow", "bla_bla_chow__s2"]
LOCKBOX_FPR_TARGETS = (0.05, 0.10)


def per_ckpt_stats(scores: pd.DataFrame, meta: pd.DataFrame) -> dict:
    df = scores.merge(meta[["frame_idx", "frame_path", "label", "cohort", "base_identity"]],
                      on=["frame_idx", "frame_path"], how="inner")
    reals = df[df["label"] == 0]["prob_fake"].dropna()
    fakes = df[df["label"] == 1]["prob_fake"].dropna()
    out = {
        "n_frames_evaluated": int(df["prob_fake"].notna().sum()),
        "n_reals": int(reals.shape[0]),
        "n_fakes": int(fakes.shape[0]),
        "score_p50_on_reals": float(reals.median()),
        "score_p95_on_reals": float(reals.quantile(0.95)),
        "score_mean_on_reals": float(reals.mean()),
        "score_p50_on_fakes": float(fakes.median()),
        "score_p05_on_fakes": float(fakes.quantile(0.05)),
        "score_mean_on_fakes": float(fakes.mean()),
    }
    # lockbox-FPR-calibrated recall at FPR=10%/5%
    lockbox = df[df["suite"] == "teams_fake_all_lockbox"] if "suite" in df.columns else \
              df[df["cohort"] == "lockbox_fake"]
    if len(lockbox):
        for target in LOCKBOX_FPR_TARGETS:
            tau_at_fpr = float(reals.quantile(1 - target))
            recall = float((lockbox["prob_fake"] > tau_at_fpr).mean())
            out[f"lockbox_recall_at_FPR_{int(target*100)}pct"] = recall
            if target == 0.10:
                out["lockbox_tau_at_FPR_10pct"] = tau_at_fpr
    # max + mean per-identity mean score (over all base_identity values)
    by_id = df.groupby("base_identity")["prob_fake"].mean()
    out["max_per_identity_mean_score"] = float(by_id.max())
    out["mean_per_identity_mean_score"] = float(by_id.mean())
    # chronic_mean/<id>
    for cid in CHRONIC_IDS:
        sub = df[df["base_identity"] == cid]
        out[f"chronic_mean/{cid}"] = float(sub["prob_fake"].mean()) if len(sub) else float("nan")
    return out


def main():
    log.info("=" * 70)
    log.info("D5 — canary _step <-> ckpt mapping verification")
    log.info("=" * 70)
    df_meta = pd.read_parquet(CANARY_PARQUET)
    df_meta = df_meta.rename(columns={"label": "label", "suite": "suite"})  # passthrough
    log.info("canary parquet: %d rows", len(df_meta))
    frame_paths = download_all_frames(df_meta)
    keep_meta = df_meta[["frame_idx", "frame_path", "label", "cohort", "base_identity",
                         "suite", "p8a_reference_score"]].copy()
    # Run inference for missing ckpts
    device = torch.device("cpu")
    for ckpt_id, path in NEW_CKPTS:
        if not path.exists():
            log.warning("missing %s -- skip", path); continue
        out_csv = SCORES_DIR / f"{ckpt_id}.csv"
        if out_csv.exists():
            log.info("[%s] skip -- output exists", ckpt_id); continue
        log.info("[%s] loading ...", ckpt_id)
        t0 = time.time()
        m = load_detector(path, device)
        log.info("[%s] loaded in %.1fs", ckpt_id, time.time() - t0)
        df_scores = score_ckpt(m, df_meta, frame_paths, device, batch_size=32)
        df_scores.to_csv(out_csv, index=False)
        log.info("[%s] -> %s", ckpt_id, out_csv)
        del m
    # Compute per-ckpt stats
    rows = []
    for ckpt_id, _ in ALL_LABELLED:
        csv_path = SCORES_DIR / f"{ckpt_id}.csv"
        if not csv_path.exists():
            log.warning("missing scores: %s -- skip", csv_path); continue
        scores = pd.read_csv(csv_path)
        stats = per_ckpt_stats(scores, keep_meta)
        stats["ckpt_id"] = ckpt_id
        stats["ckpt_step_label"] = STEP_LABELS[ckpt_id]
        rows.append(stats)
    out_df = pd.DataFrame(rows)
    cols = ["ckpt_id", "ckpt_step_label"] + [c for c in out_df.columns if c not in
                                              ("ckpt_id", "ckpt_step_label")]
    out_df = out_df[cols]
    summ_csv = OUT / "d5_ckpt_mapping_summary.csv"
    out_df.to_csv(summ_csv, index=False)
    log.info("wrote %s", summ_csv)

    # Pull canary logged stats and compute closest-ckpt per fire
    canary_csv = THIS.parent / "slot_D_canary_history.csv"
    cdf = pd.read_csv(canary_csv)
    cdf = cdf[cdf.filter(like="canary/").notna().any(axis=1)].copy()
    cdf = cdf.groupby("_step").first().reset_index().sort_values("_step")
    log.info("canary fires: %s", cdf["_step"].tolist())
    # Stats columns to compare (exact same names as the per_ckpt_stats output keys)
    cmp_keys = [k for k in out_df.columns if k.startswith("score_") or k.startswith("chronic_mean/")
                or k.startswith("lockbox_") or k.startswith("max_per_identity")
                or k.startswith("mean_per_identity")]
    # For each canary fire, compute Euclidean distance to each ckpt across cmp_keys
    fire_rows = []
    for _, fire in cdf.iterrows():
        row = {"canary_step": int(fire["_step"])}
        # Map canary key names to our column names
        canary_vals = {}
        for k in cmp_keys:
            if k.startswith("chronic_mean/"):
                ck = "canary/" + k
            elif k == "max_per_identity_mean_score":
                ck = "canary/max_per_identity_mean_score"
            elif k == "mean_per_identity_mean_score":
                ck = "canary/mean_per_identity_mean_score"
            else:
                ck = "canary/" + k
            if ck in fire.index and not pd.isna(fire[ck]):
                canary_vals[k] = float(fire[ck])
        row["n_keys_matched"] = len(canary_vals)
        # distance to each ckpt
        best_ckpt = None
        best_dist = float("inf")
        for _, ckpt in out_df.iterrows():
            diffs = []
            for k, v in canary_vals.items():
                if k in ckpt.index and not pd.isna(ckpt[k]):
                    diffs.append((float(ckpt[k]) - v) ** 2)
            d = float(np.sqrt(sum(diffs))) if diffs else float("inf")
            row[f"L2_to_{ckpt['ckpt_id']}"] = d
            if d < best_dist:
                best_dist, best_ckpt = d, ckpt["ckpt_id"]
        row["closest_ckpt"] = best_ckpt
        row["closest_L2"] = best_dist
        fire_rows.append(row)
    fire_df = pd.DataFrame(fire_rows)
    fire_csv = OUT / "d5_per_fire_match.csv"
    fire_df.to_csv(fire_csv, index=False)
    log.info("wrote %s", fire_csv)
    print()
    print("=== Per-canary-fire closest ckpt (L2 over %d shared metrics) ===" % len(canary_vals))
    show_cols = ["canary_step", "closest_ckpt", "closest_L2"] + \
                [c for c in fire_df.columns if c.startswith("L2_to_")]
    print(fire_df[show_cols].round(4).to_string(index=False))
    print()
    print("=== Per-ckpt headline stats (compare to canary fires) ===")
    show = ["ckpt_id", "ckpt_step_label", "score_p50_on_reals", "score_p95_on_reals",
            "lockbox_recall_at_FPR_10pct", "max_per_identity_mean_score"] + \
           [f"chronic_mean/{c}" for c in CHRONIC_IDS]
    print(out_df[[c for c in show if c in out_df.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
