"""
Phase E weight-delta — overlap extension for Phase A scorecard ckpts (2026-05-07).

Phase E originally ran on BUNDLE step500/1000/4000 + PAIRRANK step500/1000/6750.
Phase A's contract scorecard scores BUNDLE step500/3750/4000 + PAIRRANK
step500/6000/6750 per arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml.
Two ckpts (BUNDLE step3750, PAIRRANK step6000) are missing from the existing
weight-delta outputs.

This wrapper imports the existing logic from `compute_weight_delta.py`,
processes ONLY those two ckpts, and APPENDS rows to the existing
weight_delta_*.csv outputs (no overwrite, no recompute of the 6 existing rows).
After append, the verdict CSV is sorted by ckpt name for stable ordering.

Usage:
  cd analysis/p1_pe_eval_2026-05-07/weight_delta
  python compute_weight_delta_overlap.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Reuse the existing script's logic.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_weight_delta import (  # noqa: E402  (sys.path setup precedes import)
    BASELINE_URI,
    OUT_DIR,
    classify_layer,
    find_residual_bases,
    load_state_dict,
    reconstruct_residual,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("weight_delta_overlap")

# ---- the two ckpts missing from Phase E vs Phase A scorecard overlap --------

NEW_CKPTS: dict[str, str] = {
    "p1_bundle_step3750":   "gs://training-job-outputs/best_checkpoints/tznuar61/top_n_effort_20260506_step3750_auc0.9947_eer0.0168.pth",
    "p1_pairrank_step6000": "gs://training-job-outputs/best_checkpoints/s2mp5fxm/top_n_effort_20260507_step6000_auc0.9939_eer0.0095.pth",
}

PER_LAYER_CSV = OUT_DIR / "weight_delta_per_layer.csv"
BY_CAT_CSV = OUT_DIR / "weight_delta_by_category.csv"
VERDICT_CSV = OUT_DIR / "weight_delta_verdict.csv"


def main() -> None:
    # Sanity check: required existing CSVs must be present so we can append.
    for p in (PER_LAYER_CSV, BY_CAT_CSV, VERDICT_CSV):
        if not p.exists():
            log.error("expected existing CSV missing: %s", p)
            sys.exit(2)

    existing_verdict = pd.read_csv(VERDICT_CSV)
    existing_ckpts = set(existing_verdict["ckpt"].astype(str))
    log.info("existing verdict ckpts (%d): %s",
             len(existing_ckpts), sorted(existing_ckpts))

    todo = {k: v for k, v in NEW_CKPTS.items() if k not in existing_ckpts}
    if not todo:
        log.info("no new ckpts to process — both new ckpts already present")
        return
    log.info("will process %d new ckpts: %s", len(todo), sorted(todo))

    log.info("loading P8A baseline state_dict")
    p8a = load_state_dict(BASELINE_URI)
    bases = find_residual_bases(p8a)
    log.info("found %d SVD'd layers in P8A", len(bases))
    if not bases:
        log.error("P8A has no SVD residuals — wrong ckpt format?")
        sys.exit(2)

    new_per_layer_rows: list[dict] = []
    for ckpt_name, uri in todo.items():
        log.info("=== %s ===", ckpt_name)
        ck = load_state_dict(uri)
        ck_bases = set(find_residual_bases(ck))
        common = [b for b in bases if b in ck_bases]
        log.info("%d / %d layers in common with P8A", len(common), len(bases))

        for base in common:
            try:
                W_p8a = reconstruct_residual(p8a, base)
                W_ck = reconstruct_residual(ck, base)
                delta = (W_ck - W_p8a).norm().item()
                p8a_norm = W_p8a.norm().item()
                ck_norm = W_ck.norm().item()
                rel = delta / max(p8a_norm, 1e-9)
                new_per_layer_rows.append({
                    "ckpt": ckpt_name,
                    "layer": base,
                    "category": classify_layer(base),
                    "delta_fnorm": delta,
                    "p8a_residual_fnorm": p8a_norm,
                    "ckpt_residual_fnorm": ck_norm,
                    "relative_delta": rel,
                })
            except Exception as exc:  # noqa: BLE001
                log.warning("failed on %s: %s", base, exc)

    new_pl_df = pd.DataFrame(new_per_layer_rows)
    log.info("computed %d new per-layer rows", len(new_pl_df))

    # ---- per-layer CSV: append, schema-checked
    existing_pl = pd.read_csv(PER_LAYER_CSV)
    if list(existing_pl.columns) != list(new_pl_df.columns):
        log.error("per_layer column mismatch: existing=%s new=%s",
                  list(existing_pl.columns), list(new_pl_df.columns))
        sys.exit(2)
    combined_pl = pd.concat([existing_pl, new_pl_df], ignore_index=True)
    combined_pl.to_csv(PER_LAYER_CSV, index=False)
    log.info("wrote %s (%d rows total, +%d)",
             PER_LAYER_CSV, len(combined_pl), len(new_pl_df))

    # ---- by-category CSV: aggregate just the new ckpts, append
    new_agg = (new_pl_df.groupby(["ckpt", "category"])["delta_fnorm"]
               .agg(["count", "mean", "median", "max", "sum"])
               .reset_index())
    existing_agg = pd.read_csv(BY_CAT_CSV)
    if list(existing_agg.columns) != list(new_agg.columns):
        log.error("by_category column mismatch: existing=%s new=%s",
                  list(existing_agg.columns), list(new_agg.columns))
        sys.exit(2)
    combined_agg = pd.concat([existing_agg, new_agg], ignore_index=True)
    combined_agg = combined_agg.sort_values(["ckpt", "category"]).reset_index(drop=True)
    combined_agg.to_csv(BY_CAT_CSV, index=False)
    log.info("wrote %s (%d rows total, +%d)",
             BY_CAT_CSV, len(combined_agg), len(new_agg))

    # ---- verdict CSV: compute new rows, append, sort by ckpt name
    new_verdict_rows: list[dict] = []
    for ckpt_name in todo:
        sub = new_agg[new_agg["ckpt"] == ckpt_name].set_index("category")
        if "in_proj_qkv" not in sub.index or "out_proj" not in sub.index:
            log.warning("skipping verdict row for %s — missing in_proj_qkv or out_proj",
                        ckpt_name)
            continue
        qkv = sub.loc["in_proj_qkv", "mean"]
        op = sub.loc["out_proj", "mean"]
        mlp = sub.loc["mlp", "mean"] if "mlp" in sub.index else None
        new_verdict_rows.append({
            "ckpt": ckpt_name,
            "qkv_mean_fnorm": qkv,
            "out_proj_mean_fnorm": op,
            "mlp_mean_fnorm": mlp,
            "qkv_over_out_proj": qkv / max(op, 1e-9),
            "qkv_over_mlp": qkv / max(mlp, 1e-9) if mlp else None,
        })

    new_verdict_df = pd.DataFrame(new_verdict_rows)
    if list(existing_verdict.columns) != list(new_verdict_df.columns):
        log.error("verdict column mismatch: existing=%s new=%s",
                  list(existing_verdict.columns), list(new_verdict_df.columns))
        sys.exit(2)

    combined_verdict = pd.concat([existing_verdict, new_verdict_df], ignore_index=True)
    combined_verdict = combined_verdict.sort_values("ckpt").reset_index(drop=True)
    combined_verdict.to_csv(VERDICT_CSV, index=False)
    log.info("wrote %s (%d rows total, +%d)",
             VERDICT_CSV, len(combined_verdict), len(new_verdict_df))

    print()
    print("=" * 78)
    print(f"NEW VERDICT ROWS (+{len(new_verdict_df)}):")
    print("=" * 78)
    print(new_verdict_df.to_string(index=False))
    print()
    print("=" * 78)
    print(f"FULL VERDICT TABLE ({len(combined_verdict)} rows, sorted by ckpt):")
    print("=" * 78)
    print(combined_verdict.to_string(index=False))


if __name__ == "__main__":
    main()
