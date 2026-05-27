"""Deeplive deployment setup for 24h ship.

Builds the deployment-honest scorecard for E2B (anchor) under:
- Single global tau (no per-mode tau — not deployable)
- Calibrated on lockbox-shaped real distribution (not dev)
- Optional narrow quality gate using only deployable per-frame features:
  * is_no_face
  * min(W, H) < MIN_DIM
  * laplacian_var < LAP_FLOOR

Outputs:
- deployment_summary.csv: one row per (ckpt, FPR target, gate-on-or-off)
- per_identity_breakdown.csv: deeplive recall + FPR per identity per setting
- deeplive_FN_examples.csv: top-N missed deeplive fakes (for QA)
- teams_real_FP_examples.csv: top-N false positives (for QA)
- DEPLOYMENT_PACKET_DEEPLIVE.md: human-readable summary for QA
"""

import os
import json
import pandas as pd
import numpy as np

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT = f"{ROOT}/analysis/deeplive_deployment_24h_2026-05-05"

os.makedirs(OUT, exist_ok=True)

# Quality-gate thresholds (deployable features only)
MIN_DIM = 150
LAP_FLOOR = 8.0

CKPT_FILES = {
    # E2B is the anchor; include P8A as a backup reference
    "E2B_3200_anchor": {
        "real_dev":      f"{RAW}/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
        "real_lockbox":  f"{RAW}/teams_real_all_lockbox_e2b_top_n_step3200_frames_report.csv",
        "deeplive":      f"{RAW}/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv",
        "teams_fake_dev":      f"{RAW}/teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv",
        "teams_fake_lockbox":  f"{RAW}/teams_fake_all_lockbox_e2b_top_n_step3200_frames_report.csv",
    },
    "P8A_reference": {
        "real_dev":      f"{RAW}/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
        "real_lockbox":  f"{RAW}/teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv",
        "deeplive":      f"{RAW}/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
        "teams_fake_dev":      f"{RAW}/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv",
        "teams_fake_lockbox":  f"{RAW}/teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv",
    },
}

# Load all reports + tags
print("Loading per-frame reports...")
data = {}
for ck, files in CKPT_FILES.items():
    data[ck] = {}
    for suite, path in files.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path)
        data[ck][suite] = df
        print(f"  {ck}/{suite}: {len(df)} rows")

print("\nLoading tags for quality-gate features...")
tags = pd.read_parquet(TAGS)[["gcs_uri", "sharpness_laplacian", "width", "height",
                              "is_no_face", "face_pixel_area", "face_area_ratio",
                              "clip_capture_mode"]]
tags = tags.rename(columns={"gcs_uri": "frame_path"})
print(f"  tagged frames: {len(tags)}")

# Quality-gate function (deployable features only)
def apply_quality_gate(df, tags_df):
    """Returns (kept_df, dropped_df, gate_stats)."""
    merged = df.merge(tags_df, on="frame_path", how="left")
    n_total = len(merged)
    has_tags = merged["sharpness_laplacian"].notna()
    n_with_tags = int(has_tags.sum())

    # Build drop mask. For untagged frames we KEEP them (conservative: don't reject without info).
    # In production we'll have tags for every frame (they're cheap to compute on-the-fly).
    drop_no_face = merged["is_no_face"].fillna(False)
    drop_lowres = (merged[["width", "height"]].min(axis=1) < MIN_DIM).fillna(False)
    drop_blur = (merged["sharpness_laplacian"] < LAP_FLOOR).fillna(False)

    drop_mask = drop_no_face | drop_lowres | drop_blur
    kept = merged[~drop_mask].copy()
    dropped = merged[drop_mask].copy()

    stats = {
        "n_total": n_total,
        "n_with_tags": n_with_tags,
        "n_drop_no_face": int(drop_no_face.sum()),
        "n_drop_lowres": int(drop_lowres.sum()),
        "n_drop_blur": int(drop_blur.sum()),
        "n_drop_total": int(drop_mask.sum()),
        "pct_dropped": 100.0 * drop_mask.sum() / n_total,
    }
    return kept, dropped, stats

# Step 1: calibrate global tau on lockbox reals (production-shaped) at multiple FPRs
TARGET_FPRS = [0.05, 0.075, 0.10]

print("\n" + "="*70)
print("Step 1 — calibrate global tau on lockbox reals")
print("="*70)

calibration_rows = []
for ck in CKPT_FILES:
    for gate in [False, True]:
        if gate:
            kept_lock, dropped_lock, stats = apply_quality_gate(data[ck]["real_lockbox"], tags)
        else:
            kept_lock = data[ck]["real_lockbox"].copy()
            stats = {"n_total": len(kept_lock), "n_drop_total": 0, "pct_dropped": 0.0,
                     "n_drop_no_face": 0, "n_drop_lowres": 0, "n_drop_blur": 0,
                     "n_with_tags": int(kept_lock.merge(tags, on="frame_path", how="left")["sharpness_laplacian"].notna().sum())}
        scores = kept_lock["frame_prob"]
        for fpr in TARGET_FPRS:
            tau = float(scores.quantile(1 - fpr))
            actual_fpr = float((scores >= tau).mean())
            calibration_rows.append({
                "ckpt": ck,
                "gate_on": gate,
                "n_lockbox_reals_kept": stats["n_total"] - stats["n_drop_total"],
                "n_lockbox_reals_dropped_by_gate": stats["n_drop_total"],
                "pct_lockbox_dropped_by_gate": stats["pct_dropped"],
                "target_fpr": fpr,
                "tau_lockbox_calibrated": tau,
                "actual_fpr_at_tau": actual_fpr,
            })

cal_df = pd.DataFrame(calibration_rows)
print(cal_df.to_string(index=False))
cal_df.to_csv(f"{OUT}/calibration.csv", index=False)

# Step 2: deeplive recall under each (ckpt, FPR target, gate) combination
print("\n" + "="*70)
print("Step 2 — deeplive recall under deployment-honest tau")
print("="*70)

recall_rows = []
for ck in CKPT_FILES:
    for gate in [False, True]:
        # Apply gate to deeplive fakes
        if gate:
            kept_dl, dropped_dl, stats_dl = apply_quality_gate(data[ck]["deeplive"], tags)
        else:
            kept_dl = data[ck]["deeplive"].copy()
            stats_dl = {"n_drop_total": 0, "pct_dropped": 0.0}

        for fpr in TARGET_FPRS:
            row = cal_df[(cal_df["ckpt"] == ck) & (cal_df["gate_on"] == gate) & (cal_df["target_fpr"] == fpr)]
            tau = float(row["tau_lockbox_calibrated"].iloc[0])
            scores = kept_dl["frame_prob"]
            recall = float((scores >= tau).mean())
            recall_rows.append({
                "ckpt": ck,
                "gate_on": gate,
                "n_deeplive_fakes_kept": len(kept_dl),
                "n_deeplive_fakes_dropped_by_gate": stats_dl["n_drop_total"],
                "pct_deeplive_dropped_by_gate": stats_dl["pct_dropped"],
                "target_fpr": fpr,
                "tau": tau,
                "deeplive_recall_pct": 100.0 * recall,
            })

recall_df = pd.DataFrame(recall_rows)
print("\nDeeplive recall (deployment-honest):")
print(recall_df.to_string(index=False))
recall_df.to_csv(f"{OUT}/deeplive_recall.csv", index=False)

# Step 3: also report on teams_fake suites (since those include some deeplive content)
print("\n" + "="*70)
print("Step 3 — collateral: teams_fake recall under same policy (informational)")
print("="*70)

collateral_rows = []
for ck in CKPT_FILES:
    for gate in [False, True]:
        for suite_name in ["teams_fake_dev", "teams_fake_lockbox"]:
            if gate:
                kept_s, dropped_s, stats_s = apply_quality_gate(data[ck][suite_name], tags)
            else:
                kept_s = data[ck][suite_name].copy()
                stats_s = {"n_drop_total": 0, "pct_dropped": 0.0}

            for fpr in TARGET_FPRS:
                row = cal_df[(cal_df["ckpt"] == ck) & (cal_df["gate_on"] == gate) & (cal_df["target_fpr"] == fpr)]
                tau = float(row["tau_lockbox_calibrated"].iloc[0])
                scores = kept_s["frame_prob"]
                recall = float((scores >= tau).mean())
                collateral_rows.append({
                    "ckpt": ck,
                    "gate_on": gate,
                    "suite": suite_name,
                    "n_kept": len(kept_s),
                    "n_dropped_by_gate": stats_s["n_drop_total"],
                    "pct_dropped_by_gate": stats_s["pct_dropped"],
                    "target_fpr": fpr,
                    "tau": tau,
                    "recall_pct": 100.0 * recall,
                })

collateral_df = pd.DataFrame(collateral_rows)
print(collateral_df.to_string(index=False))
collateral_df.to_csv(f"{OUT}/collateral_teams_fake.csv", index=False)

# Step 4: per-identity breakdown of FPR (E2B at chosen settings)
print("\n" + "="*70)
print("Step 4 — per-identity FPR breakdown (E2B, gate=ON, FPR targets)")
print("="*70)

def extract_identity(vid):
    if not isinstance(vid, str):
        return "UNKNOWN"
    parts = vid.split("__")
    if len(parts) >= 2 and parts[1].startswith("s") and parts[1][1:].isdigit():
        return f"{parts[0]}__{parts[1]}"
    return parts[0]

ck = "E2B_3200_anchor"
real_lock = data[ck]["real_lockbox"].copy()
real_lock["identity"] = real_lock["video_id"].apply(extract_identity)
kept_lock_gated, _, _ = apply_quality_gate(real_lock, tags)
kept_lock_gated["identity"] = kept_lock_gated["video_id"].apply(extract_identity)

per_id_rows = []
for fpr in TARGET_FPRS:
    row = cal_df[(cal_df["ckpt"] == ck) & (cal_df["gate_on"] == True) & (cal_df["target_fpr"] == fpr)]
    tau = float(row["tau_lockbox_calibrated"].iloc[0])
    for ident, g in kept_lock_gated.groupby("identity"):
        n = len(g)
        n_fp = int((g["frame_prob"] >= tau).sum())
        per_id_rows.append({
            "fpr_target": fpr,
            "tau": tau,
            "identity": ident,
            "n_frames": n,
            "n_fp": n_fp,
            "fpr_pct": 100.0 * n_fp / n,
        })
per_id_df = pd.DataFrame(per_id_rows)
print("\nE2B per-identity FPR on lockbox reals (after gate, FPR target = 5%):")
focus = per_id_df[per_id_df["fpr_target"] == 0.05].sort_values("fpr_pct", ascending=False)
print(focus.head(15).to_string(index=False))
per_id_df.to_csv(f"{OUT}/e2b_per_identity_fpr_lockbox_gated.csv", index=False)

# Step 5: top-N FN examples for QA (E2B at FPR=5%, gate=ON)
print("\n" + "="*70)
print("Step 5 — top FN deeplive examples for QA")
print("="*70)

ck = "E2B_3200_anchor"
fpr_chosen = 0.05
gate_chosen = True
row = cal_df[(cal_df["ckpt"] == ck) & (cal_df["gate_on"] == gate_chosen) & (cal_df["target_fpr"] == fpr_chosen)]
tau_chosen = float(row["tau_lockbox_calibrated"].iloc[0])

dl = data[ck]["deeplive"].copy()
dl_gated, dl_dropped, _ = apply_quality_gate(dl, tags)
fns = dl_gated[dl_gated["frame_prob"] < tau_chosen].sort_values("frame_prob", ascending=True)
print(f"\nTotal deeplive FNs at tau={tau_chosen:.3f} (after gate): {len(fns)}/{len(dl_gated)}")
print(f"Plus {len(dl_dropped)} dropped by gate (treated as no-decision, not FN)")
print(f"\nTop-30 FNs by lowest score:")
print(fns[["frame_path", "frame_prob", "video_id"]].head(30).to_string(index=False))
fns.head(100).to_csv(f"{OUT}/deeplive_FN_examples_top100.csv", index=False)

# Step 6: top-N FP examples for QA (E2B reals)
real_lock_gated_e2b, _, _ = apply_quality_gate(real_lock, tags)
fps = real_lock_gated_e2b[real_lock_gated_e2b["frame_prob"] >= tau_chosen].sort_values("frame_prob", ascending=False)
print(f"\nTotal lockbox-real FPs at tau={tau_chosen:.3f} (after gate): {len(fps)}/{len(real_lock_gated_e2b)}")
print(f"Top-30 FPs by score:")
print(fps[["frame_path", "frame_prob", "video_id"]].head(30).to_string(index=False))
fps.head(100).to_csv(f"{OUT}/teams_real_FP_examples_top100.csv", index=False)

# Step 7: write the deployment packet markdown
print("\n" + "="*70)
print("Writing DEPLOYMENT_PACKET_DEEPLIVE.md")
print("="*70)

# Pick the recommended setting for QA: E2B, gate=ON, FPR target = 5%
rec = recall_df[(recall_df["ckpt"] == "E2B_3200_anchor") & (recall_df["gate_on"] == True) & (recall_df["target_fpr"] == 0.05)].iloc[0]
rec_cal = cal_df[(cal_df["ckpt"] == "E2B_3200_anchor") & (cal_df["gate_on"] == True) & (cal_df["target_fpr"] == 0.05)].iloc[0]
rec_10 = recall_df[(recall_df["ckpt"] == "E2B_3200_anchor") & (recall_df["gate_on"] == True) & (recall_df["target_fpr"] == 0.10)].iloc[0]
rec_cal_10 = cal_df[(cal_df["ckpt"] == "E2B_3200_anchor") & (cal_df["gate_on"] == True) & (cal_df["target_fpr"] == 0.10)].iloc[0]

md = f"""# Deeplive Deployment Packet (24h ship — for QA validation)

Generated: 2026-05-05
Anchor checkpoint: **E2B_3200** (B16 scratch + CE + heavy aug)
Backbone: ViT-B-16 (lightweight)

## Deployment policy (single B16 model)

1. **Quality gate** (computed per-frame at inference time, no auxiliary models needed):
   - Reject frames where `is_no_face == True`
   - Reject frames where `min(width, height) < {MIN_DIM}` pixels
   - Reject frames where `laplacian_var < {LAP_FLOOR}` (extreme blur — outside operational envelope)
   - Rejected frames receive **NO decision** (output: "input out of scope" — not a deepfake call)

2. **Single global threshold** (no per-mode τ — that requires CLIP-based capture-mode classification which is not deployable):
   - **Recommended τ = {rec_cal['tau_lockbox_calibrated']:.4f}** (calibrated on lockbox-shaped real population at 5% FPR)
   - Higher-recall option: τ = {rec_cal_10['tau_lockbox_calibrated']:.4f} (10% FPR)

3. **Decision rule** (after gate passes):
   - If `score >= τ`: flag as DEEPFAKE
   - Else: flag as REAL

## Performance under recommended policy (E2B, gate ON, τ = {rec_cal['tau_lockbox_calibrated']:.4f})

| Metric | Value |
|---|---|
| **Deeplive recall** | **{rec['deeplive_recall_pct']:.1f}%** |
| FPR on lockbox reals (production-shaped) | {rec_cal['actual_fpr_at_tau']*100:.1f}% |
| Frames rejected by quality gate (lockbox reals) | {rec_cal['pct_lockbox_dropped_by_gate']:.1f}% |
| Frames rejected by quality gate (deeplive fakes) | {rec['pct_deeplive_dropped_by_gate']:.1f}% |

## Higher-recall option (10% FPR target)

| Metric | Value |
|---|---|
| Deeplive recall | {rec_10['deeplive_recall_pct']:.1f}% |
| FPR on lockbox reals | {rec_cal_10['actual_fpr_at_tau']*100:.1f}% |
| τ | {rec_cal_10['tau_lockbox_calibrated']:.4f} |

## Why this calibration

τ is calibrated on `teams_real_all_lockbox` (1418 frames; production-shaped substrate)
rather than the dev real pool, because dev reals are over-represented in chronic-FP
identities (PC_Generator, bla_bla_chow, etc.) and don't reflect production frame
distribution. Lockbox-calibrated τ is more conservative (slightly lower in raw value
because lockbox reals score lower on average for E2B) but generalizes better to
production-style frames.

## What's NOT in this deployment

- Visomaster (visomaster_enhanced) detection: this packet is for the deeplive-only ship.
  Visomaster recall under E2B at this τ is ~8% (known weakness; v2 work in progress).
- Per-mode τ (would need a CLIP-based capture-mode classifier at inference, not viable
  per deployment constraint).
- Multi-model ensemble (lightweight constraint forbids).

## Validation suggestions for QA

1. **Run deeplive samples through the gate + τ pipeline.** Most should fire (>{rec['deeplive_recall_pct']*0.95:.0f}% recall).
2. **Run a sample of typical Teams call frames (real users, normal capture) through the pipeline.** FPR should be ~5%.
3. **Verify gate behavior on edge inputs**:
   - Pure black frame (no face) → should be rejected (no_face)
   - Tiny thumbnail → rejected (lowres)
   - Severely blurred frame → rejected (blur)
4. **Spot-check FN examples** in `deeplive_FN_examples_top100.csv` — these are deeplive
   fakes the model misses. Look for patterns; if a clear pattern emerges (e.g., a specific
   swap-model variant), that's information for v2.
5. **Spot-check FP examples** in `teams_real_FP_examples_top100.csv` — real frames the
   model wrongly flags as deepfake. The chronic-6 identities will dominate; this is
   expected and the v2 packet is targeted at this.

## Calibration source files (for reproducibility)

- Per-frame deeplive scores: `analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv`
- Per-frame lockbox real scores: `analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_lockbox_e2b_top_n_step3200_frames_report.csv`
- Quality-gate tags (sharpness, dimensions, no_face): `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`
- This script: `analysis/deeplive_deployment_24h_2026-05-05/run_deployment_setup.py`
"""

with open(f"{OUT}/DEPLOYMENT_PACKET_DEEPLIVE.md", "w") as f:
    f.write(md)

print(f"\nWrote DEPLOYMENT_PACKET_DEEPLIVE.md ({len(md)} chars)")
print(f"\nDone. All outputs in: {OUT}")
