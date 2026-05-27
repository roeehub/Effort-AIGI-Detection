"""Job alpha: chronic-6 ↔ capture-mode IoU + capture-mode-gate simulation.

Question: is the F1 unlock (drop chronic-6 identities → ~10x FPR drop) the same
finding as the existing webcam/screen-mode policy, or is it data-snooping?

Inputs:
- per-frame reports for P8A/E2B/E3 on teams_real_all_dev (raw_reports/)
- full_tags_2026-04-27.parquet (per-frame clip_capture_mode tags)
- chronic_offender_comparison.csv (chronic-6 identity list)

Outputs:
- chronic6_capture_mode_overlap.csv: identity-level overlap stats
- per_capture_mode_fpr.csv: FPR by (ckpt, capture_mode)
- capture_mode_gate_simulation.csv: F1-style simulation under capture-mode gate
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
CHRONIC = f"{ROOT}/analysis/job_11_identity_audit_2026-05-04/chronic_offender_comparison.csv"
OUT = f"{ROOT}/analysis/chronic6_capture_mode_iou_2026-05-05"

os.makedirs(OUT, exist_ok=True)

CKPT_FILES = {
    "P8A":      f"{RAW}/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/teams_real_all_dev_e3_top_n_step6600_frames_report.csv",
}

# tau values from f1_recall_2026-05-04 (F0 calibrated at FPR=10%)
TAU_F0 = {"P8A": 0.70503, "E2B_3200": 0.50644, "E3_6600": 0.85196}

# ----------------------------------------------------------------------
# 1. Load per-frame reports + tags, join on frame_path
# ----------------------------------------------------------------------
print("Loading per-frame reports...")
ckpt_dfs = {}
for ck, path in CKPT_FILES.items():
    df = pd.read_csv(path)
    # frame_path is full GCS URI
    df = df.rename(columns={"frame_prob": f"score_{ck}"})
    ckpt_dfs[ck] = df[["frame_path", "label", "video_id", "group_key", "family_key", f"score_{ck}"]]

merged = ckpt_dfs["P8A"]
for ck in ["E2B_3200", "E3_6600"]:
    merged = merged.merge(
        ckpt_dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner"
    )
print(f"  joined: {merged.shape[0]} frames across 3 ckpts")

# Extract identity from video_id (format: <identity_key>__s<session>__<...>)
def extract_identity(video_id: str) -> str:
    if not isinstance(video_id, str):
        return "UNKNOWN"
    # strip __session__type tail; teams uses identity_key like "PC_Generator__s22"
    # video_id pattern: identity__sNN__... or identity__...
    parts = video_id.split("__")
    if len(parts) >= 2 and parts[1].startswith("s") and parts[1][1:].isdigit():
        return f"{parts[0]}__{parts[1]}"
    return parts[0]

merged["identity"] = merged["video_id"].apply(extract_identity)
print(f"  unique identities: {merged['identity'].nunique()}")

# ----------------------------------------------------------------------
# 2. Load tags + join on frame_path (= gcs_uri)
# ----------------------------------------------------------------------
print("\nLoading tags...")
tags = pd.read_parquet(TAGS)[["gcs_uri", "clip_capture_mode", "clip_capture_mode_prob",
                              "is_likely_screen_capture", "sharpness_laplacian",
                              "face_pixel_area", "is_no_face"]]
tags = tags.rename(columns={"gcs_uri": "frame_path"})
print(f"  tags: {tags.shape[0]} frames, {tags['clip_capture_mode'].nunique()} capture modes")

merged = merged.merge(tags, on="frame_path", how="left")
covered = merged["clip_capture_mode"].notna().sum()
print(f"  capture-mode coverage on dev reals: {covered}/{merged.shape[0]} = {covered/merged.shape[0]:.1%}")

# ----------------------------------------------------------------------
# 3. Load chronic-6 list
# ----------------------------------------------------------------------
chronic_df = pd.read_csv(CHRONIC)
chronic_6 = set(chronic_df.loc[chronic_df["is_chronic_6"], "identity"].tolist())
print(f"\nChronic-6: {sorted(chronic_6)}")

# ----------------------------------------------------------------------
# 4. Per-identity dominant capture mode (only on covered frames)
# ----------------------------------------------------------------------
covered_df = merged[merged["clip_capture_mode"].notna()].copy()
identity_mode = (
    covered_df.groupby(["identity", "clip_capture_mode"])
    .size()
    .reset_index(name="n")
)
total_per_id = covered_df.groupby("identity").size().reset_index(name="n_total")
identity_mode = identity_mode.merge(total_per_id, on="identity")
identity_mode["share"] = identity_mode["n"] / identity_mode["n_total"]

# Dominant mode per identity
dom = identity_mode.sort_values(["identity", "share"], ascending=[True, False]).groupby("identity").head(1)
dom = dom.rename(columns={"clip_capture_mode": "dominant_capture_mode", "share": "dominant_share"})
dom = dom[["identity", "n_total", "dominant_capture_mode", "dominant_share"]]
print(f"\nIdentities with capture-mode coverage: {dom.shape[0]}")
print("Top 15 by frame count:")
print(dom.sort_values("n_total", ascending=False).head(15).to_string(index=False))

# ----------------------------------------------------------------------
# 5. Define webcam/screen cohort
# ----------------------------------------------------------------------
SCREEN_LIKE = {"webcam", "screen", "phone_screen", "screen_recording"}
webcam_screen_ids = set(dom.loc[dom["dominant_capture_mode"].isin(SCREEN_LIKE), "identity"])

print(f"\nWebcam/screen cohort: {len(webcam_screen_ids)} identities")
print(f"  examples: {sorted(webcam_screen_ids)[:15]}")

iou_chronic_vs_screen = (
    len(chronic_6 & webcam_screen_ids) /
    max(1, len(chronic_6 | webcam_screen_ids))
)

# Coverage check: chronic-6 identities WITH capture-mode tags
chronic_with_tags = chronic_6 & set(dom["identity"])
print(f"\nChronic-6 with capture-mode tags: {len(chronic_with_tags)}/{len(chronic_6)}")
chronic_without_tags = chronic_6 - set(dom["identity"])
print(f"  chronic-6 missing tags: {sorted(chronic_without_tags)}")

# IoU on the covered subset
chronic_6_covered = chronic_6 & set(dom["identity"])
iou_covered = (
    len(chronic_6_covered & webcam_screen_ids) /
    max(1, len(chronic_6_covered | webcam_screen_ids))
)

# Coverage / containment
chronic_in_screen = chronic_6_covered & webcam_screen_ids
print(f"\nChronic-6 (covered) ⊆ webcam/screen check:")
print(f"  chronic-6 (covered): {sorted(chronic_6_covered)}")
print(f"  in webcam/screen cohort: {sorted(chronic_in_screen)}")
print(f"  fraction of chronic-6 (covered) in screen cohort: {len(chronic_in_screen)}/{len(chronic_6_covered)}")

# ----------------------------------------------------------------------
# 6. Per-capture-mode FPR per ckpt (on covered frames)
# ----------------------------------------------------------------------
rows = []
for ck in CKPT_FILES:
    tau = TAU_F0[ck]
    for mode in ["normal_photo", "webcam", "phone_screen", "screen", "screen_recording", "ALL_COVERED", "F0_ALL"]:
        if mode == "ALL_COVERED":
            sub = merged[merged["clip_capture_mode"].notna()]
        elif mode == "F0_ALL":
            sub = merged
        else:
            sub = merged[merged["clip_capture_mode"] == mode]
        if len(sub) == 0:
            continue
        scores = sub[f"score_{ck}"]
        fp = (scores >= tau).sum()
        rows.append({
            "ckpt": ck,
            "capture_mode": mode,
            "n_frames": len(sub),
            "n_fp": int(fp),
            "fpr_pct": 100.0 * fp / len(sub),
            "tau": tau,
            "mean_score": float(scores.mean()),
            "p90_score": float(scores.quantile(0.90)),
        })
per_mode = pd.DataFrame(rows)
print("\nPer-capture-mode FPR @ F0 tau:")
print(per_mode.to_string(index=False))

# ----------------------------------------------------------------------
# 7. F1-style simulation under capture-mode gate
#    Build "F_capture" = drop frames whose dominant identity capture_mode in screen/webcam set
#    Recalibrate tau at FPR=10% on remaining reals; report tau, mean, p90 etc.
# ----------------------------------------------------------------------

# Per-frame capture-mode-based filter: drop frames where clip_capture_mode in SCREEN_LIKE
# (more aggressive than identity-level: we drop the screen-like FRAMES of any identity)
# Two variants:
#   (a) frame-mode: drop frames where clip_capture_mode ∈ SCREEN_LIKE
#   (b) identity-mode: drop frames belonging to identities whose dominant_capture_mode ∈ SCREEN_LIKE

merged_with_dom = merged.merge(dom[["identity", "dominant_capture_mode"]], on="identity", how="left")

filters = {
    "F0_all_covered": merged[merged["clip_capture_mode"].notna()].copy(),
    "F_capture_frame": merged.loc[
        merged["clip_capture_mode"].notna() & ~merged["clip_capture_mode"].isin(SCREEN_LIKE)
    ].copy(),
    "F_capture_identity": merged_with_dom.loc[
        merged_with_dom["clip_capture_mode"].notna() &
        ~merged_with_dom["dominant_capture_mode"].isin(SCREEN_LIKE)
    ].copy(),
    "F1_chronic6": merged[~merged["identity"].isin(chronic_6)].copy(),
}

sim_rows = []
for ck in CKPT_FILES:
    score_col = f"score_{ck}"
    for fname, fdf in filters.items():
        if len(fdf) == 0:
            continue
        scores = fdf[score_col].sort_values()
        # tau at FPR=10% (top 10% scores marked as positive)
        tau_at_10 = float(scores.quantile(0.90))
        fp_at_tau = (fdf[score_col] >= tau_at_10).sum()
        sim_rows.append({
            "ckpt": ck,
            "filter": fname,
            "n_reals": len(fdf),
            "tau_at_fpr10": tau_at_10,
            "fpr_at_tau_pct": 100.0 * fp_at_tau / len(fdf),
            "real_p50": float(fdf[score_col].quantile(0.50)),
            "real_p90": float(fdf[score_col].quantile(0.90)),
        })
sim_df = pd.DataFrame(sim_rows)
print("\nFilter simulations (real reals only — tau recalibrated):")
print(sim_df.to_string(index=False))

# ----------------------------------------------------------------------
# 8. Apply each filter's tau to each fake suite — viso recall
# ----------------------------------------------------------------------
# Need viso fake scores for each ckpt
VISO_FAKE_FILES = {
    "P8A":      f"{RAW}/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
}

viso_dfs = {}
for ck, path in VISO_FAKE_FILES.items():
    d = pd.read_csv(path)
    viso_dfs[ck] = d[["frame_path", "label", f"frame_prob"]].rename(columns={"frame_prob": f"score_{ck}"})

viso_merged = viso_dfs["P8A"]
for ck in ["E2B_3200", "E3_6600"]:
    viso_merged = viso_merged.merge(viso_dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")

print(f"\nViso fakes joined across 3 ckpts: {viso_merged.shape[0]} frames")

# Recall under each (ckpt, filter)'s tau
recall_rows = []
for ck in CKPT_FILES:
    score_col = f"score_{ck}"
    for fname in ["F0_all_covered", "F_capture_frame", "F_capture_identity", "F1_chronic6"]:
        # find tau from sim_df
        row = sim_df[(sim_df["ckpt"] == ck) & (sim_df["filter"] == fname)]
        if row.empty:
            continue
        tau = float(row["tau_at_fpr10"].iloc[0])
        recall = (viso_merged[score_col] >= tau).mean()
        recall_rows.append({
            "ckpt": ck,
            "filter": fname,
            "tau_dev_cal": tau,
            "viso_recall_pct": 100.0 * recall,
            "n_viso_fakes": len(viso_merged),
        })
recall_df = pd.DataFrame(recall_rows)
print("\nViso recall under each filter's tau:")
print(recall_df.to_string(index=False))

# ----------------------------------------------------------------------
# 9. Summary IoU + verdict numbers
# ----------------------------------------------------------------------
summary = {
    "n_chronic6_total": len(chronic_6),
    "n_chronic6_covered": len(chronic_6_covered),
    "n_chronic6_in_screen_cohort": len(chronic_in_screen),
    "frac_chronic6_covered_in_screen_cohort": len(chronic_in_screen) / max(1, len(chronic_6_covered)),
    "iou_full_chronic6_vs_screen": iou_chronic_vs_screen,
    "iou_chronic6_covered_vs_screen": iou_covered,
    "n_screen_cohort_identities": len(webcam_screen_ids),
}
import json
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary:\n{json.dumps(summary, indent=2)}")

# Save outputs
dom.to_csv(f"{OUT}/per_identity_dominant_capture_mode.csv", index=False)
per_mode.to_csv(f"{OUT}/per_capture_mode_fpr.csv", index=False)
sim_df.to_csv(f"{OUT}/capture_mode_gate_simulation.csv", index=False)
recall_df.to_csv(f"{OUT}/capture_mode_gate_viso_recall.csv", index=False)

# Identity-level overlap details
overlap_rows = []
for ident in sorted(chronic_6 | webcam_screen_ids):
    overlap_rows.append({
        "identity": ident,
        "is_chronic6": ident in chronic_6,
        "is_webcam_screen_cohort": ident in webcam_screen_ids,
        "in_both": ident in chronic_6 and ident in webcam_screen_ids,
    })
overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(f"{OUT}/chronic6_capture_mode_overlap.csv", index=False)

print(f"\nDone. Outputs in {OUT}")
