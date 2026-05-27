"""Chronic-6 fingerprint diagnostic.

Two questions:
1. Do chronic-6 share an in-frame property (other than identity / capture_mode
   itself) that distinguishes them from non-chronic webcam users?
   If yes → deployable per-frame gate without rejecting all webcam.
   If no → no gating-side fix; must be training-side.

2. What does deployment-realistic "per-mode tau calibration" buy us?
   Set tau separately per capture_mode at 10% FPR within mode → global FPR ≤ 10%
   AND every fake gets scored. Compare recall vs global tau.

Inputs:
- per-frame reports (P8A, E2B, E3) on teams_real_all_dev + viso/deeplive/teams_fake
- full_tags_2026-04-27.parquet (IQ + face + arcface tags)
- chronic_offender_comparison.csv (chronic-6 list)

Outputs:
- chronic6_vs_nonchronic_webcam_features.csv
- chronic6_gate_simulation.csv
- per_mode_tau_calibration.csv
- summary.json
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
CHRONIC = f"{ROOT}/analysis/job_11_identity_audit_2026-05-04/chronic_offender_comparison.csv"
OUT = f"{ROOT}/analysis/chronic6_fingerprint_2026-05-05"

CKPT_FILES_REAL = {
    "P8A":      f"{RAW}/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/teams_real_all_dev_e3_top_n_step6600_frames_report.csv",
}

CKPT_FILES_FAKE = {
    "viso": {
        "P8A":      f"{RAW}/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
        "E2B_3200": f"{RAW}/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
        "E3_6600":  f"{RAW}/visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
    },
    "deeplive": {
        "P8A":      f"{RAW}/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
        "E2B_3200": f"{RAW}/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv",
        "E3_6600":  f"{RAW}/deeplive_enhanced_dev_e3_top_n_step6600_frames_report.csv",
    },
    "teams_fake": {
        "P8A":      f"{RAW}/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv",
        "E2B_3200": f"{RAW}/teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv",
        "E3_6600":  f"{RAW}/teams_fake_all_dev_e3_top_n_step6600_frames_report.csv",
    },
    "teams_fake_lockbox": {
        "P8A":      f"{RAW}/teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv",
        "E2B_3200": f"{RAW}/teams_fake_all_lockbox_e2b_top_n_step3200_frames_report.csv",
        "E3_6600":  f"{RAW}/teams_fake_all_lockbox_e3_top_n_step6600_frames_report.csv",
    },
}

TAU_F0 = {"P8A": 0.70503, "E2B_3200": 0.50644, "E3_6600": 0.85196}

os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Load reals + tags + chronic list
# ----------------------------------------------------------------------
print("Loading reals...")
real_dfs = {}
for ck, path in CKPT_FILES_REAL.items():
    d = pd.read_csv(path).rename(columns={"frame_prob": f"score_{ck}"})
    real_dfs[ck] = d[["frame_path", "video_id", "label", f"score_{ck}"]]

reals = real_dfs["P8A"]
for ck in ["E2B_3200", "E3_6600"]:
    reals = reals.merge(real_dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")

def extract_identity(vid):
    if not isinstance(vid, str):
        return "UNKNOWN"
    parts = vid.split("__")
    if len(parts) >= 2 and parts[1].startswith("s") and parts[1][1:].isdigit():
        return f"{parts[0]}__{parts[1]}"
    return parts[0]

reals["identity"] = reals["video_id"].apply(extract_identity)
print(f"  reals: {reals.shape[0]} frames, {reals['identity'].nunique()} identities")

print("Loading tags...")
tag_cols = ["gcs_uri", "clip_capture_mode", "sharpness_laplacian", "jpeg_qf_estimate",
            "brightness_v_mean", "brightness_v_std", "contrast_rms", "saturation_s_mean",
            "face_pixel_area", "face_area_ratio", "face_count",
            "yaw_deg", "pitch_deg", "roll_deg",
            "is_clipped_highlights", "is_low_quality", "is_pose_extreme", "is_no_face",
            "is_likely_screen_capture", "width", "height"]
tags = pd.read_parquet(TAGS)[tag_cols].rename(columns={"gcs_uri": "frame_path"})
reals = reals.merge(tags, on="frame_path", how="left")
print(f"  reals with tag coverage: {reals['clip_capture_mode'].notna().sum()}")

chronic_df = pd.read_csv(CHRONIC)
chronic_6 = set(chronic_df.loc[chronic_df["is_chronic_6"], "identity"])
print(f"\nChronic-6: {sorted(chronic_6)}")

# ----------------------------------------------------------------------
# Q1: Chronic-6 vs non-chronic webcam fingerprint
# ----------------------------------------------------------------------

SCREEN_LIKE = {"webcam", "screen", "phone_screen", "screen_recording"}
FEATURES = ["sharpness_laplacian", "jpeg_qf_estimate", "brightness_v_mean", "brightness_v_std",
            "contrast_rms", "saturation_s_mean", "face_pixel_area", "face_area_ratio",
            "face_count", "yaw_deg", "pitch_deg", "roll_deg", "width", "height"]

# Cohorts:
# A: chronic-6 frames (within webcam/screen-like mode)
# B: non-chronic-6 frames in webcam/screen-like mode (the "good" webcam users)
# C: normal_photo frames

covered = reals[reals["clip_capture_mode"].notna()].copy()
covered["is_chronic_6"] = covered["identity"].isin(chronic_6)
covered["is_screen_like_mode"] = covered["clip_capture_mode"].isin(SCREEN_LIKE)

A = covered[covered["is_chronic_6"] & covered["is_screen_like_mode"]].copy()
B = covered[~covered["is_chronic_6"] & covered["is_screen_like_mode"]].copy()
C = covered[covered["clip_capture_mode"] == "normal_photo"].copy()

print(f"\nCohort sizes:")
print(f"  A (chronic-6 in screen-like): {len(A)}")
print(f"  B (non-chronic in screen-like): {len(B)}")
print(f"  C (normal_photo): {len(C)}")

# Per-feature distribution comparison: chronic-6 vs non-chronic-webcam
fp_rows = []
for feat in FEATURES:
    a_vals = A[feat].dropna()
    b_vals = B[feat].dropna()
    c_vals = C[feat].dropna()
    if len(a_vals) < 5 or len(b_vals) < 5:
        continue
    ks_ab, ks_p_ab = stats.ks_2samp(a_vals, b_vals)
    ks_ac, ks_p_ac = stats.ks_2samp(a_vals, c_vals) if len(c_vals) >= 5 else (np.nan, np.nan)
    fp_rows.append({
        "feature": feat,
        "n_A": len(a_vals),
        "n_B": len(b_vals),
        "mean_A_chronic6": float(a_vals.mean()),
        "mean_B_nonchronic_webcam": float(b_vals.mean()),
        "mean_C_normal_photo": float(c_vals.mean()) if len(c_vals) > 0 else np.nan,
        "p50_A": float(a_vals.quantile(0.50)),
        "p50_B": float(b_vals.quantile(0.50)),
        "p50_C": float(c_vals.quantile(0.50)) if len(c_vals) > 0 else np.nan,
        "delta_mean_AB": float(a_vals.mean() - b_vals.mean()),
        "ks_stat_A_vs_B": float(ks_ab),
        "ks_p_A_vs_B": float(ks_p_ab),
        "ks_stat_A_vs_C": float(ks_ac) if not np.isnan(ks_ac) else None,
        "ks_p_A_vs_C": float(ks_p_ac) if not np.isnan(ks_p_ac) else None,
    })
fp_df = pd.DataFrame(fp_rows).sort_values("ks_stat_A_vs_B", ascending=False)
print("\nChronic-6 vs non-chronic-webcam (sorted by KS separation):")
print(fp_df[["feature", "n_A", "n_B", "mean_A_chronic6", "mean_B_nonchronic_webcam",
             "ks_stat_A_vs_B", "ks_p_A_vs_B"]].to_string(index=False))
fp_df.to_csv(f"{OUT}/chronic6_vs_nonchronic_webcam_features.csv", index=False)

# Boolean tag counts
bool_cols = ["is_clipped_highlights", "is_low_quality", "is_pose_extreme", "is_no_face", "is_likely_screen_capture"]
bool_rows = []
for col in bool_cols:
    if col not in covered.columns:
        continue
    bool_rows.append({
        "tag": col,
        "frac_A_chronic6": float(A[col].fillna(False).mean()),
        "frac_B_nonchronic_webcam": float(B[col].fillna(False).mean()),
        "frac_C_normal_photo": float(C[col].fillna(False).mean()) if len(C) > 0 else np.nan,
    })
bool_df = pd.DataFrame(bool_rows)
print("\nBoolean-tag fractions per cohort:")
print(bool_df.to_string(index=False))
bool_df.to_csv(f"{OUT}/boolean_tag_fractions.csv", index=False)

# ----------------------------------------------------------------------
# Per-identity median feature values (which identities cluster on each feature)
# ----------------------------------------------------------------------
per_id_rows = []
for ident, g in covered.groupby("identity"):
    if len(g) < 5:
        continue
    row = {
        "identity": ident,
        "n_frames": len(g),
        "is_chronic_6": ident in chronic_6,
        "dominant_capture_mode": g["clip_capture_mode"].mode().iloc[0] if len(g["clip_capture_mode"].mode()) > 0 else "UNKNOWN",
    }
    for feat in FEATURES:
        if feat in g.columns:
            row[f"{feat}_p50"] = float(g[feat].median())
    per_id_rows.append(row)
per_id_df = pd.DataFrame(per_id_rows)
per_id_df.to_csv(f"{OUT}/per_identity_feature_medians.csv", index=False)
print(f"\nPer-identity median features computed for {len(per_id_df)} identities")

# ----------------------------------------------------------------------
# Q2: Per-capture-mode tau calibration
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Per-capture-mode tau calibration simulation")
print("="*60)

# For each ckpt, calibrate tau per capture_mode such that within-mode FPR = 10%
per_mode_tau = {}
for ck in CKPT_FILES_REAL:
    score_col = f"score_{ck}"
    per_mode_tau[ck] = {}
    for mode in ["normal_photo", "webcam", "phone_screen", "screen", "screen_recording"]:
        mode_reals = covered[covered["clip_capture_mode"] == mode]
        if len(mode_reals) == 0:
            per_mode_tau[ck][mode] = None
            continue
        # tau at FPR=10% within mode
        per_mode_tau[ck][mode] = float(mode_reals[score_col].quantile(0.90))

print("\nPer-mode tau (FPR=10% within mode):")
print(pd.DataFrame(per_mode_tau).T)

# Now load fake suites + their per-frame capture_modes
print("\nLoading fake suites for per-mode-tau recall computation...")
all_results = []
for suite, ck_files in CKPT_FILES_FAKE.items():
    fake_dfs = {}
    for ck, path in ck_files.items():
        d = pd.read_csv(path).rename(columns={"frame_prob": f"score_{ck}"})
        fake_dfs[ck] = d[["frame_path", "label", f"score_{ck}"]]
    f = fake_dfs["P8A"]
    for ck in ["E2B_3200", "E3_6600"]:
        f = f.merge(fake_dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")
    f = f.merge(tags, on="frame_path", how="left")
    f = f[f["label"] == 1].copy()
    n_total = len(f)
    n_with_mode = f["clip_capture_mode"].notna().sum()
    print(f"  {suite}: {n_total} fakes, {n_with_mode} with capture_mode tag")

    # Show per-mode count
    print(f"    capture_mode dist: {f['clip_capture_mode'].value_counts(dropna=False).to_dict()}")

    for ck in CKPT_FILES_REAL:
        score_col = f"score_{ck}"
        # Global tau (F0)
        global_recall = (f[score_col] >= TAU_F0[ck]).mean()
        # Per-mode tau: for each fake, look up tau by its capture_mode
        f_covered = f[f["clip_capture_mode"].notna()].copy()
        if len(f_covered) > 0:
            f_covered["tau_per_mode"] = f_covered["clip_capture_mode"].map(lambda m: per_mode_tau[ck].get(m))
            valid = f_covered["tau_per_mode"].notna()
            per_mode_recall_covered = (f_covered.loc[valid, score_col] >= f_covered.loc[valid, "tau_per_mode"]).mean()
            n_per_mode_covered = valid.sum()
        else:
            per_mode_recall_covered = np.nan
            n_per_mode_covered = 0

        # Apply per-mode-tau also to F0_global_tau on covered subset (apples-to-apples)
        if len(f_covered) > 0:
            global_recall_covered = (f_covered[score_col] >= TAU_F0[ck]).mean()
        else:
            global_recall_covered = np.nan

        all_results.append({
            "ckpt": ck,
            "suite": suite,
            "n_fakes": n_total,
            "n_fakes_covered": n_per_mode_covered,
            "global_tau": TAU_F0[ck],
            "recall_global_tau_all_fakes": float(global_recall),
            "recall_global_tau_covered_only": float(global_recall_covered),
            "recall_per_mode_tau_covered_only": float(per_mode_recall_covered),
            "lift_per_mode_vs_global_pp": float(per_mode_recall_covered - global_recall_covered) * 100 if not np.isnan(per_mode_recall_covered) else np.nan,
        })

per_mode_df = pd.DataFrame(all_results)
print("\nPer-mode-tau vs global-tau (covered fakes only, FPR=10% within each mode):")
print(per_mode_df[["ckpt", "suite", "n_fakes_covered",
                   "recall_global_tau_covered_only", "recall_per_mode_tau_covered_only",
                   "lift_per_mode_vs_global_pp"]].to_string(index=False))
per_mode_df.to_csv(f"{OUT}/per_mode_tau_calibration.csv", index=False)

# Verify global FPR under per-mode-tau
print("\nVerification: global FPR under per-mode tau (should be ≤ 10%):")
verify_rows = []
for ck in CKPT_FILES_REAL:
    score_col = f"score_{ck}"
    cov = covered.copy()
    cov["tau_per_mode"] = cov["clip_capture_mode"].map(lambda m: per_mode_tau[ck].get(m))
    fp_under_per_mode = (cov[score_col] >= cov["tau_per_mode"]).sum()
    fp_under_global = (cov[score_col] >= TAU_F0[ck]).sum()
    verify_rows.append({
        "ckpt": ck,
        "n_covered_reals": len(cov),
        "fpr_under_global_tau_pct": 100 * fp_under_global / len(cov),
        "fpr_under_per_mode_tau_pct": 100 * fp_under_per_mode / len(cov),
    })
verify_df = pd.DataFrame(verify_rows)
print(verify_df.to_string(index=False))
verify_df.to_csv(f"{OUT}/global_fpr_under_per_mode_tau.csv", index=False)

# ----------------------------------------------------------------------
# Q1 sub-test: simulated chronic-6 gate using top fingerprint features
# Pick top-3 features by KS separation, gate on simple threshold, see what we drop
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Simulated chronic-6 gate (top fingerprint features)")
print("="*60)

# Pick top-3 features where chronic-6 sit on the same side
top3 = fp_df.nlargest(3, "ks_stat_A_vs_B")
print("\nTop-3 separating features:")
print(top3[["feature", "mean_A_chronic6", "mean_B_nonchronic_webcam", "ks_stat_A_vs_B"]].to_string(index=False))

# Build a one-feature gate per top feature: define threshold at A's median
# Then count: of all reals, what fraction would be dropped by this gate? Of chronic-6 only?
gate_rows = []
for _, row in top3.iterrows():
    feat = row["feature"]
    a_med = row["mean_A_chronic6"]
    b_med = row["mean_B_nonchronic_webcam"]
    # Threshold direction: drop side where A sits
    if a_med > b_med:
        # A is HIGH on this feature; drop frames with feat >= a_med (or some fraction)
        threshold = float(A[feat].quantile(0.25))  # drop top-75% of A
        drop_above = True
        n_chronic_dropped = int((A[feat] >= threshold).sum())
        n_nonchronic_dropped = int((B[feat] >= threshold).sum())
        n_normal_dropped = int((C[feat] >= threshold).sum())
    else:
        threshold = float(A[feat].quantile(0.75))  # drop bottom-75% of A
        drop_above = False
        n_chronic_dropped = int((A[feat] <= threshold).sum())
        n_nonchronic_dropped = int((B[feat] <= threshold).sum())
        n_normal_dropped = int((C[feat] <= threshold).sum())
    gate_rows.append({
        "feature": feat,
        "threshold": threshold,
        "drop_above": drop_above,
        "n_chronic_dropped": n_chronic_dropped,
        "frac_chronic_dropped": n_chronic_dropped / len(A),
        "n_nonchronic_dropped": n_nonchronic_dropped,
        "frac_nonchronic_dropped": n_nonchronic_dropped / len(B) if len(B) > 0 else 0,
        "n_normal_dropped": n_normal_dropped,
        "frac_normal_dropped": n_normal_dropped / len(C) if len(C) > 0 else 0,
    })
gate_df = pd.DataFrame(gate_rows)
print("\nSingle-feature chronic-6 gate (threshold at chronic-6's quartile):")
print(gate_df.to_string(index=False))
gate_df.to_csv(f"{OUT}/chronic6_gate_simulation.csv", index=False)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
top_feature = fp_df.iloc[0]
summary = {
    "n_chronic6_frames_covered": int(len(A)),
    "n_nonchronic_webcam_frames": int(len(B)),
    "n_normal_photo_frames": int(len(C)),
    "top_separating_feature": str(top_feature["feature"]),
    "top_feature_KS_chronic_vs_nonchronic_webcam": float(top_feature["ks_stat_A_vs_B"]),
    "top_feature_p_value": float(top_feature["ks_p_A_vs_B"]),
    "interpretation": (
        "If top KS_stat ≥ 0.5 with p<1e-5, chronic-6 has a tight in-frame fingerprint "
        "distinguishing them from non-chronic webcam users — deployment-viable per-frame gate possible. "
        "If top KS < 0.3, chronic-6 are not separable from other webcam users by IQ/face features alone — "
        "need training-side intervention."
    ),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n{json.dumps(summary, indent=2)}")
print(f"\nDone. Outputs in {OUT}")
