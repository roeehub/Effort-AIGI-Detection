"""Viso capture-mode proxy: predict capture_mode from IQ features.

Trains a multinomial classifier on the 7334 already-tagged frames to predict
clip_capture_mode from non-CLIP IQ features. Applies to viso fakes (550) which
have no tags.

Then computes per-mode tau recall on viso under predicted modes.

This is an APPROXIMATION — true tags would require running CLIP on viso frames.
Worst-case bounds are also computed: viso recall if all viso assumed webcam,
all assumed normal_photo, all assumed phone_screen.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
CROP_ATTRS = f"{ROOT}/analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv"
OUT = f"{ROOT}/analysis/viso_capture_mode_proxy_2026-05-05"

os.makedirs(OUT, exist_ok=True)

VISO_FILES = {
    "P8A":      f"{RAW}/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
}
TAU_F0 = {"P8A": 0.70503, "E2B_3200": 0.50644, "E3_6600": 0.85196}

# Per-mode tau values from dev-calibration (from previous job)
PER_MODE_TAU_DEV = {
    "P8A":      {"normal_photo": 0.100498, "webcam": 0.987451, "phone_screen": 0.237097, "screen": 0.830916, "screen_recording": 0.036185},
    "E2B_3200": {"normal_photo": 0.122986, "webcam": 0.419271, "phone_screen": 0.308249, "screen": 0.907920, "screen_recording": 0.048251},
    "E3_6600":  {"normal_photo": 0.006568, "webcam": 0.503439, "phone_screen": 0.913548, "screen": 0.993364, "screen_recording": 0.006802},
}

# IQ features available in the tagger (and in crop_attributes for viso)
# Use the COMMON subset across both
IQ_FEATURES_TAGS = ["sharpness_laplacian", "brightness_v_mean", "saturation_s_mean",
                    "face_pixel_area", "face_area_ratio", "width", "height"]
# crop_attributes column names differ slightly:
#   tags: sharpness_laplacian, brightness_v_mean, saturation_s_mean
#   crop_attrs: laplacian_var, luma_mean, saturation_mean
# We need a common feature set computable on both. The crop_attributes file
# computed laplacian on a different basis (see memory project_sharpness_metric_bug).
# For this proxy, use whatever is computable on viso AND comparable to tags.

# Available in crop_attributes (full list): h, w, luma_mean, luma_std, luma_p10, luma_p90,
# laplacian_var, sobel_edge_mean, saturation_mean, skin_frac, seq_id, subtype, frame_num, filename

# To compare apples-to-apples, compute on tagged frames the SAME columns.
# The full_tags has sharpness_laplacian (full image), brightness_v_mean, saturation_s_mean,
# face_pixel_area, face_area_ratio, width, height. Different feature set than crop_attributes.
# We only have crop_attributes for viso, so train on the SUBSET of features that exist in BOTH.

# Common features: width/height (called h/w in crop_attributes), face_pixel_area (not in crop_attrs),
# brightness_v_mean (vs luma_mean — different normalization), saturation (different normalization)

# Approach: train a classifier using TAGS' native features on tagged frames; predict on viso
# using the same TAGS' features. But viso has no tag entries. Solution: compute the tag-style
# features on viso from local frame cache OR from the visomaster manifests.

# Faster alternative: train using ONLY width/height features (which exist in both forms),
# then validate prediction quality. This is a coarse proxy but a meaningful one.

# Even simpler: use ONLY the crop_attributes columns and join by computing the same columns
# on the tagged frames somehow. But we don't have raw frames for the tagged set.

# Cleanest path: use the small overlap of features computable from tagged-data parquet
# (which has rich features) AND crop_attributes (smaller set). Since tags has its own feature
# set, train a classifier on tags using its own features, then for viso, look up viso's
# image dimensions + sharpness laplacian if we recompute.

# Actually we DO have crop_attributes.csv for viso, which has w, h, luma_mean, laplacian_var,
# sobel_edge_mean, saturation_mean. That's enough features. We just need the SAME features
# computed on the tagged data.

# Tagged data only has: width, height, sharpness_laplacian (full-image), brightness_v_mean,
# saturation_s_mean, face_pixel_area, face_area_ratio.

# Mapping (close enough proxies):
#   crop_attrs.w        ~ tags.width
#   crop_attrs.h        ~ tags.height
#   crop_attrs.luma_mean ~ tags.brightness_v_mean (both 0-255 luma)
#   crop_attrs.laplacian_var ~ tags.sharpness_laplacian (computed differently — full image vs face)
#   crop_attrs.saturation_mean ~ tags.saturation_s_mean (saturation_mean is 0-1 vs s_mean 0-255 — need to scale)

# Use only width, height, brightness, saturation (after scaling) as the cross-compatible features
FEATURE_MAP = {
    "tags_width":       ("width", "w"),
    "tags_height":      ("height", "h"),
    "tags_brightness":  ("brightness_v_mean", "luma_mean"),
    # saturation_s_mean is on 0-255 scale, saturation_mean is 0-1; scale below
}

# ----------------------------------------------------------------------
# Build training data from tagged frames
# ----------------------------------------------------------------------
print("Loading tagged data...")
tags = pd.read_parquet(TAGS)
tags = tags[tags["clip_capture_mode"].notna()].copy()
print(f"  tagged frames with capture_mode: {len(tags)}")

# Saturation: scale to 0-1
tags["saturation_norm"] = tags["saturation_s_mean"] / 255.0

train_features = ["width", "height", "brightness_v_mean", "saturation_norm"]
X_train = tags[train_features].values
y_train = tags["clip_capture_mode"].values

print(f"\nClass counts in training:")
print(pd.Series(y_train).value_counts())

# Train multinomial logistic regression
print("\nTraining multinomial logistic regression with 5-fold CV...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs", random_state=0)

cv_scores = cross_val_score(clf, X_train_scaled, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="accuracy", n_jobs=1)
print(f"  CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

clf.fit(X_train_scaled, y_train)

# Per-class precision/recall (in-sample as quick sanity)
from sklearn.metrics import classification_report, confusion_matrix
y_train_pred = clf.predict(X_train_scaled)
print("\nIn-sample classification report:")
print(classification_report(y_train, y_train_pred, zero_division=0))

# Save the trained classifier's per-class behavior on the dev tag dataset
# (so we can quote how well it works as a proxy)

# ----------------------------------------------------------------------
# Apply to viso fakes
# ----------------------------------------------------------------------
print("\nLoading viso fake reports + crop_attributes...")
viso_dfs = {}
for ck, path in VISO_FILES.items():
    d = pd.read_csv(path).rename(columns={"frame_prob": f"score_{ck}"})
    viso_dfs[ck] = d[["frame_path", "label", f"score_{ck}"]]
viso = viso_dfs["P8A"]
for ck in ["E2B_3200", "E3_6600"]:
    viso = viso.merge(viso_dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")
viso = viso[viso["label"] == 1].copy()

# Join crop_attributes by filename
crop = pd.read_csv(CROP_ATTRS)
viso["filename"] = viso["frame_path"].apply(lambda p: os.path.basename(p))
viso = viso.merge(crop, on="filename", how="left")
print(f"  viso fakes: {len(viso)}, crop_attrs joined: {viso['luma_mean'].notna().sum()}")

# Build feature matrix in the same order
viso["width_proxy"] = viso["w"]
viso["height_proxy"] = viso["h"]
viso["brightness_proxy"] = viso["luma_mean"]
viso["saturation_norm_proxy"] = viso["saturation_mean"]  # already 0-1

X_viso = viso[["width_proxy", "height_proxy", "brightness_proxy", "saturation_norm_proxy"]].values
X_viso_scaled = scaler.transform(X_viso)

viso["predicted_capture_mode"] = clf.predict(X_viso_scaled)
proba = clf.predict_proba(X_viso_scaled)
viso["predicted_capture_mode_prob"] = proba.max(axis=1)

print("\nPredicted capture_mode distribution on viso fakes:")
print(viso["predicted_capture_mode"].value_counts())

# Per-class confidence
print("\nMean predicted-mode probability:")
for c in viso["predicted_capture_mode"].unique():
    sub = viso[viso["predicted_capture_mode"] == c]
    print(f"  {c}: n={len(sub)}, mean prob={sub['predicted_capture_mode_prob'].mean():.3f}")

# ----------------------------------------------------------------------
# Compute per-mode tau recall on viso under predicted modes
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Viso recall under per-mode tau (predicted modes)")
print("="*60)

recall_rows = []
for ck in VISO_FILES:
    score_col = f"score_{ck}"
    # Global tau
    global_recall = (viso[score_col] >= TAU_F0[ck]).mean()
    # Per-mode tau (predicted)
    viso["tau_per_mode"] = viso["predicted_capture_mode"].map(lambda m: PER_MODE_TAU_DEV[ck].get(m))
    valid = viso["tau_per_mode"].notna()
    per_mode_recall = (viso.loc[valid, score_col] >= viso.loc[valid, "tau_per_mode"]).mean()

    # Bound analysis: assume all viso = each capture mode
    bounds = {}
    for mode in PER_MODE_TAU_DEV[ck]:
        tau_m = PER_MODE_TAU_DEV[ck][mode]
        bounds[mode] = float((viso[score_col] >= tau_m).mean())

    recall_rows.append({
        "ckpt": ck,
        "n_viso": len(viso),
        "n_with_predicted_mode": int(valid.sum()),
        "viso_global_tau_recall_pct": 100 * global_recall,
        "viso_per_mode_predicted_tau_recall_pct": 100 * per_mode_recall,
        "BOUND_all_normal_photo_pct": 100 * bounds["normal_photo"],
        "BOUND_all_webcam_pct": 100 * bounds["webcam"],
        "BOUND_all_phone_screen_pct": 100 * bounds["phone_screen"],
        "BOUND_all_screen_pct": 100 * bounds["screen"],
        "BOUND_all_screen_recording_pct": 100 * bounds["screen_recording"],
    })

recall_df = pd.DataFrame(recall_rows)
print(recall_df.to_string(index=False))
recall_df.to_csv(f"{OUT}/viso_per_mode_tau_recall.csv", index=False)

# Save the per-frame predictions
viso[["frame_path", "filename", "score_P8A", "score_E2B_3200", "score_E3_6600",
      "predicted_capture_mode", "predicted_capture_mode_prob",
      "w", "h", "luma_mean", "saturation_mean", "laplacian_var"]].to_csv(
          f"{OUT}/viso_predicted_capture_modes.csv", index=False)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
summary = {
    "classifier_cv_accuracy": float(cv_scores.mean()),
    "classifier_cv_std": float(cv_scores.std()),
    "classifier_features": train_features,
    "n_training_frames": int(len(tags)),
    "n_viso_predicted": int(len(viso)),
    "viso_predicted_mode_distribution": viso["predicted_capture_mode"].value_counts().to_dict(),
    "viso_recall_per_ckpt": {r["ckpt"]: {
        "global_tau_pct": r["viso_global_tau_recall_pct"],
        "per_mode_predicted_tau_pct": r["viso_per_mode_predicted_tau_recall_pct"],
        "lift_pp": r["viso_per_mode_predicted_tau_recall_pct"] - r["viso_global_tau_recall_pct"],
    } for r in recall_rows},
    "caveats": [
        "Predicted capture_mode is a proxy — CV accuracy ~XX% on training distribution.",
        "Sharpness laplacian was NOT used (different definitions in tags vs crop_attrs).",
        "Predictions may be biased toward modes whose training-set IQ profile matches viso.",
        "Recall numbers are upper bounds if predictions are perfect; lower if predictions are wrong.",
    ],
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n{json.dumps(summary, indent=2)}")
print(f"\nDone. Outputs in {OUT}")
