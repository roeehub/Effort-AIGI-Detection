"""Codec-aug verification: TeamsCodecSimulation vs actual teams transport.

For each pair (raw_viso_X, teams_viso_X) of the same source seq_id:
  1. Apply TeamsCodecSimulation to raw_X
  2. Compute IQ features for {raw_X, raw_X+aug, teams_X}
  3. Compute Δ_aug = features(raw+aug) - features(raw)
  4. Compute Δ_teams = features(teams) - features(raw)
  5. Cosine similarity(Δ_aug, Δ_teams) per feature
  6. Per-feature ratio |aug shift| / |teams shift|

Output:
  - aug_vs_actual_per_pair.csv: per-pair feature deltas
  - aug_vs_actual_summary.csv: aggregate statistics
  - FINDINGS_AUG_VS_ACTUAL.md
"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
sys.path.insert(0, ROOT)

OUT = f"{ROOT}/analysis/codec_aug_verification_2026-05-05"
RAW_REPORT = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv"
LOCAL_CACHE = f"{OUT}/_frame_cache"
os.makedirs(LOCAL_CACHE, exist_ok=True)

N_PAIRS = 30   # sample size
SEED = 42

# ---------------------------------------------------------------------
# 1. Pick paired frames from the eval set
# ---------------------------------------------------------------------
print("Loading viso eval frames...")
df = pd.read_csv(RAW_REPORT)
df["filename"] = df["frame_path"].apply(lambda p: p.rsplit("/", 1)[-1])
def parse(fn):
    m = re.match(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png", fn)
    return (m.group(1), m.group(3)) if m else ("UNK", "UNK")
df[["subtype", "seq_id"]] = df["filename"].apply(lambda x: pd.Series(parse(x)))

raw_df = df[df["subtype"] == "raw"].set_index("seq_id")
teams_df = df[df["subtype"] == "teams"].set_index("seq_id")
shared_seqs = sorted(set(raw_df.index) & set(teams_df.index))

rng = np.random.RandomState(SEED)
sample_seqs = rng.choice(shared_seqs, size=min(N_PAIRS, len(shared_seqs)), replace=False)
print(f"Sampled {len(sample_seqs)} paired seq_ids out of {len(shared_seqs)} available pairs")

# ---------------------------------------------------------------------
# 2. Download paired frames
# ---------------------------------------------------------------------
print("\nDownloading frames...")
import subprocess
download_pairs = []
for seq in sample_seqs:
    raw_path = raw_df.loc[seq, "frame_path"]
    teams_path = teams_df.loc[seq, "frame_path"]
    raw_local = f"{LOCAL_CACHE}/raw_{seq}.png"
    teams_local = f"{LOCAL_CACHE}/teams_{seq}.png"
    download_pairs.append((seq, raw_path, raw_local, teams_path, teams_local))

# Batch download via gsutil
gcs_paths = []
local_paths = []
for seq, rp, rl, tp, tl in download_pairs:
    if not os.path.exists(rl):
        gcs_paths.append(rp); local_paths.append(rl)
    if not os.path.exists(tl):
        gcs_paths.append(tp); local_paths.append(tl)

if gcs_paths:
    print(f"  downloading {len(gcs_paths)} frames in batches of 20...")
    for i in range(0, len(gcs_paths), 20):
        batch_gcs = gcs_paths[i:i+20]
        batch_local = local_paths[i:i+20]
        # gsutil -m cp doesn't allow renaming; use one-at-a-time
        for g, l in zip(batch_gcs, batch_local):
            subprocess.run(["gsutil", "-q", "cp", g, l], check=False)
        print(f"  downloaded batch {i//20+1}/{(len(gcs_paths)+19)//20}")
else:
    print("  all frames already cached")

# Verify all pairs downloaded
ok_pairs = []
for seq, rp, rl, tp, tl in download_pairs:
    if os.path.exists(rl) and os.path.exists(tl):
        ok_pairs.append((seq, rl, tl))
    else:
        print(f"  MISSING: seq={seq}")
print(f"Downloaded pairs: {len(ok_pairs)}")

# ---------------------------------------------------------------------
# 3. Apply TeamsCodecSimulation to raw frames + compute IQ features
# ---------------------------------------------------------------------
print("\nApplying TeamsCodecSimulation + computing IQ features...")
from data.augmentations.teams_simulation import TeamsCodecSimulation
import cv2

def compute_iq_features(img_rgb_uint8: np.ndarray) -> dict:
    """Cheap deployable IQ features."""
    img = img_rgb_uint8
    h, w = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge_mean = float(np.sqrt(sobel_x**2 + sobel_y**2).mean())
    luma_mean = float(img_yuv[..., 0].mean())
    luma_p10 = float(np.percentile(img_yuv[..., 0], 10))
    luma_p90 = float(np.percentile(img_yuv[..., 0], 90))
    saturation_mean = float(img_hsv[..., 1].mean())
    contrast = float(img_yuv[..., 0].std())
    # High-frequency energy (FFT-based)
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    cy, cx = magnitude.shape[0]//2, magnitude.shape[1]//2
    # ratio of high-freq energy (outside center 1/4 of frequencies) to total
    radius = min(cy, cx) // 4
    yy, xx = np.indices(magnitude.shape)
    mask_hf = ((yy - cy)**2 + (xx - cx)**2) > radius**2
    hf_energy_ratio = float(magnitude[mask_hf].sum() / max(1, magnitude.sum()))
    return {
        "laplacian_var": laplacian_var,
        "sobel_edge_mean": sobel_edge_mean,
        "luma_mean": luma_mean,
        "luma_p10": luma_p10,
        "luma_p90": luma_p90,
        "saturation_mean": saturation_mean,
        "contrast": contrast,
        "hf_energy_ratio": hf_energy_ratio,
        "width": w,
        "height": h,
    }

# Initialize TeamsCodecSimulation with default params (matches calibration)
np.random.seed(SEED)
aug = TeamsCodecSimulation(always_apply=True, p=1.0)

rows = []
for seq, raw_local, teams_local in ok_pairs:
    raw_bgr = cv2.imread(raw_local)
    teams_bgr = cv2.imread(teams_local)
    if raw_bgr is None or teams_bgr is None:
        print(f"  failed to load seq={seq}")
        continue
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    teams_rgb = cv2.cvtColor(teams_bgr, cv2.COLOR_BGR2RGB)
    # Apply aug to raw (need to match teams resolution — resize raw if needed)
    # Actually we want to test aug as-is on raw
    aug_rgb = aug.apply(raw_rgb)

    # Compute features
    f_raw = compute_iq_features(raw_rgb)
    f_aug = compute_iq_features(aug_rgb)
    f_teams = compute_iq_features(teams_rgb)

    row = {"seq_id": seq}
    for k in f_raw:
        row[f"raw_{k}"] = f_raw[k]
        row[f"aug_{k}"] = f_aug[k]
        row[f"teams_{k}"] = f_teams[k]
        # Deltas
        row[f"delta_aug_{k}"] = f_aug[k] - f_raw[k]
        row[f"delta_teams_{k}"] = f_teams[k] - f_raw[k]
    rows.append(row)

per_pair_df = pd.DataFrame(rows)
per_pair_df.to_csv(f"{OUT}/aug_vs_actual_per_pair.csv", index=False)
print(f"\n  computed features for {len(per_pair_df)} pairs")

# ---------------------------------------------------------------------
# 4. Aggregate analysis
# ---------------------------------------------------------------------
features = ["laplacian_var", "sobel_edge_mean", "luma_mean", "luma_p10", "luma_p90",
            "saturation_mean", "contrast", "hf_energy_ratio"]

summary_rows = []
for feat in features:
    delta_aug = per_pair_df[f"delta_aug_{feat}"].values
    delta_teams = per_pair_df[f"delta_teams_{feat}"].values

    # Mean shifts
    mean_aug_shift = float(np.mean(delta_aug))
    mean_teams_shift = float(np.mean(delta_teams))

    # Direction agreement: fraction of pairs where signs match
    sign_match = float(np.mean(np.sign(delta_aug) == np.sign(delta_teams)))

    # Cosine similarity across pairs
    if np.linalg.norm(delta_aug) > 0 and np.linalg.norm(delta_teams) > 0:
        cosine_sim = float(np.dot(delta_aug, delta_teams) /
                           (np.linalg.norm(delta_aug) * np.linalg.norm(delta_teams)))
    else:
        cosine_sim = 0.0

    # Magnitude ratio (|aug| / |teams|)
    if abs(mean_teams_shift) > 1e-6:
        magnitude_ratio = abs(mean_aug_shift) / abs(mean_teams_shift)
    else:
        magnitude_ratio = float("nan")

    # Pearson correlation
    if np.std(delta_aug) > 0 and np.std(delta_teams) > 0:
        pearson_r = float(np.corrcoef(delta_aug, delta_teams)[0, 1])
    else:
        pearson_r = 0.0

    summary_rows.append({
        "feature": feat,
        "mean_raw_value": float(per_pair_df[f"raw_{feat}"].mean()),
        "mean_aug_shift": mean_aug_shift,
        "mean_teams_shift": mean_teams_shift,
        "ratio_aug_to_teams_pct": 100 * magnitude_ratio if not np.isnan(magnitude_ratio) else None,
        "sign_match_frac": sign_match,
        "cosine_similarity_across_pairs": cosine_sim,
        "pearson_r_per_pair": pearson_r,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/aug_vs_actual_summary.csv", index=False)
print("\n=== AUG vs ACTUAL TEAMS TRANSPORT — per feature ===")
print(summary_df.to_string(index=False))

# Calibration target reminder (from teams_simulation.py:6-22)
expected = {
    "laplacian_var": -0.509,        # -50.9% sharpness
    "sobel_edge_mean": -0.388,      # ~ -38% from inline pipeline comment
    "luma_mean": +0.190,            # +19% brightness
    "saturation_mean": +0.035,      # +3.5% contrast (proxied via saturation)
    "hf_energy_ratio": -0.774,      # -77.4% HF energy
}

print("\n=== Calibration sanity check (mean aug shift / mean raw value, vs expected target) ===")
for feat, target_frac in expected.items():
    raw_val = per_pair_df[f"raw_{feat}"].mean()
    aug_shift = per_pair_df[f"delta_aug_{feat}"].mean()
    teams_shift = per_pair_df[f"delta_teams_{feat}"].mean()
    aug_pct = 100.0 * aug_shift / raw_val if raw_val != 0 else None
    teams_pct = 100.0 * teams_shift / raw_val if raw_val != 0 else None
    expected_pct = 100.0 * target_frac
    print(f"  {feat:20s} raw={raw_val:8.2f} aug_shift_pct={aug_pct:+8.1f}% teams_shift_pct={teams_pct:+8.1f}% target_pct={expected_pct:+8.1f}%")

# Verdict
print("\n=== VERDICT ===")
strong_features = []
weak_features = []
for _, row in summary_df.iterrows():
    if row["sign_match_frac"] >= 0.7 and row["cosine_similarity_across_pairs"] >= 0.5:
        strong_features.append(row["feature"])
    elif row["sign_match_frac"] >= 0.5 or row["cosine_similarity_across_pairs"] >= 0.3:
        pass  # neutral
    else:
        weak_features.append(row["feature"])

print(f"  Features where aug WELL captures teams direction: {strong_features}")
print(f"  Features where aug POORLY captures teams direction: {weak_features}")

if len(strong_features) >= 4:
    print("\n  ==> CODEC AUG IS A FAITHFUL SIMULATION on most features. RECOMMEND launch.")
elif len(strong_features) >= 2:
    print("\n  ==> CODEC AUG is partially faithful. Worth launching but expect partial benefit.")
else:
    print("\n  ==> CODEC AUG does NOT capture the actual teams transport direction. DO NOT launch.")

print(f"\nDone. Outputs in {OUT}")
