"""Build unified IQ-tagged dataframe spanning train / dev / lockbox / chronic-6.

Sources:
- analysis/iq_data_atlas_2026-05-08/_cache/*.parquet  (7-axis IQ schema, ~14k frames)
- analysis/lockbox_tagging/full_tags_2026-04-27.parquet (identity_key + richer schema, 7334 frames)

Output:
- artifacts/unified_tags.parquet (one row per frame, harmonized 7-axis schema, with
  pool, role, label, split, and identity_key where derivable)
"""
from __future__ import annotations

import glob
import os
import pandas as pd
import numpy as np

OUT_DIR = "analysis/joint_marginal_audit_2026-05-19/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

ATLAS_DIR = "analysis/iq_data_atlas_2026-05-08/_cache"
FULL_TAGS = "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"

# 7 atlas axes that constitute the "image-property" feature space
AXES = ["lap_var", "luma_mean", "color_a_dev", "color_b_dev", "saturation_mean", "min_dim", "skin_frac"]

# chronic-6 identity name patterns; we look for these substrings in identity_key
# (codebase memory references full list; Roy_D is missing from full_tags but listed)
CHRONIC_PATTERNS = ["PC_Generator", "Roy_D", "bla_bla_chow", "dor_shkedi", "xiang", "Cam_Test", "Chikara_Takahashi"]

# Atlas role mapping → standardized buckets
ROLE_MAP = {
    "train_real": "train_real",
    "train_fake": "train_fake",
    "dev_real":   "dev_real",
    "dev_fake":   "dev_fake",
    "lockbox_real": "lockbox_real",
    "lockbox_fake": "lockbox_fake",
    "hdtf_real":  None,  # role-suffix carries dev/lockbox info via pool name
    "hdtf_fake":  None,
}

def derive_bucket(pool: str, role: str) -> str:
    """Map (pool, role) → standardized bucket. None if unclassifiable."""
    if role in ("train_real", "train_fake", "dev_real", "dev_fake", "lockbox_real", "lockbox_fake"):
        return role
    if role == "hdtf_real":
        if "lockbox" in pool:
            return "lockbox_real"
        else:
            return "dev_real"
    if role == "hdtf_fake":
        if "lockbox" in pool:
            return "lockbox_fake"
        else:
            return "dev_fake"
    return None


# === Load atlas data ===
files = sorted(glob.glob(os.path.join(ATLAS_DIR, "*.parquet")))
print(f"Loading {len(files)} atlas parquets ...")
atlas_dfs = []
for f in files:
    df = pd.read_parquet(f)
    df["bucket"] = df.apply(lambda r: derive_bucket(r["pool"], r["role"]), axis=1)
    atlas_dfs.append(df)
atlas = pd.concat(atlas_dfs, ignore_index=True)
print(f"  atlas combined: {len(atlas)} rows")

# === Load full_tags and add identity_key by joining on frame_path ===
ft = pd.read_parquet(FULL_TAGS)
ft_short = ft[["gcs_uri", "identity_key", "split", "label"]].rename(columns={"gcs_uri": "frame_path"})
atlas = atlas.merge(ft_short, on="frame_path", how="left", suffixes=("", "_ft"))
print(f"  atlas with identity_key joined: {atlas['identity_key'].notna().sum()} rows match full_tags")

# === Tag chronic-6 membership ===
def is_chronic(idkey):
    if pd.isna(idkey):
        return False
    s = str(idkey)
    return any(p.lower() in s.lower() for p in CHRONIC_PATTERNS)

atlas["is_chronic"] = atlas["identity_key"].apply(is_chronic)
print(f"  atlas rows flagged chronic: {atlas['is_chronic'].sum()}")

# === Drop rows with no derivable bucket or with NaN in any axis ===
n_before = len(atlas)
atlas = atlas[atlas["bucket"].notna()].copy()
for a in AXES:
    atlas = atlas[atlas[a].notna()].copy()
print(f"  dropped {n_before - len(atlas)} rows missing bucket or axes; kept {len(atlas)}")

# === Per-bucket summary ===
print("\n=== Per-bucket coverage ===")
summary = atlas.groupby("bucket").agg(
    n=("frame_path", "count"),
    n_chronic=("is_chronic", "sum"),
).reset_index()
print(summary.to_string(index=False))

# === Also keep a 'chronic_lockbox_real' synthetic bucket ===
mask = (atlas["bucket"] == "lockbox_real") & (atlas["is_chronic"])
print(f"\nchronic ∩ lockbox_real: {mask.sum()} frames")
mask = (atlas["bucket"] == "dev_real") & (atlas["is_chronic"])
print(f"chronic ∩ dev_real: {mask.sum()} frames")

# === Save ===
out_path = os.path.join(OUT_DIR, "unified_tags.parquet")
atlas.to_parquet(out_path, index=False)
print(f"\nWrote {out_path}  ({len(atlas)} rows, {len(atlas.columns)} cols)")
print(f"Columns: {atlas.columns.tolist()}")

# === Also save a "chronic_pool" using full_tags directly for chronic identities ===
# This gives us a richer chronic-6 sample than the atlas-joined subset
ft_chronic = ft[ft["identity_key"].apply(is_chronic)].copy()
ft_chronic["bucket"] = ft_chronic.apply(
    lambda r: f"{r['split']}_{r['label']}", axis=1
)
# Harmonize axes from full_tags richer schema:
# - lap_var ≈ sharpness_laplacian (assumed full-image laplacian on the saved crop, same as atlas)
# - luma_mean ≈ brightness_v_mean
# - saturation_mean ≈ saturation_s_mean
# - min_dim = min(width, height)
# - color_a_dev / color_b_dev: not in full_tags → leave NaN (we'll handle in coverage analysis)
# - skin_frac ≈ face_area_ratio
ft_chronic_h = pd.DataFrame({
    "frame_path": ft_chronic["gcs_uri"],
    "identity_key": ft_chronic["identity_key"],
    "bucket": ft_chronic["bucket"],
    "is_chronic": True,
    "lap_var": ft_chronic["sharpness_laplacian"],
    "luma_mean": ft_chronic["brightness_v_mean"],
    "saturation_mean": ft_chronic["saturation_s_mean"],
    "min_dim": ft_chronic[["width", "height"]].min(axis=1),
    "skin_frac": ft_chronic["face_area_ratio"],
    "color_a_dev": np.nan,
    "color_b_dev": np.nan,
})
out_path_c = os.path.join(OUT_DIR, "chronic_full_tags.parquet")
ft_chronic_h.to_parquet(out_path_c, index=False)
print(f"\nWrote {out_path_c}  ({len(ft_chronic_h)} chronic rows from full_tags)")
print("chronic full_tags bucket counts:")
print(ft_chronic_h["bucket"].value_counts().to_string())
