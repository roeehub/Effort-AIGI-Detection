"""Crop-tightness distribution in the training/dev data.

Reads `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`. Computes face
size distributions by (split, label, method) — the question is whether the
real and fake samples are systematically at different face-pixel-areas, which
would indicate the camera-signature shortcut is partially label-leaked
through crop tightness.

Also probes a few image-dimension columns if available to compute
face-to-frame-area ratio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT_DIR = REPO / "analysis/crop_shortcut_2026-04-27"


def quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "p05": float(s.quantile(0.05)),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
    }


def main() -> None:
    df = pd.read_parquet(PARQUET)
    print(f"[crop-dist] loaded {len(df)} rows, splits: {df['split'].value_counts().to_dict()}")
    print(f"[crop-dist] columns mentioning face/area/bbox: "
          f"{[c for c in df.columns if any(k in c.lower() for k in ['face', 'bbox', 'area', 'crop', 'sharp', 'image'])]}")

    # 1. By (split, label)
    print("\n=== A. face_pixel_area by (split, label) ===")
    rows = []
    for (split, label), g in df.groupby(["split", "label"]):
        q = quantiles(g["face_pixel_area"])
        q.update({"split": split, "label": label})
        rows.append(q)
    a = pd.DataFrame(rows).sort_values(["split", "label"])
    print(a.to_string(index=False, float_format=lambda x: f"{x:.0f}" if abs(x) > 100 else f"{x:.3f}"))

    # 2. By method (top 12)
    print("\n=== B. face_pixel_area by method (top 12 by sample size) ===")
    rows = []
    for m, g in df.groupby("method"):
        q = quantiles(g["face_pixel_area"])
        q["method"] = m
        q["label"] = g["label"].iloc[0]
        rows.append(q)
    b = pd.DataFrame(rows).sort_values("n", ascending=False).head(12)
    cols = ["method", "label", "n", "p10", "p25", "p50", "p75", "p90", "mean"]
    print(b[cols].to_string(index=False, float_format=lambda x: f"{x:.0f}" if abs(x) > 100 else f"{x:.3f}"))

    # 3. Direct label-comparison on dev split — is there a leak?
    print("\n=== C. Dev split: real vs fake — face_pixel_area distribution ===")
    dev = df[df["split"] == "dev"]
    real = dev[dev["label"] == "real"]["face_pixel_area"].dropna()
    fake = dev[dev["label"] == "fake"]["face_pixel_area"].dropna()
    print(f"  REAL (n={len(real)}): mean={real.mean():.0f}, p10={real.quantile(0.1):.0f}, p50={real.quantile(0.5):.0f}, p90={real.quantile(0.9):.0f}")
    print(f"  FAKE (n={len(fake)}): mean={fake.mean():.0f}, p10={fake.quantile(0.1):.0f}, p50={fake.quantile(0.5):.0f}, p90={fake.quantile(0.9):.0f}")
    # Effect size (Cohen's d-ish, since means + std)
    pooled_std = float(np.sqrt(((len(real) - 1) * real.std() ** 2 + (len(fake) - 1) * fake.std() ** 2) / max(len(real) + len(fake) - 2, 1)))
    cohen_d = (real.mean() - fake.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d: {cohen_d:.3f}  (|d|>0.5 = label leak; |d|>0.8 = strong leak)")

    # 4. Bin-and-compare: what fraction of (real, fake) fall in each face_size bucket?
    print("\n=== D. Dev split: face_pixel_area buckets — compare label distributions ===")
    bins = [0, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 1e9]
    bin_labels = ["<5k", "5-10k", "10-20k", "20-30k", "30-50k", "50-75k", "75-100k", "100-150k", "150k+"]
    real_b = pd.cut(real, bins=bins, labels=bin_labels).value_counts().sort_index()
    fake_b = pd.cut(fake, bins=bins, labels=bin_labels).value_counts().sort_index()
    real_pct = real_b / len(real) * 100
    fake_pct = fake_b / len(fake) * 100
    bins_df = pd.DataFrame({"real_n": real_b, "real_pct": real_pct, "fake_n": fake_b, "fake_pct": fake_pct})
    print(bins_df.to_string(float_format=lambda x: f"{x:.1f}"))

    # 5. Per-method: where is each fake method's face size distribution?
    print("\n=== E. Per-fake-method face_pixel_area summary ===")
    fake_dev = dev[dev["label"] == "fake"]
    rows = []
    for m, g in fake_dev.groupby("method"):
        if len(g) < 20:
            continue
        rows.append({
            "method": m, "n": len(g),
            "p10": float(g["face_pixel_area"].quantile(0.1)),
            "p50": float(g["face_pixel_area"].quantile(0.5)),
            "p90": float(g["face_pixel_area"].quantile(0.9)),
            "mean": float(g["face_pixel_area"].mean()),
        })
    e = pd.DataFrame(rows).sort_values("p50")
    print(e.to_string(index=False, float_format=lambda x: f"{x:.0f}" if abs(x) > 100 else f"{x:.3f}"))

    # 6. Production-honest pool comparison — derive face_pixel_area on the 180 frames if possible
    # (We can compute via a quick mediapipe pass; but the prod frames are all face crops at 186-238px,
    # so they're full-face — face_pixel_area ≈ image_area.)
    print("\n=== F. Production-honest pool: image-area distribution (proxy for max possible face size) ===")
    prod_root = REPO / "analysis/deployment_honest_eval_2026-04-27/_prod_cache/frames"
    if prod_root.exists():
        from PIL import Image
        rows = []
        for tag_dir in prod_root.iterdir():
            if not tag_dir.is_dir():
                continue
            for fp in sorted(tag_dir.glob("*.png")):
                img = Image.open(fp)
                w, h = img.size
                rows.append({
                    "tag": tag_dir.name,
                    "image_w": w, "image_h": h, "image_area": w * h,
                })
        prod = pd.DataFrame(rows)
        print(f"  loaded {len(prod)} production frames")
        for tag, g in prod.groupby("tag"):
            print(f"    {tag}: n={len(g)}  area p10={g['image_area'].quantile(0.1):.0f}  p50={g['image_area'].quantile(0.5):.0f}  p90={g['image_area'].quantile(0.9):.0f}")
    else:
        print(f"  prod cache not found at {prod_root}")

    # Save tabular
    a.to_csv(OUT_DIR / "training_data_face_size_by_split_label.csv", index=False)
    b.to_csv(OUT_DIR / "training_data_face_size_by_method.csv", index=False)
    e.to_csv(OUT_DIR / "training_data_fake_method_face_size.csv", index=False)
    print(f"\n[crop-dist] saved CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
