"""Compute per-frame attributes on already-cached frames.

Walks _frame_cache/<group>/*.{png,jpg,jpeg}, computes:
  - skin_frac (BT.601 YCrCb mask, reused from crop_attribute_audit.py)
  - laplacian_var (sharpness)
  - luma_mean (brightness)
Writes outputs/per_frame_attrs.csv.

Resizes to 224x224 (BILINEAR) for consistency with what the model sees.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DIAG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DIAG_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis" / "score_distribution_2026-05-02"))
from crop_attribute_audit import skin_mask_fraction, luminance, laplacian_var  # noqa: E402

CACHE = DIAG_ROOT / "_frame_cache"
OUT = DIAG_ROOT / "outputs"
LOG = DIAG_ROOT / "run.log"

GROUPS = ["eval_viso_fake", "eval_real", "train_viso_fake", "train_viso_real"]


def per_frame_attrs(p: Path) -> dict:
    img = Image.open(p).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img)
    luma = luminance(arr)
    return {
        "skin_frac": skin_mask_fraction(arr),
        "laplacian_var": laplacian_var(luma),
        "luma_mean": float(luma.mean()),
    }


def main():
    t0 = time.time()
    with open(LOG, "a") as log:
        log.write(f"\n=== 01c_compute_attrs.py @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    rows = []
    for g in GROUPS:
        gdir = CACHE / g
        if not gdir.exists():
            print(f"[skip] no {gdir}", flush=True); continue
        files = [p for p in sorted(gdir.iterdir())
                 if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        label = "fake" if g.endswith("_fake") else "real"
        n_done = 0
        for p in files:
            try:
                attrs = per_frame_attrs(p)
            except Exception as e:
                print(f"  [skip] {p.name}: {e}", flush=True); continue
            attrs.update({
                "frame_path": str(p),
                "group": g,
                "label": label,
            })
            rows.append(attrs)
            n_done += 1
        print(f"[{g}] processed {n_done}/{len(files)} files (t={time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True, parents=True)
    out_csv = OUT / "per_frame_attrs.csv"
    df.to_csv(out_csv, index=False)
    counts = df["group"].value_counts().to_dict()
    print(f"[wrote] {out_csv} (n={len(df)}, by-group={counts})", flush=True)
    with open(LOG, "a") as log:
        log.write(f"[01c] wrote {out_csv} (n={len(df)}, by-group={counts}); elapsed={time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
