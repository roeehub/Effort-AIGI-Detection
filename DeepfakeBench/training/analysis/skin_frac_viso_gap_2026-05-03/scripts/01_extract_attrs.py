"""Step 1: Cache frames + extract per-frame image-quality attributes.

Three groups (target ~550 / ~550 / ~500):
  - eval_viso_fake : 550 viso-enhanced fakes from visomaster_enhanced_macro_dev CSV
  - eval_real      : 550 random rows of teams_real_all_dev CSV (seed=737)
  - train_viso_fake: 500 enhanced viso fakes sampled from
                     gs://visomaster-enhanced-face-cropped/samples/visomaster_*

We also pull a small auxiliary train_viso_real group (250) from the base bucket
(reals are paired with the base, not enhanced) so KS for {train_viso_fake vs
train_real} can run within the training distribution.

Reuses skin_mask_fraction() / luminance() from
analysis/score_distribution_2026-05-02/crop_attribute_audit.py.

Cache: _frame_cache/{group}/<safe_name>.png
Output: outputs/per_frame_attrs.csv
        outputs/_uri_lists/<group>.txt   (provenance)

Constraints:
  - Local Mac, n_jobs=1 in any sklearn/joblib (none used here).
  - gsutil parallel cap = 8 worker threads.
  - Tolerate partial download failures; need >=400 frames per primary group.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Reuse upstream helpers
SCRIPT_DIR = Path(__file__).resolve().parent
DIAG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = DIAG_ROOT.parent.parent  # .../training
sys.path.insert(0, str(REPO_ROOT / "analysis" / "score_distribution_2026-05-02"))
from crop_attribute_audit import skin_mask_fraction, luminance, laplacian_var  # noqa: E402

CACHE = DIAG_ROOT / "_frame_cache"
OUT = DIAG_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
URI_LISTS = OUT / "_uri_lists"
URI_LISTS.mkdir(parents=True, exist_ok=True)

EVAL_FAKE_CSV = (
    REPO_ROOT
    / "analysis/score_distribution_2026-05-02/raw_reports"
    / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"
)
EVAL_REAL_CSV = (
    REPO_ROOT
    / "analysis/score_distribution_2026-05-02/raw_reports"
    / "teams_real_all_dev_p8a_reference_step5000_frames_report.csv"
)

TRAIN_VISO_ENHANCED_PREFIX = "gs://visomaster-enhanced-face-cropped/samples/"
TRAIN_VISO_BASE_PREFIX = "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/"

SEED = 737
PARALLEL_WORKERS = 8

TARGETS = {
    "eval_viso_fake":  550,
    "eval_real":       550,
    "train_viso_fake": 500,
    "train_viso_real": 250,  # small auxiliary group, train side
}


def safe_name(uri: str) -> str:
    return uri.replace("gs://", "").replace("/", "__")


def fetch_one(uri: str, dest: Path) -> Tuple[str, bool]:
    if dest.exists() and dest.stat().st_size > 0:
        return (uri, True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["gsutil", "-q", "cp", uri, str(dest)],
            capture_output=True, timeout=60,
        )
        ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
        return (uri, ok)
    except Exception:
        return (uri, False)


def parallel_fetch(uri_list: List[str], group_dir: Path, group: str, log) -> List[Path]:
    print(f"[fetch:{group}] {len(uri_list)} URIs into {group_dir}", file=log, flush=True)
    print(f"[fetch:{group}] {len(uri_list)} URIs", flush=True)
    paths: List[Path] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futs = {}
        for uri in uri_list:
            dest = group_dir / safe_name(uri)
            futs[ex.submit(fetch_one, uri, dest)] = dest
        ok = 0
        for i, fut in enumerate(as_completed(futs)):
            uri, success = fut.result()
            if success:
                paths.append(futs[fut])
                ok += 1
            if (i + 1) % 50 == 0:
                print(f"  [fetch:{group}] {i+1}/{len(uri_list)} ok={ok}", flush=True)
    print(f"[fetch:{group}] done; {len(paths)}/{len(uri_list)} ok", file=log, flush=True)
    print(f"[fetch:{group}] done; {len(paths)}/{len(uri_list)} ok", flush=True)
    return paths


def list_train_viso_enhanced_dirs() -> List[str]:
    """List all sample dirs under enhanced bucket."""
    r = subprocess.run(
        ["gsutil", "ls", TRAIN_VISO_ENHANCED_PREFIX],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        print(f"[list] WARN gsutil ls returned {r.returncode}: {r.stderr[:200]}", flush=True)
        return []
    return [
        l.strip()
        for l in r.stdout.split("\n")
        if l.strip().endswith("/") and "/visomaster_" in l
    ]


def list_train_viso_base_dirs() -> List[str]:
    r = subprocess.run(
        ["gsutil", "ls", TRAIN_VISO_BASE_PREFIX],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        print(f"[list] WARN gsutil ls returned {r.returncode}: {r.stderr[:200]}", flush=True)
        return []
    # only visomaster_* dirs (the bucket also has deeplive)
    return [
        l.strip()
        for l in r.stdout.split("\n")
        if l.strip().endswith("/") and "/visomaster_" in l
    ]


def build_train_uris(rng: random.Random, group: str, target_n: int) -> List[str]:
    """Build a URI list of frame_NNNN.png by random sampling sample dirs.

    Each sample dir holds frames/fake/frame_0001..0010.png (typically).
    We sample a unique (sample_dir, frame_index) without repeats.
    """
    if group == "train_viso_fake":
        dirs = list_train_viso_enhanced_dirs()
        sub = "frames/fake/"
    elif group == "train_viso_real":
        dirs = list_train_viso_base_dirs()
        sub = "frames/real/"
    else:
        raise ValueError(group)
    rng.shuffle(dirs)
    # We assume up to 10 frames per sample dir (filenames frame_0001..frame_0010).
    # To keep gsutil ls cost low, just guess frame indices and rely on
    # parallel_fetch to silently drop misses; we oversample by 1.6x.
    candidates: List[str] = []
    for d in dirs:
        for k in range(1, 11):
            candidates.append(f"{d}{sub}frame_{k:04d}.png")
    rng.shuffle(candidates)
    # Oversample 1.5x to absorb missing frames (some dirs may have <10)
    return candidates[: int(target_n * 1.6)]


def per_frame_attrs(p: Path) -> Dict[str, float]:
    arr = np.asarray(Image.open(p).convert("RGB"))
    if arr.shape[0] != 224 or arr.shape[1] != 224:
        # resize using PIL for consistency
        img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img)
    luma = luminance(arr)
    return {
        "skin_frac": skin_mask_fraction(arr),
        "laplacian_var": laplacian_var(luma),
        "luma_mean": float(luma.mean()),
    }


def main():
    log_path = DIAG_ROOT / "run.log"
    log = open(log_path, "a")
    print(f"\n=== 01_extract_attrs.py @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", file=log)
    print(f"=== 01_extract_attrs.py @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    rng = random.Random(SEED)

    # ----- Build URI lists -----
    print("[build] eval_viso_fake from CSV", file=log)
    eval_fake_df = pd.read_csv(EVAL_FAKE_CSV)
    eval_fake_uris = eval_fake_df["frame_path"].tolist()
    if len(eval_fake_uris) > TARGETS["eval_viso_fake"]:
        eval_fake_uris = eval_fake_uris[: TARGETS["eval_viso_fake"]]

    print("[build] eval_real (sampled) from CSV", file=log)
    eval_real_df = pd.read_csv(EVAL_REAL_CSV)
    n_target = min(TARGETS["eval_real"], len(eval_real_df))
    eval_real_df = eval_real_df.sample(n=n_target, random_state=SEED)
    eval_real_uris = eval_real_df["frame_path"].tolist()

    print("[build] train_viso_fake from enhanced bucket", file=log)
    train_fake_uris = build_train_uris(rng, "train_viso_fake", TARGETS["train_viso_fake"])
    print("[build] train_viso_real from base bucket", file=log)
    train_real_uris = build_train_uris(rng, "train_viso_real", TARGETS["train_viso_real"])

    groups = {
        "eval_viso_fake":  eval_fake_uris,
        "eval_real":       eval_real_uris,
        "train_viso_fake": train_fake_uris,
        "train_viso_real": train_real_uris,
    }

    # Persist URI provenance
    for g, uris in groups.items():
        (URI_LISTS / f"{g}.txt").write_text("\n".join(uris) + "\n")
        print(f"  [{g}] {len(uris)} URIs queued", flush=True)
        print(f"  [{g}] {len(uris)} URIs queued", file=log)

    # ----- Fetch in parallel per group -----
    fetched: Dict[str, List[Path]] = {}
    for g, uris in groups.items():
        gdir = CACHE / g
        gdir.mkdir(parents=True, exist_ok=True)
        fetched[g] = parallel_fetch(uris, gdir, g, log)

    # ----- Compute attributes -----
    print("[attrs] computing per-frame", flush=True)
    rows = []
    for g, paths in fetched.items():
        label = "fake" if g.endswith("_fake") else "real"
        for p in paths:
            try:
                attrs = per_frame_attrs(p)
            except Exception as e:
                print(f"  [attrs] skip {p.name}: {e}", file=log)
                continue
            attrs.update({
                "frame_path": str(p),
                "group": g,
                "label": label,
            })
            rows.append(attrs)
    df = pd.DataFrame(rows)
    out_csv = OUT / "per_frame_attrs.csv"
    df.to_csv(out_csv, index=False)
    counts = df["group"].value_counts().to_dict()
    print(f"[attrs] wrote {out_csv} (n={len(df)}, by-group={counts})", flush=True)
    print(f"[attrs] wrote {out_csv} (n={len(df)}, by-group={counts})", file=log)

    # Sanity gate
    primary = ["eval_viso_fake", "eval_real", "train_viso_fake"]
    insufficient = [g for g in primary if counts.get(g, 0) < 400]
    if insufficient:
        print(f"[gate] WARN insufficient frames in: {insufficient}", flush=True)
        print(f"[gate] WARN insufficient frames in: {insufficient}", file=log)

    log.close()


if __name__ == "__main__":
    main()
