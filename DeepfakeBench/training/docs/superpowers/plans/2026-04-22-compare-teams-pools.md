# compare_teams_pools.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file local diagnostic that downloads representative samples from `teams_ood` and `proper_visomaster_teams` pools, runs five passes (image stats, face geometry, model embeddings, hard-sample selection, HTML report), and produces a self-contained HTML report explaining the distributional differences.

**Architecture:** One file `analysis/compare_teams_pools.py` with eight banner-marked sections (CLI, sampling, pass 1–5, main). Sister test file `tests/test_compare_teams_pools.py` for the logic-bearing units (sampling determinism, KS calc, hard-sample selection, parquet schema). Reuses the checkpoint-load pattern from `retro_score_value_composite.py:94-145`.

**Tech Stack:** Python 3, torch (CPU), numpy, pandas, pyarrow, matplotlib, opencv-python, Pillow, mediapipe, umap-learn, google-cloud-storage. All pip-only.

**Spec:** `docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md`

---

### Task 1: Module skeleton + CLI scaffolding + paths

**Files:**
- Create: `analysis/__init__.py` (empty)
- Create: `analysis/compare_teams_pools.py`
- Create: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

`tests/test_compare_teams_pools.py`:
```python
"""Tests for analysis/compare_teams_pools.py — local Teams-pool diagnostic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analysis" / "compare_teams_pools.py"


def test_cli_help_runs():
    """--help should exit 0 and mention all required CLI flags."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    for flag in ("--checkpoint", "--videos-per-group", "--frames-per-video",
                 "--output-dir", "--skip-pass", "--cache-dir", "--seed"):
        assert flag in out.stdout, f"missing flag {flag}"


def test_output_dir_layout(tmp_path):
    """Importing the module + calling _make_run_dir produces a timestamped subdir."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    run_dir = ctp._make_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    # ISO-ish timestamp prefix
    assert len(run_dir.name) >= 15
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -30
```
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.compare_teams_pools'`.

- [ ] **Step 3: Write minimal implementation**

`analysis/__init__.py`:
```python
```

`analysis/compare_teams_pools.py`:
```python
"""
compare_teams_pools.py — local diagnostic for distributional differences
between the `teams_ood` OOD pool and the `proper_visomaster_teams` training
pool. Spec: docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md.

Usage:
    python analysis/compare_teams_pools.py \\
        --checkpoint gs://.../value_composite_*.pth \\
        --videos-per-group 150 --frames-per-video 8 \\
        --output-dir scratch/teams_pool_diff
"""
from __future__ import annotations

# === Section 1: CLI + paths =================================================

import argparse
import datetime as _dt
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("compare_teams_pools")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "teams_pool_diff"
DEFAULT_OUTPUT_DIR = Path("scratch") / "teams_pool_diff"
DEFAULT_CHECKPOINT = (
    "gs://training-job-outputs/phase2r13_experiments/w92amaaa/"
    "value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth"
)
DEFAULT_SEED = 737
KNOWN_PASSES = ("stats", "geometry", "model", "gallery")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_teams_pools",
        description="Diagnose distributional differences between teams_ood "
                    "and proper_visomaster_teams pools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="GCS URI of the checkpoint to use for embeddings/confidence.")
    p.add_argument("--videos-per-group", type=int, default=150)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--skip-pass", action="append", default=[],
                   choices=KNOWN_PASSES,
                   help="Skip a pass by name (repeatable).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _make_run_dir(output_dir: Path) -> Path:
    """Create a timestamped subdir under output_dir and return it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "thumbnails").mkdir()
    return run_dir


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# === Section 8: main() ======================================================


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.log_level)
    run_dir = _make_run_dir(args.output_dir)
    logger.info("Run dir: %s", run_dir)
    logger.info("Skipped passes: %s", args.skip_pass or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -20
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/__init__.py analysis/compare_teams_pools.py tests/test_compare_teams_pools.py docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md docs/superpowers/plans/2026-04-22-compare-teams-pools.md
git commit -m "$(cat <<'EOF'
Scaffold compare_teams_pools.py diagnostic + spec/plan

Single-file local diagnostic that will surface distributional differences
between teams_ood and proper_visomaster_teams pools across image stats,
face geometry, and model embedding/confidence — to inform packet-4
training-regime adjustments.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Sampling + GCS download (cached)

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 2)
- Modify: `tests/test_compare_teams_pools.py` (add sampling tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_sample_video_ids_is_deterministic():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    pool = [f"vid_{i:04d}" for i in range(1000)]
    a = ctp._sample_video_ids(pool, n=50, seed=737)
    b = ctp._sample_video_ids(pool, n=50, seed=737)
    c = ctp._sample_video_ids(pool, n=50, seed=738)
    assert a == b
    assert a != c
    assert len(set(a)) == 50
    assert all(v in pool for v in a)


def test_sample_video_ids_handles_undersized_pool():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    pool = [f"vid_{i:02d}" for i in range(10)]
    out = ctp._sample_video_ids(pool, n=50, seed=737)
    assert len(out) == 10
    assert sorted(out) == sorted(pool)


def test_pool_definitions_complete():
    """All 4 groups defined with bucket + label."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    groups = ctp.GROUP_DEFINITIONS
    expected = {
        "teams_ood_real", "teams_ood_fake",
        "proper_visomaster_teams_real", "proper_visomaster_teams_fake",
    }
    assert set(groups.keys()) == expected
    for g in groups.values():
        assert g.label in ("real", "fake")
        assert len(g.buckets) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compare_teams_pools.py -v -k "sample or pool_def" 2>&1 | head -30
```
Expected: 3 failures (`AttributeError: module ... has no attribute '_sample_video_ids'`).

- [ ] **Step 3: Write the implementation**

Insert after `_configure_logging` and before `# === Section 8: main()`:
```python
# === Section 2: Sampling + GCS download (cached) ============================

import dataclasses
import hashlib
import io
import random
from typing import Iterable

# Lazy GCS import — only imported when actually downloading.
_GCS_CLIENT = None


@dataclasses.dataclass(frozen=True)
class GroupDef:
    name: str
    label: str  # "real" | "fake"
    buckets: tuple[str, ...]  # gs:// URIs to the *parent* directory containing
                              # video subfolders, each holding cropped frames.


GROUP_DEFINITIONS: dict[str, GroupDef] = {
    "teams_ood_real": GroupDef(
        name="teams_ood_real",
        label="real",
        buckets=(
            "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/"
            "samples",
        ),
    ),
    "teams_ood_fake": GroupDef(
        name="teams_ood_fake",
        label="fake",
        buckets=(
            "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/"
            "samples",
        ),
    ),
    "proper_visomaster_teams_real": GroupDef(
        name="proper_visomaster_teams_real",
        label="real",
        buckets=(
            "gs://hdtf_visomaster_cropped_frames_teams",
            "gs://quickclips_visomaster_cropped_frames_teams",
        ),
    ),
    "proper_visomaster_teams_fake": GroupDef(
        name="proper_visomaster_teams_fake",
        label="fake",
        buckets=(
            "gs://hdtf_visomaster_cropped_frames_teams",
            "gs://quickclips_visomaster_cropped_frames_teams",
        ),
    ),
}


def _sample_video_ids(pool: Iterable[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of n video ids from pool. Returns sorted IDs.

    If pool has fewer than n items, returns the whole pool sorted.
    """
    items = sorted(set(pool))
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    return sorted(rng.sample(items, n))


def _gcs_client():
    """Lazy GCS client; surfaces a helpful auth message on failure."""
    global _GCS_CLIENT
    if _GCS_CLIENT is not None:
        return _GCS_CLIENT
    try:
        from google.cloud import storage  # noqa: WPS433
        _GCS_CLIENT = storage.Client()
        return _GCS_CLIENT
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "GCS client init failed: %s. "
            "Run `gcloud auth application-default login` and retry.",
            exc,
        )
        raise SystemExit(2)


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"not a gs:// URI: {uri}"
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix


def _list_video_dirs(gs_root: str, label: str, max_blobs: int = 50000) -> list[str]:
    """List video subdirectories under <gs_root>/<...>/frames/<label>/.

    Returns deterministic list of video folder names (the folder right above
    the per-frame jpgs). The exact bucket layout for `live-deepfake-methods`
    is `<root>/<capture>/frames/<label>/<frame>.jpg`, while `proper_visomaster`
    is `<bucket_root>/<video_id>/<frame>.jpg` with no per-label subdir
    (real/fake split is determined by the bucket itself).
    """
    client = _gcs_client()
    bucket_name, prefix = _parse_gs_uri(gs_root)
    bucket = client.bucket(bucket_name)
    seen: set[str] = set()
    iterator = client.list_blobs(bucket, prefix=prefix, max_results=max_blobs)
    for blob in iterator:
        # Path: <prefix>/.../<video_id>/<frame>.jpg
        # For deeplive: prefix/<capture>/frames/<label>/<vid>/<frame>.jpg
        # For visomaster: prefix/<vid>/<frame>.jpg
        rel = blob.name[len(prefix):].lstrip("/")
        parts = rel.split("/")
        # Filter by label only when the bucket has a /frames/<label>/ marker.
        if "frames" in parts:
            i = parts.index("frames")
            if i + 1 < len(parts) and parts[i + 1] == label and i + 2 < len(parts):
                seen.add("/".join(parts[: i + 3]))
        else:
            if len(parts) >= 2:
                seen.add(parts[0])
    return sorted(seen)


def _download_frames_for_video(
    gs_root: str, video_dir: str, n_frames: int, cache_dir: Path,
) -> list[Path]:
    """Download up to n_frames evenly-spaced frames from a single video subdir.

    Returns local file paths in order. Idempotent: skips downloads if the
    cache already has the same file count for this video.
    """
    client = _gcs_client()
    bucket_name, prefix = _parse_gs_uri(gs_root)
    full_prefix = f"{prefix}/{video_dir}".rstrip("/")
    cache_key = hashlib.blake2b(f"{gs_root}|{video_dir}".encode(), digest_size=8).hexdigest()
    local_dir = cache_dir / cache_key
    local_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(local_dir.glob("*.jpg"))
    if len(existing) >= n_frames:
        return existing[:n_frames]
    bucket = client.bucket(bucket_name)
    blobs = sorted(
        b for b in client.list_blobs(bucket, prefix=full_prefix + "/")
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not blobs:
        return []
    # Evenly spaced indices.
    if len(blobs) <= n_frames:
        chosen = blobs
    else:
        step = len(blobs) / n_frames
        chosen = [blobs[int(i * step)] for i in range(n_frames)]
    out: list[Path] = []
    for blob in chosen:
        ext = Path(blob.name).suffix or ".jpg"
        local_path = local_dir / f"{Path(blob.name).stem}{ext}"
        if not local_path.exists():
            blob.download_to_filename(str(local_path))
        out.append(local_path)
    return out


def collect_frames(
    *, videos_per_group: int, frames_per_video: int, cache_dir: Path, seed: int,
) -> "pd.DataFrame":
    """Top-level: enumerate, sample, download, return a DataFrame.

    Columns: group, label, video_id, frame_path
    """
    import pandas as pd  # local import keeps --help fast
    rows: list[dict] = []
    for gname, gdef in GROUP_DEFINITIONS.items():
        all_videos: list[tuple[str, str]] = []  # (gs_root, video_dir)
        for gs_root in gdef.buckets:
            try:
                vids = _list_video_dirs(gs_root, gdef.label)
            except Exception as exc:  # noqa: BLE001
                logger.warning("listing %s/%s failed: %s", gs_root, gdef.label, exc)
                continue
            all_videos.extend((gs_root, v) for v in vids)
        keys = [f"{r}|{v}" for r, v in all_videos]
        chosen_keys = set(_sample_video_ids(keys, n=videos_per_group, seed=seed))
        for (gs_root, video_dir), key in zip(all_videos, keys):
            if key not in chosen_keys:
                continue
            paths = _download_frames_for_video(
                gs_root, video_dir, frames_per_video, cache_dir,
            )
            for p in paths:
                rows.append({
                    "group": gname,
                    "label": gdef.label,
                    "video_id": video_dir,
                    "frame_path": str(p),
                })
        logger.info("group=%s sampled %d videos", gname,
                    len({r["video_id"] for r in rows if r["group"] == gname}))
    return pd.DataFrame(rows)


```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -30
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add cached GCS sampler

Deterministic per-group video sampling (seed-stable, undersized-pool safe)
plus a GCS lister/downloader keyed by blake2b(gs_root|video_dir) for the
local cache. Lazy GCS client surfaces auth instructions on failure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Pass 1 — image stats

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 3)
- Modify: `tests/test_compare_teams_pools.py` (add stats tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_compute_image_stats_synthetic():
    """A pure-grey 100x100 frame: brightness=0.5, contrast=0, sharpness=0."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.full((100, 100, 3), 128, dtype=np.uint8)  # mid-grey
    stats = ctp._image_stats_from_array(img, file_size_bytes=10000)
    assert 0.45 < stats["brightness"] < 0.55
    assert stats["contrast"] < 0.01
    assert stats["sharpness"] < 0.5
    assert stats["resolution_w"] == 100
    assert stats["resolution_h"] == 100
    assert stats["bytes_per_pixel"] == pytest.approx(1.0, abs=1e-6)


def test_pairwise_ks_distance_basic():
    """KS-distance between identical samples is 0; between disjoint is 1."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert ctp._ks_distance(a, b) == 0.0
    assert ctp._ks_distance(a, c) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compare_teams_pools.py -v -k "image_stats or ks" 2>&1 | head -20
```
Expected: 2 failures.

- [ ] **Step 3: Write the implementation**

Insert after Section 2 and before `# === Section 8: main()`:
```python
# === Section 3: Pass 1 — image stats ========================================


def _colorfulness(img_rgb):
    """Hasler-Süsstrunk colorfulness metric on a HxWx3 uint8 RGB image."""
    import numpy as np
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma_rgyb = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mu_rgyb = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return sigma_rgyb + 0.3 * mu_rgyb


def _image_stats_from_array(img_rgb, *, file_size_bytes: int) -> dict:
    """Per-frame low-level stats. Input: HxWx3 uint8 RGB array."""
    import cv2
    import numpy as np
    h, w = img_rgb.shape[:2]
    luma = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    laplacian = cv2.Laplacian(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F,
    )
    return {
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "sharpness": float(laplacian.var()),
        "colorfulness": float(_colorfulness(img_rgb)),
        "saturation": float(sat.mean()),
        "resolution_w": int(w),
        "resolution_h": int(h),
        "bytes_per_pixel": float(file_size_bytes / max(1, w * h)),
        "warmth": float(img_rgb[..., 0].mean() - img_rgb[..., 2].mean()),
    }


def _ks_distance(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance (max |F_a(x) - F_b(x)|)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def run_pass_stats(frames_df, *, run_dir: Path) -> dict:
    """Pass 1: per-frame low-level stats + per-group aggregates + KS pairs."""
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    rows = []
    for _, row in frames_df.iterrows():
        path = Path(row["frame_path"])
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
            s = _image_stats_from_array(img, file_size_bytes=path.stat().st_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("stats failed for %s: %s", path, exc)
            continue
        s.update({k: row[k] for k in ("group", "label", "video_id", "frame_path")})
        rows.append(s)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns
                    if c not in ("group", "label", "video_id", "frame_path")]
    agg: dict[str, dict] = {}
    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        agg[g] = {f: {
            "mean": float(sub[f].mean()),
            "median": float(sub[f].median()),
            "p5": float(sub[f].quantile(0.05)),
            "p95": float(sub[f].quantile(0.95)),
            "n": int(sub[f].count()),
        } for f in feature_cols}
    pairs = (
        ("teams_ood_real", "proper_visomaster_teams_real"),
        ("teams_ood_fake", "proper_visomaster_teams_fake"),
    )
    ks: dict[str, dict] = {}
    for left, right in pairs:
        if left not in df["group"].unique() or right not in df["group"].unique():
            continue
        ks_pair: dict = {}
        for f in feature_cols:
            ks_pair[f] = _ks_distance(df[df["group"] == left][f].to_numpy(),
                                      df[df["group"] == right][f].to_numpy())
        ks[f"{left}__vs__{right}"] = ks_pair
    out = {"per_group": agg, "ks": ks, "feature_cols": feature_cols,
           "n_frames": int(len(df))}
    (run_dir / "stats_pass_stats.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(run_dir / "raw_stats.parquet", index=False)
    return out


```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -20
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add Pass 1 (low-level image stats + KS-distance)

Per-frame brightness/contrast/sharpness/colorfulness/saturation/resolution/
bytes-per-pixel/warmth, aggregated per group, with two-sample KS distance
across the cross-pool same-label pairs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Pass 2 — face geometry (mediapipe)

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 4)
- Modify: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_face_geometry_no_face_returns_nan_row():
    """A pure-grey image has no face — should return face_detected=False."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    geom = ctp._face_geometry_from_array(img)
    assert geom["face_detected"] is False
    assert geom["face_bbox_area_ratio"] != geom["face_bbox_area_ratio"]  # NaN
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_compare_teams_pools.py -v -k "face_geometry" 2>&1 | head -20
```
Expected: 1 failure.

- [ ] **Step 3: Write the implementation**

Insert after Section 3:
```python
# === Section 4: Pass 2 — face geometry (mediapipe) ==========================

# MediaPipe face_mesh has 468 landmarks. We use a small canonical subset for pose.
# Landmark indices: nose_tip=1, chin=152, left_eye_outer=33, right_eye_outer=263,
# left_mouth=61, right_mouth=291.
_POSE_INDICES = (1, 152, 33, 263, 61, 291)


def _face_geometry_from_array(img_rgb) -> dict:
    """Per-frame face geometry via MediaPipe face_mesh."""
    import math
    import numpy as np
    nan = float("nan")
    out = {
        "face_detected": False,
        "face_bbox_area_ratio": nan,
        "inter_eye_distance_ratio": nan,
        "yaw_deg": nan,
        "pitch_deg": nan,
        "roll_deg": nan,
        "landmark_confidence": nan,
    }
    try:
        import mediapipe as mp  # noqa: WPS433
    except ImportError:
        logger.warning("mediapipe not installed — geometry pass returns no-face rows")
        return out
    h, w = img_rgb.shape[:2]
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
    )
    try:
        result = mesh.process(img_rgb)
    finally:
        mesh.close()
    if not result.multi_face_landmarks:
        return out
    lms = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bbox_area = float(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))
    crop_area = float(w * h)
    left_eye = pts[33]
    right_eye = pts[263]
    ied = float(np.linalg.norm(right_eye - left_eye))
    # Roll = angle of the eye line vs. horizontal.
    roll = math.degrees(math.atan2(
        float(right_eye[1] - left_eye[1]),
        float(right_eye[0] - left_eye[0]),
    ))
    # Yaw proxy: x-asymmetry of nose vs. eye midpoint.
    nose = pts[1]
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2)
    yaw = math.degrees(math.atan2(
        float(nose[0] - eye_mid_x),
        max(1e-3, ied / 2),
    ))
    # Pitch proxy: y-distance nose-to-eye-mid normalized by IED.
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2)
    pitch = math.degrees(math.atan2(
        float(nose[1] - eye_mid_y),
        max(1e-3, ied / 2),
    ))
    out.update({
        "face_detected": True,
        "face_bbox_area_ratio": bbox_area / max(1.0, crop_area),
        "inter_eye_distance_ratio": ied / max(1.0, float(w)),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "landmark_confidence": 1.0,  # face_mesh doesn't expose per-face score
    })
    return out


def run_pass_geometry(frames_df, *, run_dir: Path) -> dict:
    """Pass 2: per-frame face geometry + per-group aggregates + KS pairs."""
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    rows = []
    for _, row in frames_df.iterrows():
        path = Path(row["frame_path"])
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
            g = _face_geometry_from_array(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("geometry failed for %s: %s", path, exc)
            continue
        g.update({k: row[k] for k in ("group", "label", "video_id", "frame_path")})
        rows.append(g)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns
                    if c not in ("group", "label", "video_id", "frame_path",
                                 "face_detected")]
    agg: dict[str, dict] = {}
    for grp in sorted(df["group"].unique()):
        sub = df[df["group"] == grp]
        agg[grp] = {
            "face_detected_rate": float(sub["face_detected"].mean()),
            **{f: {
                "mean": float(sub[f].mean()),
                "median": float(sub[f].median()),
                "p5": float(sub[f].quantile(0.05)),
                "p95": float(sub[f].quantile(0.95)),
                "n": int(sub[f].count()),
            } for f in feature_cols},
        }
    pairs = (
        ("teams_ood_real", "proper_visomaster_teams_real"),
        ("teams_ood_fake", "proper_visomaster_teams_fake"),
    )
    ks: dict[str, dict] = {}
    for left, right in pairs:
        if left not in df["group"].unique() or right not in df["group"].unique():
            continue
        ks[f"{left}__vs__{right}"] = {
            f: _ks_distance(df[df["group"] == left][f].to_numpy(),
                            df[df["group"] == right][f].to_numpy())
            for f in feature_cols
        }
    out = {"per_group": agg, "ks": ks, "feature_cols": feature_cols,
           "n_frames": int(len(df))}
    (run_dir / "stats_pass_geometry.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(run_dir / "raw_geometry.parquet", index=False)
    return out


```

- [ ] **Step 4: Install mediapipe and run tests**

```bash
pip install mediapipe 2>&1 | tail -3
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -20
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add Pass 2 (face geometry via mediapipe)

Per-frame face_mesh detection extracting bbox-area ratio, inter-eye
distance, yaw/pitch/roll from a small landmark subset. Returns NaN row
on no-face; aggregates include face_detected_rate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Pass 3 — model embeddings + confidence

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 5)
- Modify: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_preprocess_image_shape():
    """Preprocess should output a (3, 224, 224) tensor with ImageNet stats."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    t = ctp._preprocess_image(img)
    assert tuple(t.shape) == (3, 224, 224)
    # ImageNet-normalized data centered around 0
    assert -3.0 < float(t.mean()) < 3.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_compare_teams_pools.py -v -k "preprocess" 2>&1 | head -20
```
Expected: 1 failure.

- [ ] **Step 3: Write the implementation**

Insert after Section 4:
```python
# === Section 5: Pass 3 — model embeddings + confidence ======================

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_image(img_rgb):
    """Resize to 224x224 and ImageNet-normalize. Returns (3, 224, 224) tensor."""
    import cv2
    import numpy as np
    import torch
    img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    for c in range(3):
        arr[..., c] = (arr[..., c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _download_checkpoint_if_gcs(uri: str, cache_dir: Path) -> Path:
    """If uri is a gs:// path, download to cache_dir and return local path."""
    if not uri.startswith("gs://"):
        return Path(uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(uri).name
    if local.exists():
        logger.info("checkpoint cache hit: %s", local)
        return local
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(local))
    logger.info("checkpoint downloaded: %s", local)
    return local


def _load_model_for_inference(ckpt_local: Path):
    """Load checkpoint via the same path retro_score_value_composite uses.

    Returns (model, embedding_fn). embedding_fn takes a (B, 3, 224, 224)
    tensor and returns (embeddings, fake_probs).
    """
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    # Resolve REPO_ROOT relative to this file (analysis/ is one dir down).
    from retro_score_value_composite import _load_state_dict_into_model  # noqa: WPS433
    saved = torch.load(ckpt_local, map_location="cpu", weights_only=False)
    saved_cfg = saved.get("config") if isinstance(saved, dict) else None
    if saved_cfg is None:
        raise RuntimeError(
            "Checkpoint has no embedded `config` — cannot reconstruct backbone. "
            "Diagnostic requires checkpoints saved by the live training pipeline."
        )
    from detectors import DETECTOR  # noqa: WPS433
    model_cls = DETECTOR[saved_cfg["model_name"]]
    model = model_cls(saved_cfg)
    _load_state_dict_into_model(model, saved, saved_cfg, logger)
    model.eval()

    @torch.no_grad()
    def embedding_fn(batch):
        out = model({"image": batch.to("cpu")}, inference=True)
        # Convention in this repo: out["feat"] is the penultimate embedding,
        # out["prob"] is the sigmoid fake-probability.
        feat = out.get("feat")
        if feat is None:
            feat = out.get("embedding")
        prob = out.get("prob")
        if prob is None:
            logits = out.get("cls", out.get("logits"))
            prob = torch.sigmoid(logits[:, -1] if logits.ndim == 2 else logits)
        return feat.cpu().numpy(), prob.cpu().numpy()

    return model, embedding_fn


def run_pass_model(
    frames_df, *, run_dir: Path, checkpoint_uri: str, cache_dir: Path,
    batch_size: int = 16,
) -> dict:
    """Pass 3: load model, score every frame, write embeddings + probs."""
    import json
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    ckpt_local = _download_checkpoint_if_gcs(checkpoint_uri, cache_dir / "checkpoints")
    _, embedding_fn = _load_model_for_inference(ckpt_local)

    paths = frames_df["frame_path"].tolist()
    n = len(paths)
    embeddings: list = []
    probs: list = []
    valid_idx: list[int] = []
    batch: list = []
    batch_idx: list[int] = []
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
            t = _preprocess_image(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preprocess failed for %s: %s", path, exc)
            continue
        batch.append(t)
        batch_idx.append(i)
        if len(batch) == batch_size:
            stacked = torch.stack(batch, dim=0)
            emb, pb = embedding_fn(stacked)
            embeddings.append(emb)
            probs.append(pb)
            valid_idx.extend(batch_idx)
            batch.clear()
            batch_idx.clear()
            logger.info("model pass: %d/%d frames scored", len(valid_idx), n)
    if batch:
        stacked = torch.stack(batch, dim=0)
        emb, pb = embedding_fn(stacked)
        embeddings.append(emb)
        probs.append(pb)
        valid_idx.extend(batch_idx)
    if not embeddings:
        out = {"per_group": {}, "centroid_distances": {}, "n_frames": 0}
        (run_dir / "stats_pass_model.json").write_text(json.dumps(out, indent=2))
        return out
    emb_arr = np.concatenate(embeddings, axis=0)
    prob_arr = np.concatenate(probs, axis=0)
    df = frames_df.iloc[valid_idx].reset_index(drop=True).copy()
    df["fake_prob"] = prob_arr
    for d in range(emb_arr.shape[1]):
        df[f"emb_{d:03d}"] = emb_arr[:, d]
    df.to_parquet(run_dir / "raw_scores.parquet", index=False)

    per_group: dict[str, dict] = {}
    centroids: dict[str, np.ndarray] = {}
    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        emb = sub[[c for c in df.columns if c.startswith("emb_")]].to_numpy()
        centroids[g] = emb.mean(axis=0)
        per_group[g] = {
            "fake_prob_mean": float(sub["fake_prob"].mean()),
            "fake_prob_median": float(sub["fake_prob"].median()),
            "embedding_dispersion_l2": float(
                np.linalg.norm(emb - centroids[g], axis=1).mean()
            ),
            "n": int(len(sub)),
        }
    centroid_distances: dict[str, float] = {}
    keys = sorted(centroids.keys())
    for i, gi in enumerate(keys):
        for gj in keys[i + 1:]:
            ci, cj = centroids[gi], centroids[gj]
            cos = float(
                np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-12)
            )
            centroid_distances[f"{gi}__{gj}"] = 1.0 - cos
    out = {
        "per_group": per_group,
        "centroid_distances": centroid_distances,
        "n_frames": int(len(df)),
        "embedding_dim": int(emb_arr.shape[1]),
    }
    (run_dir / "stats_pass_model.json").write_text(json.dumps(out, indent=2))
    return out


```

- [ ] **Step 4: Run preprocess test**

```bash
pytest tests/test_compare_teams_pools.py -v -k "preprocess" 2>&1 | head -20
```
Expected: passes (other model tests are integration-only since we don't have a checkpoint locally).

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add Pass 3 (model embeddings + fake-prob)

Reuses _load_state_dict_into_model from retro_score_value_composite for
the checkpoint-load path. CPU inference in batches of 16; emits
raw_scores.parquet with one row per frame plus 512d embedding cols.
Per-group centroid + cosine-distance summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Pass 4 — hard-sample selection

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 6)
- Modify: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_hard_sample_selection_buckets():
    """Selection produces 3 buckets (wrong/uncertain/right) per group."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import pandas as pd
    rng_rows = []
    for g in ("teams_ood_real", "proper_visomaster_teams_fake"):
        label = "real" if "_real" in g else "fake"
        target = 0.0 if label == "real" else 1.0
        for i in range(200):
            rng_rows.append({
                "group": g, "label": label,
                "video_id": f"v{i:03d}", "frame_path": f"/tmp/{g}_{i}.jpg",
                "fake_prob": float((i % 100) / 100.0),
                "_target": target,
            })
    df = pd.DataFrame(rng_rows).drop(columns="_target")
    sel = ctp._select_hard_samples(df, per_bucket=5, seed=737)
    for g in ("teams_ood_real", "proper_visomaster_teams_fake"):
        sub = sel[sel["group"] == g]
        for bucket in ("confidently_wrong", "uncertain", "confidently_right"):
            assert (sub["bucket"] == bucket).sum() <= 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_compare_teams_pools.py -v -k "hard_sample" 2>&1 | head -20
```
Expected: 1 failure (`AttributeError`).

- [ ] **Step 3: Write the implementation**

Insert after Section 5:
```python
# === Section 6: Pass 4 — hard-sample selection ==============================


def _select_hard_samples(scored_df, *, per_bucket: int = 30, seed: int = 737):
    """Select 3 buckets of frames per group: wrong, uncertain, right.

    `scored_df` must contain columns: group, label, video_id, frame_path, fake_prob.
    Returns a DataFrame with an added `bucket` column.
    """
    import numpy as np
    import pandas as pd
    out_rows = []
    rng = np.random.default_rng(seed)
    for g in sorted(scored_df["group"].unique()):
        sub = scored_df[scored_df["group"] == g].copy()
        label = sub["label"].iloc[0] if len(sub) else "real"
        target = 0.0 if label == "real" else 1.0
        sub["err"] = (sub["fake_prob"] - target).abs()
        # Confidently wrong: err > 0.9, take top per_bucket by err desc.
        wrong = sub[sub["err"] > 0.9].sort_values("err", ascending=False).head(per_bucket)
        wrong = wrong.assign(bucket="confidently_wrong")
        # Uncertain: prob in [0.4, 0.6], sort by closeness to 0.5.
        uncert_pool = sub[(sub["fake_prob"] >= 0.4) & (sub["fake_prob"] <= 0.6)].copy()
        uncert_pool["dist_to_half"] = (uncert_pool["fake_prob"] - 0.5).abs()
        uncert = uncert_pool.sort_values("dist_to_half").head(per_bucket).drop(columns="dist_to_half")
        uncert = uncert.assign(bucket="uncertain")
        # Confidently right: err < 0.1, random sample.
        right_pool = sub[sub["err"] < 0.1]
        if len(right_pool) > per_bucket:
            ix = rng.choice(len(right_pool), size=per_bucket, replace=False)
            right = right_pool.iloc[sorted(ix)]
        else:
            right = right_pool
        right = right.assign(bucket="confidently_right")
        out_rows.append(pd.concat([wrong, uncert, right], ignore_index=True))
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).drop(columns=["err"], errors="ignore")


def run_pass_gallery(scored_df, *, run_dir: Path, per_bucket: int = 30) -> dict:
    """Pass 4: select hard samples and copy thumbnails into run_dir/thumbnails/."""
    import json
    import shutil
    from PIL import Image
    if scored_df is None or len(scored_df) == 0:
        out = {"selected": 0, "note": "no scored frames — model pass skipped or failed"}
        (run_dir / "stats_pass_gallery.json").write_text(json.dumps(out, indent=2))
        return out
    sel = _select_hard_samples(scored_df, per_bucket=per_bucket, seed=DEFAULT_SEED)
    thumb_dir = run_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    rows = []
    for _, row in sel.iterrows():
        src = Path(row["frame_path"])
        dst_dir = thumb_dir / row["group"] / row["bucket"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{row['video_id']}__{src.stem}.jpg"
        try:
            with Image.open(src) as im:
                im.thumbnail((224, 224))
                im.convert("RGB").save(dst, format="JPEG", quality=85)
        except Exception as exc:  # noqa: BLE001
            logger.warning("thumbnail failed for %s: %s", src, exc)
            continue
        rows.append({
            "group": row["group"], "label": row["label"], "bucket": row["bucket"],
            "video_id": row["video_id"], "fake_prob": float(row["fake_prob"]),
            "thumbnail_rel_path": str(dst.relative_to(run_dir)),
        })
    out = {"selected": len(rows), "rows": rows, "per_bucket": per_bucket}
    (run_dir / "stats_pass_gallery.json").write_text(json.dumps(out, indent=2))
    return out


```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -25
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add Pass 4 (hard-sample selection + thumbnails)

Selects 3 buckets per group (confidently wrong, uncertain, confidently
right) at per_bucket=30 each (90 per group, 360 total). Writes 224px
thumbnails into run_dir/thumbnails/<group>/<bucket>/ for the gallery.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Pass 5 — HTML report

**Files:**
- Modify: `analysis/compare_teams_pools.py` (add Section 7)
- Modify: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_html_report_renders_with_minimal_inputs(tmp_path):
    """Empty inputs should still produce a parseable HTML with all sections."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "thumbnails").mkdir()
    html_path = ctp.build_html_report(
        run_dir=run_dir,
        stats_section={"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0},
        geometry_section={"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0},
        model_section={"per_group": {}, "centroid_distances": {}, "n_frames": 0,
                       "embedding_dim": 0},
        gallery_section={"selected": 0, "rows": [], "per_bucket": 0},
    )
    assert html_path.exists()
    html = html_path.read_text()
    for section in ("Executive summary", "Image stats", "Face geometry",
                    "Model section", "Failures gallery"):
        assert section in html
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_compare_teams_pools.py -v -k "html_report" 2>&1 | head -20
```
Expected: 1 failure.

- [ ] **Step 3: Write the implementation**

Insert after Section 6:
```python
# === Section 7: Pass 5 — HTML report ========================================


def _ks_summary_table(stats_section: dict, geometry_section: dict) -> str:
    """Build the executive-summary KS-distance table."""
    rows: list[tuple[str, str, str, float]] = []  # (pass, pair, feature, ks)
    for source_name, section in (("stats", stats_section), ("geometry", geometry_section)):
        for pair, ks_dict in (section or {}).get("ks", {}).items():
            for feat, ks in ks_dict.items():
                if ks != ks:  # NaN
                    continue
                rows.append((source_name, pair, feat, float(ks)))
    rows.sort(key=lambda r: r[3], reverse=True)
    lines = ['<table class="summary"><thead><tr>'
             '<th>Pass</th><th>Pair</th><th>Feature</th><th>KS distance</th>'
             '</tr></thead><tbody>']
    for src, pair, feat, ks in rows:
        cls = ' class="hi"' if ks > 0.2 else ""
        lines.append(f'<tr{cls}><td>{src}</td><td>{pair}</td>'
                     f'<td>{feat}</td><td>{ks:.3f}</td></tr>')
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _per_group_table(section: dict, title: str) -> str:
    pg = (section or {}).get("per_group", {})
    if not pg:
        return f"<p><em>No data for {title}.</em></p>"
    feature_cols = (section or {}).get("feature_cols", [])
    head = "<th>Group</th>" + "".join(f"<th>{f} (mean)</th>" for f in feature_cols)
    body = []
    for g, stats in pg.items():
        cells = [f"<td>{g}</td>"]
        for f in feature_cols:
            v = stats.get(f, {}).get("mean") if isinstance(stats.get(f), dict) else None
            cells.append(f"<td>{v:.3f}</td>" if isinstance(v, (int, float)) else "<td>-</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<h3>{title} — per-group means</h3>"
            f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def _gallery_section(gallery_section: dict, run_dir: Path) -> str:
    rows = (gallery_section or {}).get("rows", [])
    if not rows:
        return "<p><em>No gallery data — model pass was skipped or empty.</em></p>"
    by_group: dict[str, dict[str, list]] = {}
    for r in rows:
        by_group.setdefault(r["group"], {}).setdefault(r["bucket"], []).append(r)
    parts = []
    for grp in sorted(by_group):
        parts.append(f"<h3>{grp}</h3>")
        for bucket in ("confidently_wrong", "uncertain", "confidently_right"):
            items = by_group[grp].get(bucket, [])
            parts.append(f"<h4>{bucket} ({len(items)} frames)</h4>")
            parts.append('<div class="thumbs">')
            for r in items:
                rel = r["thumbnail_rel_path"]
                parts.append(
                    f'<figure><img src="{rel}" alt="{grp}/{bucket}">'
                    f'<figcaption>p={r["fake_prob"]:.2f}<br>'
                    f'{r["video_id"]}</figcaption></figure>'
                )
            parts.append("</div>")
    return "\n".join(parts)


def build_html_report(
    *,
    run_dir: Path,
    stats_section: dict,
    geometry_section: dict,
    model_section: dict,
    gallery_section: dict,
) -> Path:
    """Compose the single-file HTML report. Returns the written path."""
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
           margin: 2em auto; padding: 0 1em; color: #222; }
    h1 { border-bottom: 2px solid #888; padding-bottom: 0.3em; }
    h2 { margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
    table { border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }
    th, td { border: 1px solid #ccc; padding: 0.3em 0.6em; text-align: left; }
    th { background: #f4f4f4; }
    tr.hi td { background: #fff4e0; font-weight: bold; }
    .thumbs { display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 1.5em; }
    .thumbs figure { margin: 0; padding: 0; text-align: center; }
    .thumbs img { width: 112px; height: 112px; object-fit: cover; border: 1px solid #ddd; }
    .thumbs figcaption { font-size: 0.7em; color: #555; max-width: 112px; }
    """
    n_stats = (stats_section or {}).get("n_frames", 0)
    n_geom = (geometry_section or {}).get("n_frames", 0)
    n_model = (model_section or {}).get("n_frames", 0)
    centroid_html = ""
    cd = (model_section or {}).get("centroid_distances", {})
    if cd:
        centroid_html = "<h3>Centroid cosine distances</h3><table><tbody>" + "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(cd.items())
        ) + "</tbody></table>"
    body = f"""
    <h1>Teams pool comparison: teams_ood vs proper_visomaster_teams</h1>
    <p>Run dir: <code>{run_dir}</code></p>
    <p>Frames analyzed — stats: {n_stats}, geometry: {n_geom}, model: {n_model}.</p>

    <h2>Executive summary</h2>
    <p>KS distances across cross-pool same-label pairs. Rows highlighted in
    orange exceed 0.2 (a notable distributional shift).</p>
    {_ks_summary_table(stats_section, geometry_section)}

    <h2>Image stats</h2>
    {_per_group_table(stats_section, "Image stats")}

    <h2>Face geometry</h2>
    {_per_group_table(geometry_section, "Face geometry")}

    <h2>Model section</h2>
    <h3>Per-group fake-probability</h3>
    {_per_group_table({"per_group": (model_section or {}).get("per_group", {}),
                        "feature_cols": ["fake_prob_mean", "fake_prob_median",
                                          "embedding_dispersion_l2"]},
                       "Model")}
    {centroid_html}

    <h2>Failures gallery</h2>
    {_gallery_section(gallery_section, run_dir)}
    """
    html = f"<!doctype html><meta charset='utf-8'><title>teams pool diff</title>" \
           f"<style>{css}</style>{body}"
    out = run_dir / "report.html"
    out.write_text(html)
    return out


```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -25
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: add Pass 5 (single-file HTML report)

Self-contained report with executive-summary KS table, per-group means
for stats + geometry, model centroid-distance table, and the failures
gallery. Thumbnails referenced via run_dir-relative paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Wire main() + smoke run

**Files:**
- Modify: `analysis/compare_teams_pools.py` (replace minimal main with full pipeline)
- Modify: `tests/test_compare_teams_pools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_teams_pools.py`:
```python
def test_main_with_skip_all_passes_runs_clean(tmp_path, monkeypatch):
    """--skip-pass {stats,geometry,model,gallery} should produce empty report."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp

    # Stub collect_frames so we don't hit GCS.
    import pandas as pd
    monkeypatch.setattr(ctp, "collect_frames",
                        lambda **kw: pd.DataFrame(
                            columns=["group", "label", "video_id", "frame_path"]))
    rc = ctp.main([
        "--output-dir", str(tmp_path),
        "--skip-pass", "stats",
        "--skip-pass", "geometry",
        "--skip-pass", "model",
        "--skip-pass", "gallery",
    ])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    assert (runs[0] / "report.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_compare_teams_pools.py -v -k "skip_all_passes" 2>&1 | head -20
```
Expected: failure (`AssertionError: not (runs[0] / "report.html").exists()`).

- [ ] **Step 3: Replace `main()` with the full pipeline**

Replace the `def main(argv: list[str] | None = None) -> int:` block with:
```python
def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.log_level)
    run_dir = _make_run_dir(args.output_dir)
    logger.info("Run dir: %s", run_dir)
    logger.info("Skipped passes: %s", args.skip_pass or "(none)")

    frames_df = collect_frames(
        videos_per_group=args.videos_per_group,
        frames_per_video=args.frames_per_video,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    logger.info("collected %d frames across %d groups",
                len(frames_df), frames_df["group"].nunique() if len(frames_df) else 0)

    stats_section: dict = {}
    geometry_section: dict = {}
    model_section: dict = {}
    gallery_section: dict = {}

    if "stats" not in args.skip_pass and len(frames_df):
        try:
            stats_section = run_pass_stats(frames_df, run_dir=run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("stats pass failed: %s", exc)
            stats_section = {"error": str(exc)}

    if "geometry" not in args.skip_pass and len(frames_df):
        try:
            geometry_section = run_pass_geometry(frames_df, run_dir=run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("geometry pass failed: %s", exc)
            geometry_section = {"error": str(exc)}

    scored_df = None
    if "model" not in args.skip_pass and len(frames_df):
        try:
            model_section = run_pass_model(
                frames_df, run_dir=run_dir, checkpoint_uri=args.checkpoint,
                cache_dir=args.cache_dir,
            )
            import pandas as pd
            scored_path = run_dir / "raw_scores.parquet"
            if scored_path.exists():
                scored_df = pd.read_parquet(scored_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("model pass failed: %s", exc)
            model_section = {"error": str(exc)}

    if "gallery" not in args.skip_pass and scored_df is not None:
        try:
            gallery_section = run_pass_gallery(scored_df, run_dir=run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("gallery pass failed: %s", exc)
            gallery_section = {"error": str(exc)}

    html_path = build_html_report(
        run_dir=run_dir,
        stats_section=stats_section,
        geometry_section=geometry_section,
        model_section=model_section,
        gallery_section=gallery_section,
    )
    import json
    summary = {
        "run_dir": str(run_dir),
        "n_frames_collected": int(len(frames_df)),
        "report_path": str(html_path),
        "skipped_passes": list(args.skip_pass),
        "checkpoint": args.checkpoint,
    }
    (run_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    logger.info("done — report at %s", html_path)
    return 0
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_compare_teams_pools.py -v 2>&1 | head -25
```
Expected: 11 passed.

- [ ] **Step 5: Smoke run end-to-end (small sample, then real)**

```bash
# Tiny smoke first — 2 videos * 2 frames per group, no model.
python analysis/compare_teams_pools.py \
  --videos-per-group 2 --frames-per-video 2 \
  --output-dir scratch/teams_pool_diff \
  --skip-pass model --skip-pass gallery 2>&1 | tail -20
```
Expected: exit 0, run dir created, report.html written, sections present.

If smoke passes, the real run is the user's call (it pulls ~250 MB and runs ~10 min model inference). Document the command in the final summary.

- [ ] **Step 6: Commit**

```bash
git add analysis/compare_teams_pools.py tests/test_compare_teams_pools.py
git commit -m "$(cat <<'EOF'
compare_teams_pools: wire main() pipeline + run smoke

End-to-end orchestration of all 5 passes with per-pass try/except so a
single failure renders as 'pass failed' in the report rather than
crashing the run. Each pass writes its own stats_pass_*.json so reruns
with --skip-pass can resume from cached artifacts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**
- Data plane (4 groups, sampling, cache) — Task 2 ✓
- Pass 1 stats — Task 3 ✓
- Pass 2 geometry — Task 4 ✓
- Pass 3 model embeddings + confidence — Task 5 ✓
- Pass 4 hard-sample selection — Task 6 ✓
- Pass 5 HTML report — Task 7 ✓
- CLI w/ all flags — Task 1 ✓
- Output: report.html + stats.json + raw_scores.parquet + thumbnails/ — Tasks 5, 6, 7, 8 ✓
- Error handling per spec (GCS auth, mediapipe no-face, checkpoint mismatch, per-pass try/except) — Tasks 2, 4, 5, 8 ✓
- Acceptance criteria 1, 2, 3, 4, 5 — covered by Task 8 smoke run ✓

**Placeholder scan:** No "TBD", "TODO", or "implement later". Each step shows code or exact commands.

**Type/name consistency:** `_sample_video_ids`, `_image_stats_from_array`, `_face_geometry_from_array`, `_preprocess_image`, `_select_hard_samples`, `build_html_report`, `run_pass_*`, `collect_frames`, `GROUP_DEFINITIONS`, `GroupDef` are used consistently across tasks. `cache_dir` and `output_dir` parameter names match across functions.

**Known limitations:**
- The model checkpoint requires `detectors.DETECTOR` registry + a config block in the saved file. If the checkpoint format changes, Task 5 will need an update. Mitigated by the explicit `RuntimeError` rather than a silent partial load.
- MediaPipe install on Apple Silicon can be flaky on Python 3.12; the spec calls out CPU-only and pip-only — if the user is on a fresh env, `pip install mediapipe` may need a Python downgrade. The geometry pass degrades gracefully (returns no-face rows) if mediapipe is unavailable.
- `_list_video_dirs` walks at most 50000 blobs per bucket. If `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` exceeds that, raise the cap or paginate properly. Not a blocker for the diagnostic but a known scaling cliff.
