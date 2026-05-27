"""Sample N frames from every data source used by packet-5 training/eval, compute
image-quality + feature stats, and write a per-source comparison.

Question it answers: why doesn't a model trained on DeepLive + proper_visomaster
generalize to teams_ood / wma_failure? Hypothesis from user: the OOD captures are
from an older pipeline with different resolution / compression / sharpness.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_CSV = HERE / "per_image_stats.csv"
OUT_SUMMARY = HERE / "per_source_summary.csv"
MANIFEST = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/arena/manifests/"
    "proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
)

N_PER_SOURCE = 30  # frames to sample per source
OVERSAMPLE = 2     # request 2x URIs to tolerate manifest staleness / missing objects
SEED = 737

# Strategy categorization.
# IMPORTANT: `quality_enhancement` is NON-enhanced despite the name (user correction).
DEEPLIVE_NON_ENHANCED = ("edge_cases", "minimal_processing", "quality_enhancement")
DEEPLIVE_ENHANCED = ("edge_cases_enhanced", "minimal_processing_enhanced")
VISOMASTER_IN_LIVE_BUCKET = (  # in DeepLive bucket, currently `visomaster.enabled: false`
    "visomaster_CSCS", "visomaster_GhostFace-v1", "visomaster_GhostFace-v2",
    "visomaster_GhostFace-v3", "visomaster_InStyleSwapper256-A",
    "visomaster_InStyleSwapper256-B", "visomaster_InStyleSwapper256-C",
    "visomaster_Inswapper128", "visomaster_SimSwap512",
)


@dataclass
class Source:
    name: str           # short label written to stats
    provenance: str     # user_created_training / user_created_ood_old_system / external / candidate
    label: str          # fake / real / mixed
    uris: list[str]     # resolved gs:// frame URIs (already sampled)


def _gs_ls(uri: str) -> list[str]:
    out = subprocess.run(
        ["gsutil", "ls", uri],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _rnd(seq, n, rng):
    if len(seq) <= n:
        return list(seq)
    return rng.sample(seq, n)


def _session_matches_strategy(session_url: str, strategies: tuple[str, ...]) -> bool:
    """A session URL like gs://BUCKET/samples/STRATEGY_NNNN/ matches if its
    STRATEGY prefix is in `strategies`. This distinguishes e.g. `edge_cases`
    (non-enhanced) from `edge_cases_enhanced` — longest-match is required."""
    name = session_url.rstrip("/").split("/")[-1]
    # find longest prefix among strategies that matches name up to _digits
    best = None
    for s in strategies:
        if name.startswith(s + "_"):
            rest = name[len(s) + 1:]
            # rest must be digits-only (session number), else it's a longer strategy
            if rest.isdigit():
                if best is None or len(s) > len(best):
                    best = s
    return best is not None


def _resolve_deeplive_like(bucket: str, strategies: tuple[str, ...],
                            subdirs_to_try: int, child: str,
                            n: int, rng: random.Random) -> list[str]:
    """For buckets structured as gs://{bucket}/samples/{STRATEGY_NNNN}/frames/{real|fake}/frame_*.png.
    Filter session dirs by strategy family, then pick frames until n collected."""
    session_dirs = _gs_ls(f"gs://{bucket}/samples/")
    if not session_dirs:
        return []
    session_dirs = [s for s in session_dirs if s.endswith("/")]
    matching = [s for s in session_dirs if _session_matches_strategy(s, strategies)]
    if not matching:
        return []
    chosen_sessions = _rnd(matching, subdirs_to_try, rng)
    uris: list[str] = []
    for sess in chosen_sessions:
        frames = _gs_ls(f"{sess}frames/{child}/")
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        if not frames:
            continue
        pick = _rnd(frames, max(2, n // subdirs_to_try), rng)
        uris.extend(pick)
        if len(uris) >= n:
            break
    return uris[:n]


def build_proper_lane_uris(
    manifest_path: Path, lane: str, label_filter: str, n: int, rng: random.Random
) -> list[str]:
    with open(manifest_path) as f:
        m = json.load(f)
    videos = m["videos"]

    def _label_match(v):
        if label_filter == "fake":
            return v.get("label") == "fake" and v.get("lane") == lane
        # real -- proper_real_{clean,teams} lanes pair with proper_visomaster_{clean,teams}
        if lane.endswith("_teams"):
            return v.get("lane") == "proper_real_teams"
        if lane.endswith("_clean"):
            return v.get("lane") == "proper_real_clean"
        return False

    if label_filter == "fake":
        matched = [v for v in videos if v.get("lane") == lane]
    else:
        # for real we want reals that share identity with the target lane's videos
        target_ids = {v.get("identity_id") for v in videos if v.get("lane") == lane}
        matched = [
            v for v in videos
            if v.get("label") == "real"
            and v.get("identity_id") in target_ids
            and (
                (lane.endswith("_teams") and v.get("lane") == "proper_real_teams")
                or (lane.endswith("_clean") and v.get("lane") == "proper_real_clean")
            )
        ]

    if not matched:
        return []
    rng.shuffle(matched)
    uris: list[str] = []
    for v in matched:
        paths = v.get("frame_paths") or []
        if not paths:
            continue
        uris.append(rng.choice(paths))
        if len(uris) >= n:
            break
    return uris


def build_external_vcd_uris(n: int, rng: random.Random) -> list[str]:
    top = _gs_ls("gs://effort-collected-data/real/VCD/")
    top = [d for d in top if d.endswith("/")]
    picks = _rnd(top, 8, rng)
    uris = []
    for d in picks:
        frames = _gs_ls(d)
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        if not frames:
            # might have one more level
            subs = [x for x in frames if x.endswith("/")]
            for s in subs[:2]:
                frames.extend([f for f in _gs_ls(s) if f.endswith((".png", ".jpg", ".jpeg"))])
        if frames:
            uris.extend(_rnd(frames, max(3, n // 8), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_external_avspeech_uris(n: int, rng: random.Random) -> list[str]:
    # Structure: gs://BUCKET/real/external_youtube_avspeech/{video_id}_person_N/person_N_NNNN.png
    top = _gs_ls("gs://effort-collected-data/real/external_youtube_avspeech/")
    top = [d for d in top if d.endswith("/")]
    picks = _rnd(top, 16, rng)
    uris = []
    for d in picks:
        frames = _gs_ls(d)
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        if frames:
            uris.extend(_rnd(frames, max(2, n // 16), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_wma_fake_uris(n: int, rng: random.Random) -> list[str]:
    top = _gs_ls("gs://effort-collected-data/wma_validation/enhanced_fake/")
    top = [d for d in top if d.endswith("/")]
    picks = _rnd(top, 6, rng)
    uris = []
    for d in picks:
        frames = _gs_ls(d)
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        if not frames:
            subs = [x for x in _gs_ls(d) if x.endswith("/")]
            for s in subs[:3]:
                frames.extend([f for f in _gs_ls(s) if f.endswith((".png", ".jpg", ".jpeg"))])
        if frames:
            uris.extend(_rnd(frames, max(5, n // 6), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_enhanced_v2_uris(n: int, rng: random.Random) -> list[str]:
    methods = _gs_ls("gs://visomaster-enhanced-face-cropped-v2/fake/")
    methods = [d for d in methods if d.endswith("/")]
    picks = _rnd(methods, 12, rng)
    uris = []
    for d in picks:
        frames = _gs_ls(d)
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        if frames:
            uris.extend(_rnd(frames, max(2, n // 12), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def discover_sources() -> list[Source]:
    rng = random.Random(SEED)
    n = N_PER_SOURCE * OVERSAMPLE  # request more; downloads may fail

    sources: list[Source] = []
    DL_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
    TV2_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"

    # ==== DeepLive bucket splits ====
    # quality_enhancement is NON-enhanced despite the name (user correction 2026-04-23)
    for label in ("fake", "real"):
        uris = _resolve_deeplive_like(DL_BUCKET, DEEPLIVE_NON_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"deeplive_non_enh_{label}", "user_created_training__fam=deeplive_non_enhanced", label, uris))
        uris = _resolve_deeplive_like(DL_BUCKET, DEEPLIVE_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"deeplive_enh_{label}", "user_created_training__fam=deeplive_enhanced", label, uris))
        # visomaster_* sessions in the DeepLive bucket — currently `visomaster.enabled: false`
        uris = _resolve_deeplive_like(DL_BUCKET, VISOMASTER_IN_LIVE_BUCKET, 8, label, n, rng)
        sources.append(Source(f"dl_bucket_visomaster_{label}", "user_created_UNUSED__visomaster_in_dl_bucket", label, uris))

    # ==== Teams-v2 bucket splits ====
    # Teams-v2 has DeepLive (no _enhanced) + visomaster_* sessions; all bundled under deeplive_teams_*
    for label in ("fake", "real"):
        uris = _resolve_deeplive_like(TV2_BUCKET, DEEPLIVE_NON_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"tv2_deeplive_{label}", "user_created_training+ood__fam=deeplive_teams", label, uris))
        uris = _resolve_deeplive_like(TV2_BUCKET, VISOMASTER_IN_LIVE_BUCKET, 8, label, n, rng)
        sources.append(Source(f"tv2_visomaster_{label}", "user_created_training+ood__fam=deeplive_teams", label, uris))

    # ==== proper_visomaster manifest lanes ====
    for lane in ("proper_visomaster_clean", "proper_visomaster_teams",
                 "proper_visomaster_enhanced_clean", "proper_visomaster_enhanced_teams"):
        prov = "user_created_UNUSED__proper" if lane == "proper_visomaster_enhanced_clean" else "user_created_training__proper"
        uris = build_proper_lane_uris(MANIFEST, lane, "fake", n, rng)
        sources.append(Source(f"{lane}_fake", prov, "fake", uris))
    # paired reals for clean + teams
    for lane in ("proper_visomaster_clean", "proper_visomaster_teams"):
        uris = build_proper_lane_uris(MANIFEST, lane, "real", n, rng)
        short = lane.replace("proper_visomaster_", "proper_real_") + "__paired"
        sources.append(Source(short, "user_created_training__proper", "real", uris))

    # ==== Candidate new bucket ====
    uris = build_enhanced_v2_uris(n, rng)
    sources.append(Source("visomaster_enhanced_v2_fake", "candidate_new_data", "fake", uris))

    # ==== External (used in OOD gate) ====
    uris = build_external_vcd_uris(n, rng)
    sources.append(Source("external_vcd_real", "external_collected__ood_gate", "real", uris))
    uris = build_external_avspeech_uris(n, rng)
    sources.append(Source("external_youtube_avspeech_real", "external_collected__ood_gate", "real", uris))
    uris = build_wma_fake_uris(n, rng)
    sources.append(Source("wma_failure_fake", "external_collected__ood_gate", "fake", uris))

    return sources


def download_all(sources: list[Source]) -> dict[str, list[Path]]:
    """Parallel download using gsutil -m cp. Returns per-source list of local paths."""
    local_by_source: dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {}
        for src in sources:
            local_dir = CACHE_DIR / src.name
            local_dir.mkdir(exist_ok=True)
            if not src.uris:
                local_by_source[src.name] = []
                continue
            # write uri list file
            list_file = local_dir / "_uris.txt"
            list_file.write_text("\n".join(src.uris))
            # download if not already present
            missing = [u for u in src.uris if not (local_dir / Path(u).name).exists()]
            if missing:
                futs[pool.submit(_gs_copy, missing, local_dir)] = src
            else:
                local_by_source[src.name] = sorted(local_dir.glob("*.png")) + sorted(local_dir.glob("*.jpg")) + sorted(local_dir.glob("*.jpeg"))

        for fut in as_completed(futs):
            src = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  ! download error {src.name}: {e}", file=sys.stderr)
            local_dir = CACHE_DIR / src.name
            files = sorted(local_dir.glob("*.png")) + sorted(local_dir.glob("*.jpg")) + sorted(local_dir.glob("*.jpeg"))
            local_by_source[src.name] = files
    return local_by_source


def _gs_copy(uris: list[str], local_dir: Path) -> None:
    # Batch: pipe URIs to `gsutil -m cp -I <local_dir>`
    p = subprocess.run(
        ["gsutil", "-m", "-q", "cp", "-I", str(local_dir)],
        input="\n".join(uris), text=True,
        capture_output=True, check=False, timeout=600,
    )
    if p.returncode != 0:
        # Log stderr but keep going; some uris may have failed auth / existence.
        print(f"  gsutil cp warning: rc={p.returncode} tail={p.stderr[-300:]}", file=sys.stderr)


def compute_stats(path: Path) -> dict:
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            fmt = im.format
            # Try to extract JPEG quality if JPEG
            jpeg_q = None
            if fmt == "JPEG" and hasattr(im, "quantization") and im.quantization:
                # Rough proxy: mean of the first quantization table (lower = higher quality)
                qt = im.quantization[0]
                jpeg_q_proxy = float(np.mean(qt))
                jpeg_q = jpeg_q_proxy
        arr_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr_bgr is None:
            return {"path": str(path), "error": "cv2_read_failed"}
        arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)

        # Resolution / size
        stats: dict = {
            "path": str(path),
            "width": w,
            "height": h,
            "aspect": round(w / max(h, 1), 3),
            "pixels": w * h,
            "file_size_bytes": path.stat().st_size,
            "format": fmt,
            "jpeg_q_proxy": jpeg_q,
        }
        # Color (BGR→RGB already)
        stats["mean_r"] = float(arr_rgb[..., 0].mean())
        stats["mean_g"] = float(arr_rgb[..., 1].mean())
        stats["mean_b"] = float(arr_rgb[..., 2].mean())
        stats["std_luma"] = float(gray.std())
        stats["mean_luma"] = float(gray.mean())
        # Sharpness: Laplacian variance (higher = sharper)
        stats["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Edge density: Canny edges per pixel
        edges = cv2.Canny(gray, 80, 200)
        stats["edge_density"] = float((edges > 0).mean())
        # Noise estimate via high-pass residual std (low-pass subtracted)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
        stats["noise_std"] = float(residual.std())
        # Blockiness proxy: difference between 8-boundary and non-8-boundary row variance
        # (useful to detect JPEG block artifacts)
        g = gray.astype(np.float32)
        row_diffs = np.abs(g[:, 1:] - g[:, :-1])
        col_diffs = np.abs(g[1:, :] - g[:-1, :])
        # at 8-pixel boundaries
        r_bnd = row_diffs[:, 7::8].mean() if row_diffs.shape[1] > 8 else float("nan")
        r_non = row_diffs[:, [i for i in range(row_diffs.shape[1]) if (i + 1) % 8 != 0]].mean() if row_diffs.shape[1] > 8 else float("nan")
        c_bnd = col_diffs[7::8, :].mean() if col_diffs.shape[0] > 8 else float("nan")
        c_non = col_diffs[[i for i in range(col_diffs.shape[0]) if (i + 1) % 8 != 0], :].mean() if col_diffs.shape[0] > 8 else float("nan")
        stats["blockiness"] = float(((r_bnd - r_non) + (c_bnd - c_non)) / 2.0) if r_bnd == r_bnd and c_bnd == c_bnd else float("nan")
        return stats
    except Exception as e:
        return {"path": str(path), "error": str(e)[:200]}


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate stats per source."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        buckets[r["source"]].append(r)
    out = []
    for src, items in buckets.items():
        def col(k):
            vals = [it[k] for it in items if isinstance(it.get(k), (int, float)) and it.get(k) == it.get(k)]
            return np.array(vals) if vals else np.array([])
        row = {
            "source": src,
            "n": len(items),
            "label": items[0].get("label"),
            "provenance": items[0].get("provenance"),
            "format_mix": "/".join(sorted({it["format"] for it in items if it.get("format")})),
        }
        for k in ("width", "height", "pixels", "file_size_bytes",
                  "jpeg_q_proxy", "mean_luma", "std_luma",
                  "laplacian_var", "edge_density", "noise_std", "blockiness"):
            arr = col(k)
            if arr.size:
                row[f"{k}_mean"] = round(float(arr.mean()), 3)
                row[f"{k}_med"] = round(float(np.median(arr)), 3)
                row[f"{k}_std"] = round(float(arr.std()), 3)
            else:
                row[f"{k}_mean"] = row[f"{k}_med"] = row[f"{k}_std"] = None
        out.append(row)
    return out


def main() -> None:
    print("[1/4] Discovering sources + sampling URIs …")
    sources = discover_sources()
    for s in sources:
        print(f"  - {s.name:45s} {s.provenance:32s} {s.label:5s} → {len(s.uris)} uris")

    print("\n[2/4] Downloading frames in parallel …")
    local = download_all(sources)
    for name, paths in local.items():
        print(f"  - {name:45s} {len(paths)} local files")

    print("\n[3/4] Computing per-image stats …")
    rows: list[dict] = []
    for s in sources:
        for p in local.get(s.name, []):
            st = compute_stats(p)
            st["source"] = s.name
            st["label"] = s.label
            st["provenance"] = s.provenance
            rows.append(st)
    print(f"  computed stats for {len(rows)} images")

    # Write per-image CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {OUT_CSV}")

    print("\n[4/4] Summarizing per source …")
    summary = summarize(rows)
    # Order columns: descriptive first, then stats in groups
    stat_order = []
    for base in ("width", "height", "pixels", "file_size_bytes",
                 "jpeg_q_proxy", "mean_luma", "std_luma",
                 "laplacian_var", "edge_density", "noise_std", "blockiness"):
        stat_order.extend([f"{base}_mean", f"{base}_med", f"{base}_std"])
    cols = ["source", "n", "label", "provenance", "format_mix"] + stat_order
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow({c: row.get(c) for c in cols})
    print(f"  wrote {OUT_SUMMARY}")

    # Print compact summary to stdout
    print("\n=== PER-SOURCE SUMMARY (key columns) ===")
    hdr = f"{'source':45s} {'n':>3s} {'w_med':>6s} {'h_med':>6s} {'file_kb':>8s} {'jpegQ':>6s} {'lumaM':>6s} {'lumaS':>6s} {'sharp':>8s} {'edges':>6s} {'noise':>6s} {'block':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        line = (
            f"{r['source']:45s} {r['n']:>3d} "
            f"{(r.get('width_med') or 0):>6.0f} "
            f"{(r.get('height_med') or 0):>6.0f} "
            f"{(r.get('file_size_bytes_med') or 0) / 1024.0:>8.1f} "
            f"{(r.get('jpeg_q_proxy_med') or 0):>6.1f} "
            f"{(r.get('mean_luma_med') or 0):>6.1f} "
            f"{(r.get('std_luma_med') or 0):>6.1f} "
            f"{(r.get('laplacian_var_med') or 0):>8.1f} "
            f"{(r.get('edge_density_med') or 0):>6.3f} "
            f"{(r.get('noise_std_med') or 0):>6.2f} "
            f"{(r.get('blockiness_med') or 0):>6.2f}"
        )
        print(line)


if __name__ == "__main__":
    main()
