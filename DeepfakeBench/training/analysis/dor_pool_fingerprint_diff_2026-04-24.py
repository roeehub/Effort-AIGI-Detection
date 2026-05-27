#!/usr/bin/env python3
"""Lighting + codec fingerprint diff across 3 Dor pools.

Pools (30 frames each):
  A. today_failing   — /tmp/dor_session_20260424/dor_shkedi/ (PNG)
  B. dor_shkedi_lb   — lockbox frames where identity_key=dor_shkedi (JPG, via GCS)
  C. real_dor_lb     — lockbox frames where identity_key=real_dor (JPG, via GCS)

Outputs:
  analysis/dor_pool_fingerprints_2026-04-24.csv
  analysis/dor_pool_fingerprints_2026-04-24.png  (2 rows × 5 cols strip-plots)
  analysis/dor_pool_fingerprints_2026-04-24.summary.json

Decision rule (printed):
  - cluster_on_lighting   → V1 (lighting-aggressive) is strongest
  - cluster_on_codec      → V2 (codec-aggressive)
  - both                  → V3 (combined)
  - neither               → stop, hypothesis rejected
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.visualize_teams_target_domain_manifest import _read_bytes_from_path  # noqa: E402

LOCAL_CSV = Path("/tmp/r13_analysis/frames/teams_real_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv")
MANIFEST = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
TODAY_DIR = Path("/tmp/dor_session_20260424/dor_shkedi")

OUT_DIR = REPO_ROOT / "analysis"
OUT_CSV = OUT_DIR / "dor_pool_fingerprints_2026-04-24.csv"
OUT_PNG = OUT_DIR / "dor_pool_fingerprints_2026-04-24.png"
OUT_SUMMARY = OUT_DIR / "dor_pool_fingerprints_2026-04-24.summary.json"
CACHE = OUT_DIR / "_fingerprint_cache_2026-04-24"

N_PER_POOL = 30
SEED = 42


def cache_gcs(gs_path: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(gs_path.encode()).hexdigest()
    local = CACHE / f"{key}.bin"
    if not local.exists():
        local.write_bytes(_read_bytes_from_path(gs_path))
    return local


def load_lockbox_paths() -> Dict[str, List[str]]:
    m = json.loads(MANIFEST.read_text())
    v2id = {v["video_id"]: v["identity_key"] for v in m["videos"]}
    rows = list(csv.DictReader(open(LOCAL_CSV)))
    ds_rows = [r for r in rows if v2id.get(r["video_id"]) == "dor_shkedi"]
    rd_rows = [r for r in rows if v2id.get(r["video_id"]) == "real_dor"]
    rng = random.Random(SEED)
    ds_paths = rng.sample([r["frame_path"] for r in ds_rows], min(N_PER_POOL, len(ds_rows)))
    rd_paths = rng.sample([r["frame_path"] for r in rd_rows], min(N_PER_POOL, len(rd_rows)))
    return {"dor_shkedi_lb": ds_paths, "real_dor_lb": rd_paths}


def today_paths() -> List[str]:
    files = sorted(TODAY_DIR.glob("frame_*.png"))
    rng = random.Random(SEED)
    return [str(p) for p in rng.sample(files, min(N_PER_POOL, len(files)))]


def stats_for_frame(local_path: Path, is_png: bool) -> Dict:
    raw = local_path.read_bytes()
    img = Image.open(local_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)  # HWC, 0-255
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Luminance (BT.601)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # Chroma
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    # Specular hotspots — connected components of Y>250
    hot = y > 250
    # Quick 4-connected component count (sufficient for a proxy)
    try:
        from scipy.ndimage import label
        n_hot, _ = label(hot)
        if isinstance(n_hot, tuple):
            n_hot = n_hot[1]
        else:
            # scipy returns (labeled, num); but our destructuring above failed due to type
            labeled, n_hot = label(hot)
    except Exception:
        # fallback: approximate with hotpixel count / avg blob size
        n_hot = int(hot.sum() / 25)

    # DCT-HF ratio (grayscale FFT high-freq energy / total)
    gray = y
    fft = np.fft.fft2(gray)
    mag = np.abs(fft)
    total = mag.sum() + 1e-9
    hf = mag[h // 4:, w // 4:].sum()
    dct_hf_ratio = float(hf / total)

    jpeg_quant_sig = None
    if not is_png:
        try:
            qt = Image.open(local_path).quantization  # dict[int, list[int]]
            if qt:
                # Use table 0 mean + std as short signature
                qt0 = np.asarray(qt[0], dtype=np.float32)
                jpeg_quant_sig = f"m{qt0.mean():.1f}_s{qt0.std():.1f}"
        except Exception:
            pass

    return {
        # Lighting
        "mean_lum": float(y.mean()),
        "std_lum": float(y.std()),
        "highlight_clip_pct": float((y > 245).mean() * 100),
        "shadow_clip_pct": float((y < 10).mean() * 100),
        "dynamic_range": float(np.percentile(y, 99) - np.percentile(y, 1)),
        "wb_rb_ratio": float(r.mean() / (b.mean() + 1e-9)),
        "specular_hotspots": int(n_hot),
        # Codec
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_cb": float(cb.mean()),
        "mean_cr": float(cr.mean()),
        "dct_hf_ratio": dct_hf_ratio,
        "bits_per_pixel": float(len(raw) * 8 / (w * h)),
        "jpeg_quant_sig": jpeg_quant_sig,
        "width": w,
        "height": h,
    }


def main():
    print("Loading lockbox frame paths ...")
    lb = load_lockbox_paths()
    today = today_paths()
    print(f"  today_failing:  {len(today)} PNGs")
    print(f"  dor_shkedi_lb:  {len(lb['dor_shkedi_lb'])} JPG GCS paths")
    print(f"  real_dor_lb:    {len(lb['real_dor_lb'])} JPG GCS paths")

    rows = []
    print("\nFetching + measuring today_failing (PNG, local) ...")
    for p in today:
        s = stats_for_frame(Path(p), is_png=True)
        s["pool"] = "today_failing"
        s["source"] = p
        rows.append(s)

    print("Fetching + measuring dor_shkedi_lb (JPG via GCS) ...")
    for gs in lb["dor_shkedi_lb"]:
        local = cache_gcs(gs)
        s = stats_for_frame(local, is_png=False)
        s["pool"] = "dor_shkedi_lb"
        s["source"] = gs
        rows.append(s)

    print("Fetching + measuring real_dor_lb (JPG via GCS) ...")
    for gs in lb["real_dor_lb"]:
        local = cache_gcs(gs)
        s = stats_for_frame(local, is_png=False)
        s["pool"] = "real_dor_lb"
        s["source"] = gs
        rows.append(s)

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    # Summary stats per pool
    pools = ["today_failing", "dor_shkedi_lb", "real_dor_lb"]
    metrics_lighting = ["mean_lum", "highlight_clip_pct", "shadow_clip_pct", "dynamic_range", "wb_rb_ratio", "specular_hotspots"]
    metrics_codec = ["dct_hf_ratio", "mean_cb", "mean_cr", "bits_per_pixel"]

    summary: Dict = {"pools": {}}
    for pool in pools:
        sub = [r for r in rows if r["pool"] == pool]
        psum = {"n": len(sub)}
        for m in metrics_lighting + metrics_codec:
            vals = [float(r[m]) for r in sub if r.get(m) is not None]
            if vals:
                psum[m] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        summary["pools"][pool] = psum

    # Decision rule: for each metric, is (today, dor_shkedi_lb) cluster distinct from real_dor_lb?
    # Use: |mean(cluster_failing) - mean(real_dor_lb)| > 0.5 * pooled_std
    decision: Dict[str, Dict] = {"lighting": {}, "codec": {}}
    for group_name, metrics in [("lighting", metrics_lighting), ("codec", metrics_codec)]:
        for m in metrics:
            t_vals = [float(r[m]) for r in rows if r["pool"] == "today_failing" and r.get(m) is not None]
            d_vals = [float(r[m]) for r in rows if r["pool"] == "dor_shkedi_lb" and r.get(m) is not None]
            c_vals = [float(r[m]) for r in rows if r["pool"] == "real_dor_lb" and r.get(m) is not None]
            if not (t_vals and d_vals and c_vals):
                continue
            t_mean = statistics.mean(t_vals)
            d_mean = statistics.mean(d_vals)
            c_mean = statistics.mean(c_vals)
            failing_mean = (t_mean + d_mean) / 2
            c_std = statistics.stdev(c_vals) if len(c_vals) > 1 else 1.0
            delta = abs(failing_mean - c_mean)
            sep = delta / (c_std + 1e-9)
            decision[group_name][m] = {
                "today_failing_mean": t_mean,
                "dor_shkedi_lb_mean": d_mean,
                "real_dor_lb_mean": c_mean,
                "sep_from_clean_in_std": sep,
                "separating": sep > 1.0,
                "same_direction": (t_mean > c_mean) == (d_mean > c_mean),
            }

    lighting_hits = sum(1 for d in decision["lighting"].values() if d["separating"] and d["same_direction"])
    codec_hits = sum(1 for d in decision["codec"].values() if d["separating"] and d["same_direction"])
    verdict = {
        "lighting_hits": lighting_hits,
        "codec_hits": codec_hits,
        "recommended_variant": None,
    }
    if lighting_hits >= 2 and codec_hits >= 2:
        verdict["recommended_variant"] = "V3_combined"
    elif lighting_hits >= 2:
        verdict["recommended_variant"] = "V1_lighting_aggressive"
    elif codec_hits >= 2:
        verdict["recommended_variant"] = "V2_codec_aggressive"
    else:
        verdict["recommended_variant"] = "STOP_signature_hypothesis_rejected"

    summary["decision"] = decision
    summary["verdict"] = verdict
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_SUMMARY}")

    # Strip-plot PNG
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows_m = [metrics_lighting[:5], metrics_codec[:5]]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        colors = {"today_failing": "#FFB428", "dor_shkedi_lb": "#D64E4E", "real_dor_lb": "#24AA55"}
        for ri, group_metrics in enumerate(rows_m):
            for ci, m in enumerate(group_metrics):
                ax = axes[ri, ci]
                for pi, pool in enumerate(pools):
                    vals = [float(r[m]) for r in rows if r["pool"] == pool and r.get(m) is not None]
                    if not vals:
                        continue
                    xs = np.random.default_rng(SEED + pi).normal(pi, 0.08, size=len(vals))
                    ax.scatter(xs, vals, c=colors[pool], alpha=0.7, s=16, label=pool if (ri, ci) == (0, 0) else None)
                    ax.scatter([pi], [statistics.mean(vals)], marker="_", c="black", s=400)
                ax.set_xticks(range(len(pools)))
                ax.set_xticklabels([p.replace("_", "\n") for p in pools], fontsize=8)
                ax.set_title(m, fontsize=10)
                ax.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="upper right", fontsize=7)
        fig.suptitle(
            f"Dor pool fingerprint diff  —  lighting (top) / codec (bottom)  —  verdict: {verdict['recommended_variant']}"
            f"  (lighting hits={lighting_hits}, codec hits={codec_hits})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=110)
        print(f"Wrote {OUT_PNG}")
    except Exception as e:
        print(f"matplotlib strip-plot skipped: {e}")

    # Print a compact summary
    print("\n=== SUMMARY ===")
    for pool in pools:
        print(f"\n[{pool}] n={summary['pools'][pool]['n']}")
        for m in metrics_lighting + metrics_codec:
            if m in summary["pools"][pool]:
                v = summary["pools"][pool][m]
                print(f"  {m:<22} {v['mean']:>9.3f} ± {v['std']:<7.3f}")

    print(f"\n>>> Verdict: {verdict['recommended_variant']}")
    print(f"    Lighting separating metrics (≥1σ, same direction): {lighting_hits}")
    print(f"    Codec    separating metrics (≥1σ, same direction): {codec_hits}")


if __name__ == "__main__":
    main()
