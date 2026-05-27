#!/usr/bin/env python3
"""Generalized lighting + codec fingerprint diff across N arbitrary image pools.

Forked from analysis/dor_pool_fingerprint_diff_2026-04-24.py. The per-frame
metric computation is kept byte-identical to the original so old verdicts
remain reproducible.

Config (YAML):

    output_prefix: dor_roee_multi_pair_2026-04-24
    seed: 42
    n_per_pool: 30
    separation_sigma: 1.0          # (optional) threshold for "separating"
    cache_dir: analysis/_fingerprint_cache_2026-04-24  # (optional)

    pools:
      dor_webcam_today:
        source: local_dir          # local_dir | local_files | gcs_prefix | lockbox_csv_identity
        path: /tmp/dor_session_20260424/dor_webcam
        pattern: "frame_*.png"     # for local_dir / gcs_prefix (optional, default "*")
      dor_laptop_today:
        source: local_dir
        path: /tmp/dor_session_20260424/dor_laptop
        pattern: "frame_*.png"
      gs_demo:
        source: gcs_prefix
        prefix: "gs://bucket/path/to/frames/"
        pattern: "*.jpg"
      dor_shkedi_lb:
        source: lockbox_csv_identity
        csv_path: /tmp/r13_analysis/frames/teams_real_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv
        manifest_path: arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json
        identity_key: dor_shkedi

    pairs:
      - name: dor_webcam_vs_laptop
        failing: dor_webcam_today
        clean: dor_laptop_today
      - name: dor_webcam_vs_lockbox_clean
        failing: dor_webcam_today
        clean: real_dor_lb

Outputs (under analysis/):
    {output_prefix}.csv                      # per-frame rows, all pools
    {output_prefix}.summary.json             # per-pool stats, per-pair decisions, cross-pair consistency
    {output_prefix}.{pair_name}.png          # strip-plots per pair (lighting top / codec bottom)
    {output_prefix}.cross_pair_summary.png   # cross-pair consistency bar chart
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.visualize_teams_target_domain_manifest import (  # noqa: E402
    _get_gcs_bucket,
    _read_bytes_from_path,
    _split_gs_uri,
)

OUT_DIR = REPO_ROOT / "analysis"

METRICS_LIGHTING = ["mean_lum", "highlight_clip_pct", "shadow_clip_pct", "dynamic_range", "wb_rb_ratio", "specular_hotspots"]
METRICS_CODEC = ["dct_hf_ratio", "mean_cb", "mean_cr", "bits_per_pixel"]
ALL_METRICS = METRICS_LIGHTING + METRICS_CODEC


# ---------- pool resolution ----------

def _cache_path(cache_dir: Path, key: str, suffix: str = ".bin") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}{suffix}"


def _materialize_gcs(gs_path: str, cache_dir: Path) -> Path:
    local = _cache_path(cache_dir, gs_path)
    if not local.exists():
        local.write_bytes(_read_bytes_from_path(gs_path))
    return local


def _list_gcs_prefix(prefix: str, pattern: str) -> List[str]:
    """Return sorted gs:// URIs under prefix matching pattern (fnmatch on basename)."""
    import fnmatch
    bucket_name, blob_prefix = _split_gs_uri(prefix.rstrip("/") + "/")
    bucket = _get_gcs_bucket(bucket_name)
    out: List[str] = []
    for blob in bucket.list_blobs(prefix=blob_prefix):
        if blob.name.endswith("/"):
            continue
        base = blob.name.rsplit("/", 1)[-1]
        if pattern and not fnmatch.fnmatch(base, pattern):
            continue
        out.append(f"gs://{bucket_name}/{blob.name}")
    return sorted(out)


def _resolve_pool(label: str, spec: Dict, seed: int, n_per_pool: int, cache_dir: Path) -> List[Tuple[str, Path, Optional[float]]]:
    """Return list of (display_source, local_path, score_or_none) for up to n_per_pool frames.

    `display_source` is the original reference (gs:// URI, CSV frame_path, or absolute local path).
    `local_path` is always a local file usable by PIL.
    `score` is the per-frame model score when the source provides it (currently only
    combined_frame_tags); otherwise None.
    """
    source = spec["source"]
    rng = random.Random(seed)

    if source == "local_dir":
        root = Path(spec["path"]).expanduser()
        pattern = spec.get("pattern", "*")
        files = sorted(root.glob(pattern))
        if not files:
            raise ValueError(f"Pool '{label}': no files matched {root}/{pattern}")
        chosen = rng.sample(files, min(n_per_pool, len(files)))
        return [(str(p), p, None) for p in chosen]

    if source == "local_files":
        files = [Path(p).expanduser() for p in spec["paths"]]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise ValueError(f"Pool '{label}': missing files: {missing[:3]}...")
        chosen = rng.sample(files, min(n_per_pool, len(files)))
        return [(str(p), p, None) for p in chosen]

    if source == "gcs_prefix":
        prefix = spec["prefix"]
        pattern = spec.get("pattern", "*")
        uris = _list_gcs_prefix(prefix, pattern)
        if not uris:
            raise ValueError(f"Pool '{label}': no GCS blobs under {prefix} matching {pattern}")
        chosen = rng.sample(uris, min(n_per_pool, len(uris)))
        return [(u, _materialize_gcs(u, cache_dir), None) for u in chosen]

    if source == "lockbox_csv_identity":
        csv_path = Path(spec["csv_path"]).expanduser()
        manifest_path = Path(spec["manifest_path"])
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        identity_key = spec["identity_key"]
        manifest = json.loads(manifest_path.read_text())
        v2id = {v["video_id"]: v["identity_key"] for v in manifest["videos"]}
        rows = list(csv.DictReader(open(csv_path)))
        matching = [r["frame_path"] for r in rows if v2id.get(r["video_id"]) == identity_key]
        if not matching:
            raise ValueError(f"Pool '{label}': no CSV rows with identity_key={identity_key}")
        chosen = rng.sample(matching, min(n_per_pool, len(matching)))
        resolved: List[Tuple[str, Path, Optional[float]]] = []
        for ref in chosen:
            if ref.startswith("gs://"):
                resolved.append((ref, _materialize_gcs(ref, cache_dir), None))
            else:
                resolved.append((ref, Path(ref), None))
        return resolved

    if source == "combined_frame_tags":
        manifest_path = Path(spec["manifest"]).expanduser()
        tag_name = spec["tag"]
        base_prefix = spec["base_gcs_prefix"].rstrip("/") + "/"
        manifest = json.loads(manifest_path.read_text())
        matching_tag = next((t for t in manifest["tags"] if t["name"] == tag_name), None)
        if matching_tag is None:
            names = [t["name"] for t in manifest["tags"]]
            raise ValueError(f"Pool '{label}': tag '{tag_name}' not in manifest (have: {names})")
        items = list(matching_tag["items"])
        if not items:
            raise ValueError(f"Pool '{label}': tag '{tag_name}' has zero items")
        chosen = rng.sample(items, min(n_per_pool, len(items)))
        resolved: List[Tuple[str, Path, Optional[float]]] = []
        for it in chosen:
            gs_uri = f"{base_prefix}{it['export_path']}"
            local = _materialize_gcs(gs_uri, cache_dir)
            score = it.get("score")
            resolved.append((gs_uri, local, float(score) if score is not None else None))
        return resolved

    raise ValueError(f"Pool '{label}': unknown source type '{source}'")


# ---------- per-frame metrics (byte-identical to original) ----------

def stats_for_frame(local_path: Path) -> Dict:
    """Compute lighting + codec fingerprint metrics for one frame.

    Kept numerically identical to the original dor_pool_fingerprint_diff_2026-04-24.py
    except that `is_png` is auto-inferred from the actual image format (the original
    required the caller to pass it explicitly).
    """
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
    try:
        from scipy.ndimage import label
        n_hot, _ = label(hot)
        if isinstance(n_hot, tuple):
            n_hot = n_hot[1]
        else:
            labeled, n_hot = label(hot)
    except Exception:
        n_hot = int(hot.sum() / 25)

    # DCT-HF ratio (grayscale FFT high-freq energy / total)
    gray = y
    fft = np.fft.fft2(gray)
    mag = np.abs(fft)
    total = mag.sum() + 1e-9
    hf = mag[h // 4:, w // 4:].sum()
    dct_hf_ratio = float(hf / total)

    jpeg_quant_sig: Optional[str] = None
    fmt = (Image.open(local_path).format or "").upper()
    if fmt in ("JPEG", "MPO"):
        try:
            qt = Image.open(local_path).quantization  # dict[int, list[int]]
            if qt:
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


# ---------- per-pair and cross-pair analysis ----------

def _pair_decision(rows: List[Dict], failing_pool: str, clean_pool: str, sigma: float) -> Dict[str, Dict]:
    """Per-metric decision for one (failing, clean) pair.

    Separation denominator is max(clean_std, failing_std, 1e-6). Using the max
    of the two within-pool stds (rather than clean_std alone, as the original
    Dor script did) avoids billion-σ blow-ups when one pool is degenerate
    (e.g. real_dor_lb has highlight_clip_pct=0 for every frame → std=0).
    Reports `std_floor_hit: true` when the floor kicks in so degenerate
    separations can be recognised downstream.
    """
    decision: Dict[str, Dict] = {"lighting": {}, "codec": {}}
    for group_name, metrics in (("lighting", METRICS_LIGHTING), ("codec", METRICS_CODEC)):
        for m in metrics:
            f_vals = [float(r[m]) for r in rows if r["pool"] == failing_pool and r.get(m) is not None]
            c_vals = [float(r[m]) for r in rows if r["pool"] == clean_pool and r.get(m) is not None]
            if not (f_vals and c_vals):
                continue
            f_mean = statistics.mean(f_vals)
            c_mean = statistics.mean(c_vals)
            c_std = statistics.stdev(c_vals) if len(c_vals) > 1 else 0.0
            f_std = statistics.stdev(f_vals) if len(f_vals) > 1 else 0.0
            raw_denom = max(c_std, f_std)
            denom = max(raw_denom, 1e-6)
            delta = f_mean - c_mean
            sep = abs(delta) / denom
            decision[group_name][m] = {
                "failing_mean": f_mean,
                "clean_mean": c_mean,
                "failing_std": f_std,
                "clean_std": c_std,
                "delta": delta,
                "sep_from_clean_in_std": sep,
                "std_floor_hit": raw_denom < 1e-6,
                "separating": sep > sigma,
                "direction": "failing_gt_clean" if delta > 0 else ("failing_lt_clean" if delta < 0 else "equal"),
            }
    return decision


def _cross_pair_consistency(per_pair_decisions: Dict[str, Dict]) -> Dict:
    """For each metric, count pairs where it separates + whether direction is consistent across pairs."""
    metric_to_pairs: Dict[str, List[Tuple[str, Dict]]] = {m: [] for m in ALL_METRICS}
    for pair_name, d in per_pair_decisions.items():
        for group in ("lighting", "codec"):
            for m, info in d[group].items():
                metric_to_pairs[m].append((pair_name, info))

    summary: Dict[str, Dict] = {}
    for m, pair_infos in metric_to_pairs.items():
        if not pair_infos:
            continue
        separating = [(pn, i) for (pn, i) in pair_infos if i["separating"]]
        directions = {i["direction"] for (_, i) in separating if i["direction"] != "equal"}
        directions_all = {i["direction"] for (_, i) in pair_infos if i["direction"] != "equal"}
        consistent = len(directions) <= 1
        consistent_all = len(directions_all) <= 1
        max_sep = max((i["sep_from_clean_in_std"] for (_, i) in pair_infos), default=0.0)
        summary[m] = {
            "n_pairs_total": len(pair_infos),
            "n_pairs_separating": len(separating),
            "n_pairs_separating_same_dir": len(separating) if consistent else max(
                sum(1 for (_, i) in separating if i["direction"] == d) for d in directions
            ) if directions else 0,
            "direction_consistent_across_separating": consistent,
            "direction_consistent_across_all": consistent_all,
            "max_sep_from_clean_in_std": max_sep,
            "dominant_direction": (
                max(directions, key=lambda d: sum(1 for (_, i) in separating if i["direction"] == d))
                if directions else "equal"
            ),
            "per_pair": {pn: i for (pn, i) in pair_infos},
        }
    return summary


def _corr_pearson_spearman(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Pearson and Spearman correlation. Spearman uses rank-Pearson fallback if scipy missing."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        spearman = float(rho)
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


def _score_correlations_per_pool(rows: List[Dict], pools: List[str]) -> Dict[str, Dict]:
    """For each pool with scores, correlate each metric with per-frame score.

    Returns {pool: {metric: {"pearson": r, "spearman": rho, "n": n}}}.
    """
    out: Dict[str, Dict] = {}
    for pool in pools:
        sub = [r for r in rows if r["pool"] == pool and r.get("score") is not None]
        if len(sub) < 3:
            continue
        scores = [float(r["score"]) for r in sub]
        pool_info: Dict = {"n": len(sub), "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
                           "score_min": float(np.min(scores)), "score_max": float(np.max(scores))}
        metric_corr: Dict[str, Dict] = {}
        for m in ALL_METRICS:
            vals = [float(r[m]) for r in sub if r.get(m) is not None]
            if len(vals) != len(scores):
                continue
            p, s = _corr_pearson_spearman(vals, scores)
            metric_corr[m] = {"pearson": p, "spearman": s, "n": len(vals)}
        pool_info["metrics"] = metric_corr
        out[pool] = pool_info
    return out


def _cross_pool_score_correlation_summary(per_pool_corr: Dict[str, Dict]) -> List[Dict]:
    """Aggregate per-pool score correlations across pools.

    For each metric, average |spearman| across pools (weighted by n), and count
    pools where |spearman| >= 0.3 (medium-effect threshold). Metrics with
    consistently high |correlation| across pools are the ones the model
    tracks at frame level — independent of pool-level means.
    """
    pool_names = list(per_pool_corr.keys())
    out = []
    for m in ALL_METRICS:
        rhos = []
        strong_pools = 0
        directions = []
        for pool in pool_names:
            info = per_pool_corr[pool].get("metrics", {}).get(m)
            if info is None:
                continue
            rho = info["spearman"]
            rhos.append(rho)
            if abs(rho) >= 0.3:
                strong_pools += 1
                directions.append("pos" if rho > 0 else "neg")
        if not rhos:
            continue
        mean_abs = float(np.mean([abs(r) for r in rhos]))
        mean_rho = float(np.mean(rhos))
        consistent_sign = len(set(directions)) <= 1 if directions else True
        out.append({
            "metric": m,
            "mean_abs_spearman": mean_abs,
            "mean_spearman": mean_rho,
            "n_pools_strong": strong_pools,
            "n_pools_total": len(rhos),
            "dominant_direction": "higher_metric_higher_score" if mean_rho > 0 else (
                "higher_metric_lower_score" if mean_rho < 0 else "none"
            ),
            "direction_consistent_across_strong_pools": consistent_sign,
        })
    return sorted(out, key=lambda x: -x["mean_abs_spearman"])


def _robust_shortcut_metrics(cross: Dict, min_pair_fraction: float = 2 / 3) -> List[Dict]:
    """Metrics that separate in ≥min_pair_fraction of pairs, same direction."""
    out = []
    for m, info in cross.items():
        total = info["n_pairs_total"]
        if total == 0:
            continue
        same_dir_count = info["n_pairs_separating_same_dir"]
        if info["direction_consistent_across_separating"] and same_dir_count / total >= min_pair_fraction:
            out.append({
                "metric": m,
                "pairs_hit": same_dir_count,
                "pairs_total": total,
                "direction": info["dominant_direction"],
                "max_sep": info["max_sep_from_clean_in_std"],
            })
    return sorted(out, key=lambda x: (-x["pairs_hit"], -x["max_sep"]))


# ---------- plotting ----------

def _strip_plot_pair(rows: List[Dict], failing: str, clean: str, title: str, out_path: Path, seed: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  strip-plot skipped ({out_path.name}): {e}")
        return

    pools = [failing, clean]
    rows_m = [METRICS_LIGHTING[:5], METRICS_CODEC[:5]]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    colors = {failing: "#D64E4E", clean: "#24AA55"}
    for ri, group_metrics in enumerate(rows_m):
        for ci, m in enumerate(group_metrics):
            ax = axes[ri, ci]
            for pi, pool in enumerate(pools):
                vals = [float(r[m]) for r in rows if r["pool"] == pool and r.get(m) is not None]
                if not vals:
                    continue
                xs = np.random.default_rng(seed + pi).normal(pi, 0.08, size=len(vals))
                ax.scatter(xs, vals, c=colors[pool], alpha=0.7, s=16, label=pool if (ri, ci) == (0, 0) else None)
                ax.scatter([pi], [statistics.mean(vals)], marker="_", c="black", s=400)
            ax.set_xticks(range(len(pools)))
            ax.set_xticklabels([p.replace("_", "\n") for p in pools], fontsize=8)
            ax.set_title(m, fontsize=10)
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _score_corr_plot(cross_pool_corr: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  score-corr plot skipped: {e}")
        return
    if not cross_pool_corr:
        return
    metrics = [c["metric"] for c in cross_pool_corr]
    mean_abs = [c["mean_abs_spearman"] for c in cross_pool_corr]
    mean_rho = [c["mean_spearman"] for c in cross_pool_corr]
    strong = [c["n_pools_strong"] for c in cross_pool_corr]
    total = cross_pool_corr[0]["n_pools_total"]
    colors = ["#D64E4E" if m in METRICS_LIGHTING else "#4E6ED6" for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(range(len(metrics)), mean_abs, color=colors)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean |Spearman ρ| across pools (score vs metric)")
    ax1.set_title("Frame-level score correlation (red=lighting, blue=codec)")
    for i, (ma, s) in enumerate(zip(mean_abs, strong)):
        ax1.text(ma + 0.005, i, f"  {s}/{total} strong", va="center", fontsize=8)
    ax1.axvline(0.3, color="gray", linestyle=":", alpha=0.5, label="|ρ|=0.3 (medium)")
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="|ρ|=0.5 (large)")
    ax1.legend(loc="lower right", fontsize=8)

    ax2.barh(range(len(metrics)), mean_rho, color=colors)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels(metrics, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean Spearman ρ across pools (signed)")
    ax2.set_title("Direction of correlation (positive: metric↑ ⇒ score↑)")
    ax2.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _cross_pair_plot(cross: Dict, n_pairs: int, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  cross-pair plot skipped: {e}")
        return

    metrics = [m for m in ALL_METRICS if m in cross]
    if not metrics:
        return
    hits = [cross[m]["n_pairs_separating_same_dir"] for m in metrics]
    consistent = [cross[m]["direction_consistent_across_separating"] for m in metrics]
    colors = ["#D64E4E" if m in METRICS_LIGHTING else "#4E6ED6" for m in metrics]
    edges = ["black" if c else "lightgray" for c in consistent]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bars = ax.bar(range(len(metrics)), hits, color=colors, edgecolor=edges, linewidth=1.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(f"# pairs separating in same direction (of {n_pairs})")
    ax.set_title("Cross-pair consistency — robust shortcut signatures (red=lighting, blue=codec)")
    ax.axhline(n_pairs, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(0, n_pairs + 0.5)
    for bar, m in zip(bars, metrics):
        info = cross[m]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{info['dominant_direction'][:1].upper()}\n{info['max_sep_from_clean_in_std']:.1f}σ",
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- driver ----------

def run(config_path: Path) -> Dict:
    cfg = yaml.safe_load(config_path.read_text())

    seed = int(cfg.get("seed", 42))
    n_per_pool = int(cfg.get("n_per_pool", 30))
    sigma = float(cfg.get("separation_sigma", 1.0))
    output_prefix = cfg.get("output_prefix") or f"fingerprint_diff_{config_path.stem}"
    cache_dir_str = cfg.get("cache_dir", "analysis/_fingerprint_cache_2026-04-24")
    cache_dir = (REPO_ROOT / cache_dir_str) if not Path(cache_dir_str).is_absolute() else Path(cache_dir_str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{output_prefix}.csv"
    out_summary = OUT_DIR / f"{output_prefix}.summary.json"
    out_xpair_png = OUT_DIR / f"{output_prefix}.cross_pair_summary.png"

    pool_specs: Dict[str, Dict] = cfg["pools"]
    pair_specs: List[Dict] = cfg["pairs"]

    # Validate pairs reference known pools.
    for pair in pair_specs:
        for key in ("failing", "clean"):
            if pair[key] not in pool_specs:
                raise ValueError(f"Pair '{pair['name']}': {key}='{pair[key]}' not declared in pools")

    # Resolve each pool once; compute metrics once per frame.
    all_rows: List[Dict] = []
    per_pool_sources: Dict[str, int] = {}
    any_scores = False
    for label, spec in pool_specs.items():
        print(f"Pool '{label}' ({spec['source']}) ...")
        resolved = _resolve_pool(label, spec, seed, n_per_pool, cache_dir)
        per_pool_sources[label] = len(resolved)
        print(f"  -> {len(resolved)} frames")
        for display_source, local_path, score in resolved:
            s = stats_for_frame(local_path)
            s["pool"] = label
            s["source"] = display_source
            s["score"] = score
            if score is not None:
                any_scores = True
            all_rows.append(s)

    # CSV
    fields = list(all_rows[0].keys())
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out_csv}")

    # Per-pool stats
    pool_stats: Dict[str, Dict] = {}
    for label in pool_specs:
        sub = [r for r in all_rows if r["pool"] == label]
        psum = {"n": len(sub)}
        for m in ALL_METRICS:
            vals = [float(r[m]) for r in sub if r.get(m) is not None]
            if vals:
                psum[m] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                }
        pool_stats[label] = psum

    # Per-pair decisions + per-pair plot
    per_pair_decisions: Dict[str, Dict] = {}
    for pair in pair_specs:
        name = pair["name"]
        failing = pair["failing"]
        clean = pair["clean"]
        decision = _pair_decision(all_rows, failing, clean, sigma)
        per_pair_decisions[name] = decision

        lighting_hits = sum(1 for d in decision["lighting"].values() if d["separating"])
        codec_hits = sum(1 for d in decision["codec"].values() if d["separating"])
        title = (
            f"Pair: {name}  —  failing={failing}  clean={clean}  "
            f"—  lighting hits={lighting_hits}, codec hits={codec_hits}"
        )
        pair_png = OUT_DIR / f"{output_prefix}.{name}.png"
        _strip_plot_pair(all_rows, failing, clean, title, pair_png, seed)

    # Cross-pair consistency
    cross = _cross_pair_consistency(per_pair_decisions)
    robust = _robust_shortcut_metrics(cross)
    _cross_pair_plot(cross, len(pair_specs), out_xpair_png)

    # Per-frame score correlation (only if any pool provided scores)
    pool_names = list(pool_specs.keys())
    per_pool_corr: Dict[str, Dict] = {}
    cross_pool_corr: List[Dict] = []
    if any_scores:
        per_pool_corr = _score_correlations_per_pool(all_rows, pool_names)
        cross_pool_corr = _cross_pool_score_correlation_summary(per_pool_corr)

    summary = {
        "config_path": str(config_path),
        "output_prefix": output_prefix,
        "seed": seed,
        "n_per_pool": n_per_pool,
        "separation_sigma": sigma,
        "pools": pool_stats,
        "pairs": [
            {"name": p["name"], "failing": p["failing"], "clean": p["clean"]}
            for p in pair_specs
        ],
        "per_pair_decisions": per_pair_decisions,
        "cross_pair_consistency": cross,
        "robust_shortcut_metrics": robust,
        "per_pool_score_correlation": per_pool_corr,
        "cross_pool_score_correlation": cross_pool_corr,
    }
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_summary}")

    # Console report
    print("\n=== POOL STATS ===")
    for label in pool_specs:
        print(f"\n[{label}] n={pool_stats[label]['n']}")
        for m in ALL_METRICS:
            if m in pool_stats[label]:
                v = pool_stats[label][m]
                print(f"  {m:<22} {v['mean']:>9.3f} ± {v['std']:<7.3f}")

    print("\n=== PER-PAIR DECISIONS ===")
    for pair in pair_specs:
        name = pair["name"]
        d = per_pair_decisions[name]
        lighting_hits = sum(1 for di in d["lighting"].values() if di["separating"])
        codec_hits = sum(1 for di in d["codec"].values() if di["separating"])
        print(f"\n[{name}]  failing={pair['failing']}  clean={pair['clean']}")
        print(f"  lighting separating (>{sigma}σ): {lighting_hits}")
        print(f"  codec    separating (>{sigma}σ): {codec_hits}")
        for group in ("lighting", "codec"):
            for m, info in d[group].items():
                if info["separating"]:
                    arrow = "↑" if info["delta"] > 0 else "↓"
                    print(f"    {group:8s} {m:<22} failing {arrow} clean  ({info['sep_from_clean_in_std']:.2f}σ)")

    print("\n=== CROSS-PAIR CONSISTENCY ===")
    print(f"Pairs analyzed: {len(pair_specs)}")
    header = f"  {'metric':<22} {'hits/total':>11}  {'direction':<20}  {'max σ':>7}  consistent"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in ALL_METRICS:
        if m not in cross:
            continue
        info = cross[m]
        cons = "yes" if info["direction_consistent_across_separating"] else "no"
        print(
            f"  {m:<22} {info['n_pairs_separating_same_dir']:>3}/{info['n_pairs_total']:<5}  "
            f"{info['dominant_direction']:<20}  {info['max_sep_from_clean_in_std']:>6.2f}   {cons}"
        )

    print("\n=== ROBUST SHORTCUT METRICS (≥2/3 pairs, same direction) ===")
    if not robust:
        print("  (none)")
    else:
        for r in robust:
            print(f"  {r['metric']:<22} {r['pairs_hit']}/{r['pairs_total']} pairs  "
                  f"dir={r['direction']}  max={r['max_sep']:.2f}σ")

    if any_scores:
        print("\n=== PER-POOL SCORE CORRELATION (Spearman ρ, |ρ|≥0.3 bolded with *) ===")
        for pool in pool_names:
            if pool not in per_pool_corr:
                continue
            info = per_pool_corr[pool]
            print(f"\n[{pool}]  n={info['n']}  score range=[{info['score_min']:.3f}, {info['score_max']:.3f}]  mean={info['score_mean']:.3f}")
            metric_corrs = info["metrics"]
            ranked = sorted(metric_corrs.items(), key=lambda kv: -abs(kv[1]["spearman"]))
            for m, c in ranked:
                marker = "*" if abs(c["spearman"]) >= 0.3 else " "
                print(f"  {marker} {m:<22} pearson={c['pearson']:+.3f}  spearman={c['spearman']:+.3f}")

        print("\n=== CROSS-POOL SCORE CORRELATION (ranked by mean |ρ|) ===")
        header = f"  {'metric':<22} {'mean|ρ|':>8}  {'mean ρ':>8}  {'strong pools':>12}  direction"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for c in cross_pool_corr:
            print(
                f"  {c['metric']:<22} {c['mean_abs_spearman']:>8.3f}  {c['mean_spearman']:>+8.3f}  "
                f"{c['n_pools_strong']:>5}/{c['n_pools_total']:<5}  {c['dominant_direction']}"
                f"{' ' if c['direction_consistent_across_strong_pools'] else ' [mixed]'}"
            )

        # A simple score-corr bar plot
        out_corr_png = OUT_DIR / f"{output_prefix}.score_correlation.png"
        _score_corr_plot(cross_pool_corr, out_corr_png)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="YAML config path")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
