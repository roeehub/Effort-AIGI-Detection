"""
Flask server for the Training Data Viewer.

Endpoints:
  GET  /                           → Dashboard HTML
  GET  /api/discovery-status       → Discovery progress
  POST /api/discover               → Start or force-refresh discovery
  GET  /api/stats                  → Aggregated statistics JSON
  GET  /api/samples                → Paginated sample listing with filters
  GET  /api/sample/<source>/<sid>  → Single sample detail
  GET  /api/frame/<bucket>/<path>  → Proxied GCS frame (cached locally)
  DELETE /api/cache                → Clear local cache
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, jsonify, request, send_file, send_from_directory
from google.cloud import storage
from PIL import Image

from .discovery import discover_all, CACHE_DIR, samples_from_dicts
from .families import infer_family_key
from .splitting import (
    SampleInfo,
    presplit_external_reals,
    split_samples_by_identity,
)
from .visomaster_policy import (
    TEAMS_FAKE_FAMILY,
    TEAMS_SOURCE,
    load_visomaster_bad_data_policy,
    policy_summary_for_response,
)

logger = logging.getLogger(__name__)

FRAME_CACHE_DIR = Path(".viewer_cache/frames").resolve()

app = Flask(__name__, template_folder="templates", static_folder="templates")

# ── Global State ────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "config": {},
    "samples": [],          # List[SampleInfo]
    "discovery_running": False,
    "discovery_progress": {},  # source → (current, total)
    "discovery_done": False,
    "discovery_error": None,
    "stats_cache": None,
    "visomaster_policy": None,
}
_lock = threading.Lock()


def _progress_cb(source: str, current: int, total: int):
    with _lock:
        _state["discovery_progress"][source] = (current, total)


# ── Discovery thread ────────────────────────────────────────────────────────

def _run_discovery(force: bool = False):
    with _lock:
        _state["discovery_running"] = True
        _state["discovery_done"] = False
        _state["discovery_error"] = None
        _state["discovery_progress"] = {}
        _state["stats_cache"] = None
        _state["visomaster_policy"] = None

    try:
        config = _state["config"]
        policy = load_visomaster_bad_data_policy(config)
        samples = discover_all(config, progress_cb=_progress_cb, force_refresh=force)

        # Apply external real pre-split
        cp = config.get("combined_paired", {})
        ext_reals_cfg = cp.get("external_training_reals", [])
        if ext_reals_cfg:
            first = ext_reals_cfg[0] if ext_reals_cfg else {}
            samples = presplit_external_reals(
                samples,
                identity_train_fraction=float(first.get("identity_train_fraction", 0.40)),
                identity_split_seed=int(first.get("identity_split_seed", 737)),
            )

        # Assign family keys
        for s in samples:
            fk = infer_family_key(s.label, s.method, s.source)
            if s.sampling_family_key:
                s.extra["effective_family"] = s.sampling_family_key
            else:
                s.extra["effective_family"] = fk
            s.extra["family_key"] = fk

        # Run identity split
        seed = int(cp.get("split_seed", cp.get("seed", config.get("seed", 737))))
        train_split = float(cp.get("train_split", 0.85))
        val_split = float(cp.get("val_split", 0.10))
        split_samples_by_identity(samples, train_split, val_split, seed)

        # Pre-compute frame_url for each sample
        for s in samples:
            s.extra["frame_url"] = _frame_url_for_sample(s)

        with _lock:
            _state["samples"] = samples
            _state["discovery_done"] = True
            _state["discovery_running"] = False
            _state["visomaster_policy"] = policy_summary_for_response(policy)
        logger.info("Discovery complete: %d samples", len(samples))

    except Exception as exc:
        logger.exception("Discovery failed")
        with _lock:
            _state["discovery_error"] = str(exc)
            _state["discovery_running"] = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _filter_samples(params: dict) -> List[SampleInfo]:
    samples = _state["samples"]
    source = params.get("source")
    method = params.get("method")
    family = params.get("family")
    split = params.get("split")
    label = params.get("label")
    identity = params.get("identity")
    search = params.get("search")
    tier = params.get("tier")

    filtered = samples
    if source:
        filtered = [s for s in filtered if s.source == source]
    if method:
        filtered = [s for s in filtered if s.method == method]
    if family:
        filtered = [s for s in filtered if s.extra.get("effective_family") == family]
    if split:
        filtered = [s for s in filtered if s.split == split]
    if label is not None and label != "":
        label_int = int(label)
        filtered = [s for s in filtered if s.label == label_int]
    if tier:
        filtered = [s for s in filtered if s.tier == tier]
    if identity:
        filtered = [s for s in filtered if identity in s.identity]
    if search:
        search_l = search.lower()
        filtered = [s for s in filtered if search_l in s.sample_id.lower()
                    or search_l in s.method.lower()
                    or search_l in s.identity.lower()
                    or search_l in s.original_video_name.lower()]
    return filtered


def _compute_stats() -> Dict[str, Any]:
    with _lock:
        if _state["stats_cache"]:
            return _state["stats_cache"]

    samples = _state["samples"]
    if not samples:
        return {"total": 0}

    config = _state["config"]
    cp = config.get("combined_paired", {})
    family_weights = cp.get("sampling", {}).get("family_weights", {})

    # Count by various dimensions
    by_source = Counter()
    by_method = Counter()
    by_family = Counter()
    by_split = Counter()
    by_label = Counter()
    by_family_split = defaultdict(Counter)
    by_source_label = defaultdict(Counter)
    by_method_split = defaultdict(Counter)
    by_family_label = defaultdict(Counter)
    by_tier = Counter()
    identities_per_source = defaultdict(set)
    identities_per_family = defaultdict(set)
    family_weight_aliases = defaultdict(Counter)
    method_sources = defaultdict(Counter)
    method_fake_families = defaultdict(Counter)
    method_all_families = defaultdict(Counter)

    for s in samples:
        fam = s.extra.get("effective_family", "unknown")
        by_source[s.source] += 1
        by_method[s.method] += 1
        by_family[fam] += 1
        by_split[s.split] += 1
        by_label[s.label] += 1
        by_family_split[fam][s.split] += 1
        by_source_label[s.source][s.label] += 1
        by_method_split[s.method][s.split] += 1
        by_family_label[fam][s.label] += 1
        if s.tier:
            by_tier[s.tier] += 1
        identities_per_source[s.source].add(s.identity)
        identities_per_family[fam].add(s.identity)
        if s.extra.get("family_weight_alias"):
            family_weight_aliases[fam][s.extra["family_weight_alias"]] += 1
        method_sources[s.method][s.source] += 1
        method_all_families[s.method][fam] += 1
        if s.label == 1:
            method_fake_families[s.method][fam] += 1

    # Compute effective sampling distribution
    # Normalize: for each family, weight × count in training set
    train_by_family = {
        fam: counts.get("train", 0)
        for fam, counts in by_family_split.items()
    }
    weighted = {}
    for fam, count in train_by_family.items():
        alias = None
        if family_weight_aliases.get(fam):
            alias = family_weight_aliases[fam].most_common(1)[0][0]
        w = family_weights.get(fam, family_weights.get(alias, 1.0))
        weighted[fam] = count * w
    total_weighted = sum(weighted.values()) or 1
    effective_pct = {fam: v / total_weighted * 100 for fam, v in weighted.items()}

    # Target domain analysis
    target_families = {
        "deeplive_teams_fake",
        "deeplive_teams_real",
        "visomaster_enhanced_fake",
        TEAMS_FAKE_FAMILY,
    }
    target_pct = sum(effective_pct.get(f, 0) for f in target_families)

    # Identity overlap: identities that appear across multiple sources
    all_identities = defaultdict(set)
    for s in samples:
        all_identities[s.identity].add(s.source)
    multi_source_ids = {
        ident: list(sources)
        for ident, sources in all_identities.items()
        if len(sources) > 1
    }

    # Enhancer distribution (for visomaster_enhanced and teams_enhanced)
    enhancer_counts = Counter()
    for s in samples:
        if s.enhancer and s.label == 1:
            enhancer_counts[s.enhancer] += 1

    # Swap model distribution
    swap_model_counts = Counter()
    for s in samples:
        if s.swap_model and s.label == 1:
            swap_model_counts[s.swap_model] += 1

    # Teams coverage: how many realpool identities have teams counterpart
    realpool_ids = set()
    teams_ids = set()
    for s in samples:
        if s.identity.startswith("realpool_"):
            realpool_ids.add(s.identity)
            if s.source in {"deeplive_teams", TEAMS_SOURCE}:
                teams_ids.add(s.identity)
    teams_coverage = len(teams_ids) / len(realpool_ids) * 100 if realpool_ids else 0

    method_meta = {}
    for method in by_method:
        source = method_sources[method].most_common(1)[0][0] if method_sources[method] else "unknown"
        family_counter = method_fake_families[method] or method_all_families[method]
        family = family_counter.most_common(1)[0][0] if family_counter else "unknown"
        method_meta[method] = {
            "source": source,
            "family": family,
        }

    stats = {
        "total": len(samples),
        "by_source": dict(by_source),
        "by_method": dict(by_method),
        "by_family": dict(by_family),
        "by_split": dict(by_split),
        "by_label": {"real": by_label.get(0, 0), "fake": by_label.get(1, 0)},
        "by_family_split": {k: dict(v) for k, v in by_family_split.items()},
        "by_source_label": {k: {"real": v.get(0, 0), "fake": v.get(1, 0)}
                            for k, v in by_source_label.items()},
        "by_method_split": {k: dict(v) for k, v in by_method_split.items()},
        "by_family_label": {k: {"real": v.get(0, 0), "fake": v.get(1, 0)}
                            for k, v in by_family_label.items()},
        "identities_per_source": {k: len(v) for k, v in identities_per_source.items()},
        "identities_per_family": {k: len(v) for k, v in identities_per_family.items()},
        "family_weights": family_weights,
        "effective_pct": effective_pct,
        "target_domain_pct": target_pct,
        "teams_coverage_pct": teams_coverage,
        "multi_source_identities": len(multi_source_ids),
        "enhancer_counts": dict(enhancer_counts),
        "swap_model_counts": dict(swap_model_counts),
        "tier_counts": dict(by_tier),
        "method_meta": method_meta,
        "family_weight_aliases": {
            fam: aliases.most_common(1)[0][0]
            for fam, aliases in family_weight_aliases.items()
            if aliases
        },
        "visomaster_policy": _state.get("visomaster_policy"),
        "split_config": {
            "seed": int(cp.get("split_seed", cp.get("seed", 737))),
            "train_split": float(cp.get("train_split", 0.85)),
            "val_split": float(cp.get("val_split", 0.10)),
            "test_split": float(cp.get("test_split", 0.05)),
        },
    }

    with _lock:
        _state["stats_cache"] = stats
    return stats


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/discovery-status")
def discovery_status():
    with _lock:
        return jsonify({
            "running": _state["discovery_running"],
            "done": _state["discovery_done"],
            "error": _state["discovery_error"],
            "progress": _state["discovery_progress"],
            "total_samples": len(_state["samples"]),
        })


@app.route("/api/discover", methods=["POST"])
def trigger_discovery():
    force = request.json.get("force", False) if request.is_json else False
    with _lock:
        if _state["discovery_running"]:
            return jsonify({"status": "already_running"}), 409

    t = threading.Thread(target=_run_discovery, args=(force,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/stats")
def get_stats():
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425
    return jsonify(_compute_stats())


@app.route("/api/samples")
def get_samples():
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    filtered = _filter_samples(request.args)
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 40))

    start = (page - 1) * per_page
    end = start + per_page
    page_samples = filtered[start:end]

    return jsonify({
        "total": len(filtered),
        "page": page,
        "per_page": per_page,
        "has_more": end < len(filtered),
        "samples": [asdict(s) for s in page_samples],
    })


@app.route("/api/sample/<source>/<sample_id>")
def get_sample_detail(source, sample_id):
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    for s in _state["samples"]:
        if s.source == source and s.sample_id == sample_id:
            return jsonify(asdict(s))
    return jsonify({"error": "not found"}), 404


@app.route("/api/frame/<bucket>/<path:blob_path>")
def proxy_frame(bucket, blob_path):
    """Proxy GCS frame and cache locally. Returns a thumbnail by default."""
    thumb = request.args.get("full") != "1"
    thumb_size = int(request.args.get("size", 224))

    # Check local cache first
    cache_key = f"{bucket}/{blob_path}"
    if thumb:
        cache_key += f"__thumb{thumb_size}"
    local_path = FRAME_CACHE_DIR / cache_key
    if local_path.exists():
        return send_file(str(local_path.resolve()), mimetype="image/jpeg")

    # Download from GCS
    try:
        client = _get_gcs_client()
        b = client.bucket(bucket)
        blob = b.blob(blob_path)
        data = blob.download_as_bytes()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    if thumb:
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        except Exception:
            pass  # serve original if resize fails

    # Cache
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)

    return send_file(io.BytesIO(data), mimetype="image/jpeg")


@app.route("/api/cache", methods=["DELETE"])
def clear_cache():
    cache_root = Path(".viewer_cache")
    if cache_root.exists():
        shutil.rmtree(cache_root)
    return jsonify({"status": "cleared"})


@app.route("/api/cache-size")
def cache_size():
    cache_root = Path(".viewer_cache")
    if not cache_root.exists():
        return jsonify({"bytes": 0, "human": "0 B"})
    total = sum(f.stat().st_size for f in cache_root.rglob("*") if f.is_file())
    human = _human_size(total)
    return jsonify({"bytes": total, "human": human})


@app.route("/api/filter-options")
def filter_options():
    """Return unique values for all filter dropdowns."""
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425
    samples = _state["samples"]
    return jsonify({
        "sources": sorted(set(s.source for s in samples)),
        "methods": sorted(set(s.method for s in samples)),
        "families": sorted(set(s.extra.get("effective_family", "unknown") for s in samples)),
        "splits": sorted(set(s.split for s in samples if s.split)),
        "tiers": sorted(set(s.tier for s in samples if s.tier)),
        "labels": [0, 1],
    })


@app.route("/api/method-health")
def method_health():
    """Rich per-method health breakdown for data quality assessment."""
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    samples = _state["samples"]
    config = _state["config"]
    cp = config.get("combined_paired", {})
    family_weights = cp.get("sampling", {}).get("family_weights", {})

    # Gather per-method data in one pass
    method_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "real": 0, "fake": 0,
        "train": 0, "val": 0, "test": 0,
        "identities": set(), "frame_counts": [],
        "tiers": Counter(), "sources": Counter(), "families": Counter(),
        "enhancers": Counter(), "swap_models": Counter(),
        "family_weight_aliases": Counter(),
    })

    for s in samples:
        m = method_data[s.method]
        m["total"] += 1
        if s.label == 0:
            m["real"] += 1
        else:
            m["fake"] += 1
        if s.split == "train":
            m["train"] += 1
        elif s.split in ("val_in_dist", "val"):
            m["val"] += 1
        elif s.split == "test":
            m["test"] += 1
        m["identities"].add(s.identity)
        if s.frame_count:
            m["frame_counts"].append(s.frame_count)
        if s.tier:
            m["tiers"][s.tier] += 1
        m["sources"][s.source] += 1
        fam = s.extra.get("effective_family", "unknown")
        if s.label == 1:
            m["families"][fam] += 2
        else:
            m["families"][fam] += 1
        if s.extra.get("family_weight_alias"):
            m["family_weight_aliases"][s.extra["family_weight_alias"]] += 1
        if s.enhancer:
            m["enhancers"][s.enhancer] += 1
        if s.swap_model:
            m["swap_models"][s.swap_model] += 1

    # Build response rows with health scoring
    rows = []
    for method, d in method_data.items():
        total = d["total"]
        real = d["real"]
        fake = d["fake"]
        n_ids = len(d["identities"])
        fcs = d["frame_counts"]

        # Balance ratio: 0 = perfectly balanced, 1 = all one class
        balance_ratio = abs(real - fake) / total if total else 0

        # Samples per identity
        samples_per_id = total / n_ids if n_ids else 0

        # Frame count stats
        avg_frames = sum(fcs) / len(fcs) if fcs else 0
        min_frames = min(fcs) if fcs else 0
        max_frames = max(fcs) if fcs else 0

        # Health checks → list of issues
        issues = []
        if total < 50:
            issues.append("very_few_samples")
        elif total < 200:
            issues.append("few_samples")
        if n_ids < 5:
            issues.append("low_identity_diversity")
        if balance_ratio > 0.8 and total > 20:
            issues.append("severe_class_imbalance")
        elif balance_ratio > 0.5 and total > 20:
            issues.append("class_imbalance")
        if fcs and max_frames - min_frames > 20:
            issues.append("frame_count_variance")

        # Tier issues for visomaster
        tier_counts = dict(d["tiers"])
        if tier_counts:
            tier_total = sum(tier_counts.values())
            artifact_pct = tier_counts.get("ARTIFACT", 0) / tier_total if tier_total else 0
            strong_pct = tier_counts.get("STRONG", 0) / tier_total if tier_total else 0
            if artifact_pct > 0.3:
                issues.append("high_artifact_rate")
            if strong_pct + artifact_pct > 0.5:
                issues.append("low_quality_majority")

        # Overall health: green, yellow, red
        critical = {"severe_class_imbalance", "very_few_samples",
                    "low_identity_diversity", "high_artifact_rate"}
        warning = {"class_imbalance", "few_samples", "frame_count_variance",
                   "low_quality_majority"}
        if any(i in critical for i in issues):
            health = "red"
        elif any(i in warning for i in issues):
            health = "yellow"
        else:
            health = "green"

        # Effective weight
        fam = d["families"].most_common(1)[0][0] if d["families"] else "unknown"
        weight_alias = d["family_weight_aliases"].most_common(1)[0][0] if d["family_weight_aliases"] else None
        weight = family_weights.get(fam, family_weights.get(weight_alias, 1.0))

        rows.append({
            "method": method,
            "source": d["sources"].most_common(1)[0][0] if d["sources"] else "unknown",
            "family": fam,
            "total": total,
            "real": real,
            "fake": fake,
            "train": d["train"],
            "val": d["val"],
            "test": d["test"],
            "n_identities": n_ids,
            "samples_per_identity": round(samples_per_id, 1),
            "avg_frames": round(avg_frames, 1),
            "min_frames": min_frames,
            "max_frames": max_frames,
            "balance_ratio": round(balance_ratio, 3),
            "tier_counts": tier_counts,
            "enhancer_counts": dict(d["enhancers"]),
            "swap_model_counts": dict(d["swap_models"]),
            "weight": weight,
            "issues": issues,
            "health": health,
        })

    rows.sort(key=lambda r: ({"red": 0, "yellow": 1, "green": 2}[r["health"]], -r["total"]))
    return jsonify({"methods": rows, "total_methods": len(rows)})


@app.route("/api/method-preview/<method>")
def method_preview(method):
    """Return random sample thumbnails for quick visual assessment of a method."""
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    import random
    samples = _state["samples"]
    method_samples = [s for s in samples if s.method == method]

    real_samples = [s for s in method_samples if s.label == 0]
    fake_samples = [s for s in method_samples if s.label == 1]

    n = int(request.args.get("n", 6))
    seed = int(request.args.get("seed", int(time.time())))
    rng = random.Random(seed)

    def pick(lst, count):
        if len(lst) <= count:
            chosen = lst[:]
        else:
            chosen = rng.sample(lst, count)
        return [{
            "sample_id": s.sample_id,
            "source": s.source,
            "identity": s.identity,
            "frame_url": s.extra.get("frame_url", ""),
            "tier": s.tier or "",
            "split": s.split or "",
            "frame_count": s.frame_count,
        } for s in chosen]

    return jsonify({
        "method": method,
        "real": pick(real_samples, n),
        "fake": pick(fake_samples, n),
        "total_real": len(real_samples),
        "total_fake": len(fake_samples),
        "seed": seed,
    })


@app.route("/api/method-gallery/<method>")
def method_gallery(method):
    """Paginated gallery: returns samples with multiple frame URLs for deep visual review."""
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    import random
    samples = _state["samples"]
    method_samples = [s for s in samples if s.method == method]

    # Filter by label
    label_filter = request.args.get("label", "")  # "real", "fake", or "" for both
    if label_filter == "real":
        method_samples = [s for s in method_samples if s.label == 0]
    elif label_filter == "fake":
        method_samples = [s for s in method_samples if s.label == 1]

    # Filter by split
    split_filter = request.args.get("split", "")
    if split_filter:
        method_samples = [s for s in method_samples if s.split == split_filter]

    # Sort options
    sort_by = request.args.get("sort", "random")  # random, identity, frames
    seed = int(request.args.get("seed", 42))
    if sort_by == "random":
        rng = random.Random(seed)
        rng.shuffle(method_samples)
    elif sort_by == "identity":
        method_samples.sort(key=lambda s: s.identity)
    elif sort_by == "frames":
        method_samples.sort(key=lambda s: s.frame_count or 0)

    # Paginate
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 24))
    total = len(method_samples)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    page_samples = method_samples[start:end]

    # Build response with multiple frames per sample
    frames_per = int(request.args.get("frames_per", 3))
    results = []
    for s in page_samples:
        max_frames = min(s.frame_count or 1, frames_per)
        frame_urls = []
        for fi in range(max_frames):
            url = _frame_url_for_sample(s, fi)
            if url:
                frame_urls.append(url)
        results.append({
            "sample_id": s.sample_id,
            "source": s.source,
            "identity": s.identity,
            "label": "fake" if s.label == 1 else "real",
            "tier": s.tier or "",
            "split": s.split or "",
            "frame_count": s.frame_count,
            "frame_urls": frame_urls,
            "enhancer": s.enhancer or "",
            "swap_model": s.swap_model or "",
        })

    return jsonify({
        "method": method,
        "total": total,
        "page": page,
        "per_page": per_page,
        "has_more": end < total,
        "total_pages": (total + per_page - 1) // per_page,
        "seed": seed,
        "samples": results,
    })


@app.route("/api/sample-detail/<sample_id>")
def sample_detail(sample_id):
    """Return detailed sample info + all frames + paired counterpart (if any)."""
    if not _state["discovery_done"]:
        return jsonify({"error": "Discovery not complete"}), 425

    samples = _state["samples"]
    frames_per = int(request.args.get("frames_per", 8))

    # Find the target sample
    target = None
    for s in samples:
        if s.sample_id == sample_id:
            target = s
            break
    if not target:
        return jsonify({"error": "Sample not found"}), 404

    def _build_sample_data(s, n_frames):
        max_frames = min(s.frame_count or 1, n_frames)
        frame_urls = []
        for fi in range(max_frames):
            url = _frame_url_for_sample(s, fi)
            if url:
                frame_urls.append(url)
        return {
            "sample_id": s.sample_id,
            "source": s.source,
            "identity": s.identity,
            "method": s.method,
            "label": "fake" if s.label == 1 else "real",
            "tier": s.tier or "",
            "split": s.split or "",
            "frame_count": s.frame_count,
            "frame_urls": frame_urls,
            "enhancer": s.enhancer or "",
            "swap_model": s.swap_model or "",
            "strategy": s.strategy or "",
            "original_video_name": s.original_video_name or "",
            "policy_label": s.extra.get("policy_label", ""),
            "policy_lane": s.extra.get("policy_lane", ""),
            "policy_action": s.extra.get("policy_action", ""),
            "policy_nominal_method": s.extra.get("policy_nominal_method", ""),
        }

    result = _build_sample_data(target, frames_per)

    # Find paired counterpart
    pair_data = None
    if target.has_pair:
        # All sources: sample_id ends with __fake or __real
        base_id = target.sample_id.replace("__fake", "").replace("__real", "")
        opposite_suffix = "__real" if target.label == 1 else "__fake"
        pair_id = base_id + opposite_suffix

        # For df40: also check via pair_id in extra
        for s in samples:
            if s.sample_id == pair_id:
                pair_data = _build_sample_data(s, frames_per)
                break

        # df40 fallback: search by pair_id + opposite label
        if pair_data is None and target.source == "df40":
            df40_pair_id = target.extra.get("pair_id", "")
            if df40_pair_id:
                for s in samples:
                    if s.source == "df40" and s.extra.get("pair_id") == df40_pair_id and s.label != target.label:
                        pair_data = _build_sample_data(s, frames_per)
                        break

    result["pair"] = pair_data
    return jsonify(result)


# ── Helpers ──────────────────────────────────────────────────────────────────

_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        proj = os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
        _gcs_client = storage.Client(project=proj)
    return _gcs_client


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ── Frame URL helpers ────────────────────────────────────────────────────────

def _frame_url_for_sample(s: SampleInfo, frame_idx: int = 0) -> str:
    """Build the /api/frame/<bucket>/<path> URL for a sample."""
    if s.source == "df40":
        # DF40: paths are blob-relative like 'fake/blendface/001_870/'
        if s.label == 1:
            prefix = s.extra.get("fake_path", "")
            frame_list = s.extra.get("fake_frames", [])
        else:
            prefix = s.extra.get("real_path", "")
            frame_list = s.extra.get("real_frames", [])
        if prefix:
            # Use actual frame filenames from the pair JSON (not sequential)
            if frame_idx < len(frame_list):
                frame_name = frame_list[frame_idx]
            else:
                frame_name = frame_list[0] if frame_list else "000.png"
            return f"/api/frame/{s.bucket}/{prefix}{frame_name}"
        return ""

    if s.source == "external":
        gcs_path = s.extra.get("gcs_path", "")
        return f"/api/frame/{s.bucket}/{gcs_path}" if gcs_path else ""

    # DeepLive, VisoMaster, Teams, Enhanced: standard layout
    base_id = s.sample_id.replace("__fake", "").replace("__real", "")
    label_dir = "fake" if s.label == 1 else "real"

    if s.source in ("visomaster_enhanced",) and s.label == 1:
        # Enhanced fakes are in the enhanced bucket
        return f"/api/frame/{s.bucket}/samples/{base_id}/frames/fake/frame_{frame_idx:04d}.png"

    ext = ".jpg" if s.source in ("deeplive_teams", TEAMS_SOURCE) else ".png"
    return f"/api/frame/{s.bucket}/samples/{base_id}/frames/{label_dir}/frame_{frame_idx:04d}{ext}"


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training Data Viewer")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--auto-discover", action="store_true", default=True,
                        help="Start discovery automatically on launch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    _state["config"] = config

    logger.info("Loaded config: %s", args.config)
    logger.info("Starting viewer at http://%s:%d", args.host, args.port)

    if args.auto_discover:
        t = threading.Thread(target=_run_discovery, daemon=True)
        t.start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
