#!/usr/bin/env python3
"""D9 source-substrate inventory audit.

Quantifies the source-substrate distribution of:
  (a) R13 training-yaml referenced training-data lanes
  (b) Dev real entries in the eval manifest
  (c) Lockbox real entries in the eval manifest

READ-ONLY. No GCS writes. CPU/IO-light.

Outputs:
  outputs/d9_train_yaml_inventory.csv
  outputs/d9_train_lane_inventory.csv
  outputs/d9_eval_real_per_video.csv
  outputs/d9_train_manifest_per_sample.csv
  outputs/d9_crosstab_per_split.csv
  outputs/d9_schema_gap.csv
  outputs/d9_train_manifest_schema.txt
  outputs/d9_eval_manifest_schema.txt
  outputs/d9_capture_mode_value_summary.txt
  _run.log
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve()
HERE = ROOT.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
LOG = HERE / "_run.log"

REPO_TRAIN = HERE.parents[1]  # .../DeepfakeBench/training
EXPERIMENTS = REPO_TRAIN / "experiments" / "phase2_round13"
EVAL_MANIFEST = REPO_TRAIN / "arena" / "manifests" / "teams_target_domain_manifest_2026-04-23_with_dor.json"
PROPER_MANIFEST = REPO_TRAIN / "arena" / "manifests" / "proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
PRIOR_GCS = REPO_TRAIN / "analysis" / "cpu_diagnostics_2026-05-12_gcs_identity_audit"
TRAIN_BUCKET = "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"

# YouTube video-ID pattern: 11 chars from [A-Za-z0-9_-]
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
YT_EMBEDDED_RE = re.compile(r"_([A-Za-z0-9_-]{11})\.mp4$")

# Eval identity tokens that should NOT match YouTube (named persons / sessions).
# These are ALL observed dev + lockbox real prefixes in
# arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json
# plus their case/spelling variants.
EVAL_NAMED_TOKENS = (
    # Dev (13 prefixes)
    "test_cam", "pc_generator", "bla_bla_chow", "md_noyn_sharker",
    "xiang_xiang2_feng", "dor", "xiang", "roy_d", "cam_test",
    "dor_shkedi", "orel", "ilan",
    # Q is a 1-char prefix; skip in token list (use prefix-field match
    # instead; classify_via_prefix below handles it).
    # Lockbox additional
    "real_dor", "chikara_takahashi",
)
# Single-letter / very short prefixes that exist in eval but would cause
# false positives if used as substring tokens. We handle these via the
# prefix-field match path (caller passes manifest 'prefix' value).
EVAL_NAMED_PREFIX_VALUES = (
    "Q",  # one-letter prefix exists in dev real_prefixes
)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], check: bool = True) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        sys.stderr.write(
            f"CMD failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\n"
        )
        raise SystemExit(2)
    return res.stdout


# -------------------- 1. Train yaml enumeration --------------------

def step1_train_yamls() -> list[Path]:
    yamls = sorted(EXPERIMENTS.glob("*.yaml"))
    log(f"Step 1: Enumerated {len(yamls)} R13 yamls in {EXPERIMENTS}")
    return yamls


# -------------------- 2. Extract training-data lane configs from yamls --------------------

def yaml_read_lanes(yaml_path: Path) -> dict[str, Any]:
    """Light yaml parser using string regex for the few keys we care about.

    Avoids pulling in pyyaml (works on CPU-only env, no extra deps).
    Returns dict {lane: enabled_bool_or_None, gcs_bucket: str_or_None, ...}.
    """
    text = yaml_path.read_text()
    info = {
        "yaml_name": yaml_path.name,
        "n_lines": len(text.splitlines()),
    }

    # Top-level data lanes (under combined_paired and combined_source).
    # We track presence + enabled flag + bucket.
    lane_candidates = [
        "teams",
        "visomaster_hints_teams",
        "proper_data",
        "external_training_reals",
        "deeplive",
        "visomaster",
        "visomaster_hints",
        "visomaster_teams_enhanced",
        "visomaster_enhanced",
    ]

    # Scoped extraction: find `<lane>:` followed by indented block
    # We just check if `enabled: true|false` appears within the immediate block.
    for lane in lane_candidates:
        # Pattern: ^  <lane>:\n(    .+\n)*
        rx = re.compile(
            rf"(?m)^( {{2,4}}){re.escape(lane)}:\s*$\n((?:\1 .*\n|\s*\n)+)"
        )
        m = rx.search(text)
        if not m:
            info[f"lane_{lane}_present"] = False
            info[f"lane_{lane}_enabled"] = None
            info[f"lane_{lane}_bucket"] = None
            continue
        info[f"lane_{lane}_present"] = True
        block = m.group(2)
        en_m = re.search(r"^\s*enabled:\s*(true|false)\b", block, re.MULTILINE)
        if en_m:
            info[f"lane_{lane}_enabled"] = (en_m.group(1) == "true")
        else:
            info[f"lane_{lane}_enabled"] = None
        bk_m = re.search(r"gcs_bucket:\s*[\"']?([^\s\"']+)", block)
        info[f"lane_{lane}_bucket"] = bk_m.group(1) if bk_m else None
        # also catch the bucket: field (for external_training_reals)
        if not info[f"lane_{lane}_bucket"]:
            bk_m = re.search(r"bucket:\s*[\"']?([^\s\"']+)", block)
            info[f"lane_{lane}_bucket"] = bk_m.group(1) if bk_m else None

    # Capture ood_monitoring.external_real_sources (NOT loss-bearing)
    rx_ood = re.compile(
        r"(?m)^  ood_monitoring:\s*$\n((?:    .*\n|\s*\n)+)"
    )
    m = rx_ood.search(text)
    info["ood_monitoring_present"] = bool(m)

    return info


def step2_train_yaml_inventory(yamls: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for yp in yamls:
        rows.append(yaml_read_lanes(yp))
    log(f"Step 2: Parsed {len(rows)} yaml lane configs")
    # Write CSV
    csv_path = OUT / "d9_train_yaml_inventory.csv"
    if rows:
        cols = sorted(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log(f"  wrote {csv_path}")
    return rows


# -------------------- 3. Summarize lane-enabled counts across yamls --------------------

def step3_lane_inventory(rows: list[dict[str, Any]]) -> None:
    """For each lane, count yamls that have it enabled=true."""
    lane_candidates = [
        "teams", "visomaster_hints_teams", "proper_data",
        "external_training_reals", "deeplive", "visomaster",
        "visomaster_hints", "visomaster_teams_enhanced",
        "visomaster_enhanced",
    ]
    summary = []
    for lane in lane_candidates:
        n_present = sum(1 for r in rows if r.get(f"lane_{lane}_present"))
        n_enabled_true = sum(1 for r in rows if r.get(f"lane_{lane}_enabled") is True)
        n_enabled_false = sum(1 for r in rows if r.get(f"lane_{lane}_enabled") is False)
        buckets = Counter(r.get(f"lane_{lane}_bucket") for r in rows if r.get(f"lane_{lane}_enabled") is True)
        buckets.pop(None, None)
        summary.append({
            "lane": lane,
            "n_yamls_with_lane_block": n_present,
            "n_yamls_enabled_true": n_enabled_true,
            "n_yamls_enabled_false": n_enabled_false,
            "n_yamls_enabled_unspecified": n_present - n_enabled_true - n_enabled_false,
            "buckets_when_enabled": ";".join(f"{k}={v}" for k, v in buckets.most_common()),
        })
    csv_path = OUT / "d9_train_lane_inventory.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    log(f"Step 3: lane inventory written -> {csv_path}")


# -------------------- 4. Classifier --------------------

def classify_video_id_or_path(s: str) -> str:
    """Classify a source-content identifier string.

    Returns one of: 'youtube_origin', 'direct_teams_capture',
                    'hdtf_corpus', 'vcd_real_external', 'other'.

    Precedence order:
      1. exact eval-prefix match (handles 1-char prefixes like "Q")
      2. HDTF tokens (avoids HDTF being treated as VCD via "real" substring)
      3. VCD (effort-collected-data)
      4. YouTube-ID embedded patterns
      5. Named-person tokens (substring match, e.g. "dor_shkedi")
      6. bare YouTube ID (11-char)
      7. "other"
    """
    if not s:
        return "other"
    s_lc = s.lower()

    # 1) exact eval-prefix match
    if s in EVAL_NAMED_PREFIX_VALUES:
        return "direct_teams_capture"

    # 2) HDTF (RD_Radio, WDA_*, WRA_*) or QCLIPS (QCLIP*) — both are
    # YouTube-derived corpora used in the proper_data lane.
    if "hdtf" in s_lc or re.match(r"^(WDA|WRA|RD)_", s, re.IGNORECASE):
        return "hdtf_corpus"
    if s_lc.startswith("qclip"):
        return "qclips_corpus"

    # 3) effort-collected VCD: real__VCD__<md5>_
    if "_vcd_" in f"_{s_lc}_" or s_lc.startswith("real__vcd"):
        return "vcd_real_external"

    # 4) YouTube ID inside a string that follows "cropped_<11char>.mp4" pattern
    m = YT_EMBEDDED_RE.search(s)
    if m:
        return "youtube_origin"

    # 5) named tokens (substring match)
    for tok in EVAL_NAMED_TOKENS:
        if tok in s_lc:
            return "direct_teams_capture"

    # 6) bare 11-char YouTube ID
    if YT_ID_RE.match(s):
        return "youtube_origin"

    # 7) sample_id pattern like "edge_cases_0000" -- yields no info on its own;
    # caller should pass us original_trimmed_video_name instead.
    if re.match(r"^(edge_cases|minimal_processing|quality_enhancement|visomaster_[A-Za-z0-9-]+)_\d+$", s):
        return "other"  # unresolved without manifest contents

    return "other"


# -------------------- 5. Eval-manifest classification --------------------

def step5_eval_real_classification() -> dict[str, Any]:
    log(f"Step 5: Loading eval manifest {EVAL_MANIFEST}")
    with EVAL_MANIFEST.open() as f:
        data = json.load(f)
    videos = data["videos"]
    log(f"  {len(videos)} video entries total")

    # Schema
    all_keys = set()
    for v in videos:
        all_keys.update(v.keys())
    (OUT / "d9_eval_manifest_schema.txt").write_text(
        f"Total videos: {len(videos)}\nUnion keys: {sorted(all_keys)}\n"
    )

    # Per-real-video classification
    rows = []
    n_real_dev = 0
    n_real_lockbox = 0
    n_real_frames_dev = 0
    n_real_frames_lockbox = 0
    src_substrate_x_split = Counter()
    src_kind_x_split = Counter()
    capture_modes_seen = Counter()
    prefix_x_split = defaultdict(Counter)
    methods_x_split = defaultdict(Counter)
    identity_keys_seen = defaultdict(set)  # split -> set(identity_key)

    for v in videos:
        if v.get("label") != "real":
            continue
        split = v.get("split", "unknown")
        n_frames = len(v.get("frame_paths", []) or [])
        if split == "dev":
            n_real_dev += 1
            n_real_frames_dev += n_frames
        elif split == "lockbox":
            n_real_lockbox += 1
            n_real_frames_lockbox += n_frames

        # Classification - use prefix + identity_key + video_id ensemble
        candidates = [
            v.get("prefix", ""),
            v.get("identity_key", ""),
            v.get("video_id", ""),
        ]
        cls = "other"
        for c in candidates:
            res = classify_video_id_or_path(c)
            if res != "other":
                cls = res
                break

        src_substrate_x_split[(split, cls)] += 1
        src_kind_x_split[(split, v.get("source_kind"))] += 1
        prefix_x_split[split][v.get("prefix")] += 1
        methods_x_split[split][v.get("method")] += 1
        identity_keys_seen[split].add(v.get("identity_key"))

        rows.append({
            "split": split,
            "label": v.get("label"),
            "video_id": v.get("video_id"),
            "identity_key": v.get("identity_key"),
            "prefix": v.get("prefix"),
            "method": v.get("method"),
            "source_kind": v.get("source_kind"),
            "n_frames": n_frames,
            "source_substrate_class": cls,
        })

    # Capture-mode: not a field in eval manifest; record None
    # We instead surface source_kind values

    # Write per-video CSV
    csv_path = OUT / "d9_eval_real_per_video.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {csv_path}")

    log(f"  Real videos: dev={n_real_dev}, lockbox={n_real_lockbox}")
    log(f"  Real frames: dev={n_real_frames_dev}, lockbox={n_real_frames_lockbox}")
    log(f"  Unique identity_keys: dev={len(identity_keys_seen['dev'])}, lockbox={len(identity_keys_seen['lockbox'])}")
    log(f"  Source substrate x split: {dict(src_substrate_x_split)}")
    log(f"  source_kind x split: {dict(src_kind_x_split)}")

    return {
        "n_real_dev": n_real_dev,
        "n_real_lockbox": n_real_lockbox,
        "n_real_frames_dev": n_real_frames_dev,
        "n_real_frames_lockbox": n_real_frames_lockbox,
        "src_substrate_x_split": dict(src_substrate_x_split),
        "src_kind_x_split": dict(src_kind_x_split),
        "prefix_x_split": {k: dict(v) for k, v in prefix_x_split.items()},
        "methods_x_split": {k: dict(v) for k, v in methods_x_split.items()},
        "identity_keys_dev": sorted(identity_keys_seen["dev"]),
        "identity_keys_lockbox": sorted(identity_keys_seen["lockbox"]),
        "schema_keys": sorted(all_keys),
    }


# -------------------- 6. Training-manifest classification --------------------

def step6_train_manifest_classification() -> dict[str, Any]:
    """Pull from teams-v2 + proper_data manifests already on disk.

    teams-v2: reuse the 250-manifest pull from the prior GCS audit (cached).
    proper_data: read the local provisional manifest JSON.
    """
    log("Step 6a: Reading prior-audit cached teams-v2 manifests")
    cached_examples = PRIOR_GCS / "outputs" / "sample_manifest_examples.json"
    picked_ids = PRIOR_GCS / "outputs" / "_picked_sample_ids.txt"
    teams_rows = []

    if cached_examples.exists():
        with cached_examples.open() as f:
            examples = json.load(f)
        log(f"  {len(examples)} examples in prior audit cache (one per strategy prefix)")
        # The prior audit only kept 10 examples (one per strategy prefix), but
        # picked 250 sample_ids. We need to re-fetch them OR accept that
        # we only have 10 cached. The schema is fully consistent across all
        # 250 per prior audit; classify the 10 we have here, and report that
        # the full 250-sample classification needs re-fetching from GCS.
    else:
        examples = {}
        log(f"  no cached examples")

    # Fetch the full 250 manifests (or whatever picked_sample_ids has)
    sample_id_count = 0
    if picked_ids.exists():
        picks = [s.strip() for s in picked_ids.read_text().splitlines() if s.strip()]
        sample_id_count = len(picks)
        log(f"  prior audit picked {sample_id_count} sample_ids; fetching manifests")
    else:
        picks = []
        log("  no _picked_sample_ids.txt -- nothing to fetch")

    n_fetched_ok = 0
    n_fetch_err = 0
    for i, sid in enumerate(picks):
        # avoid re-fetch if we already have it cached
        cached_path = OUT / "_manifest_cache" / f"{sid}.json"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        if cached_path.exists():
            try:
                m = json.loads(cached_path.read_text())
            except Exception:
                m = None
        else:
            uri = f"{TRAIN_BUCKET}/samples/{sid}/manifest.json"
            try:
                out = run(["gsutil", "cat", uri], check=False)
            except Exception:
                out = ""
            if not out.strip():
                n_fetch_err += 1
                continue
            try:
                m = json.loads(out)
                cached_path.write_text(json.dumps(m))
            except Exception:
                n_fetch_err += 1
                continue
        n_fetched_ok += 1
        if (i + 1) % 50 == 0:
            log(f"  fetched {i+1}/{len(picks)}")

        # Classify based on original_trimmed_video_name, falling back to
        # original_original_video_name then original_cropped_video_name.
        # Visomaster-strategy samples have empty trimmed_video_name; the
        # YouTube ID lives in original_video_name / cropped_video_name.
        otv = m.get("original_trimmed_video_name") or ""
        oov = m.get("original_original_video_name") or ""
        ocv = m.get("original_cropped_video_name") or ""
        cls = "other"
        used_field = "none"
        for field, val in (
            ("original_trimmed_video_name", otv),
            ("original_original_video_name", oov),
            ("original_cropped_video_name", ocv),
        ):
            if val:
                c = classify_video_id_or_path(val)
                if c != "other":
                    cls = c
                    used_field = field
                    break
        # record the actual value used for classification
        source_id_value = otv or oov or ocv
        # Frame counts
        n_real = m.get("frame_count_real") or m.get("teams_real_frames_written") or 0
        n_fake = m.get("frame_count_fake") or m.get("teams_fake_frames_written") or 0
        teams_rows.append({
            "source": "teams-v2",
            "sample_id": sid,
            "strategy": m.get("strategy") or m.get("source"),
            "original_trimmed_video_name": otv,
            "original_original_video_name": oov,
            "original_cropped_video_name": ocv,
            "classified_via_field": used_field,
            "classified_via_value": source_id_value,
            "pair_complete": m.get("pair_complete"),
            "n_real_frames": n_real,
            "n_fake_frames": n_fake,
            "source_substrate_class": cls,
        })

    log(f"  teams-v2: {n_fetched_ok} fetched, {n_fetch_err} errors")

    # Union schema
    teams_keys = set()
    for r in teams_rows:
        teams_keys.update(r.keys())

    # Now scan the proper_visomaster manifest for the proper_data lane (HDTF)
    log("Step 6b: Reading proper_visomaster manifest")
    with PROPER_MANIFEST.open() as f:
        pm = json.load(f)
    proper_videos = pm["videos"]
    proper_rows = []
    for v in proper_videos:
        if v.get("label") != "real":
            continue
        # Classify by base_capture_id (HDTF20260416_00000 / QCLIP...) first,
        # then identity_id, then identity_key. base_capture_id is most
        # reliable for distinguishing HDTF vs QCLIPS.
        candidates = [v.get("base_capture_id"), v.get("identity_id"), v.get("identity_key")]
        cls = "other"
        for c in candidates:
            if c:
                res = classify_video_id_or_path(str(c))
                if res != "other":
                    cls = res
                    break
        # Fallback: proper_data reals are always YouTube-derived corpora
        if cls == "other":
            cls = "hdtf_or_qclips_unknown"
        proper_rows.append({
            "source": "proper_data",
            "video_id": v.get("video_id"),
            "identity_id": v.get("identity_id"),
            "identity_key": v.get("identity_key"),
            "base_capture_id": v.get("base_capture_id"),
            "lane": v.get("lane"),
            "method": v.get("method"),
            "transport": v.get("transport"),
            "playback_path": v.get("playback_path"),
            "n_frames": len(v.get("frame_paths", []) or []),
            "source_substrate_class": cls,
        })

    # External_training_reals — heuristic from yaml: VCD identifier pattern
    # We can't enumerate the actual GCS objects in scope but we record the
    # identity_pattern from the yaml (per the P8A yaml) as part of the
    # source-substrate inventory.

    # Write per-sample CSV (teams-v2 sample-level + proper-video-level)
    csv_path = OUT / "d9_train_manifest_per_sample.csv"
    all_rows = []
    for r in teams_rows:
        all_rows.append({
            "source_lane": "combined_paired.teams (teams-v2)",
            "sample_or_video_id": r["sample_id"],
            "strategy_or_method": r.get("strategy"),
            "source_id_field": r["original_trimmed_video_name"],
            "n_real_frames": r["n_real_frames"],
            "n_fake_frames": r["n_fake_frames"],
            "source_substrate_class": r["source_substrate_class"],
        })
    for r in proper_rows:
        all_rows.append({
            "source_lane": f"combined_paired.proper_data ({r['lane']})",
            "sample_or_video_id": r["video_id"],
            "strategy_or_method": r["method"],
            "source_id_field": r["identity_id"],
            "n_real_frames": r["n_frames"],
            "n_fake_frames": 0,
            "source_substrate_class": r["source_substrate_class"],
        })
    if all_rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    log(f"  wrote {csv_path} ({len(all_rows)} rows)")

    # Write schema
    proper_keys = set()
    for v in proper_videos:
        proper_keys.update(v.keys())
    (OUT / "d9_train_manifest_schema.txt").write_text(
        "Training manifests carry two distinct schemas:\n\n"
        "(A) teams-v2 per-sample manifest (27-key schema from prior GCS audit):\n"
        "  fake_complete, frame_count_fake, frame_count_real, image_format,\n"
        "  original_anchor_frames, original_consecutive_frames, original_cropped,\n"
        "  original_cropped_video_name, original_frame_count_fake, original_frame_count_real,\n"
        "  original_has_landmarks, original_original_video_name, original_pipeline_version,\n"
        "  original_trimmed_video_name, pair_complete, pipeline_version, real_complete,\n"
        "  sample_id, source, strategy, teams_fake_duration_s, teams_fake_frames_discarded,\n"
        "  teams_fake_frames_written, teams_real_duration_s, teams_real_frames_discarded,\n"
        "  teams_real_frames_written, uploaded_at\n\n"
        f"(B) proper_visomaster manifest (lane=proper_data, video-record schema):\n"
        f"  {sorted(proper_keys)}\n"
    )

    return {
        "teams_v2_rows": teams_rows,
        "proper_rows": proper_rows,
        "teams_v2_n_fetched": n_fetched_ok,
        "teams_v2_n_errors": n_fetch_err,
        "teams_v2_substrate_classification": Counter(r["source_substrate_class"] for r in teams_rows),
        "proper_substrate_classification": Counter(r["source_substrate_class"] for r in proper_rows),
        "proper_schema_keys": sorted(proper_keys),
    }


# -------------------- 7. Cross-tab --------------------

def step7_crosstab(eval_result: dict, train_result: dict) -> None:
    rows = []

    # (a) Training train-bucket teams-v2 (frame-weighted)
    teams_substrate_frames = Counter()
    teams_substrate_samples = Counter()
    for r in train_result["teams_v2_rows"]:
        n_real = r.get("n_real_frames", 0) or 0
        teams_substrate_frames[r["source_substrate_class"]] += n_real
        teams_substrate_samples[r["source_substrate_class"]] += 1

    proper_substrate_frames = Counter()
    proper_substrate_samples = Counter()
    for r in train_result["proper_rows"]:
        n = r.get("n_frames", 0) or 0
        proper_substrate_frames[r["source_substrate_class"]] += n
        proper_substrate_samples[r["source_substrate_class"]] += 1

    eval_substrate_frames = defaultdict(Counter)  # split -> Counter
    eval_substrate_videos = defaultdict(Counter)

    # Build from per-video CSV we just wrote
    with (OUT / "d9_eval_real_per_video.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            cls = row["source_substrate_class"]
            n_frames = int(row["n_frames"])
            eval_substrate_videos[split][cls] += 1
            eval_substrate_frames[split][cls] += n_frames

    all_classes = sorted(set(list(teams_substrate_frames.keys())
                              + list(proper_substrate_frames.keys())
                              + list(eval_substrate_frames["dev"].keys())
                              + list(eval_substrate_frames["lockbox"].keys())))

    # Build crosstab rows: one row per source-substrate class
    crosstab = []
    teams_total_frames = sum(teams_substrate_frames.values())
    teams_total_samples = sum(teams_substrate_samples.values())
    proper_total_frames = sum(proper_substrate_frames.values())
    proper_total_samples = sum(proper_substrate_samples.values())
    dev_total_frames = sum(eval_substrate_frames["dev"].values())
    dev_total_videos = sum(eval_substrate_videos["dev"].values())
    lb_total_frames = sum(eval_substrate_frames["lockbox"].values())
    lb_total_videos = sum(eval_substrate_videos["lockbox"].values())

    def frac(x, total):
        if total == 0:
            return ""
        return f"{x / total:.4f}"

    for cls in all_classes:
        crosstab.append({
            "source_substrate_class": cls,
            "train_teamsv2_n_samples": teams_substrate_samples.get(cls, 0),
            "train_teamsv2_n_real_frames": teams_substrate_frames.get(cls, 0),
            "train_teamsv2_frame_fraction": frac(teams_substrate_frames.get(cls, 0), teams_total_frames),
            "train_proper_n_videos": proper_substrate_samples.get(cls, 0),
            "train_proper_n_real_frames": proper_substrate_frames.get(cls, 0),
            "train_proper_frame_fraction": frac(proper_substrate_frames.get(cls, 0), proper_total_frames),
            "dev_n_videos": eval_substrate_videos["dev"].get(cls, 0),
            "dev_n_frames": eval_substrate_frames["dev"].get(cls, 0),
            "dev_frame_fraction": frac(eval_substrate_frames["dev"].get(cls, 0), dev_total_frames),
            "lockbox_n_videos": eval_substrate_videos["lockbox"].get(cls, 0),
            "lockbox_n_frames": eval_substrate_frames["lockbox"].get(cls, 0),
            "lockbox_frame_fraction": frac(eval_substrate_frames["lockbox"].get(cls, 0), lb_total_frames),
        })

    # Totals row
    crosstab.append({
        "source_substrate_class": "_TOTAL_",
        "train_teamsv2_n_samples": teams_total_samples,
        "train_teamsv2_n_real_frames": teams_total_frames,
        "train_teamsv2_frame_fraction": "",
        "train_proper_n_videos": proper_total_samples,
        "train_proper_n_real_frames": proper_total_frames,
        "train_proper_frame_fraction": "",
        "dev_n_videos": dev_total_videos,
        "dev_n_frames": dev_total_frames,
        "dev_frame_fraction": "",
        "lockbox_n_videos": lb_total_videos,
        "lockbox_n_frames": lb_total_frames,
        "lockbox_frame_fraction": "",
    })

    csv_path = OUT / "d9_crosstab_per_split.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crosstab[0].keys()))
        w.writeheader()
        for r in crosstab:
            w.writerow(r)
    log(f"Step 7: crosstab written -> {csv_path}")


# -------------------- 8. Schema gap --------------------

def step8_schema_gap(eval_result: dict, train_result: dict) -> None:
    train_v2_keys = set([
        "fake_complete", "frame_count_fake", "frame_count_real", "image_format",
        "original_anchor_frames", "original_consecutive_frames", "original_cropped",
        "original_cropped_video_name", "original_frame_count_fake", "original_frame_count_real",
        "original_has_landmarks", "original_original_video_name", "original_pipeline_version",
        "original_trimmed_video_name", "pair_complete", "pipeline_version", "real_complete",
        "sample_id", "source", "strategy", "teams_fake_duration_s", "teams_fake_frames_discarded",
        "teams_fake_frames_written", "teams_real_duration_s", "teams_real_frames_discarded",
        "teams_real_frames_written", "uploaded_at",
    ])
    eval_keys = set(eval_result["schema_keys"])
    proper_keys = set(train_result["proper_schema_keys"])

    all_keys = train_v2_keys | eval_keys | proper_keys

    rows = []
    for k in sorted(all_keys):
        rows.append({
            "field_name": k,
            "in_train_teamsv2_schema": k in train_v2_keys,
            "in_train_proper_schema": k in proper_keys,
            "in_eval_schema": k in eval_keys,
        })

    csv_path = OUT / "d9_schema_gap.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Step 8: schema gap written -> {csv_path}")

    # Capture-mode value summary: scan all 3 schemas for "capture_mode"-like
    # fields
    notes = []
    notes.append("=== capture_mode / source_kind value distributions ===\n")
    notes.append("\n[A] teams-v2 train manifests: NO capture_mode field; source_kind absent.")
    notes.append("    Available origin fields: original_trimmed_video_name (YouTube-ID-style 11-char tokens).\n")

    # source_kind in eval
    with EVAL_MANIFEST.open() as f:
        ev = json.load(f)
    src_kind_real = Counter()
    src_kind_fake = Counter()
    src_kind_real_split = defaultdict(Counter)
    for v in ev["videos"]:
        if v.get("label") == "real":
            sk = v.get("source_kind")
            src_kind_real[sk] += 1
            src_kind_real_split[v.get("split")][sk] += 1
        elif v.get("label") == "fake":
            src_kind_fake[v.get("source_kind")] += 1

    notes.append("[B] eval manifest (real only): source_kind values:")
    for k, c in src_kind_real.most_common():
        notes.append(f"     {k}: {c}")
    notes.append("")
    notes.append("[B'] eval manifest (real only) source_kind x split:")
    for split, cc in src_kind_real_split.items():
        for k, c in cc.most_common():
            notes.append(f"     [{split}] {k}: {c}")
    notes.append("")
    notes.append("[C] eval manifest (fake only): source_kind values:")
    for k, c in src_kind_fake.most_common():
        notes.append(f"     {k}: {c}")
    notes.append("")
    notes.append("[D] proper_data manifest (real only): source_kind values:")
    with PROPER_MANIFEST.open() as f:
        pm = json.load(f)
    src_kind_proper = Counter()
    for v in pm["videos"]:
        if v.get("label") == "real":
            src_kind_proper[v.get("source_kind")] += 1
    for k, c in src_kind_proper.most_common():
        notes.append(f"     {k}: {c}")

    notes.append("")
    notes.append("[E] proper_data manifest (real only): playback_path values:")
    pp_proper = Counter()
    for v in pm["videos"]:
        if v.get("label") == "real":
            pp_proper[v.get("playback_path")] += 1
    for k, c in pp_proper.most_common():
        notes.append(f"     {k}: {c}")

    (OUT / "d9_capture_mode_value_summary.txt").write_text("\n".join(notes) + "\n")
    log(f"  capture-mode summary written")


# -------------------- main --------------------

def main() -> int:
    LOG.unlink(missing_ok=True)
    log("=== D9 source-substrate inventory audit ===")
    log(f"REPO_TRAIN: {REPO_TRAIN}")
    log(f"EVAL_MANIFEST: {EVAL_MANIFEST}")
    log(f"PROPER_MANIFEST: {PROPER_MANIFEST}")

    yamls = step1_train_yamls()
    rows = step2_train_yaml_inventory(yamls)
    step3_lane_inventory(rows)
    eval_result = step5_eval_real_classification()
    train_result = step6_train_manifest_classification()
    step7_crosstab(eval_result, train_result)
    step8_schema_gap(eval_result, train_result)

    # Final summary
    log("")
    log("=== HEADLINES ===")
    log(f"  R13 yamls enumerated: {len(yamls)}")
    log(f"  Eval real videos: dev={eval_result['n_real_dev']} lockbox={eval_result['n_real_lockbox']}")
    log(f"  Eval real frames: dev={eval_result['n_real_frames_dev']} lockbox={eval_result['n_real_frames_lockbox']}")
    log(f"  teams-v2 manifests classified: {train_result['teams_v2_n_fetched']} (errors {train_result['teams_v2_n_errors']})")
    log(f"  teams-v2 substrate class counts: {dict(train_result['teams_v2_substrate_classification'])}")
    log(f"  proper_data video substrate class counts: {dict(train_result['proper_substrate_classification'])}")
    log(f"  eval source_substrate x split: {eval_result['src_substrate_x_split']}")
    log(f"  eval source_kind x split: {eval_result['src_kind_x_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
