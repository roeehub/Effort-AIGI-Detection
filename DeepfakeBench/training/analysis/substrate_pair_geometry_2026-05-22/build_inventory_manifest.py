"""Build paired substrate inventory manifest (CSV + gates JSON).

Phase 0 A0.1 — Substrate-pair geometry, 2026-05-22.

Parses two on-disk manifests (no GCS calls) and emits:
  - inventory_manifest.csv : one row per paired identity x base_capture
  - gates_summary.json     : aggregate counts + phase-0 gate status

Inputs (READ-ONLY):
  - arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml
  - analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json

CSV columns (per plan spec):
  identity_id, base_capture_id, source, clean_bucket, clean_prefix,
  clean_real_frame_count, teams_bucket, teams_prefix, teams_real_frame_count, notes

Phase-0 gate: total_pairs >= 500 -> FLOOR_MET, else FLOOR_NOT_MET.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


REPO_TRAINING_ROOT = Path(__file__).resolve().parents[2]
PROVISIONAL_YAML = (
    REPO_TRAINING_ROOT
    / "arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml"
)
RESOLVER_JSON = (
    REPO_TRAINING_ROOT
    / "analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json"
)
OUT_DIR = REPO_TRAINING_ROOT / "analysis/substrate_pair_geometry_2026-05-22"
CSV_PATH = OUT_DIR / "inventory_manifest.csv"
JSON_PATH = OUT_DIR / "gates_summary.json"

PHASE_0_FLOOR = 500
SUFFICIENT_FRAMES_PER_SIDE = 5

CSV_COLUMNS = [
    "identity_id",
    "base_capture_id",
    "source",
    "clean_bucket",
    "clean_prefix",
    "clean_real_frame_count",
    "teams_bucket",
    "teams_prefix",
    "teams_real_frame_count",
    "notes",
]


def _strip_gs_prefix(gs_path: str) -> Tuple[str, str]:
    """Return (bucket, key) for a gs:// URL."""
    if not gs_path.startswith("gs://"):
        raise ValueError(f"Expected gs:// path, got: {gs_path!r}")
    no_scheme = gs_path[len("gs://"):]
    if "/" not in no_scheme:
        return no_scheme, ""
    bucket, key = no_scheme.split("/", 1)
    return bucket, key


def _parent_prefix(gs_path: str) -> Tuple[str, str]:
    """Given a gs:// frame URL, return (bucket, parent_prefix_with_trailing_slash)."""
    bucket, key = _strip_gs_prefix(gs_path)
    parent = key.rsplit("/", 1)[0] if "/" in key else ""
    if parent and not parent.endswith("/"):
        parent = parent + "/"
    return bucket, parent


def _source_label_for_base(base_id: str) -> str:
    if base_id.startswith("HDTF"):
        return "hdtf_visomaster_teams"
    if base_id.startswith("QCLIP"):
        return "quickclips_visomaster_teams"
    raise ValueError(f"Unknown base_capture_id prefix: {base_id!r}")


def _derive_viso_enhanced_identity(sample_id: str, manifest_blob: Optional[Dict[str, Any]]) -> str:
    """Mirror VisoMasterTeamsEnhancedSample.identity logic.

    Prefers manifest.original_video_name when present; falls back to
    the post-'visomaster_' token after the final underscore in sample_id.
    """
    if manifest_blob:
        ov = manifest_blob.get("original_video_name", "") or ""
        if ov:
            name = ov.rsplit("/", 1)[-1]
            if "." in name:
                name = name.rsplit(".", 1)[0]
            if name.startswith("cropped_"):
                name = name[len("cropped_"):]
            return name
    remainder = sample_id
    if remainder.startswith("visomaster_"):
        remainder = remainder[len("visomaster_"):]
    last_us = remainder.rfind("_")
    if last_us > 0:
        return remainder[last_us + 1:]
    return sample_id


def _extract_pair_rows_from_provisional(yaml_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walk provisional captures; emit one CSV row per HDTF / QCLIP capture."""
    rows: List[Dict[str, Any]] = []
    for cap in yaml_doc.get("captures", []):
        base_id = cap["base_capture_id"]
        identity_id = cap.get("identity_id", "")
        source = _source_label_for_base(base_id)

        clean_variant = None
        teams_variant = None
        for variant in cap.get("variants", []):
            if variant.get("label") != "real":
                continue
            transport = variant.get("transport")
            if transport == "clean":
                clean_variant = variant
            elif transport == "teams":
                teams_variant = variant

        if clean_variant is None or teams_variant is None:
            # Provisional yaml is built to only contain 16/16 captures, but
            # be defensive: skip any capture missing a real_clean or real_teams variant.
            continue

        clean_paths = clean_variant.get("frame_paths") or []
        teams_paths = teams_variant.get("frame_paths") or []
        if not clean_paths or not teams_paths:
            continue

        clean_bucket, clean_prefix = _parent_prefix(clean_paths[0])
        teams_bucket, teams_prefix = _parent_prefix(teams_paths[0])

        rows.append({
            "identity_id": identity_id,
            "base_capture_id": base_id,
            "source": source,
            "clean_bucket": clean_bucket,
            "clean_prefix": clean_prefix,
            "clean_real_frame_count": len(clean_paths),
            "teams_bucket": teams_bucket,
            "teams_prefix": teams_prefix,
            "teams_real_frame_count": len(teams_paths),
            "notes": "",
        })
    return rows


def _extract_pair_rows_from_resolver(resolver_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Walk enhanced resolver; emit one CSV row per teams_v2_companion sample."""
    rows: List[Dict[str, Any]] = []
    clean_bucket = "live-deepfake-methods-real-and-fake-frames-cropped"
    for r in resolver_doc.get("rows", []):
        if r.get("resolution_status") != "teams_v2_companion":
            continue

        sample_id = r["sample_id"]
        teams_bucket = r["resolved_companion_bucket"]
        companion_path = r["resolved_companion_path"]
        if not companion_path.endswith("/"):
            companion_path = companion_path + "/"
        teams_prefix = companion_path + "frames/real/"
        clean_prefix = f"samples/{sample_id}/frames/real/"
        teams_count = r.get("resolved_real_frame_count")

        # Resolver row does not embed manifest.original_video_name; identity
        # falls back to sample-id-derived token (5-digit numeric in practice).
        manifest_blob = r.get("manifest")  # not present in resolver schema
        identity_id = _derive_viso_enhanced_identity(sample_id, manifest_blob)
        if identity_id == sample_id or identity_id.isdigit():
            identity_id = sample_id

        rows.append({
            "identity_id": identity_id,
            "base_capture_id": sample_id,
            "source": "visomaster_teams_enhanced",
            "clean_bucket": clean_bucket,
            "clean_prefix": clean_prefix,
            "clean_real_frame_count": None,  # not in resolver row schema
            "teams_bucket": teams_bucket,
            "teams_prefix": teams_prefix,
            "teams_real_frame_count": teams_count,
            "notes": (
                "clean_real_frame_count unverified: resolver row carries no clean-side "
                "count; clean_real_exists=true per resolver. identity_id falls back to "
                "sample_id because resolver row carries no manifest.original_video_name."
            ),
        })
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out_row = {col: ("" if row.get(col) is None else row.get(col, "")) for col in CSV_COLUMNS}
            writer.writerow(out_row)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs_by_source: Counter = Counter()
    frames_clean_by_source: Counter = Counter()
    frames_teams_by_source: Counter = Counter()
    pairs_clean_count_missing: int = 0
    pairs_with_sufficient = 0

    for row in rows:
        src = row["source"]
        pairs_by_source[src] += 1
        clean_n = row["clean_real_frame_count"]
        teams_n = row["teams_real_frame_count"] or 0
        if clean_n is None:
            pairs_clean_count_missing += 1
        else:
            frames_clean_by_source[src] += int(clean_n)
        frames_teams_by_source[src] += int(teams_n)

        # "Sufficient" gate: at least 5 frames per side.
        clean_ok = (clean_n is None) or (int(clean_n) >= SUFFICIENT_FRAMES_PER_SIDE)
        teams_ok = int(teams_n) >= SUFFICIENT_FRAMES_PER_SIDE
        if clean_ok and teams_ok:
            pairs_with_sufficient += 1

    total_pairs = sum(pairs_by_source.values())
    total_frames_clean = sum(frames_clean_by_source.values())
    total_frames_teams = sum(frames_teams_by_source.values())
    gate_status = "FLOOR_MET" if total_pairs >= PHASE_0_FLOOR else "FLOOR_NOT_MET"

    return {
        "phase_0_gate_status": gate_status,
        "phase_0_floor_threshold": PHASE_0_FLOOR,
        "total_pairs": total_pairs,
        "pairs_by_source": dict(pairs_by_source),
        "total_frames_clean": total_frames_clean,
        "total_frames_teams": total_frames_teams,
        "frames_clean_by_source": dict(frames_clean_by_source),
        "frames_teams_by_source": dict(frames_teams_by_source),
        "pairs_clean_count_missing": pairs_clean_count_missing,
        "pairs_with_sufficient_frames": pairs_with_sufficient,
        "sufficient_frames_threshold_per_side": SUFFICIENT_FRAMES_PER_SIDE,
        "inputs": {
            "provisional_yaml": str(PROVISIONAL_YAML.relative_to(REPO_TRAINING_ROOT)),
            "enhanced_resolver_json": str(RESOLVER_JSON.relative_to(REPO_TRAINING_ROOT)),
        },
    }


def main() -> int:
    if not PROVISIONAL_YAML.is_file():
        print(f"ERROR: missing provisional yaml: {PROVISIONAL_YAML}", file=sys.stderr)
        return 2
    if not RESOLVER_JSON.is_file():
        print(f"ERROR: missing resolver json: {RESOLVER_JSON}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {PROVISIONAL_YAML}", flush=True)
    with PROVISIONAL_YAML.open("r") as f:
        yaml_doc = yaml.safe_load(f)
    n_captures = len(yaml_doc.get("captures", []))
    print(f"[load] captures parsed: {n_captures}", flush=True)

    print(f"[load] {RESOLVER_JSON}", flush=True)
    with RESOLVER_JSON.open("r") as f:
        resolver_doc = json.load(f)
    n_rows = len(resolver_doc.get("rows", []))
    print(f"[load] resolver rows: {n_rows}", flush=True)

    provisional_rows = _extract_pair_rows_from_provisional(yaml_doc)
    print(f"[extract] provisional pair rows: {len(provisional_rows)}", flush=True)

    resolver_rows = _extract_pair_rows_from_resolver(resolver_doc)
    print(f"[extract] resolver pair rows: {len(resolver_rows)}", flush=True)

    all_rows = provisional_rows + resolver_rows

    _write_csv(all_rows, CSV_PATH)
    print(f"[write] {CSV_PATH} ({len(all_rows)} rows)", flush=True)

    summary = _summarize(all_rows)
    with JSON_PATH.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[write] {JSON_PATH}", flush=True)

    print(f"[gate] phase_0_gate_status = {summary['phase_0_gate_status']}", flush=True)
    print(f"[gate] total_pairs        = {summary['total_pairs']}", flush=True)
    print(f"[gate] pairs_by_source    = {summary['pairs_by_source']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
