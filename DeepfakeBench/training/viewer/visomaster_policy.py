"""Helpers for the VisoMaster bad-data relabel policy."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


KEEP_ACTION = "keep"
IGNORE_ACTION = "ignore"
DELETE_ACTION = "delete"

BASELINE_LANE = "baseline_cropped"
TEAMS_LANE = "teams_pair_complete"

KEEP_LABEL_BASELINE = "visomaster hints"
KEEP_LABEL_TEAMS = "visomaster hints (teams)"

# Short aliases used by the viewer code.
BASELINE_LABEL = KEEP_LABEL_BASELINE
TEAMS_LABEL = KEEP_LABEL_TEAMS

BASELINE_SOURCE = "visomaster_hints"
TEAMS_SOURCE = "visomaster_hints_teams"

BASELINE_METHOD = "visomaster_hints"
TEAMS_METHOD = "visomaster_hints_teams"

BASELINE_FAKE_FAMILY = "visomaster_hints_fake"
TEAMS_FAKE_FAMILY = "visomaster_hints_teams_fake"

BASELINE_WEIGHT_ALIAS = "visomaster_fake"
TEAMS_WEIGHT_ALIAS = "deeplive_teams_fake"

TRAINING_DIR = Path(__file__).resolve().parent.parent
DEBUG_DIR = TRAINING_DIR / "debug"

MANIFEST_RE = re.compile(r"^VISOMASTER_BAD_DATA_POLICY_MANIFEST_(\d{4}-\d{2}-\d{2})\.csv$")
SUMMARY_RE = re.compile(r"^VISOMASTER_BAD_DATA_POLICY_SUMMARY_(\d{4}-\d{2}-\d{2})\.json$")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return TRAINING_DIR / p


def _find_latest(debug_dir: Path, pattern: re.Pattern[str]) -> Optional[Path]:
    candidates = []
    if not debug_dir.exists():
        return None
    for path in debug_dir.iterdir():
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        candidates.append((m.group(1), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


@dataclass(frozen=True)
class PolicyRow:
    sample_id: str
    policy_action: str
    policy_label: str
    policy_lane: str
    decision_reason: str
    selection_group: str
    selection_seed: str
    swap_model: str
    tier: str
    identity_delta: Optional[float]
    original_video_name: str
    in_cropped_bucket: bool
    in_frames_bucket: bool
    in_teams_pair_complete: bool


@dataclass(frozen=True)
class PolicyBundle:
    date_tag: str
    manifest_path: Path
    summary_path: Optional[Path]
    rows_by_sample_id: Dict[str, PolicyRow]
    summary: Dict[str, Any]

    def row_for(self, sample_id: str) -> Optional[PolicyRow]:
        return self.rows_by_sample_id.get(sample_id)

    @property
    def is_active(self) -> bool:
        return bool(self.rows_by_sample_id)


def _resolve_manifest_path(config: Optional[Dict[str, Any]]) -> Optional[Path]:
    cp = (config or {}).get("combined_paired", {})
    vm_cfg = cp.get("visomaster", {})

    explicit = (
        vm_cfg.get("bad_data_policy_manifest")
        or cp.get("visomaster_bad_data_policy_manifest")
        or cp.get("bad_data_policy_manifest")
    )
    if explicit:
        return _resolve_path(str(explicit))
    return _find_latest(DEBUG_DIR, MANIFEST_RE)


def _resolve_summary_path(config: Optional[Dict[str, Any]], date_tag: str) -> Optional[Path]:
    cp = (config or {}).get("combined_paired", {})
    vm_cfg = cp.get("visomaster", {})

    explicit = (
        vm_cfg.get("bad_data_policy_summary")
        or cp.get("visomaster_bad_data_policy_summary")
        or cp.get("bad_data_policy_summary")
    )
    if explicit:
        return _resolve_path(str(explicit))

    candidate = DEBUG_DIR / f"VISOMASTER_BAD_DATA_POLICY_SUMMARY_{date_tag}.json"
    if candidate.exists():
        return candidate
    return _find_latest(DEBUG_DIR, SUMMARY_RE)


def _extract_date_tag(path: Path) -> str:
    match = MANIFEST_RE.match(path.name)
    if match:
        return match.group(1)
    return "unknown"


@lru_cache(maxsize=8)
def _load_cached(manifest_path_str: str, summary_path_str: str) -> PolicyBundle:
    manifest_path = Path(manifest_path_str)
    summary_path = Path(summary_path_str) if summary_path_str else None
    date_tag = _extract_date_tag(manifest_path)

    rows_by_sample_id: Dict[str, PolicyRow] = {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            rows_by_sample_id[sample_id] = PolicyRow(
                sample_id=sample_id,
                policy_action=str(row.get("policy_action") or "").strip(),
                policy_label=str(row.get("policy_label") or "").strip(),
                policy_lane=str(row.get("policy_lane") or "").strip(),
                decision_reason=str(row.get("decision_reason") or "").strip(),
                selection_group=str(row.get("selection_group") or "").strip(),
                selection_seed=str(row.get("selection_seed") or "").strip(),
                swap_model=str(row.get("swap_model") or "").strip(),
                tier=str(row.get("tier") or "").strip(),
                identity_delta=_to_float(row.get("identity_delta")),
                original_video_name=str(row.get("original_video_name") or "").strip(),
                in_cropped_bucket=_to_bool(row.get("in_cropped_bucket")),
                in_frames_bucket=_to_bool(row.get("in_frames_bucket")),
                in_teams_pair_complete=_to_bool(row.get("in_teams_pair_complete")),
            )

    summary: Dict[str, Any] = {}
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        date_tag = str(summary.get("date_tag") or date_tag)

    return PolicyBundle(
        date_tag=date_tag,
        manifest_path=manifest_path,
        summary_path=summary_path if summary_path and summary_path.exists() else None,
        rows_by_sample_id=rows_by_sample_id,
        summary=summary,
    )


def load_visomaster_bad_data_policy(config: Optional[Dict[str, Any]] = None) -> Optional[PolicyBundle]:
    manifest_path = _resolve_manifest_path(config)
    if not manifest_path or not manifest_path.exists():
        return None
    summary_path = _resolve_summary_path(config, _extract_date_tag(manifest_path))
    return _load_cached(str(manifest_path), str(summary_path) if summary_path else "")


def policy_summary_for_response(policy: Optional[PolicyBundle]) -> Optional[Dict[str, Any]]:
    if not policy:
        return None

    summary = dict(policy.summary) if policy.summary else {}
    if not summary:
        action_counts = {"keep": 0, "ignore": 0, "delete": 0}
        label_counts = {KEEP_LABEL_BASELINE: 0, KEEP_LABEL_TEAMS: 0}
        for row in policy.rows_by_sample_id.values():
            action_counts[row.policy_action] = action_counts.get(row.policy_action, 0) + 1
            if row.policy_label:
                label_counts[row.policy_label] = label_counts.get(row.policy_label, 0) + 1
        summary = {
            "policy_name": "visomaster_bad_data_relabel_policy",
            "date_tag": policy.date_tag,
            "action_counts": action_counts,
            "label_counts": label_counts,
        }

    summary["active"] = True
    summary["manifest_path"] = str(policy.manifest_path)
    if policy.summary_path:
        summary["summary_path"] = str(policy.summary_path)
    return summary
