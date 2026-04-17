#!/usr/bin/env python3
"""Render and validate the checked-in WT-A policy truth artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "relaunch_handoffs"
    / "WT-A_policy_truth_artifact_2026-04-17.json"
)


def load_artifact(path: Path | None = None) -> Dict[str, Any]:
    target = path or ARTIFACT_PATH
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _approx_equal(left: float, right: float, tol: float = 0.03) -> bool:
    return math.isclose(float(left), float(right), abs_tol=tol)


def _sum_values(mapping: Dict[str, Any]) -> float:
    return float(sum(float(value) for value in mapping.values()))


def validate_artifact(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    policy = data.get("policy_awareness") or {}
    if policy.get("current_training_policy_aware") is not False:
        errors.append("current_training_policy_aware must be false for the checked-in tree")

    labels = policy.get("same_day_labels") or {}
    if labels.get("current_checked_in_track_a_configs") != "old-semantics":
        errors.append("current checked-in Track A configs must be labeled old-semantics")
    if labels.get("explicit_hint_lane_retrains_on_current_data") != "weak-signal-only":
        errors.append("explicit hint-lane retrains must be labeled weak-signal-only")
    if labels.get("policy_corrected_available_today") is not False:
        errors.append("policy_corrected_available_today must be false")

    families = data.get("classification_families") or []
    family_map = {
        family.get("family_id"): family
        for family in families
        if family.get("family_id")
    }

    base = family_map.get("mixed_p050")
    if not base:
        errors.append("missing mixed_p050 family")
        return errors

    old_counts = base.get("old_semantics_counts") or {}
    old_total_pairs = float(old_counts.get("paired_objects_total") or 0)
    old_train_pairs = float(old_counts.get("train_pairs_total") or 0)

    old_total_by_source = old_counts.get("paired_objects_total_by_source") or {}
    if not _approx_equal(_sum_values(old_total_by_source), old_total_pairs):
        errors.append("mixed_p050 old total paired-object count does not match source breakdown")

    old_train_by_source = old_counts.get("train_pairs_by_source") or {}
    if not _approx_equal(_sum_values(old_train_by_source), old_train_pairs):
        errors.append("mixed_p050 old train pair count does not match source breakdown")

    corrected = base.get("corrected_policy_reference_counts") or {}
    corrected_total_pairs = float(corrected.get("paired_objects_total") or 0)
    corrected_train_pairs = float(corrected.get("train_pairs_total") or 0)

    corrected_total_by_source = corrected.get("paired_objects_total_by_source") or {}
    if not _approx_equal(_sum_values(corrected_total_by_source), corrected_total_pairs):
        errors.append("mixed_p050 corrected total paired-object count does not match source breakdown")

    corrected_train_by_source = corrected.get("train_pairs_by_source") or {}
    if not _approx_equal(_sum_values(corrected_train_by_source), corrected_train_pairs):
        errors.append("mixed_p050 corrected train pair count does not match source breakdown")

    direct_total = (base.get("lane_contamination") or {}).get("direct_teams_total") or {}
    if int(direct_total.get("clean_teams_after_policy") or 0) != (
        int(direct_total.get("pairs_total") or 0) - int(direct_total.get("bad_visomaster_before_policy") or 0)
    ):
        errors.append("mixed_p050 direct_teams_total clean count is inconsistent")

    direct_train = (base.get("lane_contamination") or {}).get("direct_teams_train") or {}
    if int(direct_train.get("clean_teams_after_policy") or 0) != (
        int(direct_train.get("pairs_total") or 0) - int(direct_train.get("bad_visomaster_before_policy") or 0)
    ):
        errors.append("mixed_p050 direct_teams_train clean count is inconsistent")

    if int(corrected_total_by_source.get("visomaster_hints") or 0) + int(corrected_total_by_source.get("visomaster_hints_teams") or 0) != 682:
        errors.append("corrected total retained hint count must equal 682 paired objects")
    if int(corrected_train_by_source.get("visomaster_hints") or 0) + int(corrected_train_by_source.get("visomaster_hints_teams") or 0) != 571:
        errors.append("corrected train retained hint count must equal 571 paired objects")

    for family in families:
        family_id = str(family.get("family_id") or "")
        source_ref = family.get("source_truth_reference_family")
        if source_ref and source_ref not in family_map:
            errors.append(f"{family_id} references missing source truth family {source_ref}")

        exposure = family.get("vte_epoch_exposure") or {}
        total = float(exposure.get("expected_vte_selections_per_epoch") or 0.0)
        true_teams = float(exposure.get("expected_true_teams_companion_selections_per_epoch") or 0.0)
        clean_fallback = float(exposure.get("expected_clean_fallback_selections_per_epoch") or 0.0)
        branches = exposure.get("expected_branch_exposure_per_epoch") or {}
        teams_original = float(branches.get("teams_original_fake") or 0.0)
        clean_original = float(branches.get("clean_original_fake") or 0.0)
        clean_enhanced = float(branches.get("clean_enhanced_fake") or 0.0)

        if not _approx_equal(true_teams + clean_fallback, total):
            errors.append(f"{family_id} VTE total does not match teams+clean_fallback split")
        if not _approx_equal(teams_original + clean_original + clean_enhanced, total):
            errors.append(f"{family_id} branch exposure does not sum to total VTE exposure")

    return errors


def render_summary(data: Dict[str, Any]) -> str:
    lines: List[str] = []

    policy = data["policy_awareness"]
    lines.append(f"Artifact: {data['artifact_name']}")
    lines.append(f"Training policy-aware today: {policy['current_training_policy_aware']}")
    lines.append(
        "Same-day labels: current="
        f"{policy['same_day_labels']['current_checked_in_track_a_configs']}, "
        "explicit-hints="
        f"{policy['same_day_labels']['explicit_hint_lane_retrains_on_current_data']}, "
        "policy-corrected-available="
        f"{policy['same_day_labels']['policy_corrected_available_today']}"
    )
    lines.append("")

    for family in data.get("classification_families") or []:
        family_id = family["family_id"]
        lines.append(f"[{family_id}] {family['classification']} p_original={family['p_original']}")
        for config_name in family.get("promotion_relevant_configs") or family.get("family_configs") or []:
            lines.append(f"  - {config_name}")

        if "old_semantics_counts" in family:
            old_counts = family["old_semantics_counts"]
            if "train_pairs_total" in old_counts:
                lines.append(f"  old train pairs: {old_counts['train_pairs_total']}")
            if "paired_objects_total" in old_counts:
                lines.append(f"  old total paired objects: {old_counts['paired_objects_total']}")

        if "corrected_policy_reference_counts" in family:
            corrected = family["corrected_policy_reference_counts"]
            if "train_pairs_total" in corrected:
                lines.append(f"  corrected train pairs: {corrected['train_pairs_total']}")
            if "paired_objects_total" in corrected:
                lines.append(f"  corrected total paired objects: {corrected['paired_objects_total']}")
            if "status" in corrected:
                lines.append(f"  corrected status: {corrected['status']}")

        exposure = family.get("vte_epoch_exposure") or {}
        branches = exposure.get("expected_branch_exposure_per_epoch") or {}
        if branches:
            lines.append(
                "  VTE branch exposure: "
                f"teams_original={branches.get('teams_original_fake')}, "
                f"clean_original={branches.get('clean_original_fake')}, "
                f"clean_enhanced={branches.get('clean_enhanced_fake')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ARTIFACT_PATH,
        help="Path to the WT-A policy truth JSON artifact",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the artifact and exit with status 1 on any invariant failure",
    )
    args = parser.parse_args()

    data = load_artifact(args.artifact)
    errors = validate_artifact(data)

    if args.validate:
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        print(f"OK: {args.artifact}")
        return 0

    print(render_summary(data), end="")
    if errors:
        print("Validation warnings:")
        for err in errors:
            print(f"  - {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
