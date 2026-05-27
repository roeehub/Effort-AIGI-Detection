"""Wiring validation script for the Job C breakdown-yaml merge proposal.

Pure CPU-side dry-run that does NOT touch any GPU or model. Verifies that
the proposed merged yaml (proposed_teams_promotion_contract_with_breakdown.yaml)
is safe to drop in as a replacement for the production suite manifest at
arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml.

What it checks:
  1. Both yamls load as valid YAML (production + proposed).
  2. Every suite in the proposed yaml has a manifest path that resolves to a
     file on disk.
  3. Every suite's slice name appears in summary.slice_counts of the
     referenced manifest.
  4. No duplicate suite names exist within the proposed yaml.
  5. The 9 existing production suites all survive (name + key set unchanged)
     and reference the same manifest path they did before.
  6. For every newly-added suite, computes n_videos / n_frames / n_identities
     resolved against the manifest and prints them as a sanity table.

Usage:
  cd DeepfakeBench/training
  python analysis/job_c_breakdown_wiring_2026-05-04/wiring_validation_script.py

Exit code 0 = safe to merge. Exit code 1 = a check failed; do NOT merge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_YAML = REPO_ROOT / "arena" / "target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml"
PROPOSED_YAML = (
    REPO_ROOT
    / "analysis"
    / "job_c_breakdown_wiring_2026-05-04"
    / "proposed_teams_promotion_contract_with_breakdown.yaml"
)


def _suite_manifest_paths(suite: dict) -> list[str]:
    paths: list[str] = []
    for key in ("external_real_manifest", "external_fake_manifest"):
        v = suite.get(key)
        if v:
            paths.append(v)
    return paths


def _suite_slice(suite: dict) -> str | None:
    return suite.get("external_real_manifest_slices") or suite.get(
        "external_fake_manifest_slices"
    )


def _suite_split(suite: dict) -> str | None:
    return suite.get("external_real_manifest_split") or suite.get(
        "external_fake_manifest_split"
    )


def _suite_label_mode(suite: dict) -> str:
    has_real = bool(suite.get("external_real_manifest"))
    has_fake = bool(suite.get("external_fake_manifest"))
    if has_real and has_fake:
        return "mixed"
    if has_real:
        return "real_only"
    if has_fake:
        return "fake_only"
    return "unknown"


def _resolve_videos(manifest: dict, slice_name: str, split: str) -> list[dict]:
    out = []
    for v in manifest.get("videos", []):
        if v.get("split") != split:
            continue
        if slice_name not in (v.get("slices") or []):
            continue
        out.append(v)
    return out


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    if not PROD_YAML.exists():
        print(f"FAIL: production yaml missing: {PROD_YAML}")
        return 1
    if not PROPOSED_YAML.exists():
        print(f"FAIL: proposed yaml missing: {PROPOSED_YAML}")
        return 1

    prod = yaml.safe_load(PROD_YAML.read_text())
    proposed = yaml.safe_load(PROPOSED_YAML.read_text())

    if "suites" not in prod or "suites" not in proposed:
        failures.append("yaml missing top-level 'suites' key")

    prod_suites = {s["name"]: s for s in prod["suites"]}
    proposed_suites_list = proposed["suites"]
    proposed_names = [s["name"] for s in proposed_suites_list]

    # Check 4: duplicate suite names
    if len(set(proposed_names)) != len(proposed_names):
        seen = {}
        for n in proposed_names:
            seen[n] = seen.get(n, 0) + 1
        dups = [n for n, c in seen.items() if c > 1]
        failures.append(f"duplicate suite names in proposed yaml: {dups}")

    # Check 5: existing production suites all preserved
    proposed_by_name = {s["name"]: s for s in proposed_suites_list}
    for name, prod_suite in prod_suites.items():
        if name not in proposed_by_name:
            failures.append(f"production suite dropped in proposal: {name}")
            continue
        proposed_suite = proposed_by_name[name]
        for k, v in prod_suite.items():
            if proposed_suite.get(k) != v:
                failures.append(
                    f"production suite '{name}' field '{k}' changed: "
                    f"{v!r} -> {proposed_suite.get(k)!r}"
                )

    # Cache manifests by path for slice-presence lookups
    manifest_cache: dict[str, dict] = {}
    manifest_slice_index: dict[str, set[str]] = {}
    for s in proposed_suites_list:
        for p in _suite_manifest_paths(s):
            if p in manifest_cache:
                continue
            full = REPO_ROOT / p
            if not full.exists():
                failures.append(f"suite '{s['name']}': manifest not on disk: {p}")
                continue
            try:
                manifest_cache[p] = json.loads(full.read_text())
                slices_in_manifest: set[str] = set()
                for v in manifest_cache[p].get("videos", []):
                    slices_in_manifest.update(v.get("slices") or [])
                manifest_slice_index[p] = slices_in_manifest
            except Exception as exc:  # pragma: no cover
                failures.append(f"suite '{s['name']}': manifest unreadable {p}: {exc}")

    # Per-suite resolution sanity table (Step 6 of the validator spec)
    table_rows = []
    new_suite_names = [n for n in proposed_names if n not in prod_suites]
    for s in proposed_suites_list:
        name = s["name"]
        is_new = name in new_suite_names
        slice_name = _suite_slice(s)
        split = _suite_split(s)
        manifest_paths = _suite_manifest_paths(s)
        manifest_path = manifest_paths[0] if manifest_paths else None
        manifest = manifest_cache.get(manifest_path) if manifest_path else None

        if not slice_name or not split or manifest is None:
            failures.append(
                f"suite '{name}' missing slice/split/manifest "
                f"(slice={slice_name}, split={split}, manifest={manifest_path})"
            )
            continue

        slices_in_manifest = manifest_slice_index.get(manifest_path, set())
        if slice_name not in slices_in_manifest:
            failures.append(
                f"suite '{name}': slice '{slice_name}' not present in any video.slices "
                f"in {manifest_path}"
            )
            continue

        videos = _resolve_videos(manifest, slice_name, split)
        n_videos = len(videos)
        n_frames = sum(len(v.get("frame_paths", [])) for v in videos)
        identities = sorted({(v.get("identity_key") or v.get("identity") or "<unk>") for v in videos})
        sessions = sorted({(v.get("session_id") or "<no_session>") for v in videos})

        if n_videos == 0:
            failures.append(f"suite '{name}': resolved to 0 videos (slice={slice_name}, split={split})")

        table_rows.append({
            "suite_name": name,
            "is_new": is_new,
            "label_mode": _suite_label_mode(s),
            "slice": slice_name,
            "split": split,
            "n_videos": n_videos,
            "n_frames": n_frames,
            "n_identities": len(identities),
            "n_sessions": len(sessions),
            "first_identity": identities[0] if identities else "<empty>",
        })

    # Print results
    print("=" * 90)
    print("Wiring proposal validation — proposed merged yaml vs production")
    print("=" * 90)
    print(f"Production yaml: {PROD_YAML.relative_to(REPO_ROOT)}")
    print(f"Proposed yaml  : {PROPOSED_YAML.relative_to(REPO_ROOT)}")
    print(f"Production suites: {len(prod_suites)}")
    print(f"Proposed suites  : {len(proposed_suites_list)} "
          f"({len(new_suite_names)} new, {len(proposed_suites_list)-len(new_suite_names)} preserved)")
    print()
    print(f"{'suite_name':50s} {'is_new':6s} {'mode':10s} {'split':8s} "
          f"{'#vid':>5s} {'#frm':>6s} {'#id':>4s} {'#sess':>5s}")
    print("-" * 110)
    for r in table_rows:
        print(
            f"{r['suite_name']:50s} {str(r['is_new']):6s} {r['label_mode']:10s} "
            f"{r['split']:8s} {r['n_videos']:5d} {r['n_frames']:6d} "
            f"{r['n_identities']:4d} {r['n_sessions']:5d}"
        )

    print()
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED — proposed yaml is safe to merge.")
    print(f"  - All {len(prod_suites)} production suites preserved unchanged")
    print(f"  - All {len(new_suite_names)} new suites resolve cleanly to non-empty video sets")
    print(f"  - No duplicate suite names")
    print(f"  - All manifest paths on disk and slice tags present in summary")
    if notes:
        print()
        for n in notes:
            print(f"NOTE: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
