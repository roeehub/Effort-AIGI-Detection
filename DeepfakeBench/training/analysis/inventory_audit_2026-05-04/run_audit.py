"""
Inventory audit (CPU-only, metadata-only) — 2026-05-04 (Job 5).

Enumerates candidate eval cells across:
  - arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json
  - arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json
  - arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json
  - arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json (frozen, predecessor of with_dor)

Writes:
  - candidate_eval_cells.csv
  - inventory_summary.json
  - identity_overlap_report.csv

Train/eval status determined from R13 yaml scan:
  - proper_data lanes (4): all enabled in 102 yamls including R13_RLP8_01 (=P8A's training).
  - teams-v2 bucket: enabled in 147 yamls (training + ood_monitoring).
  - deeplive bucket (live-deepfake-methods-real-and-fake-frames-cropped): training in 163 yamls.
  - visomaster_enhanced_v2 (visomaster-enhanced-face-cropped-v2): NEVER referenced in any training yaml.
  - hdtf_visomaster_cropped_frames(_teams): never referenced; only via proper_data manifest path.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis" / "inventory_audit_2026-05-04"
OUT.mkdir(parents=True, exist_ok=True)

# --- Sources -----------------------------------------------------------------
PROPER = json.load(open(ROOT / "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"))
TEAMS = json.load(open(ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"))
VISO_V2 = json.load(open(ROOT / "arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json"))

# --- FP-prone identities (Job 3 / cpu_followups) -----------------------------
FP_OFFENDERS = ["bla_bla_chow", "bla_bla_chow__s2", "PC_Generator__s22",
                "PC_Generator__s45", "roy_d", "Q__s6"]


def _matches(identity: str, target: str) -> bool:
    if identity is None:
        return False
    return target.lower() in identity.lower() or identity.lower() in target.lower()


# --- Build candidate cells from proper viso ---------------------------------
proper_cells = collections.defaultdict(lambda: {
    "frames": 0, "videos": 0, "identities": set(),
    "captures": set(), "generator_methods": set(),
    "quality_bands": set(), "face_scale_bands": set(),
})
for v in PROPER["videos"]:
    key = (v["lane"], v["split"])
    d = proper_cells[key]
    d["videos"] += 1
    d["frames"] += len(v["frame_paths"])
    d["identities"].add(v["identity_id"])
    d["captures"].add(v["base_capture_id"])
    if v.get("generator_method"):
        d["generator_methods"].add(v["generator_method"])
    if v.get("quality_band"):
        d["quality_bands"].add(v["quality_band"])
    if v.get("face_scale_band"):
        d["face_scale_bands"].add(v["face_scale_band"])

# --- Teams manifest cells --------------------------------------------------
teams_cells = collections.defaultdict(lambda: {
    "frames": 0, "videos": 0, "identities": set(), "session_ids": set(),
})
for v in TEAMS["videos"]:
    key = (v["method"], v["split"], v["label"])
    d = teams_cells[key]
    d["videos"] += 1
    d["frames"] += len(v["frame_paths"])
    d["identities"].add(v.get("identity_key"))
    if v.get("session_id"):
        d["session_ids"].add(v["session_id"])

# --- Viso v2 ---------------------------------------------------------------
viso_v2_cells = collections.defaultdict(lambda: {
    "frames": 0, "videos": 0, "identities": set(), "model_folders": set(),
})
for v in VISO_V2["videos"]:
    method = v.get("method", "unknown")
    key = (method, v.get("split", "all"))
    d = viso_v2_cells[key]
    d["videos"] += 1
    # one frame per row in this manifest
    d["frames"] += len(v.get("frame_paths", [])) or 1
    d["identities"].add(v.get("identity"))
    folder = (v.get("metadata") or {}).get("model_folder")
    if folder:
        d["model_folders"].add(folder)

# --- Train/val/test status table ------------------------------------------
# Source of truth: R13 yaml audit + the loader code paths.
TRAINING_TABLE = {
    # source_id -> (training_use, citation, wireability)
    "proper_visomaster_clean": (
        "TRAIN+EVAL_LOCKBOX_LEAKED",
        "R13_RLP8_01_unfreeze_clip_codec.yaml proper_data.include_lanes (P8A-equivalent); "
        "R13_RLP6_04_add_enh_clean.yaml; 102 R13 yamls total. data/sources/proper_data.py "
        "discover_proper_data_samples ingests inventory captures with no split filter — "
        "manifest 'lockbox' rows ARE pulled into training when lane is enabled.",
        "blocked-by-leakage",
    ),
    "proper_visomaster_teams": (
        "TRAIN+EVAL_LOCKBOX_LEAKED",
        "R13_RLP8_01 proper_data.include_lanes; 102 R13 yamls.",
        "blocked-by-leakage",
    ),
    "proper_visomaster_enhanced_clean": (
        "TRAIN+EVAL_LOCKBOX_LEAKED",
        "R13_RLP8_01 proper_data.include_lanes (45-66 R13 yamls).",
        "blocked-by-leakage",
    ),
    "proper_visomaster_enhanced_teams": (
        "TRAIN+EVAL_LOCKBOX_LEAKED",
        "R13_RLP8_01 proper_data.include_lanes (45-66 R13 yamls).",
        "blocked-by-leakage",
    ),
    "proper_real_clean": (
        "TRAIN_PAIRED",
        "Real side of every fake variant; loader proper_data.py auto-pairs real_clean "
        "with each fake_clean variant. So real-clean frames are seen during fake-paired training.",
        "blocked-by-leakage",
    ),
    "proper_real_teams": (
        "TRAIN_PAIRED",
        "Real side of every fake_teams variant; auto-paired in proper_data.py.",
        "blocked-by-leakage",
    ),
    "teams_real": (
        "TRAIN+OOD",
        "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2 enabled in 147 R13 yamls "
        "(combined_paired.teams + ood_monitoring.external_real_sources/teams_ood_real).",
        "blocked-by-leakage (already in scorecard)",
    ),
    "teams_capture_*": (
        "EVAL_ONLY (manifest-only, fake side)",
        "Source: gs://teams-faces-data-test-2914-fake-4420-real-feb-28; teams target manifest. "
        "Used in scorecard suites teams_fake_all_dev/lockbox.",
        "(already wired)",
    ),
    "deeplive_enhanced": (
        "TRAIN+EVAL",
        "deeplive bucket (live-deepfake-methods-real-and-fake-frames-cropped) is training "
        "data with include_strategies covering both clean+enhanced pairs.",
        "(already wired in scorecard)",
    ),
    "visomaster_enhanced_macro": (
        "TRAIN+EVAL",
        "live-deepfake-methods-real-and-fake-frames-cropped is training data.",
        "(already wired in scorecard)",
    ),
    "visomaster_enhanced_v2_*": (
        "EVAL_NEW_CANDIDATE",
        "Bucket gs://visomaster-enhanced-face-cropped-v2 has NO training-side reference in "
        "experiments/. Manifest exists at arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json. "
        "Bucket comparison 2026-04-23 (analysis/bucket_comparison_2026-04-23/REPORT.md) lists it as "
        "'candidate new bucket'.",
        "yaml-only-wireable",
    ),
    "visomaster_hints / visomaster_hints_teams": (
        "DISABLED_BAD_DATA",
        "Memory project_visomaster_hints_lanes_bad_data.md + R13_P14_DATA_FIX.yaml comments: "
        "weak-signal residue from face_parser bug; net training drag (RLP1 ablation Δ -0.0077..-0.0115); "
        "teams variant has co-encoded codec×partial-swap shortcut. Policy: never enable in training. "
        "These are filter-slices of the existing live-deepfake-methods-real-and-fake-frames-cropped "
        "bucket, not separate physical buckets.",
        "do-not-wire (verified bad)",
    ),
    "df40_teams": (
        "DOES_NOT_EXIST",
        "df40 fakes (df40-frames-recropped-rfa85) are training-side; no manifest declares any "
        "df40 frames captured through the Teams pipeline. teams_with_dor manifest method_counts "
        "contain zero df40 entries.",
        "n/a — no data exists",
    ),
}

# --- Compose CSV rows ------------------------------------------------------
rows = []

def add_row(name, source, n_frames, n_identities, n_envs, status, citation, wireability,
            n_videos=None, n_captures=None, sub_methods=None, label="?"):
    rows.append({
        "canonical_name": name,
        "source_location": source,
        "label": label,
        "n_videos": n_videos or "",
        "n_frames": n_frames,
        "n_identities": n_identities,
        "n_source_environments": n_envs,
        "n_captures": n_captures or "",
        "n_sub_methods": sub_methods or "",
        "train_eval_status": status,
        "status_citation": citation,
        "wireability": wireability,
    })

# --- Proper viso cells -------------------------------------------------
LANE_LABEL = {
    "proper_real_clean": "real",
    "proper_real_teams": "real",
    "proper_visomaster_clean": "fake",
    "proper_visomaster_teams": "fake",
    "proper_visomaster_enhanced_clean": "fake",
    "proper_visomaster_enhanced_teams": "fake",
}
for (lane, split), d in sorted(proper_cells.items()):
    name = f"{lane}_{split}"
    status, cite, wire = TRAINING_TABLE[lane]
    n_envs_proxy = (
        f"{len(d['quality_bands'])}q×{len(d['face_scale_bands'])}fs="
        f"{len(d['quality_bands'])*len(d['face_scale_bands'])}"
    )
    add_row(
        name=name,
        source="arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json "
               "(buckets: hdtf_visomaster_cropped_frames[_teams])",
        n_frames=d["frames"],
        n_identities=len(d["identities"]),
        n_envs=n_envs_proxy,
        n_videos=d["videos"],
        n_captures=len(d["captures"]),
        sub_methods=len(d["generator_methods"]) or "",
        status=status,
        citation=cite,
        wireability=wire,
        label=LANE_LABEL[lane],
    )

# --- Teams cells (already in scorecard for reference) ---------------------
for (method, split, label), d in sorted(teams_cells.items()):
    if method.startswith("teams_capture_"):
        status_key = "teams_capture_*"
    elif method == "teams_real":
        status_key = "teams_real"
    elif method == "deeplive_enhanced":
        status_key = "deeplive_enhanced"
    elif method == "visomaster_enhanced_macro":
        status_key = "visomaster_enhanced_macro"
    else:
        status_key = method
    status, cite, wire = TRAINING_TABLE.get(status_key, ("UNCLASSIFIED", "", "unknown"))
    add_row(
        name=f"{method}_{split}",
        source="arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json "
               "(bucket: teams-faces-data-test-2914-fake-4420-real-feb-28)",
        n_frames=d["frames"],
        n_identities=len(d["identities"]),
        n_envs=len(d["session_ids"]) or "1",
        n_videos=d["videos"],
        status=status,
        citation=cite,
        wireability=wire,
        label=label,
    )

# --- Viso v2 cells -------------------------------------------------------
total_v2_videos = total_v2_frames = 0
v2_methods = set()
v2_method_folders = collections.Counter()
for (method, split), d in sorted(viso_v2_cells.items()):
    total_v2_videos += d["videos"]
    total_v2_frames += d["frames"]
    v2_methods.add(method)
    v2_method_folders[method] = len(d["model_folders"])
    status, cite, wire = TRAINING_TABLE["visomaster_enhanced_v2_*"]
    add_row(
        name=f"{method}_{split}",
        source="arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json "
               "(bucket: visomaster-enhanced-face-cropped-v2)",
        n_frames=d["frames"],
        n_identities="N/A (1-frame-per-video)",
        n_envs=len(d["model_folders"]) or "1",
        n_videos=d["videos"],
        sub_methods=1,
        status=status,
        citation=cite,
        wireability=wire,
        label="fake",
    )

# --- Composite v2 single cell  --------------------------------------------
add_row(
    name="visomaster_enhanced_v2_ALL_fake (single-cell rollup)",
    source="arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json",
    n_frames=total_v2_frames,
    n_identities="N/A (per-frame; ~2073 source clips at most, identity not preserved)",
    n_envs=f"{len(v2_methods)} sub-methods",
    n_videos=total_v2_videos,
    sub_methods=len(v2_methods),
    status=TRAINING_TABLE["visomaster_enhanced_v2_*"][0],
    citation=TRAINING_TABLE["visomaster_enhanced_v2_*"][1],
    wireability=TRAINING_TABLE["visomaster_enhanced_v2_*"][2],
    label="fake",
)

# --- Write CSV ------------------------------------------------------------
csv_path = OUT / "candidate_eval_cells.csv"
fieldnames = [
    "canonical_name", "source_location", "label",
    "n_videos", "n_frames", "n_identities",
    "n_source_environments", "n_captures", "n_sub_methods",
    "train_eval_status", "status_citation", "wireability",
]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {csv_path}: {len(rows)} cells")

# --- Identity overlap audit ----------------------------------------------
overlap_rows = []
proper_identities = set(v["identity_id"] for v in PROPER["videos"])
teams_identities = set(v["identity_key"] for v in TEAMS["videos"])

for cell_lane, cell_split in sorted(set((v["lane"], v["split"]) for v in PROPER["videos"])):
    cell_ids = set(
        v["identity_id"] for v in PROPER["videos"]
        if v["lane"] == cell_lane and v["split"] == cell_split
    )
    for ident in sorted(cell_ids):
        is_fp = any(_matches(ident, fp) for fp in FP_OFFENDERS)
        also_in_training = "yes (proper_data lanes are in P8A training)"
        also_in_teams_substrate = ident in teams_identities or any(
            ident.lower() in t.lower() for t in teams_identities
        )
        overlap_rows.append({
            "cell": f"{cell_lane}_{cell_split}",
            "identity": ident,
            "also_in_FP_offenders": "yes" if is_fp else "no",
            "also_in_teams_scorecard_substrate": "yes" if also_in_teams_substrate else "no",
            "also_in_training": also_in_training,
        })

# Teams-side substrate identities (already-wired) — for cross-ref
for (method, split, label), d in sorted(teams_cells.items()):
    if not method.startswith("teams_") and method not in ("deeplive_enhanced", "visomaster_enhanced_macro"):
        continue
    cell = f"{method}_{split}"
    for ident in sorted(i for i in d["identities"] if i):
        is_fp = any(_matches(ident, fp) for fp in FP_OFFENDERS)
        if method.startswith("teams_capture_") or method == "teams_real":
            also_train = "yes (teams-v2 bucket is training+ood)"
        elif method == "deeplive_enhanced":
            also_train = "yes (deeplive bucket is training)"
        elif method == "visomaster_enhanced_macro":
            also_train = "yes (deeplive bucket viso strategies are training)"
        else:
            also_train = "?"
        overlap_rows.append({
            "cell": cell,
            "identity": ident,
            "also_in_FP_offenders": "yes" if is_fp else "no",
            "also_in_teams_scorecard_substrate": "self",
            "also_in_training": also_train,
        })

with open(OUT / "identity_overlap_report.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "cell", "identity", "also_in_FP_offenders",
        "also_in_teams_scorecard_substrate", "also_in_training",
    ])
    w.writeheader()
    for r in overlap_rows:
        w.writerow(r)
print(f"Wrote identity_overlap_report.csv: {len(overlap_rows)} rows")

# --- Summary JSON --------------------------------------------------------
summary = {
    "audit_date": "2026-05-04",
    "scope": "metadata-only inventory audit of unwired eval cells",
    "manifests_audited": [
        "proper_visomaster_target_domain_manifest_2026-04-19_provisional.json",
        "teams_target_domain_manifest_2026-04-23_with_dor.json",
        "visomaster_enhanced_v2_manifest_2026-04-13.json",
    ],
    "totals": {
        "proper_viso_videos": len(PROPER["videos"]),
        "proper_viso_frames": sum(len(v["frame_paths"]) for v in PROPER["videos"]),
        "proper_viso_distinct_identity_ids": len(proper_identities),
        "proper_viso_distinct_captures": len(set(v["base_capture_id"] for v in PROPER["videos"])),
        "teams_videos": len(TEAMS["videos"]),
        "teams_frames": sum(len(v["frame_paths"]) for v in TEAMS["videos"]),
        "viso_v2_videos": len(VISO_V2["videos"]),
        "viso_v2_frames": total_v2_frames,
        "viso_v2_methods": len(v2_methods),
    },
    "proper_viso_breakdown_by_lane_split": {
        f"{lane}__{split}": {
            "videos": d["videos"],
            "frames": d["frames"],
            "identities": len(d["identities"]),
            "captures": len(d["captures"]),
            "generator_methods": len(d["generator_methods"]),
        }
        for (lane, split), d in sorted(proper_cells.items())
    },
    "viso_v2_breakdown_by_method": {
        m: {"folders": v2_method_folders.get(m, 0)}
        for m in sorted(v2_methods)
    },
    "training_data_attribution": {
        k: {"status": v[0], "citation": v[1], "wireability": v[2]}
        for k, v in TRAINING_TABLE.items()
    },
    "fp_offender_overlap_with_proper_viso": {
        fp: {
            "in_proper_viso_identity_ids": any(_matches(i, fp) for i in proper_identities),
            "in_teams_substrate": any(_matches(i, fp) for i in teams_identities),
        }
        for fp in FP_OFFENDERS
    },
    "df40_teams_search_result": "no df40 method appears in teams target manifest method_counts; "
                                 "no df40-via-teams variant exists",
    "p8a_training_provenance": {
        "p8a_run_id": "9lmvb5b4",
        "p8a_yaml": "experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml",
        "p8a_proper_data_lanes": [
            "proper_visomaster_clean", "proper_visomaster_teams",
            "proper_visomaster_enhanced_teams", "proper_visomaster_enhanced_clean",
        ],
        "p8a_parent": "h2pdu6i5/step23500 (RLP6_04, also trained on all 4 proper viso lanes)",
        "lockbox_filter_active_at_training": False,
        "implication": "Lockbox-marked proper viso captures HAVE been seen by P8A training. "
                        "Identity leakage applies to all 705 proper viso identity_ids if used "
                        "as held-out eval against P8A or any P8A descendant.",
    },
}
with open(OUT / "inventory_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Wrote inventory_summary.json")
print(f"\nTotals: {json.dumps(summary['totals'], indent=2)}")
