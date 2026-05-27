"""Suite composition audit (no model inference).

Builds three artifacts:
  - frames_tagged.csv      — every frame from raw_reports/*_frames_report.csv with
                             added columns: suite, ckpt, source_method, enhancement,
                             teams_passthrough, identity.
  - composition_table.csv  — group-by counts per (suite × source_method × enhancement
                             × teams_passthrough × label) cell with n_unique_identities.
  - composition_method_only.csv — sanity grouping per (suite × method (raw)).

Pure metadata join. No GPU. No sklearn.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
OUT = ROOT / "analysis" / "suite_composition_audit_2026-05-04"
OUT.mkdir(parents=True, exist_ok=True)

SUITE_YAML = ROOT / "arena" / "target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml"
MANIFEST_JSON = ROOT / "arena" / "manifests" / "teams_target_domain_manifest_2026-04-23_with_dor.json"
RAW_DIR = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports"


# ---------- Tagging helpers ----------------------------------------------------

# A method (string from manifest / report) -> (source_method, enhancement, teams_passthrough)
# Per the user's clarified taxonomy:
#   * deeplive / visomaster / df40 are "creation methods"
#   * "teams" is a transport pipeline (codec/capture/compression) applied on top
#   * teams_capture_*  → real footage captured *through* the Teams pipeline; the
#                       label in the manifest is "fake" because the original face
#                       was swapped by an upstream creation method (these are
#                       sessions captured via Teams). The creation method is not
#                       individually tagged in the manifest record, so we mark
#                       source_method="teams_capture_session" with enhancement
#                       unknown. teams_passthrough=yes.
#   * teams_flat_xiang_xiang2_feng → flat-upload Xiang fake; treat as
#                       source_method="other_flat" (creation method opaque) with
#                       teams_passthrough=no (flat-upload, did not go through
#                       Teams capture pipeline). enhancement unknown.
#   * deeplive_enhanced → source_method=deeplive, enhancement=enhanced,
#                        teams_passthrough=no (flat-upload of upscaled deeplive).
#   * visomaster_enhanced_macro → source_method=visomaster, enhancement=enhanced,
#                                teams_passthrough=no.
#   * teams_real → bare real footage captured through Teams pipeline (passthrough=yes)
#   * dor_real_passthrough → real footage of "dor", Teams passthrough=yes.

def tag_from_method(method: str, frame_path: str = "", video_id: str = "") -> dict:
    m = (method or "").lower()
    p = (frame_path or "").lower()
    v = (video_id or "").lower()

    out = {
        "source_method": "unknown",
        "enhancement": "unknown",
        "teams_passthrough": "unknown",
    }

    if m == "deeplive_enhanced":
        out.update(source_method="deeplive", enhancement="enhanced", teams_passthrough="no")
    elif m == "visomaster_enhanced_macro":
        out.update(source_method="visomaster", enhancement="enhanced", teams_passthrough="no")
    elif m == "teams_real":
        out.update(source_method="real", enhancement="n/a", teams_passthrough="yes")
    elif m == "dor_real_passthrough":
        out.update(source_method="real", enhancement="n/a", teams_passthrough="yes")
    elif m.startswith("teams_capture_"):
        # Teams-pipeline session capture; original face was swapped upstream by an
        # opaque creation method. Cannot disambiguate which creation method without
        # extra metadata, so source_method is "teams_capture_session".
        out.update(
            source_method="teams_capture_session",
            enhancement="unknown",
            teams_passthrough="yes",
        )
    elif m == "teams_flat_xiang_xiang2_feng":
        # Flat-upload of an externally generated fake (Xiang Xiang2 Feng). Did NOT
        # go through Teams capture pipeline (flat upload). Creation tool unspecified.
        out.update(
            source_method="other_flat_fake",
            enhancement="unknown",
            teams_passthrough="no",
        )
    else:
        # Try to recover from path tokens.
        if "deeplive" in p or "deeplive" in v:
            out.update(source_method="deeplive", enhancement="unknown", teams_passthrough="unknown")
        elif "visomaster" in p or "visomaster" in v:
            out.update(source_method="visomaster", enhancement="unknown", teams_passthrough="unknown")

    # Stamp explicit "enhanced" tag if path/video says so but method didn't.
    if out["enhancement"] == "unknown" and ("enhanced" in p or "enhanced" in v):
        out["enhancement"] = "enhanced"

    return out


_IDENTITY_PATTERNS = [
    (re.compile(r"^([A-Za-z0-9_]+?)__seq\d+"), 1),
    (re.compile(r"^([A-Za-z0-9_]+?)__s\d+__seg_"), 1),
    (re.compile(r"^([A-Za-z0-9_]+?)__seg_"), 1),
    (re.compile(r"^(deeplive_[A-Za-z0-9_]+?)__"), 1),
    (re.compile(r"^([A-Za-z0-9_]+?)__"), 1),
]


def identity_from_video_id(video_id: str, fallback_ikey: str = "") -> str:
    if not video_id:
        return fallback_ikey or "unknown"
    for pat, grp in _IDENTITY_PATTERNS:
        m = pat.match(video_id)
        if m:
            ident = m.group(grp)
            # strip trailing __sNN session token to coarsen identity
            ident = re.sub(r"__s\d+$", "", ident)
            return ident
    return fallback_ikey or video_id


# ---------- Load reference data -----------------------------------------------

print("[1/4] Loading suite yaml + manifest …")
with open(SUITE_YAML) as f:
    suites_doc = yaml.safe_load(f)

suites_meta = {s["name"]: s for s in suites_doc["suites"]}
print(f"  suites in yaml: {len(suites_meta)} -> {list(suites_meta)}")

with open(MANIFEST_JSON) as f:
    manifest = json.load(f)
videos = manifest["videos"]
print(f"  manifest videos: {len(videos)}")

# Build a frame_path -> manifest_record lookup so we can recover identity_key,
# session_id, and slices from the manifest if needed.
frame_to_record: dict[str, dict] = {}
for v in videos:
    for fp in v.get("frame_paths") or []:
        frame_to_record[fp] = v
print(f"  manifest frame_path keys: {len(frame_to_record)}")


# ---------- Scan the 45 frames_report CSVs ------------------------------------

print("[2/4] Tagging frames from 45 raw_reports CSVs …")

# suite naming convention in filename:  {SUITE}_{CKPT_LABEL}_frames_report.csv
# where SUITE is one of the 9 suites and CKPT_LABEL is one of:
#   p8a_reference_step5000
#   e2b_top_n_step3200
#   e3_top_n_step4800
#   e3_top_n_step6600
#   e3_top_n_step7200
KNOWN_SUITES = sorted(
    {
        "deeplive_enhanced_dev",
        "visomaster_enhanced_macro_dev",
        "teams_fake_all_dev",
        "teams_fake_all_lockbox",
        "teams_real_all_dev",
        "teams_real_all_lockbox",
        "teams_real_dor_dev",
        "teams_real_lighting_extreme_dev",
        "teams_real_poor_quality_dev",
    },
    key=len,
    reverse=True,  # match longest first
)


def parse_filename(fname: str) -> tuple[str, str]:
    base = fname[:-len("_frames_report.csv")]
    for suite in KNOWN_SUITES:
        if base.startswith(suite + "_"):
            ckpt = base[len(suite) + 1:]
            return suite, ckpt
    raise ValueError(f"Could not parse suite from {fname}")


report_files = sorted(RAW_DIR.glob("*_frames_report.csv"))
print(f"  found {len(report_files)} frames_report files")

dfs = []
for fp in report_files:
    suite, ckpt = parse_filename(fp.name)
    df = pd.read_csv(fp)
    df["suite"] = suite
    df["ckpt"] = ckpt
    dfs.append(df)

frames = pd.concat(dfs, ignore_index=True)
print(f"  concatenated rows: {len(frames):,}")

# Tag source_method / enhancement / teams_passthrough.
# Vectorise by precomputing a method->tag map then applying.
unique_methods = frames["method"].fillna("").unique().tolist()
method_tag_map = {m: tag_from_method(m) for m in unique_methods}
tag_df = pd.DataFrame.from_records(
    [method_tag_map[m] for m in frames["method"].fillna("")],
    index=frames.index,
)
frames = pd.concat([frames, tag_df], axis=1)

# Refine via path tokens for any rows still unknown.
def _refine_row(row):
    if row["source_method"] != "unknown" and row["enhancement"] != "unknown":
        return row
    refined = tag_from_method(row["method"] or "", row["frame_path"] or "", row["video_id"] or "")
    if row["source_method"] == "unknown":
        row["source_method"] = refined["source_method"]
    if row["enhancement"] == "unknown":
        row["enhancement"] = refined["enhancement"]
    if row["teams_passthrough"] == "unknown":
        row["teams_passthrough"] = refined["teams_passthrough"]
    return row


needs_refine = (frames["source_method"] == "unknown") | (frames["enhancement"] == "unknown")
if needs_refine.any():
    print(f"  refining {needs_refine.sum():,} rows via path tokens …")
    frames.loc[needs_refine] = frames.loc[needs_refine].apply(_refine_row, axis=1)

# Identity extraction.
print("[3/4] Extracting identity …")

# Use manifest's identity_key when present (more accurate than parsing video_id).
manifest_ikey = []
for fp in frames["frame_path"]:
    rec = frame_to_record.get(fp)
    manifest_ikey.append(rec["identity_key"] if rec and "identity_key" in rec else "")
frames["manifest_identity_key"] = manifest_ikey

frames["identity"] = [
    ikey if ikey else identity_from_video_id(vid)
    for ikey, vid in zip(frames["manifest_identity_key"], frames["video_id"].fillna(""))
]

# ---------- Write frames_tagged.csv -------------------------------------------

frames_tagged_cols = [
    "suite",
    "ckpt",
    "method",
    "label",
    "video_id",
    "frame_path",
    "frame_prob",
    "group_key",
    "family_key",
    "source_method",
    "enhancement",
    "teams_passthrough",
    "identity",
]
frames_tagged = frames[frames_tagged_cols].copy()
frames_tagged_path = OUT / "frames_tagged.csv"
frames_tagged.to_csv(frames_tagged_path, index=False)
print(f"  wrote {frames_tagged_path}  rows={len(frames_tagged):,}")


# ---------- Composition table -------------------------------------------------

print("[4/4] Building composition_table.csv …")

# Use one ckpt per suite (the rows replicate across ckpts), so we de-dupe to a
# single ckpt to count "physical" frames inside each suite. Pick the
# p8a_reference_step5000 slice if present, else the first ckpt available.
def _pick_ref_ckpt(group: pd.DataFrame) -> str:
    ckpts = group["ckpt"].unique().tolist()
    if "p8a_reference_step5000" in ckpts:
        return "p8a_reference_step5000"
    return sorted(ckpts)[0]


ref_ckpts = (
    frames_tagged.groupby("suite", group_keys=False)
    .apply(lambda g: pd.Series({"ref_ckpt": _pick_ref_ckpt(g)}))
    .reset_index()
)

frames_one_ckpt = frames_tagged.merge(ref_ckpts, on="suite")
frames_one_ckpt = frames_one_ckpt[frames_one_ckpt["ckpt"] == frames_one_ckpt["ref_ckpt"]].drop(columns=["ref_ckpt"])
print(f"  de-duplicated to one ckpt per suite: rows={len(frames_one_ckpt):,}")

comp = (
    frames_one_ckpt.groupby(
        ["suite", "label", "source_method", "enhancement", "teams_passthrough"],
        dropna=False,
    )
    .agg(n_frames=("frame_path", "count"), n_unique_identities=("identity", "nunique"))
    .reset_index()
    .sort_values(["suite", "label", "source_method", "enhancement", "teams_passthrough"])
)
comp_path = OUT / "composition_table.csv"
comp.to_csv(comp_path, index=False)
print(f"  wrote {comp_path}  cells={len(comp)}")

# Sanity: per (suite × method) breakdown.
comp_method = (
    frames_one_ckpt.groupby(["suite", "label", "method"], dropna=False)
    .agg(n_frames=("frame_path", "count"), n_unique_identities=("identity", "nunique"))
    .reset_index()
    .sort_values(["suite", "label", "method"])
)
comp_method_path = OUT / "composition_method_only.csv"
comp_method.to_csv(comp_method_path, index=False)
print(f"  wrote {comp_method_path}  cells={len(comp_method)}")

# Coverage stats for FINDINGS.
known_sm = (frames_one_ckpt["source_method"] != "unknown").mean() * 100
known_enh = frames_one_ckpt["enhancement"].isin(["enhanced", "clean", "n/a"]).mean() * 100
known_tp = frames_one_ckpt["teams_passthrough"].isin(["yes", "no"]).mean() * 100
print()
print("Coverage:")
print(f"  source_method known:     {known_sm:.1f}%")
print(f"  enhancement resolved:    {known_enh:.1f}%")
print(f"  teams_passthrough known: {known_tp:.1f}%")

# Stats blob for findings.
stats_blob = {
    "n_files": len(report_files),
    "n_rows_tagged": int(len(frames_tagged)),
    "n_rows_one_ckpt": int(len(frames_one_ckpt)),
    "suites": sorted(frames_tagged["suite"].unique().tolist()),
    "ckpts": sorted(frames_tagged["ckpt"].unique().tolist()),
    "coverage_pct": {
        "source_method_known": float(round(known_sm, 2)),
        "enhancement_resolved": float(round(known_enh, 2)),
        "teams_passthrough_known": float(round(known_tp, 2)),
    },
    "teams_fake_all_dev_method_breakdown": comp_method[comp_method.suite == "teams_fake_all_dev"].to_dict("records"),
    "teams_fake_all_lockbox_method_breakdown": comp_method[comp_method.suite == "teams_fake_all_lockbox"].to_dict("records"),
    "all_suite_total_frames": frames_one_ckpt.groupby("suite").size().to_dict(),
    "unique_identities_per_suite": frames_one_ckpt.groupby("suite")["identity"].nunique().to_dict(),
}
with open(OUT / "audit_stats.json", "w") as f:
    json.dump(stats_blob, f, indent=2, default=str)
print(f"  wrote {OUT / 'audit_stats.json'}")
print("done.")
