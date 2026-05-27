"""Train-overlap audit for D5 recurring residual identities — reproducibility script.

Question: do `real_dor`, `Cam_Test`, `PC_Generator` (the three recurring top-residual
identities from D5 §4.4) appear in P8A's training pool?

Approach (CPU only, no GCS calls):
1. Enumerate eval-side provenance per identity from the three eval manifests.
2. Grep all 158 R13 yamls for the eval bucket; verify 0 matches.
3. Catalog P8A yaml's training/OOD/readout buckets.

Outputs: outputs/per_identity_eval_provenance.csv

See sibling TRAIN_OVERLAP_FACTS_2026-05-12.md for analysis.
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis" / "train_overlap_audit_2026-05-12" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_TEAMS = ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
P8A_YAML = ROOT / "experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml"
R13_YAMLS_DIR = ROOT / "experiments/phase2_round13"

IDENTS = ["real_dor", "Cam_Test", "PC_Generator"]


def main() -> None:
    # === Step 1: per-identity eval provenance ===
    with EVAL_TEAMS.open() as f:
        teams = json.load(f)

    rows = []
    for ident in IDENTS:
        cells_label: Counter = Counter()
        sessions: set = set()
        sources: set = set()
        first_path = ""
        for v in teams["videos"]:
            ik = (v.get("identity_key") or "") + " " + (v.get("prefix") or "")
            method = v.get("method") or ""
            if ident.lower() in ik.lower() or ident.lower() in method.lower():
                cells_label[(v.get("split"), v.get("label"))] += 1
                sid = v.get("session_id")
                if sid:
                    sessions.add(sid)
                if v.get("frame_paths"):
                    bucket = v["frame_paths"][0].split("/")[2]
                    sources.add(bucket)
                    if not first_path:
                        first_path = v["frame_paths"][0]
        rows.append({
            "identity": ident,
            "n_videos_total": sum(cells_label.values()),
            "sessions": ";".join(sorted(s for s in sessions if s)),
            "source_buckets_in_eval_manifest": ";".join(sorted(sources)),
            "cells_label_breakdown": json.dumps({str(k): v for k, v in cells_label.items()}),
            "first_frame_path": first_path,
        })

    csv_path = OUT / "per_identity_eval_provenance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[step 1] wrote {csv_path}")
    for r in rows:
        print(f"  {r}")

    # === Step 2: grep eval bucket across all R13 yamls ===
    eval_bucket = "teams-faces-data-test-2914"
    grep_proc = subprocess.run(
        ["grep", "-rn", eval_bucket, str(R13_YAMLS_DIR)],
        capture_output=True, text=True, check=False,
    )
    n_matches = len([l for l in grep_proc.stdout.splitlines() if l.strip()])
    print(f"[step 2] grep '{eval_bucket}' across R13 yamls: {n_matches} matches")

    # === Step 3: catalog P8A bucket references ===
    print(f"[step 3] P8A yaml bucket references (R13_RLP8_01_unfreeze_clip_codec.yaml):")
    grep_p8a = subprocess.run(
        ["grep", "-nE", r"(gcs_)?bucket:?\s+\"?[a-zA-Z0-9_-]+", str(P8A_YAML)],
        capture_output=True, text=True, check=False,
    )
    for line in grep_p8a.stdout.splitlines():
        if line.strip():
            print(f"  {line}")


if __name__ == "__main__":
    main()
