#!/usr/bin/env python3
"""GCS identity audit for P8A training bucket teams-v2.

Tests whether 8 eval-side identities appear in teams-v2 training-bucket
sample_ids or per-sample manifests.

READ-ONLY. No GCS writes. Bound by sample-size caps.

Usage:
    python3 run_gcs_audit.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

BUCKET = "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

IDENTITY_PATTERNS = [
    "real_dor", "Dor", "dor", "roee_dor", "Roee", "roee",
    "Cam_Test", "cam_test", "CamTest", "test_cam", "Test_Cam",
    "PC_Generator", "pc_generator", "pcgen", "PC_Gen",
    "dor_shkedi", "Roy_D", "roy_d",
    "bla_bla_chow", "Md_noyn_Sharker", "noyn", "Md_noyn",
    "xiang",
]

PER_STRATEGY_SAMPLE_CAP = 25  # 10 prefixes * 25 = up to ~250 manifests


def run(cmd: list[str], check: bool = True, capture: bool = True) -> str:
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if check and res.returncode != 0:
        sys.stderr.write(f"CMD failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\n")
        if check:
            raise SystemExit(2)
    return res.stdout


def main() -> int:
    # --- 0) auth ---
    auth = run(["gcloud", "auth", "list"], check=False)
    (OUT / "auth.txt").write_text(auth)

    # --- 1) bucket metadata ---
    md = run(["gsutil", "cat", f"{BUCKET}/metadata.json"], check=False)
    (OUT / "bucket_metadata.json").write_text(md)

    # --- 2) enumerate sample_ids ---
    listing_path = OUT / "training_bucket_sample_ids.txt"
    listing_text = run(["gsutil", "ls", f"{BUCKET}/samples/"], check=True)
    listing_path.write_text(listing_text)
    lines = [ln for ln in listing_text.splitlines() if ln.strip()]
    total_samples = len(lines)
    print(f"Enumerated {total_samples} sample paths")

    # extract bare sample_ids
    sample_ids = []
    for ln in lines:
        m = re.search(r"/samples/([^/]+)/?$", ln.rstrip("/"))
        if m:
            sample_ids.append(m.group(1))
    print(f"Parsed {len(sample_ids)} sample_ids")

    # --- 3) listing-level grep ---
    listing_hits = {}
    listing_examples = defaultdict(list)
    for p in IDENTITY_PATTERNS:
        rx = re.compile(re.escape(p), re.IGNORECASE)
        matched = [sid for sid in sample_ids if rx.search(sid)]
        listing_hits[p] = len(matched)
        listing_examples[p] = matched[:3]

    # --- 4) stratified manifest pull ---
    by_prefix = defaultdict(list)
    prefixes = [
        "edge_cases", "minimal_processing", "quality_enhancement",
        "visomaster_CSCS", "visomaster_GhostFace-v1", "visomaster_GhostFace-v2",
        "visomaster_GhostFace-v3", "visomaster_InStyleSwapper256-A",
        "visomaster_InStyleSwapper256-B", "visomaster_Inswapper128",
    ]
    for sid in sample_ids:
        for pfx in prefixes:
            if sid.startswith(pfx + "_"):
                by_prefix[pfx].append(sid)
                break

    picked = []
    for pfx in prefixes:
        picked.extend(by_prefix[pfx][:PER_STRATEGY_SAMPLE_CAP])
    print(f"Picked {len(picked)} sample_ids for manifest inspection")
    (OUT / "_picked_sample_ids.txt").write_text("\n".join(picked) + "\n")

    # Fetch manifests via gsutil cat (one call per sample — bounded ~250 calls).
    manifests = {}  # sid -> dict|None
    errors = 0
    for i, sid in enumerate(picked):
        uri = f"{BUCKET}/samples/{sid}/manifest.json"
        out = run(["gsutil", "cat", uri], check=False)
        if not out.strip():
            errors += 1
            manifests[sid] = None
            continue
        try:
            manifests[sid] = json.loads(out)
        except json.JSONDecodeError:
            errors += 1
            manifests[sid] = None
        if (i + 1) % 50 == 0:
            print(f"  fetched {i+1}/{len(picked)}")
    pulled = sum(1 for v in manifests.values() if v is not None)
    print(f"Pulled {pulled} manifests, {errors} errors")

    # --- 5) manifest-level grep ---
    manifest_hits = {}
    manifest_examples = defaultdict(list)
    for p in IDENTITY_PATTERNS:
        # token-ish match: non-alphanumeric boundary or start/end
        rx = re.compile(r"(?:^|[^A-Za-z0-9])" + re.escape(p) + r"(?:[^A-Za-z0-9]|$)",
                        re.IGNORECASE)
        n = 0
        ex = []
        for sid, m in manifests.items():
            if m is None:
                continue
            # serialize as JSON text and search
            blob = json.dumps(m)
            if rx.search(blob):
                n += 1
                if len(ex) < 3:
                    # find which field matched (rough)
                    for k, v in m.items():
                        if isinstance(v, str) and rx.search(v):
                            ex.append(f"{sid}::{k}={v}")
                            break
                    else:
                        ex.append(f"{sid}::<unknown_field>")
        manifest_hits[p] = n
        manifest_examples[p] = ex

    # --- 6) write final CSV ---
    csv_path = OUT / "identity_matches.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity_pattern", "n_listing_matches", "n_manifest_matches",
            "listing_examples", "manifest_examples",
        ])
        for p in IDENTITY_PATTERNS:
            w.writerow([
                p,
                listing_hits[p],
                manifest_hits[p],
                ";".join(listing_examples[p]),
                ";".join(manifest_examples[p]),
            ])

    # --- 7) example manifests ---
    examples_path = OUT / "sample_manifest_examples.json"
    example_subset = {}
    take_each = 1  # 1 from each prefix
    for pfx in prefixes:
        for sid in by_prefix[pfx][:take_each]:
            if manifests.get(sid) is not None:
                example_subset[sid] = manifests[sid]
    examples_path.write_text(json.dumps(example_subset, indent=2))

    # --- 8) audit field schema across all pulled manifests ---
    all_fields = set()
    for m in manifests.values():
        if m is not None:
            all_fields.update(m.keys())
    field_audit_path = OUT / "manifest_field_schema.txt"
    field_audit_path.write_text(
        f"Total manifests inspected: {pulled}\n"
        f"All keys observed (union):\n  " + "\n  ".join(sorted(all_fields)) + "\n"
    )

    # --- 9) summary ---
    print("\n=== SUMMARY ===")
    print(f"Total samples in teams-v2: {total_samples}")
    print(f"Manifests inspected: {pulled} ({errors} errors)")
    print("Listing-level hits:")
    for p in IDENTITY_PATTERNS:
        if listing_hits[p] > 0:
            print(f"  {p}: {listing_hits[p]}")
    print("Manifest-level hits (token-ish):")
    for p in IDENTITY_PATTERNS:
        if manifest_hits[p] > 0:
            print(f"  {p}: {manifest_hits[p]}  examples={manifest_examples[p]}")
    if all(v == 0 for v in listing_hits.values()) and \
       all(v == 0 for v in manifest_hits.values()):
        print("ZERO listing-level and ZERO manifest-level matches across all 23 patterns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
