"""Build a video inventory for the T5C showcase, labeling each video by method.

Sources:
- arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json
  Gives the 70-method visomaster enumeration (9 swap models x 7 restorers + clean variants)
  with per-video records. Each fake record carries video_id = "{base_capture_id}__{combo}__{transport}"
  which maps to:
    HDTF{date}_NNNNN  -> gs://hdtf_visomaster_videos/fakes/{base}.mp4 (+ .json sidecar)
    QCLIP{date}R2[X]_NNNNN -> gs://quickclips_visomaster_videos/fakes/{base}.mp4 (+ .json)
  Only "clean" transport has full-frame mp4s. Teams transport exists only as cropped frames.

- gsutil ls gs://live-deepfake-methods-real-and-fake-videos/samples/
  Enumerates 6773 sample dirs. Each holds {sample_id}/fake.mp4 + real.mp4. The sample_id prefix
  encodes the method: edge_cases_* / minimal_processing_* / quality_enhancement_* are deeplive;
  visomaster_{generator}_* are the 9 visomaster swap models (clean only).
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = ROOT / "analysis" / "demo_video_inventory_2026-05-14"
MANIFEST = ROOT / "arena" / "manifests" / "proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"

RESTORERS = (
    "codeformer",
    "gfpgan_v1_4",
    "gpen_2048",
    "gpen_1024",
    "gpen_512",
    "gpen_256",
    "restoreformer",
)

SWAP_MODELS = (
    "ghostface_v1",
    "ghostface_v2",
    "ghostface_v3",
    "instyleswapper256_a",
    "instyleswapper256_b",
    "instyleswapper256_c",
    "inswapper128",
    "simswap512",
    "cscs",
)


def split_combo(combo: str) -> tuple[str, str]:
    """combo is e.g. 'cscs' or 'ghostface_v1_codeformer'. Return (swap_model, restorer)."""
    for r in RESTORERS:
        if combo.endswith("_" + r):
            return combo[: -(len(r) + 1)], r
    return combo, "none"


def base_to_video_uri(base_capture_id: str) -> tuple[str, str, str]:
    """Map a base_capture_id to (source_bucket, video_mp4_uri, sidecar_json_uri)."""
    if base_capture_id.startswith("HDTF"):
        bucket = "hdtf_visomaster_videos"
    elif base_capture_id.startswith("QCLIP"):
        bucket = "quickclips_visomaster_videos"
    else:
        raise ValueError(f"unknown base_capture_id prefix: {base_capture_id}")
    return (
        bucket,
        f"gs://{bucket}/fakes/{base_capture_id}.mp4",
        f"gs://{bucket}/fakes/{base_capture_id}.json",
    )


def build_visomaster_inventory() -> tuple[list[dict], dict[str, int]]:
    with MANIFEST.open() as f:
        manifest = json.load(f)

    rows: list[dict] = []
    counts: Counter = Counter()
    for v in manifest["videos"]:
        if v["label"] != "fake":
            continue
        vid = v["video_id"]
        parts = vid.split("__")
        if len(parts) != 3:
            raise ValueError(f"unexpected video_id: {vid}")
        base, combo, transport = parts
        swap, restorer = split_combo(combo)
        bucket, mp4_uri, sidecar_uri = base_to_video_uri(base)
        row = {
            "method": v["method"],
            "lane": v["lane"],
            "transport": transport,
            "combo": combo,
            "swap_model": swap,
            "restorer": restorer,
            "base_capture_id": base,
            "source_bucket": bucket,
            "video_id": vid,
            "video_mp4_gcs_uri": mp4_uri if transport == "clean" else "",
            "sidecar_json_gcs_uri": sidecar_uri if transport == "clean" else "",
            "identity_key": v.get("identity_key", ""),
            "split": v.get("split", ""),
        }
        rows.append(row)
        counts[v["method"]] += 1
    return rows, dict(counts)


DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-videos"
DEEPLIVE_STRATEGIES_REGULAR = {"edge_cases", "minimal_processing"}
DEEPLIVE_STRATEGIES_ENHANCED = {"quality_enhancement"}

# Same bucket also holds visomaster clean videos with a different prefix scheme.
# We capture them separately as the bucket-side visomaster_clean fallback pool.
VISO_BUCKET_PREFIX_TO_SWAP = {
    "visomaster_CSCS": "cscs",
    "visomaster_GhostFace-v1": "ghostface_v1",
    "visomaster_GhostFace-v2": "ghostface_v2",
    "visomaster_GhostFace-v3": "ghostface_v3",
    "visomaster_InStyleSwapper256-A": "instyleswapper256_a",
    "visomaster_InStyleSwapper256-B": "instyleswapper256_b",
    "visomaster_InStyleSwapper256-C": "instyleswapper256_c",
    "visomaster_Inswapper128": "inswapper128",
    "visomaster_SimSwap512": "simswap512",
}


def list_deeplive_bucket_samples() -> list[str]:
    """Return sample_ids under gs://live-deepfake-methods-real-and-fake-videos/samples/."""
    result = subprocess.run(
        ["gsutil", "ls", f"gs://{DEEPLIVE_BUCKET}/samples/"],
        check=True,
        capture_output=True,
        text=True,
    )
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip().rstrip("/")
        if not line.startswith("gs://"):
            continue
        ids.append(line.rsplit("/", 1)[-1])
    return ids


SAMPLE_ID_NUM = re.compile(r"^(.*)_(\d+)$")


def classify_sample(sample_id: str) -> tuple[str, str, str] | None:
    """Return (family, method, strategy_or_swap) for a deeplive-bucket sample_id."""
    m = SAMPLE_ID_NUM.match(sample_id)
    if not m:
        return None
    prefix = m.group(1)
    if prefix in DEEPLIVE_STRATEGIES_REGULAR:
        return ("deeplive", "deeplive_regular", prefix)
    if prefix in DEEPLIVE_STRATEGIES_ENHANCED:
        return ("deeplive", "deeplive_enhanced", prefix)
    if prefix in VISO_BUCKET_PREFIX_TO_SWAP:
        swap = VISO_BUCKET_PREFIX_TO_SWAP[prefix]
        return ("visomaster_bucket_clean", f"visomaster_bucket_clean__{swap}", swap)
    if prefix == "random_mixed":
        return ("deeplive", "deeplive_random_mixed", prefix)
    return None


def build_deeplive_inventory() -> tuple[list[dict], list[dict], dict[str, int], dict[str, int]]:
    sample_ids = list_deeplive_bucket_samples()
    print(f"deeplive bucket: {len(sample_ids)} sample dirs", file=sys.stderr)
    deeplive_rows: list[dict] = []
    viso_bucket_rows: list[dict] = []
    dl_counts: Counter = Counter()
    vb_counts: Counter = Counter()
    unclassified: list[str] = []
    for sid in sample_ids:
        cls = classify_sample(sid)
        if cls is None:
            unclassified.append(sid)
            continue
        family, method, sub = cls
        row = {
            "method": method,
            "family": family,
            "sub": sub,
            "sample_id": sid,
            "fake_mp4_gcs_uri": f"gs://{DEEPLIVE_BUCKET}/samples/{sid}/fake.mp4",
            "real_mp4_gcs_uri": f"gs://{DEEPLIVE_BUCKET}/samples/{sid}/real.mp4",
        }
        if family == "deeplive":
            deeplive_rows.append(row)
            dl_counts[method] += 1
        else:
            viso_bucket_rows.append(row)
            vb_counts[method] += 1
    if unclassified:
        print(f"WARN: {len(unclassified)} unclassified sample_ids, e.g. {unclassified[:5]}", file=sys.stderr)
    return deeplive_rows, viso_bucket_rows, dict(dl_counts), dict(vb_counts)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary(
    viso_counts: dict[str, int],
    deeplive_counts: dict[str, int],
    viso_bucket_counts: dict[str, int],
    path: Path,
) -> None:
    THRESHOLD = 20
    lines: list[str] = []
    lines.append("# Demo Video Inventory — Per-Method Counts (2026-05-14)")
    lines.append("")
    lines.append("Inventory of full-frame fake videos labeled by method.")
    lines.append("")
    lines.append("Sources:")
    lines.append("- `visomaster_videos.csv` — from proper_visomaster_target_domain_manifest_2026-04-19_provisional.json. Has the 70-method (swap x restorer) enumeration. Clean transport has mp4 URIs; teams transport has only cropped frames (mp4 URI empty).")
    lines.append("- `deeplive_videos.csv` — from gsutil ls on the deeplive videos bucket. Strategies: edge_cases + minimal_processing -> regular, quality_enhancement -> enhanced.")
    lines.append("- `visomaster_bucket_videos.csv` — bonus pool of CLEAN visomaster videos in the deeplive bucket under `visomaster_{generator}_*` prefixes. No restorer info available, covers only the 9 base swap models.")
    lines.append("")
    lines.append(f"Threshold for showcase target: **>= {THRESHOLD} videos per method**.")
    lines.append("")

    # Deeplive
    lines.append("## Deeplive (live-deepfake-methods-real-and-fake-videos bucket)")
    lines.append("")
    lines.append("| method | videos | clears >=20 |")
    lines.append("|---|---|---|")
    for m, n in sorted(deeplive_counts.items()):
        flag = "YES" if n >= THRESHOLD else "no"
        lines.append(f"| {m} | {n} | {flag} |")
    lines.append("")

    # Visomaster manifest-based (70 methods) — split clean vs teams
    clean = {k: v for k, v in viso_counts.items() if "_clean" in k and "_teams" not in k}
    teams = {k: v for k, v in viso_counts.items() if "_teams" in k}

    lines.append("## Visomaster CLEAN (full-frame mp4 available)")
    lines.append("")
    lines.append("| method | videos | clears >=20 |")
    lines.append("|---|---|---|")
    for m, n in sorted(clean.items()):
        flag = "YES" if n >= THRESHOLD else "no"
        lines.append(f"| {m} | {n} | {flag} |")
    lines.append("")

    lines.append("## Visomaster TEAMS (cropped frames only, no full mp4)")
    lines.append("")
    lines.append("| method | videos | clears >=20 |")
    lines.append("|---|---|---|")
    for m, n in sorted(teams.items()):
        flag = "YES" if n >= THRESHOLD else "no"
        lines.append(f"| {m} | {n} | {flag} |")
    lines.append("")

    # Visomaster bucket fallback (9 swap models, no restorers)
    lines.append("## Visomaster bucket fallback (9 base swap models, clean, NO restorer breakdown)")
    lines.append("")
    lines.append("| method | videos | clears >=20 |")
    lines.append("|---|---|---|")
    for m, n in sorted(viso_bucket_counts.items()):
        flag = "YES" if n >= THRESHOLD else "no"
        lines.append(f"| {m} | {n} | {flag} |")
    lines.append("")

    # Totals
    n_clears = sum(1 for n in viso_counts.values() if n >= THRESHOLD)
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Visomaster methods >= {THRESHOLD}: **{n_clears} / {len(viso_counts)}**")
    lines.append(f"- Visomaster CLEAN methods >= {THRESHOLD}: **{sum(1 for n in clean.values() if n >= THRESHOLD)} / {len(clean)}**")
    lines.append(f"- Deeplive methods >= {THRESHOLD}: **{sum(1 for n in deeplive_counts.values() if n >= THRESHOLD)} / {len(deeplive_counts)}**")
    lines.append(f"- Visomaster bucket fallback methods >= {THRESHOLD}: **{sum(1 for n in viso_bucket_counts.values() if n >= THRESHOLD)} / {len(viso_bucket_counts)}**")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building visomaster inventory from manifest...", file=sys.stderr)
    viso_rows, viso_counts = build_visomaster_inventory()
    write_csv(viso_rows, OUT_DIR / "visomaster_videos.csv")
    print(f"  wrote {len(viso_rows)} visomaster fake video records", file=sys.stderr)

    print("Listing deeplive bucket samples...", file=sys.stderr)
    deeplive_rows, viso_bucket_rows, dl_counts, vb_counts = build_deeplive_inventory()
    write_csv(deeplive_rows, OUT_DIR / "deeplive_videos.csv")
    write_csv(viso_bucket_rows, OUT_DIR / "visomaster_bucket_videos.csv")
    print(f"  wrote {len(deeplive_rows)} deeplive records, {len(viso_bucket_rows)} visomaster-bucket records", file=sys.stderr)

    write_summary(viso_counts, dl_counts, vb_counts, OUT_DIR / "summary.md")
    print(f"  wrote {OUT_DIR / 'summary.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
