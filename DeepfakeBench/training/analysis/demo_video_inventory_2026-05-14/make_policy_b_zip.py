"""Bundle the Policy-B subset of the demo video set into a single zip.

Policy B:
- Visomaster `none` (no restorer): 40 per swap (cap; if dir has <40, take all).
- Visomaster with restorer: take ALL.
- Deeplive regular + enhanced: 40 each (cap).
- Reals: take ALL (HDTF + QuickClips, no cap).

Selection within a dir is deterministic: sorted by filename ascending, first N.
Stored (no-compression) zip because mp4 is already compressed.

Output: dataset/demo_videos_2026-05-14_policy_b.zip
Includes a filtered MANIFEST.csv at the top of the archive.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


SRC = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/dataset/demo_videos_2026-05-14")
ZIP_PATH = SRC.parent / "demo_videos_2026-05-14_policy_b.zip"
SRC_MANIFEST = SRC / "MANIFEST.csv"

VISO_NONE_CAP = 40
DEEPLIVE_CAP = 40


def select_files() -> list[Path]:
    selected: list[Path] = []

    # Visomaster
    viso_root = SRC / "visomaster"
    for swap_dir in sorted(viso_root.iterdir()):
        if not swap_dir.is_dir():
            continue
        for rest_dir in sorted(swap_dir.iterdir()):
            if not rest_dir.is_dir():
                continue
            mp4s = sorted(rest_dir.glob("*.mp4"))
            if rest_dir.name == "none":
                selected.extend(mp4s[:VISO_NONE_CAP])
            else:
                selected.extend(mp4s)

    # Deeplive
    for split in ("regular", "enhanced"):
        d = SRC / "deeplive" / split
        mp4s = sorted(d.glob("*.mp4"))
        selected.extend(mp4s[:DEEPLIVE_CAP])

    # Reals (no cap)
    for src in ("hdtf", "quickclips"):
        d = SRC / "reals" / src
        selected.extend(sorted(d.glob("*.mp4")))

    return selected


def filtered_manifest(selected_rel: set[str]) -> list[dict]:
    rows = list(csv.DictReader(SRC_MANIFEST.open()))
    return [r for r in rows if r["local_path"] in selected_rel]


def main() -> None:
    selected = select_files()
    selected_rel = {str(p.relative_to(SRC)) for p in selected}
    print(f"selected {len(selected)} files", file=sys.stderr)

    # Categorize for sanity
    from collections import Counter
    cat = Counter()
    for r in selected_rel:
        if r.startswith("visomaster/"):
            parts = r.split("/")
            cat[f"viso/{parts[1]}/{parts[2]}"] += 1
        else:
            parts = r.split("/", 2)
            cat["/".join(parts[:2])] += 1
    for k, v in sorted(cat.items()):
        print(f"  {k}: {v}", file=sys.stderr)

    # Write filtered manifest into a temp file
    rows = filtered_manifest(selected_rel)
    print(f"filtered manifest: {len(rows)} rows", file=sys.stderr)
    if len(rows) != len(selected):
        print(f"WARN: manifest rows ({len(rows)}) != selected files ({len(selected)}); checking...", file=sys.stderr)
        missing = selected_rel - {r["local_path"] for r in rows}
        for m in list(missing)[:10]:
            print(f"  no manifest row for: {m}", file=sys.stderr)

    tmp_man = SRC / "MANIFEST_policy_b.csv"
    with tmp_man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_path", "method", "source_id", "gcs_uri"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {tmp_man}", file=sys.stderr)

    # Remove old zip if present
    if ZIP_PATH.exists():
        print(f"removing existing {ZIP_PATH}", file=sys.stderr)
        ZIP_PATH.unlink()

    # Build file list relative to SRC for zip
    file_list = [str(p.relative_to(SRC)) for p in selected] + ["MANIFEST_policy_b.csv"]
    list_file = SRC / "_ziplist.txt"
    list_file.write_text("\n".join(file_list) + "\n")

    print(f"zipping {len(file_list)} entries to {ZIP_PATH}...", file=sys.stderr)
    # -0: store (no compression) -- mp4 is already compressed
    # -@ : read names from stdin
    # We use the list file via xargs/redirect
    cmd = ["zip", "-0", "-@", str(ZIP_PATH)]
    res = subprocess.run(cmd, cwd=str(SRC), stdin=list_file.open(), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"zip failed: {res.stderr[:500]}", file=sys.stderr)
        sys.exit(2)
    print("zip done", file=sys.stderr)

    # Cleanup intermediate files (but keep them in zip)
    list_file.unlink()

    # Final size
    size_gb = ZIP_PATH.stat().st_size / (1024 ** 3)
    print(f"final zip: {ZIP_PATH} ({size_gb:.2f} GB)", file=sys.stderr)


if __name__ == "__main__":
    main()
