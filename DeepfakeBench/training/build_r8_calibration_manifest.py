#!/usr/bin/env python3
"""
Build a calibration manifest for R8 threshold calibration.

Expected default folder layout:

  <root_dir>/real/<identity>/**/*.jpg
  <root_dir>/fake/<identity>/**/*.jpg

Output CSV columns:
  sample_id,path,label,identity,source

Where:
  - label: 0 for real, 1 for fake
  - path: absolute path to image file
  - identity: identity token extracted from folder name
  - source: the immediate folder after <real|fake>/<identity>/ (or "unknown")
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


IMAGE_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class ManifestRow:
    sample_id: str
    path: str
    label: int
    identity: str
    source: str


def _discover_images(root: Path, exts: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _source_from_path(rel_parts: Sequence[str]) -> str:
    # rel parts are expected to begin after label dir: identity/<maybe source>/.../file
    if len(rel_parts) >= 2:
        return rel_parts[1]
    return "unknown"


def _collect_rows(
    *,
    base_dir: Path,
    label_name: str,
    label_value: int,
    exts: Sequence[str],
    max_frames_per_identity: int | None,
    seed: int,
) -> List[ManifestRow]:
    label_dir = base_dir / label_name
    if not label_dir.exists():
        return []

    image_paths = _discover_images(label_dir, exts)
    by_identity: Dict[str, List[Path]] = {}

    for p in image_paths:
        try:
            rel = p.relative_to(label_dir)
        except Exception:
            continue
        parts = rel.parts
        if not parts:
            continue
        identity = parts[0]
        by_identity.setdefault(identity, []).append(p)

    rng = random.Random(seed)
    rows: List[ManifestRow] = []

    for identity, items in sorted(by_identity.items()):
        ordered = sorted(items)
        if max_frames_per_identity is not None and len(ordered) > max_frames_per_identity:
            rng.shuffle(ordered)
            ordered = ordered[:max_frames_per_identity]
            ordered.sort()

        for idx, path in enumerate(ordered):
            rel = path.relative_to(label_dir)
            source = _source_from_path(rel.parts)
            sample_id = f"{label_name}_{identity}_{idx:04d}"
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    path=str(path.resolve()),
                    label=label_value,
                    identity=identity,
                    source=source,
                )
            )

    return rows


def _write_csv(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "path", "label", "identity", "source"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "path": r.path,
                    "label": r.label,
                    "identity": r.identity,
                    "source": r.source,
                }
            )


def _summarize(rows: List[ManifestRow]) -> str:
    n_total = len(rows)
    n_real = sum(1 for r in rows if r.label == 0)
    n_fake = sum(1 for r in rows if r.label == 1)

    real_ids = {r.identity for r in rows if r.label == 0}
    fake_ids = {r.identity for r in rows if r.label == 1}

    return (
        f"samples={n_total} real={n_real} fake={n_fake} "
        f"real_ids={len(real_ids)} fake_ids={len(fake_ids)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R8 calibration manifest from folder tree")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Data root directory. Expected subfolders real/ and fake/ by default.")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--real_dir_name", type=str, default="real")
    parser.add_argument("--fake_dir_name", type=str, default="fake")
    parser.add_argument("--extensions", type=str, default=",".join(IMAGE_EXTS_DEFAULT),
                        help="Comma-separated list of image extensions")
    parser.add_argument("--max_frames_per_identity", type=int, default=None,
                        help="Optional cap per identity per class")
    parser.add_argument("--min_frames_per_identity", type=int, default=1,
                        help="Drop identities with fewer than this many frames in a class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    exts = tuple(x.strip().lower() for x in args.extensions.split(",") if x.strip())
    if not exts:
        raise ValueError("No valid extensions provided")

    real_rows = _collect_rows(
        base_dir=root_dir,
        label_name=args.real_dir_name,
        label_value=0,
        exts=exts,
        max_frames_per_identity=args.max_frames_per_identity,
        seed=args.seed,
    )
    fake_rows = _collect_rows(
        base_dir=root_dir,
        label_name=args.fake_dir_name,
        label_value=1,
        exts=exts,
        max_frames_per_identity=args.max_frames_per_identity,
        seed=args.seed + 1,
    )

    if not real_rows:
        raise RuntimeError(f"No real images found under: {root_dir / args.real_dir_name}")
    if not fake_rows:
        raise RuntimeError(f"No fake images found under: {root_dir / args.fake_dir_name}")

    # Filter identities by min frame count inside each class.
    def _filter(rows: List[ManifestRow], min_n: int) -> List[ManifestRow]:
        if min_n <= 1:
            return rows
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r.identity] = counts.get(r.identity, 0) + 1
        keep = {k for k, v in counts.items() if v >= min_n}
        return [r for r in rows if r.identity in keep]

    real_rows = _filter(real_rows, args.min_frames_per_identity)
    fake_rows = _filter(fake_rows, args.min_frames_per_identity)

    rows = real_rows + fake_rows
    rows.sort(key=lambda r: (r.label, r.identity, r.path))

    out_csv = Path(args.output_csv).resolve()
    _write_csv(out_csv, rows)

    print("[build_r8_calibration_manifest] done")
    print(f"  root_dir   : {root_dir}")
    print(f"  output_csv : {out_csv}")
    print(f"  summary    : {_summarize(rows)}")


if __name__ == "__main__":
    main()
