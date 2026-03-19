#!/usr/bin/env python3
"""
Offline Teams codec simulation — applies TeamsCodecSimulation to a folder
of images and writes augmented copies to an output directory.

Usage:
    python apply_teams_augmentation.py \
        --input-dir  "/path/to/raw/faces" \
        --output-dir "/path/to/teams_augmented"

Each PNG/JPG is read, passed through the full Teams pipeline
(brightness+contrast boost → Gaussian blur → JPEG compression → bilateral
deblocking) and saved as PNG in the output directory with the same filename.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Direct import of the teams_simulation module to avoid pulling in
# the full training package (which requires torch, etc.)
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _SCRIPT_DIR.parent
_TEAMS_SIM = _TRAINING_DIR / "data" / "augmentations" / "teams_simulation.py"

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("teams_simulation", _TEAMS_SIM)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
TeamsCodecSimulation = _mod.TeamsCodecSimulation

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Teams codec simulation to a folder of images."
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing source images (PNG/JPG)."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write augmented images."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"No image files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} images in {input_dir}")
    print(f"Output → {output_dir}")

    # Teams simulation with always_apply so every image gets augmented
    transform = TeamsCodecSimulation(always_apply=True, p=1.0)

    np.random.seed(args.seed)
    ok, fail = 0, 0
    for i, img_path in enumerate(files, 1):
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cv2.imread returned None for {img_path.name}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Apply Teams codec simulation (expects uint8 RGB)
            augmented = transform.apply(rgb)

            # Save as PNG (lossless) to preserve exactly what was produced
            out_path = output_dir / img_path.name
            bgr_out = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr_out)
            ok += 1
        except Exception as exc:
            print(f"  FAIL [{i}/{len(files)}] {img_path.name}: {exc}")
            fail += 1

        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] processed ({ok} ok, {fail} fail)")

    print(f"\nDone: {ok} augmented, {fail} failed → {output_dir}")


if __name__ == "__main__":
    main()
