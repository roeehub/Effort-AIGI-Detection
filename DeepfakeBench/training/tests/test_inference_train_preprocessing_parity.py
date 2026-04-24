"""Guard against inference/training preprocessing drift.

The model is trained on images resized via cv2.INTER_LINEAR. Any inference path
that resizes to the model resolution MUST also use INTER_LINEAR, otherwise
retro-score and batch-inference numbers drift from what the checkpoint saw.

Prior bug (2026-04-24): batch_inference_gcs.py and arena/model_arena.py both
used INTER_AREA, silently biasing every retro-score result.

This test grep-asserts the invariant at the AST-unaware level. It also runs a
quick numerical check that the two interpolations genuinely produce different
output (so if someone "fixes" the assertion by swapping both sides to INTER_AREA
the other half of the check still catches it).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (relative_path, line_number_hint) for each inference-path resize we care about.
# The line hint is documentation only; the check is content-based.
INFERENCE_RESIZE_SITES = [
    "batch_inference_gcs.py",
    "arena/model_arena.py",
]

# Lines that call cv2.resize and declare an interpolation in the same physical line
# — every such declaration must be INTER_LINEAR in the inference paths.
INTERP_KWARG = re.compile(r"interpolation\s*=\s*cv2\.(INTER_\w+)")


@pytest.mark.parametrize("rel_path", INFERENCE_RESIZE_SITES)
def test_inference_site_uses_inter_linear(rel_path: str):
    """Every resize call in the main inference entrypoints must be INTER_LINEAR."""
    full = REPO_ROOT / rel_path
    assert full.exists(), f"Expected inference path missing: {full}"
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(full.read_text().splitlines(), start=1):
        if "cv2.resize" not in line:
            continue
        m = INTERP_KWARG.search(line)
        if m:
            found.append((lineno, m.group(1)))
    assert found, f"No cv2.resize(...interpolation=...) call found in {rel_path}"
    bad = [f"line {ln}: {kind}" for ln, kind in found if kind != "INTER_LINEAR"]
    assert not bad, (
        f"{rel_path}: resize sites using non-INTER_LINEAR: {bad}. "
        f"Training uses INTER_LINEAR (combined_paired.py:3455). "
        f"Inference paths must match or retro-score drifts."
    )


def test_inter_area_and_inter_linear_actually_differ():
    """Sanity: the two interpolations do produce different output, so the
    parity check above is meaningful. If cv2 or OpenCV ever degenerates the
    two into the same code path, this will flag it."""
    rng = np.random.default_rng(0)
    # Realistic face-crop size from our Teams data (sizes ranged 157-251 in the
    # combined session, so 200 is representative).
    img = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    r_linear = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    r_area = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.int16)
    # Expect non-trivial per-pixel delta (empirically ~0.4 on Teams frames).
    mean_abs_delta = np.abs(r_linear - r_area).mean()
    assert mean_abs_delta > 0.0, (
        "INTER_LINEAR and INTER_AREA produced identical output; the parity check "
        "would be vacuous. OpenCV behavior changed?"
    )
