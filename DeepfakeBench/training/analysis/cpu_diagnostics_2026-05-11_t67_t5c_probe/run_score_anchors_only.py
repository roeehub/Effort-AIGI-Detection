"""Test run — score JUST anchors on the cohorts, as a calibration smoke test.

Once it works for anchors we know the pipeline works; then add the ckpts.
"""
from __future__ import annotations
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from run_score_probe import (
    ANCHOR_CKPTS, build_cohort_table, load_effort_model, score_frames, OUT
)
import logging, numpy as np, pandas as pd, torch

logger = logging.getLogger("anchors-smoke")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    cohort_df = build_cohort_table()
    logger.info("cohort: %d frames; counts: %s", len(cohort_df), cohort_df["cohort"].value_counts().to_dict())
    paths = [Path(p) for p in cohort_df["frame_path"].tolist()]

    scored = cohort_df.copy()
    for label, path in ANCHOR_CKPTS.items():
        if not path.exists():
            logger.warning("anchor %s missing at %s", label, path)
            continue
        logger.info("[%s] loading", label)
        model = load_effort_model(path, device)
        logger.info("[%s] scoring %d frames", label, len(paths))
        scored[label] = score_frames(model, paths, device, batch_size=16)
        del model

    scored.to_csv(OUT / "anchor_scores_smoke.csv", index=False)
    logger.info("wrote anchor_scores_smoke.csv")


if __name__ == "__main__":
    raise SystemExit(main())
