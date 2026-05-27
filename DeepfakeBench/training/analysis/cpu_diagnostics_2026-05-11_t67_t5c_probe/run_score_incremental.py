"""Score ckpts AS THEY LAND on the cohorts. Appends to outputs/per_ckpt_*.csv.

Polls the _ckpts directories for completed .pth files. Once a ckpt is detected
(file exists, no .gstmp present for same prefix, size > 800MB),
loads it once, scores all cohorts, writes results. Then deletes from-disk to
free space if --delete-after-score.

Idempotent: skips ckpts already in per_ckpt_cohort_scores.csv.
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from run_score_probe import (
    ANCHOR_CKPTS, CANDIDATE_CKPTS, build_cohort_table,
    load_effort_model, score_frames, OUT
)

logger = logging.getLogger("score-incremental")


def detect_ready_ckpts(candidate_map: Dict[str, Path]) -> Dict[str, Path]:
    """Return ckpts whose .pth file exists AND no concurrent .gstmp."""
    ready = {}
    for label, path in candidate_map.items():
        if not path.exists():
            continue
        # Check for any .gstmp in same directory matching this filename
        sib_gstmp = path.with_suffix(path.suffix + "_.gstmp")
        if sib_gstmp.exists():
            continue
        size = path.stat().st_size
        if size < 800_000_000:
            continue
        ready[label] = path
    return ready


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll_interval_sec", type=int, default=20)
    ap.add_argument("--max_wait_sec", type=int, default=3600)
    ap.add_argument("--include_anchors", action="store_true", default=True)
    ap.add_argument("--delete_after_score", action="store_true", default=False)
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    cohort_df = build_cohort_table()
    logger.info("cohort: %d frames; counts: %s", len(cohort_df), cohort_df["cohort"].value_counts().to_dict())
    paths = [Path(p) for p in cohort_df["frame_path"].tolist()]

    SCORES_CSV = OUT / "per_ckpt_cohort_scores.csv"

    # Load existing if present (idempotent reruns)
    if SCORES_CSV.exists():
        scored = pd.read_csv(SCORES_CSV)
        logger.info("loaded existing scores (%d frames × %d cols)", len(scored), len(scored.columns))
    else:
        scored = cohort_df.copy()

    # Score anchors first (already local)
    if args.include_anchors:
        for label, path in ANCHOR_CKPTS.items():
            if label in scored.columns and scored[label].notna().any():
                logger.info("[%s] already scored, skip", label)
                continue
            if not path.exists():
                logger.warning("[%s] anchor missing", label)
                continue
            logger.info("[%s] scoring (anchor)", label)
            try:
                model = load_effort_model(path, device)
                scored[label] = score_frames(model, paths, device, batch_size=16)
                del model
                scored.to_csv(SCORES_CSV, index=False)
                logger.info("[%s] wrote scores", label)
            except Exception as exc:
                logger.error("[%s] failed: %s", label, exc)

    # Poll for ready candidates
    t0 = time.time()
    scored_set = {c for c in scored.columns if c in CANDIDATE_CKPTS and scored[c].notna().any()}
    expected_set = set(CANDIDATE_CKPTS.keys())
    while True:
        ready = detect_ready_ckpts(CANDIDATE_CKPTS)
        to_score = [(lbl, p) for lbl, p in ready.items() if lbl not in scored_set]
        for label, path in to_score:
            logger.info("[%s] ckpt ready, scoring", label)
            try:
                model = load_effort_model(path, device)
                scored[label] = score_frames(model, paths, device, batch_size=16)
                del model
                scored_set.add(label)
                scored.to_csv(SCORES_CSV, index=False)
                logger.info("[%s] wrote scores", label)
                if args.delete_after_score:
                    path.unlink(missing_ok=True)
                    logger.info("[%s] deleted ckpt to save disk", label)
            except Exception as exc:
                logger.error("[%s] failed: %s", label, exc)
                scored_set.add(label)  # don't retry
        if expected_set.issubset(scored_set):
            logger.info("all candidates scored; exiting")
            break
        if (time.time() - t0) > args.max_wait_sec:
            logger.warning("max_wait_sec exceeded (%ds); exiting with %d/%d scored",
                           args.max_wait_sec, len(scored_set), len(expected_set))
            break
        time.sleep(args.poll_interval_sec)

    # Compute summary stats now
    all_ckpts = list(ANCHOR_CKPTS.keys()) + list(CANDIDATE_CKPTS.keys())
    score_cols = [c for c in scored.columns if c in all_ckpts]
    stat_rows = []
    fpr_rows = []
    DEPLOYMENT_TAUS = [0.5, 0.7, 0.9, 0.92]
    for ckpt in score_cols:
        for cohort_name, sub in scored.groupby("cohort"):
            s = sub[ckpt].dropna().values.astype(float)
            if len(s) == 0:
                continue
            stat_rows.append({
                "ckpt": ckpt, "cohort": cohort_name, "n": len(s),
                "mean": float(s.mean()),
                "p25": float(np.percentile(s, 25)),
                "p50": float(np.percentile(s, 50)),
                "p75": float(np.percentile(s, 75)),
                "p90": float(np.percentile(s, 90)),
                "std": float(s.std()),
            })
            label_count = sub["label"].iloc[0]
            for tau in DEPLOYMENT_TAUS:
                fpr_rows.append({
                    "ckpt": ckpt, "cohort": cohort_name, "tau": tau,
                    "n": len(s), "label": int(label_count),
                    "frac_above_tau": float((s >= tau).sum() / len(s)),
                    "n_above_tau": int((s >= tau).sum()),
                })
    pd.DataFrame(stat_rows).to_csv(OUT / "per_ckpt_cohort_stats.csv", index=False)
    pd.DataFrame(fpr_rows).to_csv(OUT / "per_ckpt_deployment_fpr.csv", index=False)
    logger.info("wrote stats + FPR csvs")


if __name__ == "__main__":
    raise SystemExit(main())
