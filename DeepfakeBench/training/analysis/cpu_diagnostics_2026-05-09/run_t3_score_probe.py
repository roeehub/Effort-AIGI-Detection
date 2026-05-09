"""T3 (Stage 3) score probe — same pattern as
analysis/stage2_cpu_2026-05-09/run_score_probe.py but for the three T3 slots.

Designed to run incrementally: as each Vertex job's ckpts land in GCS, this
script downloads them, scores the 388-frame Dor cohort + 130-frame Roy_D set,
and appends results to the same outputs directory.

Usage (after at least one slot has saved ckpts):
    python3 analysis/cpu_diagnostics_2026-05-09/run_t3_score_probe.py \\
        --slots 1,2,3 --steps 500,1500,2500,4500

If a ckpt is not yet uploaded the script skips it cleanly. Re-running picks
up newly-arrived ckpts.

Outputs (under analysis/cpu_diagnostics_2026-05-09/outputs/):
    t3_dor_cohort_scores.csv          — per-frame, all ckpts (replaces on each run)
    t3_per_cohort_stats.csv
    t3_roy_d_per_frame.csv
    t3_roy_d_ckpt_stats.csv
    t3_correlation_vs_p8a.csv
    T3_SCORE_PROBE_FACTS_2026-05-09.md  — auto-generated FACTS-only summary
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
CKPT_DIR = THIS_DIR / "_ckpts_t3"
OUTPUTS = THIS_DIR / "outputs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

DOR_CACHE_DIR = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache"
DOR_MANIFEST = DOR_CACHE_DIR / "cohort_manifest.csv"
FRAMES_DIR = DOR_CACHE_DIR / "frames"

ROY_D_DIR = REPO_ROOT / "analysis" / "stage2_cpu_2026-05-09" / "_roy_d_frames"
ROY_D_REFERENCE_CSV = (
    REPO_ROOT / "analysis" / "p1_pe_eval_2026-05-07" / "roy_d_regression"
    / "roy_d_per_frame_scores.csv"
)

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Vertex jobs from the 14:48 CEST 2026-05-09 launches.
SLOT_JOBS = {
    1: {
        "wandb_project": "enhanced-aug-test",
        "vertex_job_name": "exp-R13_T3_SLOT1_DROP_HIGH_IQ_TEAMS_REALS_2026-05-09-20260509-144802",
        "vertex_job_id": "8339710278371377152",
        "region": "us-east1",
        "wandb_run_id": "bxnuz22g",
    },
    2: {
        "wandb_project": "enhanced-aug-test",
        "vertex_job_name": "exp-R13_T3_SLOT2_IQ_MATCHED_REALS_2026-05-09-20260509-144820",
        "vertex_job_id": "7520601259871567872",
        "region": "us-west4",
        "wandb_run_id": "jevz8h45",
    },
    3: {
        "wandb_project": "enhanced-aug-test",
        "vertex_job_name": "exp-R13_T3_SLOT3_PER_METHOD_IQ_MATCH_2026-05-09-20260509-144837",
        "vertex_job_id": "6693065086040276992",
        "region": "us-central1",
        "wandb_run_id": "v1u6mcwe",
    },
}

REFERENCE_SCORE_COLS = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "P2D": "p2_d_fourier_periodic_step3000",
}

logger = logging.getLogger("t3-probe")


def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    import yaml
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def score_frames(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    out_probs = np.full(len(paths), np.nan, dtype=np.float64)
    pending: List[Tuple[int, torch.Tensor]] = []

    def flush(batch):
        if not batch:
            return
        idx = [b[0] for b in batch]
        x = torch.stack([b[1] for b in batch]).to(device)
        with torch.no_grad():
            data = {"image": x}
            try:
                pred = model(data, inference=True)
            except TypeError:
                pred = model(data)
            if isinstance(pred, dict):
                logits = pred.get("cls", pred.get("logits", pred.get("score")))
            else:
                logits = pred
            if logits is None:
                raise RuntimeError("model returned no logits")
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for ii, pp in zip(idx, probs):
            out_probs[ii] = float(pp)

    for i, p in enumerate(paths):
        t = load_and_preprocess(Path(p))
        if t is not None:
            pending.append((i, t))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
    flush(pending)
    return out_probs


def find_wandb_run_id(slot_info: Dict) -> str | None:
    """Return the wandb run id baked into SLOT_JOBS (or None if absent)."""
    return slot_info.get("wandb_run_id")


def download_ckpts_for_slot(slot: int, wandb_run_id: str, steps: List[int]) -> Dict[int, Path]:
    """Download the requested step ckpts from
    gs://training-job-outputs/best_checkpoints/<wandb_run_id>/.
    Returns dict step → local Path. Skips missing ckpts cleanly.
    """
    paths: Dict[int, Path] = {}
    base = f"gs://training-job-outputs/best_checkpoints/{wandb_run_id}/"
    for step in steps:
        # Convention for periodic ckpts (matches Stage 2's _ckpts/ filenames):
        # periodic_step{N}_*.pth
        local_path = CKPT_DIR / f"slot{slot}_step{step}.pth"
        if local_path.exists():
            paths[step] = local_path
            continue
        # List bucket for matching files.
        cmd = ["gsutil", "ls", f"{base}*step{step:04d}*.pth"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            # Try non-zero-padded form too.
            cmd2 = ["gsutil", "ls", f"{base}*step{step}_*.pth"]
            try:
                result = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            except Exception:
                continue
            if result.returncode != 0 or not result.stdout.strip():
                logger.info("[slot %d step %d] not yet uploaded; skipping", slot, step)
                continue
        first_uri = result.stdout.strip().splitlines()[0]
        cp_cmd = ["gsutil", "cp", first_uri, str(local_path)]
        cp_result = subprocess.run(cp_cmd, capture_output=True, text=True, timeout=600)
        if cp_result.returncode == 0 and local_path.exists():
            paths[step] = local_path
            logger.info("[slot %d step %d] downloaded %s", slot, step, first_uri)
        else:
            logger.warning("[slot %d step %d] cp failed: %s", slot, step,
                           cp_result.stderr[:200] if cp_result.stderr else "?")
    return paths


def run_dor_probe(ckpt_label_to_path: Dict[str, Path]) -> pd.DataFrame:
    """Score all ckpts on the 388-frame Dor cohort. Reuses Stage 2 cache."""
    if not DOR_MANIFEST.exists():
        logger.error("missing %s", DOR_MANIFEST)
        return pd.DataFrame()
    cohort = pd.read_csv(DOR_MANIFEST)
    cohort["local_path"] = cohort["frame_path"].apply(
        lambda u: FRAMES_DIR / u.split("/")[-1]
    )
    cohort["frame_present"] = cohort["local_path"].apply(
        lambda p: p.exists() and p.stat().st_size > 0
    )
    cohort = cohort[cohort["frame_present"]].reset_index(drop=True)
    logger.info("dor cohort: %d frames", len(cohort))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    paths = [Path(p) for p in cohort["local_path"].tolist()]

    score_cols: Dict[str, np.ndarray] = {}
    for label, ckpt_path in ckpt_label_to_path.items():
        logger.info("[%s] loading", label)
        model = load_effort_model(ckpt_path, device)
        logger.info("[%s] scoring %d frames", label, len(paths))
        probs = score_frames(model, paths, device, batch_size=32)
        score_cols[label] = probs
        del model

    per_frame = cohort[["frame_path", "label", "cohort"]].copy()
    for label, col in REFERENCE_SCORE_COLS.items():
        if col in cohort.columns:
            per_frame[label] = cohort[col].values
    for label, probs in score_cols.items():
        per_frame[label] = probs
    return per_frame


def run_roy_d_probe(ckpt_label_to_path: Dict[str, Path]) -> pd.DataFrame:
    """Score 130-frame Roy_D set. Falls back gracefully if frames missing."""
    if not ROY_D_DIR.exists():
        logger.warning("Roy_D frames dir missing: %s", ROY_D_DIR)
        return pd.DataFrame()
    roy_paths = sorted(ROY_D_DIR.glob("*.jpg")) + sorted(ROY_D_DIR.glob("*.jpeg")) + sorted(ROY_D_DIR.glob("*.png"))
    if not roy_paths:
        logger.warning("no Roy_D frames found in %s", ROY_D_DIR)
        return pd.DataFrame()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = pd.DataFrame({"frame_path": [str(p) for p in roy_paths]})
    for label, ckpt_path in ckpt_label_to_path.items():
        model = load_effort_model(ckpt_path, device)
        probs = score_frames(model, [Path(p) for p in roy_paths], device, batch_size=32)
        rows[label] = probs
        del model

    # Merge reference scores from existing CSV if available.
    if ROY_D_REFERENCE_CSV.exists():
        ref = pd.read_csv(ROY_D_REFERENCE_CSV)
        # Reference table has frame_path or filename as the join key.
        if "frame_path" in ref.columns:
            for col in REFERENCE_SCORE_COLS.values():
                if col in ref.columns:
                    rows = rows.merge(ref[["frame_path", col]], on="frame_path", how="left")
    return rows


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="1,2,3", help="comma-sep slot indices")
    ap.add_argument("--steps", default="500,1000,1500,2500,3500,4500",
                    help="comma-sep step values to download")
    ap.add_argument("--wandb_run_ids", default="",
                    help="optional comma-sep slotN=runid pairs (e.g. 1=abc12345)")
    args = ap.parse_args()

    slots = [int(s) for s in args.slots.split(",") if s.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]

    # Parse user-provided run IDs.
    run_id_overrides: Dict[int, str] = {}
    if args.wandb_run_ids.strip():
        for pair in args.wandb_run_ids.split(","):
            slot_str, _, run_id = pair.partition("=")
            if slot_str.strip().lstrip("0").isdigit() and run_id.strip():
                run_id_overrides[int(slot_str.strip())] = run_id.strip()

    ckpt_label_to_path: Dict[str, Path] = {}
    missing_run_ids = []
    for slot in slots:
        run_id = run_id_overrides.get(slot)
        if run_id is None:
            run_id = find_wandb_run_id(SLOT_JOBS[slot])
        if run_id is None:
            missing_run_ids.append(slot)
            continue
        slot_paths = download_ckpts_for_slot(slot, run_id, steps)
        for step, path in slot_paths.items():
            ckpt_label_to_path[f"T3_SLOT{slot}_step{step}"] = path

    if missing_run_ids:
        logger.warning(
            "missing wandb_run_id for slots %s — re-run with --wandb_run_ids 1=<id>,2=<id>,3=<id>",
            missing_run_ids,
        )

    if not ckpt_label_to_path:
        logger.warning("no ckpts to score; exiting cleanly")
        return 0

    logger.info("=== running Dor cohort probe (%d ckpts) ===", len(ckpt_label_to_path))
    dor_df = run_dor_probe(ckpt_label_to_path)
    if len(dor_df) > 0:
        dor_df.to_csv(OUTPUTS / "t3_dor_cohort_scores.csv", index=False)
        logger.info("wrote t3_dor_cohort_scores.csv (%d rows)", len(dor_df))

    logger.info("=== running Roy_D probe ===")
    roy_df = run_roy_d_probe(ckpt_label_to_path)
    if len(roy_df) > 0:
        roy_df.to_csv(OUTPUTS / "t3_roy_d_per_frame.csv", index=False)
        logger.info("wrote t3_roy_d_per_frame.csv (%d rows)", len(roy_df))

    # Summary stats per ckpt × cohort.
    if len(dor_df) > 0:
        rows = []
        for ckpt_label in dor_df.columns:
            if ckpt_label in ("frame_path", "label", "cohort"):
                continue
            for cohort_name in dor_df["cohort"].unique():
                sub = dor_df[dor_df["cohort"] == cohort_name]
                scores = sub[ckpt_label].dropna().values.astype(float)
                if len(scores) == 0:
                    continue
                rows.append({
                    "ckpt": ckpt_label, "cohort": cohort_name,
                    "n": len(scores),
                    "mean": float(scores.mean()),
                    "p10": float(np.percentile(scores, 10)),
                    "p50": float(np.percentile(scores, 50)),
                    "p90": float(np.percentile(scores, 90)),
                    "std": float(scores.std()),
                })
        pd.DataFrame(rows).to_csv(OUTPUTS / "t3_per_cohort_stats.csv", index=False)
        logger.info("wrote t3_per_cohort_stats.csv")

    print("\n[done] Re-run with --slots / --steps / --wandb_run_ids to extend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
