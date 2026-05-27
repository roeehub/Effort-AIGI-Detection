"""Stage 2 score-distribution probe (CPU/MPS) — step500 dose-response check.

Goal: for each Stage 2 step500 ckpt, score the 388-frame Dor cohort and check
whether any slot reproduces P2-D's identity-cluster collapse pattern OR
BUNDLE_step500's 800× score-variance compression on Roy_D-style identities.

Cohort: re-uses analysis/dor_encoder_axis_2026-05-08/_cache (388 frames,
5 cohorts: DOR_REAL_DEV, DOR_FAKE_DEV, DOR_REAL_LOCKBOX, NON_DOR_REAL_DEV,
NON_DOR_FAKE_DEV).

Per-ckpt computation:
  - Score each frame (final softmax fake-prob).
  - Per-cohort score stats: p10/p50/p90/std/mean.
  - Pearson r vs P8A baseline scores (per-cohort and overall).
  - Per-cohort score range (max - min).

Outputs:
  outputs/stage2_dor_cohort_scores.csv  — per-frame scores all ckpts
  outputs/stage2_per_cohort_stats.csv   — per-cohort × per-ckpt stats
  outputs/stage2_correlation_vs_p8a.csv — per-cohort Pearson r
  STAGE2_SCORE_PROBE_FACTS_2026-05-09.md — FACTS doc

CKPT registry (extend as more ckpts complete):
  Stage 2 step500: S1, S2, S3 (FT-from-P8A; this script)
  Reference baselines: P8A, E2B, P2D (already cached in dor_encoder_axis output)
"""
from __future__ import annotations

import json
import logging
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
CKPT_DIR = THIS_DIR / "_ckpts"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DOR_CACHE_DIR = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache"
DOR_MANIFEST = DOR_CACHE_DIR / "cohort_manifest.csv"
FRAMES_DIR = DOR_CACHE_DIR / "frames"

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# (label, ckpt_local_path)
CKPTS_TO_SCORE: Dict[str, Path] = {
    "S1_step500": CKPT_DIR / "S1_step500.pth",
    "S1_step2500": CKPT_DIR / "S1_step2500.pth",
    "S1_step4500": CKPT_DIR / "S1_step4500.pth",
    "S2_step500": CKPT_DIR / "S2_step500.pth",
    "S2_step2500": CKPT_DIR / "S2_step2500.pth",
    "S2_step4500": CKPT_DIR / "S2_step4500.pth",
    "S3_step500": CKPT_DIR / "S3_step500.pth",
    "S3_step2500": CKPT_DIR / "S3_step2500.pth",
    "S3_step4500": CKPT_DIR / "S3_step4500.pth",
}

# Reference scores ALREADY in the cached cohort manifest (CSV columns).
REFERENCE_COLS = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "P2D": "p2_d_fourier_periodic_step3000",
}

logger = logging.getLogger("stage2-probe")


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
    """Returns per-frame fake-class softmax probability."""
    out_probs = np.full(len(paths), np.nan, dtype=np.float64)
    pending: List[Tuple[int, torch.Tensor]] = []

    def flush(batch: List[Tuple[int, torch.Tensor]]):
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
                raise RuntimeError(f"model returned no logits: keys={list(pred.keys()) if isinstance(pred, dict) else None}")
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    # 1. Load cached cohort manifest. Use only frames that were actually cached locally.
    if not DOR_MANIFEST.exists():
        logger.error("missing %s — run dor_encoder_axis build_cohort_features first", DOR_MANIFEST)
        return 2
    cohort = pd.read_csv(DOR_MANIFEST)

    # The local frame paths follow the GCS basename convention.
    cohort["local_path"] = cohort["frame_path"].apply(
        lambda u: FRAMES_DIR / u.split("/")[-1]
    )
    cohort["frame_present"] = cohort["local_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    n_total = len(cohort)
    cohort = cohort[cohort["frame_present"]].reset_index(drop=True)
    logger.info("cohort: %d frames present locally (of %d in manifest)", len(cohort), n_total)
    logger.info("by cohort:\n%s", cohort["cohort"].value_counts().to_string())

    # 2. Pick device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device: %s", device)

    # 3. Score with each Stage 2 ckpt.
    score_cols: Dict[str, np.ndarray] = {}
    paths = [Path(p) for p in cohort["local_path"].tolist()]

    for label, ckpt_path in CKPTS_TO_SCORE.items():
        if not ckpt_path.exists():
            logger.warning("ckpt missing: %s — skipping %s", ckpt_path, label)
            continue
        logger.info("[%s] loading %s", label, ckpt_path)
        model = load_effort_model(ckpt_path, device)
        logger.info("[%s] scoring %d frames…", label, len(paths))
        probs = score_frames(model, paths, device, batch_size=32)
        score_cols[label] = probs
        n_valid = int(np.isfinite(probs).sum())
        logger.info("[%s] mean=%.3f median=%.3f std=%.3f valid=%d/%d",
                    label, np.nanmean(probs), np.nanmedian(probs), np.nanstd(probs), n_valid, len(probs))
        del model

    # 4. Build the full per-frame table (Stage 2 + reference cols).
    per_frame = cohort[["frame_path", "label", "cohort"]].copy()
    for label, col in REFERENCE_COLS.items():
        per_frame[label] = cohort[col].values
    for label, probs in score_cols.items():
        per_frame[label] = probs
    per_frame.to_csv(OUTPUTS / "stage2_dor_cohort_scores.csv", index=False)

    # 5. Per-cohort score statistics.
    rows = []
    for ckpt_label in list(REFERENCE_COLS.keys()) + list(score_cols.keys()):
        if ckpt_label not in per_frame.columns:
            continue
        for cohort_name in per_frame["cohort"].unique():
            sub = per_frame[per_frame["cohort"] == cohort_name]
            scores = sub[ckpt_label].dropna().values.astype(float)
            if len(scores) == 0:
                continue
            rows.append({
                "ckpt": ckpt_label,
                "cohort": cohort_name,
                "n": len(scores),
                "mean": float(scores.mean()),
                "p10": float(np.percentile(scores, 10)),
                "p50": float(np.percentile(scores, 50)),
                "p90": float(np.percentile(scores, 90)),
                "std": float(scores.std()),
                "range": float(scores.max() - scores.min()),
                "min": float(scores.min()),
                "max": float(scores.max()),
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUTPUTS / "stage2_per_cohort_stats.csv", index=False)

    # 6. Pearson r vs P8A per-cohort (and overall).
    corr_rows = []
    p8a_col = "P8A"
    for ckpt_label in list(REFERENCE_COLS.keys()) + list(score_cols.keys()):
        if ckpt_label == p8a_col or ckpt_label not in per_frame.columns:
            continue
        # Overall.
        sub = per_frame.dropna(subset=[p8a_col, ckpt_label])
        if len(sub) > 1:
            r_overall = float(np.corrcoef(sub[p8a_col], sub[ckpt_label])[0, 1])
        else:
            r_overall = float("nan")
        corr_rows.append({"ckpt": ckpt_label, "cohort": "ALL", "n": len(sub), "pearson_r_vs_P8A": r_overall})
        for cohort_name in per_frame["cohort"].unique():
            sub = per_frame[per_frame["cohort"] == cohort_name].dropna(subset=[p8a_col, ckpt_label])
            if len(sub) > 1 and sub[ckpt_label].std() > 1e-9 and sub[p8a_col].std() > 1e-9:
                r = float(np.corrcoef(sub[p8a_col], sub[ckpt_label])[0, 1])
            else:
                r = float("nan")
            corr_rows.append({"ckpt": ckpt_label, "cohort": cohort_name, "n": len(sub), "pearson_r_vs_P8A": r})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUTPUTS / "stage2_correlation_vs_p8a.csv", index=False)

    # 7. Console headlines.
    print("\n" + "=" * 72)
    print("Per-cohort score p50 (median fake-prob)")
    print("=" * 72)
    pivot = stats_df.pivot_table(index="cohort", columns="ckpt", values="p50")
    pd.options.display.float_format = "{:.4f}".format
    print(pivot.to_string())

    print("\n" + "=" * 72)
    print("Per-cohort score std")
    print("=" * 72)
    pivot = stats_df.pivot_table(index="cohort", columns="ckpt", values="std")
    print(pivot.to_string())

    print("\n" + "=" * 72)
    print("Pearson r vs P8A (DOR_REAL_LOCKBOX is the load-bearing identity-collapse signal)")
    print("=" * 72)
    pivot = corr_df.pivot_table(index="cohort", columns="ckpt", values="pearson_r_vs_P8A")
    print(pivot.to_string())

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
