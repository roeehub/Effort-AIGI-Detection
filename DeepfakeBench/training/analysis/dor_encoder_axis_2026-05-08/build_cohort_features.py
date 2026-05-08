"""Build Dor encoder-axis cohort + extract per-frame CLS features.

Goal: characterize WHY P2-D-step3000 lost P8A's signature dor invariance
(J4: real FPR 8% → 46%, fake recall 86% → 62%).

Cohorts (frame_path GCS URIs from the P2 scoreboard,
`analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/scoreboard.parquet`):
  - DOR_REAL_DEV: 50 frames in `teams_real_dor_dev` (label=0)
  - DOR_FAKE_DEV: 78 frames in `teams_capture_dor_shkedi_dev` (label=1)
  - DOR_REAL_LOCKBOX: 100 random frames from the dor_shkedi subset of
    `teams_real_all_lockbox` (label=0)
  - NON_DOR_REAL_DEV: 80 frames sampled from `teams_real_all_dev` excluding
    any frame_path containing "dor" (label=0; control)
  - NON_DOR_FAKE_DEV: 80 frames sampled from `teams_fake_all_dev` excluding
    any frame_path containing "dor" (label=1; control)

Per-frame extraction:
  - layer 11 CLS (from OpenCLIP transformer.resblocks[11], pre-projection 768-d)
  - final CLS (post backbone.proj 512-d, what the head sees)
  - per-frame frame_prob from the scoreboard (already-computed)

Forwards model on each unique GCS URI for each of {P8A, E2B, P2_D_step3000}.

CPU/MPS only. Idempotent: per-cohort × per-ckpt cache files at
`_cache/cohort_features__{ckpt_label}.npz`.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = THIS_DIR / "_cache"
FRAMES_CACHE_DIR = LOCAL_CACHE_DIR / "frames"
PRIOR_CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
PERLAYER_CACHE_DIR = (
    REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

SCOREBOARD_PARQUET = (
    REPO_ROOT
    / "analysis"
    / "p2_eval_2026-05-08"
    / "p2_deeper_analysis"
    / "outputs"
    / "scoreboard.parquet"
)

# ckpt label → (ckpt_path, scoreboard_col)
CKPTS_INFO = {
    "P8A": {
        "path": PRIOR_CACHE_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        "score_col": "p8a_reference_step5000",
    },
    "E2B": {
        "path": REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache" / "e2b_top_n_step3200.pth",
        "score_col": "e2b_top_n_step3200",
    },
    "P2D": {
        "path": REPO_ROOT / "analysis" / "p2_eval_2026-05-08" / "d1_d4_cpu" / "ckpts" / "slotD_periodic_step3000.pth",
        "score_col": "p2_d_fourier_periodic_step3000",
    },
}

logger = logging.getLogger("dor-cohort")


# -----------------------------------------------------------------------------
# Cohort selection.
# -----------------------------------------------------------------------------
def build_cohort(seed: int = 0) -> pd.DataFrame:
    sb = pd.read_parquet(SCOREBOARD_PARQUET)
    rows = []

    # DOR_REAL_DEV
    sub = sb[sb["suite"] == "teams_real_dor_dev"][
        ["frame_path", "label", "p8a_reference_step5000", "e2b_top_n_step3200",
         "p2_d_fourier_periodic_step3000"]
    ].copy()
    sub["cohort"] = "DOR_REAL_DEV"
    rows.append(sub)

    # DOR_FAKE_DEV
    sub = sb[sb["suite"] == "teams_capture_dor_shkedi_dev"][
        ["frame_path", "label", "p8a_reference_step5000", "e2b_top_n_step3200",
         "p2_d_fourier_periodic_step3000"]
    ].copy()
    sub["cohort"] = "DOR_FAKE_DEV"
    rows.append(sub)

    # DOR_REAL_LOCKBOX (subset of teams_real_all_lockbox where path contains dor_shkedi)
    lb_all = sb[sb["suite"] == "teams_real_all_lockbox"]
    lb_dor = lb_all[lb_all["frame_path"].str.contains("dor_shkedi", case=False, na=False)]
    sub = lb_dor.sample(n=min(100, len(lb_dor)), random_state=seed)[
        ["frame_path", "label", "p8a_reference_step5000", "e2b_top_n_step3200",
         "p2_d_fourier_periodic_step3000"]
    ].copy()
    sub["cohort"] = "DOR_REAL_LOCKBOX"
    rows.append(sub)

    # NON_DOR_REAL_DEV
    real_dev = sb[sb["suite"] == "teams_real_all_dev"]
    non_dor_real = real_dev[~real_dev["frame_path"].str.contains("dor", case=False, na=False)].sample(
        n=80, random_state=seed
    )[["frame_path", "label", "p8a_reference_step5000", "e2b_top_n_step3200",
       "p2_d_fourier_periodic_step3000"]].copy()
    non_dor_real["cohort"] = "NON_DOR_REAL_DEV"
    rows.append(non_dor_real)

    # NON_DOR_FAKE_DEV
    fake_dev = sb[sb["suite"] == "teams_fake_all_dev"]
    non_dor_fake = fake_dev[~fake_dev["frame_path"].str.contains("dor", case=False, na=False)].sample(
        n=80, random_state=seed
    )[["frame_path", "label", "p8a_reference_step5000", "e2b_top_n_step3200",
       "p2_d_fourier_periodic_step3000"]].copy()
    non_dor_fake["cohort"] = "NON_DOR_FAKE_DEV"
    rows.append(non_dor_fake)

    cohort = pd.concat(rows, ignore_index=True)
    cohort = cohort.drop_duplicates(subset=["frame_path", "cohort"]).reset_index(drop=True)
    return cohort


# -----------------------------------------------------------------------------
# GCS frame download.
# -----------------------------------------------------------------------------
def gcs_frame_localpath(uri: str) -> Path:
    basename = uri.split("/")[-1]
    return FRAMES_CACHE_DIR / basename


def download_frames_parallel(uris: List[str], workers: int = 12) -> Dict[str, Path]:
    """Download URIs to FRAMES_CACHE_DIR. Returns mapping uri → Path (only successful)."""
    FRAMES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_download = []
    out: Dict[str, Path] = {}
    for u in uris:
        p = gcs_frame_localpath(u)
        if p.exists() and p.stat().st_size > 0:
            out[u] = p
        else:
            to_download.append(u)
    if not to_download:
        logger.info("all %d frames already cached", len(uris))
        return out
    logger.info("downloading %d/%d frames…", len(to_download), len(uris))

    def fetch(u):
        p = gcs_frame_localpath(u)
        try:
            subprocess.run(
                ["gsutil", "-q", "cp", u, str(p)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return u, p
        except Exception as e:
            return u, None

    succ = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (u, p) in enumerate(ex.map(fetch, to_download)):
            if p is not None and p.exists():
                out[u] = p
                succ += 1
            if (k + 1) % 50 == 0:
                logger.info("  downloaded %d/%d", k + 1, len(to_download))
    logger.info("download complete: %d/%d successful", succ, len(to_download))
    return out


# -----------------------------------------------------------------------------
# Model load + feature extraction (same shape as iq_perlayer extractor).
# -----------------------------------------------------------------------------
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


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks under model.backbone(.visual)")


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_features(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (final_cls (B,512), layer11_cls (B,768), valid_mask (B,))."""
    resblocks = get_resblocks(model)
    layer11_capture: List[np.ndarray] = []

    def hook11(_module, _input, output):
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]
            else:
                cls = output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        layer11_capture.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[11].register_forward_hook(hook11)
    try:
        valid: List[bool] = []
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is None:
                valid.append(False)
                pending.append((i, None))
                continue
            valid.append(True)
            pending.append((i, t))

        finals = []
        for j in range(0, len(pending), batch_size):
            chunk = [c for c in pending[j : j + batch_size] if c[1] is not None]
            if not chunk:
                # all invalid in this chunk; nothing to do
                continue
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.inference_mode():
                bb_out = model.backbone(batch)  # dict with 'pooler_output' key (B, 512)
                if isinstance(bb_out, dict):
                    final = bb_out["pooler_output"]
                else:
                    final = bb_out
            finals.append(final.detach().cpu().to(torch.float32).numpy())
            if (j // batch_size) % 4 == 0:
                logger.info("    batch %d/%d (final shape %s)", j // batch_size + 1, (len(pending) + batch_size - 1) // batch_size, final.shape)
        finals_arr = np.concatenate(finals, axis=0) if finals else np.zeros((0, 512), dtype=np.float32)
        layer11_arr = np.concatenate(layer11_capture, axis=0) if layer11_capture else np.zeros((0, 768), dtype=np.float32)
    finally:
        handle.remove()

    valid_mask = np.array(valid, dtype=bool)
    n_valid = int(valid_mask.sum())
    if finals_arr.shape[0] != n_valid:
        logger.warning("finals %d != n_valid %d (some invalid frames in last batch)",
                       finals_arr.shape[0], n_valid)
    return finals_arr, layer11_arr, valid_mask


def cohort_cache_path(label: str) -> Path:
    return LOCAL_CACHE_DIR / f"cohort_features__{label}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description="Dor cohort feature extraction (check c)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpts", default=None,
                    help="comma-separated subset of {P8A, E2B, P2D}; default = all 3")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    # Build cohort manifest.
    cohort = build_cohort(seed=args.seed)
    logger.info("cohort: %d unique frames across cohorts:\n%s",
                len(cohort), cohort.groupby("cohort").size().to_string())
    cohort_csv = LOCAL_CACHE_DIR / "cohort_manifest.csv"
    cohort.to_csv(cohort_csv, index=False)

    # Download frames.
    uris = cohort["frame_path"].drop_duplicates().tolist()
    uri_to_path = download_frames_parallel(uris)
    cohort["local_path"] = cohort["frame_path"].map(lambda u: str(uri_to_path.get(u, "")))
    valid_cohort = cohort[cohort["local_path"] != ""].reset_index(drop=True)
    logger.info("cohort frames with successful download: %d/%d", len(valid_cohort), len(cohort))

    # Determine which ckpts to extract.
    if args.ckpts:
        wanted = [c.strip() for c in args.ckpts.split(",") if c.strip()]
        ckpts = {k: v for k, v in CKPTS_INFO.items() if k in wanted}
    else:
        ckpts = CKPTS_INFO

    # Per ckpt feature extraction.
    for label, info in ckpts.items():
        cache_p = cohort_cache_path(label)
        if cache_p.exists() and not args.no_cache:
            blob = np.load(cache_p)
            if blob["final_cls"].shape[0] == len(valid_cohort):
                logger.info("[%s] cohort features cached (%d frames)", label, blob["final_cls"].shape[0])
                continue
            else:
                logger.info("[%s] cached cohort size %d != current %d; re-extracting",
                            label, blob["final_cls"].shape[0], len(valid_cohort))

        ckpt_path = info["path"]
        if not ckpt_path.exists():
            logger.error("ckpt missing: %s", ckpt_path)
            return 2
        logger.info("[%s] loading model from %s", label, ckpt_path)
        model = load_effort_model(ckpt_path, device)

        local_paths = [Path(p) for p in valid_cohort["local_path"].tolist()]
        finals, layer11, valid_mask = extract_features(
            model, local_paths, device, batch_size=args.batch_size
        )
        # Reduce dataframe to only valid rows
        valid_cohort_arr = valid_cohort[valid_mask].reset_index(drop=True)
        if finals.shape[0] != len(valid_cohort_arr):
            logger.warning("finals=%d, valid_cohort_arr=%d", finals.shape[0], len(valid_cohort_arr))

        np.savez_compressed(
            cache_p,
            final_cls=finals.astype(np.float32),
            layer11_cls=layer11.astype(np.float32),
            valid_mask=valid_mask,
        )
        logger.info("[%s] saved %d frames features", label, finals.shape[0])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save the cohort manifest with valid_mask aligned to first ckpt's extraction.
    # (Note: valid_mask should be identical across ckpts since image loading
    # depends only on local_path validity.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
