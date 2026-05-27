"""may6 production-drift retest for the 4 r13-overnight candidate ckpts.

Scores the 92 fresh real Xinhe-may6 frames per ckpt at tau=0.5. Reuses the
precedent harness arena/model_arena.load_model for preprocessing parity with
training (INTER_LINEAR resize, BGR2RGB, CLIP normalization). For the two LoRA
ckpts (Slot 1, Slot 2), we apply the LoRA wrapper AFTER building the base
model and BEFORE loading the state_dict, mirroring train_sweep.py's order
post-checkpoint-load. The LoRA block is not stored inside `model_config` in
the saved ckpts so we inject it explicitly from the experiment yaml
specification (rank=16, alpha=32, target_layers=[10,11],
target_modules=["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]).

Output:
  outputs/scores_<ckpt_alias>.csv  — same schema as
    analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_<ckpt>.csv
  outputs/may6_retest_table.csv     — summary n_fired @ tau=0.5 + percentiles

The 4 baseline ckpts (P8A, E2B, T3_S1_step1500, T3_S1_step2500) already have
score CSVs at analysis/xinhe_cross_camera_audit_2026-05-06/outputs/, computed
with the same harness on the same 92 may6 frames; we read them rather than
re-scoring.

Usage:
    cd analysis/r13_overnight_may6_retest_2026-05-13
    python run_may6_retest.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import (  # noqa: E402
    CLIP_MEAN, CLIP_STD,
    _download_checkpoint,
)
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("may6_retest")

RAW_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"
PRIOR_OUT_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

# Device: prefer MPS on Mac; fall back to CPU if user task surfaces contention.
# 92 frames * 4 ckpts is feasible on CPU in <30 min per task spec.
try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    DEVICE = torch.device("cpu")

RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 2  # memory:feedback_sklearn_njobs guides against -1 on this Mac

# (ckpt_alias, gcs_uri, lora_cfg or None)
LORA_CFG_R13 = {
    "rank": 16,
    "alpha": 32,
    "target_layers": [10, 11],
    "target_modules": ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"],
}

# Adjacent atlas job (r13_overnight_atlas_2026-05-13) pulls the same 4 ckpts.
# If they are already on disk there, reuse — avoids a ~4 GB redundant pull.
LOCAL_CACHE_DIR = THIS_DIR / "_ckpt_cache"
ATLAS_CACHE_DIR = REPO_ROOT / "analysis/r13_overnight_atlas_2026-05-13/_cache"

CANDIDATES: list[tuple[str, str, dict | None]] = [
    ("SLOT1_LORA_P8A_STEP2000",
     "gs://training-job-outputs/best_checkpoints/gf6l06rf/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth",
     LORA_CFG_R13),
    ("SLOT2_LORA_T5C_STEP1500",
     "gs://training-job-outputs/best_checkpoints/912kd88q/periodic_effort_20260513_step1500_auc0.9941_eer0.0198.pth",
     LORA_CFG_R13),
    ("SLOT3_T5C_JITTER030_STEP4500",
     "gs://training-job-outputs/best_checkpoints/502dcznh/top_n_effort_20260513_step4500_auc0.9923_eer0.0304.pth",
     None),
    ("SLOT4_B16_SCRATCH_FOURIER_STEP10000",
     "gs://training-job-outputs/best_checkpoints/qrpf5dtr/top_n_effort_20260513_step10000_auc0.9902_eer0.0259.pth",
     None),
]


def _resolve_ckpt(ckpt_uri: str) -> str:
    """Look in the local + adjacent atlas job's cache first; fall back to the
    GCS pull used by arena._download_checkpoint."""
    basename = ckpt_uri.split("/")[-1]
    local = LOCAL_CACHE_DIR / basename
    if local.exists():
        log.info("  using local-cache ckpt -> %s", local)
        return str(local)
    atlas = ATLAS_CACHE_DIR / basename
    if atlas.exists():
        log.info("  using atlas-cache ckpt -> %s", atlas)
        return str(atlas)
    return _download_checkpoint(ckpt_uri)

# Baseline aliases — already-scored CSVs live in PRIOR_OUT_DIR
BASELINES = ["P8A", "E2B", "T3_S1_STEP1500", "T3_S1_STEP2500"]


class LocalFrameDataset(Dataset):
    """Local-disk version of GCSFrameDataset — same preprocessing as
    arena/inference.LocalFrameDataset in the precedent harness."""

    def __init__(self, frame_paths: list[Path], resolution: int = RESOLUTION):
        self.paths = list(frame_paths)
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning("decode-fail: %s", path)
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def collect_may6_frames() -> list[Path]:
    """Return the 92 may6 frames in sorted order — identical ordering to the
    precedent harness (analysis/xinhe_cross_camera_audit_2026-05-06/
    run_local_inference.py::collect_frames)."""
    paths = sorted(RAW_DIR.glob("*.png")) + sorted(RAW_DIR.glob("*.jpg"))
    log.info("collected: may6=%d frames from %s", len(paths), RAW_DIR)
    return paths


def _load_model_with_optional_lora(
    checkpoint_path: str,
    detector_config_path: Path,
    train_config_path: Path,
    device: torch.device,
    lora_cfg: dict | None,
) -> torch.nn.Module:
    """Build the EffortDetector base, apply LoRA wrapping if requested, THEN
    load the state_dict. This sequence mirrors what train_sweep.py does
    post-checkpoint-load: LoRA wrapping happens BEFORE the trained LoRA
    weights are read back into the model.

    NOTE: when lora_cfg is non-None, the saved state_dict contains both
    base-encoder weights + LoRA A/B weights. Loading with strict=False
    after wrapping accepts both populations; missing or unexpected keys
    are surfaced in the log lines below for verification.
    """
    import yaml

    with open(detector_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # ArcFace scale restore (preserved from arena.load_model semantics)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    # ===== Apply LoRA BEFORE load_state_dict for LoRA-trained ckpts =====
    if lora_cfg is not None:
        from detectors.lora_adapter import (
            apply_lora_to_openclip_visual,
            count_lora_parameters,
            freeze_base_clip_encoder,
        )

        visual = model.backbone.visual
        n_wrapped = apply_lora_to_openclip_visual(
            visual,
            target_layers=list(lora_cfg["target_layers"]),
            rank=int(lora_cfg["rank"]),
            alpha=float(lora_cfg["alpha"]),
            target_modules=tuple(lora_cfg["target_modules"]),
        )
        # Freeze base (matches train_sweep.py freeze_base=True default for R13_LORA);
        # for inference this is cosmetic but mirrors training semantics so
        # named_parameters etc. match.
        trainable, total = freeze_base_clip_encoder(visual)
        n_lora, _ = count_lora_parameters(visual)
        log.info(
            "[LoRA] applied: layers=%s rank=%d alpha=%g wrapped=%d "
            "lora_params=%d trainable=%d/%d",
            lora_cfg["target_layers"], lora_cfg["rank"], lora_cfg["alpha"],
            n_wrapped, n_lora, trainable, total,
        )
        # After wrapping, the LoRA params live on CPU by default; align to device.
        model.to(device)

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    # Audit: for LoRA-trained ckpts, we expect zero unexpected lora_A / lora_B.
    lora_missing = [k for k in missing if "lora_" in k.lower()]
    lora_unexpected = [k for k in unexpected if "lora_" in k.lower()]
    log.info(
        "  state_dict load: %d missing, %d unexpected (lora_missing=%d, lora_unexpected=%d)",
        len(missing), len(unexpected), len(lora_missing), len(lora_unexpected),
    )
    if lora_missing:
        log.warning("  lora_missing examples: %s", lora_missing[:3])
    if lora_unexpected:
        log.warning("  lora_unexpected examples: %s", lora_unexpected[:3])

    model.eval()
    return model


def score_one_ckpt(
    ckpt_alias: str,
    ckpt_uri: str,
    lora_cfg: dict | None,
    frames: list[Path],
) -> Path:
    """Download + load + score one ckpt on the 92 may6 frames. Returns CSV path."""
    out_csv = OUT_DIR / f"scores_{ckpt_alias}.csv"
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", ckpt_alias, out_csv)
        return out_csv

    log.info("[%s] resolving checkpoint ...", ckpt_alias)
    t0 = time.time()
    local_ckpt = _resolve_ckpt(ckpt_uri)
    log.info("[%s] resolved in %.1fs -> %s", ckpt_alias, time.time() - t0, local_ckpt)

    log.info("[%s] loading model on %s (lora_cfg=%s) ...",
             ckpt_alias, DEVICE, "ON" if lora_cfg else "OFF")
    t0 = time.time()
    model = _load_model_with_optional_lora(
        str(local_ckpt), DETECTOR_CONFIG, TRAIN_CONFIG, DEVICE, lora_cfg,
    )
    log.info("[%s] model loaded in %.1fs", ckpt_alias, time.time() - t0)

    paths = list(frames)
    dataset = LocalFrameDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    n = len(paths)
    probs = np.zeros(n, dtype=np.float32)
    log.info("[%s] running inference: %d frames, batch=%d, num_workers=%d, device=%s",
             ckpt_alias, n, BATCH_SIZE, NUM_WORKERS, DEVICE)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        elapsed = time.time() - t0
        done = (batch_idx + 1) * BATCH_SIZE
        log.info("[%s] batch %d (~%d/%d) elapsed=%.1fs (%.1f fps)",
                 ckpt_alias, batch_idx + 1, min(done, n), n,
                 elapsed, min(done, n) / max(elapsed, 1e-3))

    log.info("[%s] inference done in %.1fs", ckpt_alias, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for path, prob in zip(paths, probs):
            w.writerow(["may6_falseflag", str(path), path.name, f"{prob:.6f}"])
    log.info("[%s] wrote -> %s", ckpt_alias, out_csv)

    del model
    return out_csv


def _load_score_csv(p: Path) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(p)
    df = df[df["population"] == "may6_falseflag"]
    return df["prob_fake"].to_numpy()


def make_summary_table() -> None:
    """Aggregate all 4 baselines + 4 candidates into outputs/may6_retest_table.csv."""
    rows = []

    # baselines from precedent harness
    for alias in BASELINES:
        csv_path = PRIOR_OUT_DIR / f"scores_{alias}.csv"
        if not csv_path.exists():
            log.warning("missing baseline CSV: %s", csv_path)
            continue
        probs = _load_score_csv(csv_path)
        rows.append({
            "ckpt": alias,
            "source": "baseline (precedent harness)",
            "n": int(len(probs)),
            "n_fired@0.5": int((probs > 0.5).sum()),
            "p50": float(np.median(probs)),
            "p90": float(np.quantile(probs, 0.9)),
            "p99": float(np.quantile(probs, 0.99)),
            "max": float(probs.max()),
        })

    # candidates
    for alias, _uri, _lc in CANDIDATES:
        csv_path = OUT_DIR / f"scores_{alias}.csv"
        if not csv_path.exists():
            log.warning("missing candidate CSV: %s", csv_path)
            continue
        probs = _load_score_csv(csv_path)
        rows.append({
            "ckpt": alias,
            "source": "this retest",
            "n": int(len(probs)),
            "n_fired@0.5": int((probs > 0.5).sum()),
            "p50": float(np.median(probs)),
            "p90": float(np.quantile(probs, 0.9)),
            "p99": float(np.quantile(probs, 0.99)),
            "max": float(probs.max()),
        })

    out_csv = OUT_DIR / "may6_retest_table.csv"
    fieldnames = ["ckpt", "source", "n", "n_fired@0.5", "p50", "p90", "p99", "max"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote summary -> %s", out_csv)

    log.info("=" * 80)
    log.info("may6 retest summary")
    log.info("=" * 80)
    hdr = "  {:<40s} {:>8s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "ckpt", "n", "n@0.5", "p50", "p90", "p99", "max"
    )
    log.info(hdr)
    for r in rows:
        log.info(
            "  %-40s %8d %10d %8.4f %8.4f %8.4f %8.4f",
            r["ckpt"], r["n"], r["n_fired@0.5"],
            r["p50"], r["p90"], r["p99"], r["max"],
        )


def main() -> int:
    log.info("=" * 70)
    log.info("may6 production-drift retest — 4 r13-overnight candidate ckpts")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 70)
    if not DETECTOR_CONFIG.exists():
        log.error("missing detector config: %s", DETECTOR_CONFIG)
        return 2
    if not TRAIN_CONFIG.exists():
        log.error("missing train config: %s", TRAIN_CONFIG)
        return 2

    frames = collect_may6_frames()
    if not frames:
        log.error("no may6 frames found in %s", RAW_DIR)
        return 2
    if len(frames) != 92:
        log.warning("expected 92 may6 frames, found %d", len(frames))

    for alias, uri, lora_cfg in CANDIDATES:
        try:
            score_one_ckpt(alias, uri, lora_cfg, frames)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED: %s", alias, exc, exc_info=True)

    make_summary_table()
    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
