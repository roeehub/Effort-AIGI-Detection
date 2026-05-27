"""Quick eval of Slot 1 head-retrain step250 checkpoint.

Scores the head-retrain step250 ckpt (W&B run `fz84lq5k`) on three populations:
  1. may6 production-drift frames (92 local PNGs)
  2. teams_real_all_lockbox dor_shkedi cohort (1138 video clips, 1170 frames)
  3. teams_real_all_dev full slice (3253 video clips, 4564 frames)

Compares against Slot 1's as-is top_n_step2000 ckpt at the same τ. Mirrors
the may6 retest harness (analysis/r13_overnight_may6_retest_2026-05-13/
run_may6_retest.py) for LoRA wrapping order: build base model -> apply LoRA
-> load_state_dict (strict=False). LoRA cfg matches Slot 1 (rank=16, alpha=32,
target_layers=[10,11], target_modules=[attn.in_proj/out_proj, mlp.c_fc/c_proj]).

The head_only_retrain ckpt has the SAME architecture as Slot 1; only the
head.weight/head.bias rows are different (re-init + finetuned for 250 steps).
LoRA params are frozen during head retrain so they carry through unchanged.

Outputs:
  outputs/scores_step250_may6.csv
  outputs/scores_step250_lockbox_dor.csv     (per-frame on dor_shkedi 1170 frames)
  outputs/scores_step250_dev_real_all.csv    (per-frame on 4564 dev real frames)
  outputs/scores_step250_lockbox_all.csv     (per-frame on full 1418 lockbox real frames)
  outputs/summary_step250_vs_slot1.csv       (side-by-side comparison)

Usage:
    cd analysis/r13_overnight_head_retrain_step250_eval_2026-05-13
    python run_step250_eval.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
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
    GCSFrameDataset, FrameRecord,
    _download_checkpoint,
)
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("step250_eval")


# ---- Paths / config -----------------------------------------------------

OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = THIS_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

MANIFEST_PATH = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
MAY6_RAW_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"

STEP250_GS = "gs://training-job-outputs/best_checkpoints/fz84lq5k/top_n_effort_20260513_step250_auc0.9847_eer0.0437.pth"
STEP250_LOCAL = CACHE_DIR / "top_n_effort_20260513_step250_auc0.9847_eer0.0437.pth"

SLOT1_AS_IS_GS = "gs://training-job-outputs/best_checkpoints/gf6l06rf/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth"
SLOT1_AS_IS_LOCAL_ATLAS = REPO_ROOT / "analysis/r13_overnight_atlas_2026-05-13/_cache/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth"

# LoRA cfg (matches Slot 1 / head-retrain yaml)
LORA_CFG_R13 = {
    "rank": 16,
    "alpha": 32,
    "target_layers": [10, 11],
    "target_modules": ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"],
}

RESOLUTION = 224
BATCH_SIZE = 16
NUM_WORKERS = 12  # GCS-download bound; lots of workers parallelize fetch

# Device
try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    DEVICE = torch.device("cpu")


# ---- Local frame loader (matches may6 retest harness) -------------------

class LocalFrameDataset(Dataset):
    """Local-disk version — same preprocessing as GCSFrameDataset."""

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


# ---- Model loading (LoRA-aware, mirrors may6 retest) --------------------

def _load_model_with_lora(checkpoint_path: str, device: torch.device,
                          lora_cfg: dict = LORA_CFG_R13) -> torch.nn.Module:
    """Build base -> apply LoRA -> load state_dict (strict=False)."""
    import yaml

    with open(DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG, "r") as f:
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

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

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
        trainable, total = freeze_base_clip_encoder(visual)
        n_lora, _ = count_lora_parameters(visual)
        log.info(
            "[LoRA] applied: layers=%s rank=%d alpha=%g wrapped=%d "
            "lora_params=%d trainable=%d/%d",
            lora_cfg["target_layers"], lora_cfg["rank"], lora_cfg["alpha"],
            n_wrapped, n_lora, trainable, total,
        )
        model.to(device)

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
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


# ---- Manifest helpers ---------------------------------------------------

def load_manifest_videos(split: str, slice_name: str,
                         identity_key_filter: str | None = None) -> list[dict]:
    """Return manifest video records that match split + slice (+ optional identity_key)."""
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    videos = m["videos"]
    out = []
    for v in videos:
        if v["split"] != split:
            continue
        if slice_name not in v.get("slices", []):
            continue
        if identity_key_filter is not None:
            ik = v.get("identity_key", "")
            if not ik.startswith(identity_key_filter):
                continue
        out.append(v)
    return out


def videos_to_records(videos: list[dict], label: int) -> list[FrameRecord]:
    """Flatten manifest videos into FrameRecord per frame_path."""
    recs: list[FrameRecord] = []
    for v in videos:
        for fp in v["frame_paths"]:
            # parse gs://bucket/blob_path
            assert fp.startswith("gs://"), fp
            no_scheme = fp[5:]
            bucket, blob_path = no_scheme.split("/", 1)
            recs.append(FrameRecord(
                bucket=bucket,
                blob_path=blob_path,
                label=label,
                method=v.get("method", "unknown"),
                video_id=v["video_id"],
                frame_name=Path(blob_path).name,
                strategy="",
                extra={"identity_key": v.get("identity_key", "")},
            ))
    return recs


# ---- Scoring -------------------------------------------------------------

def score_records(model: torch.nn.Module, records: list[FrameRecord],
                  out_csv: Path, label_pop: str) -> Path:
    """Score a list of GCS records, write per-frame CSV.

    Skips work if out_csv already exists.
    """
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", out_csv.name, out_csv)
        return out_csv

    n = len(records)
    log.info("[%s] scoring %d frames ...", out_csv.name, n)

    dataset = GCSFrameDataset(records, resolution=RESOLUTION)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    probs = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    last_log = t0
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        now = time.time()
        # progress every ~30s
        if (now - last_log) >= 30 or (batch_idx + 1) % 25 == 0:
            done = (batch_idx + 1) * BATCH_SIZE
            elapsed = now - t0
            fps = min(done, n) / max(elapsed, 1e-3)
            log.info("[%s] batch %d ~%d/%d elapsed=%.1fs (%.1f fps)",
                     out_csv.name, batch_idx + 1, min(done, n), n, elapsed, fps)
            last_log = now

    log.info("[%s] inference done in %.1fs", out_csv.name, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "population", "video_id", "identity_key", "gcs_uri",
            "frame_name", "label", "prob_fake",
        ])
        for rec, prob in zip(records, probs):
            w.writerow([
                label_pop, rec.video_id, rec.extra.get("identity_key", ""),
                f"gs://{rec.bucket}/{rec.blob_path}", rec.frame_name,
                rec.label, f"{prob:.6f}",
            ])
    log.info("[%s] wrote -> %s", out_csv.name, out_csv)
    return out_csv


def score_may6_local(model: torch.nn.Module, out_csv: Path) -> Path:
    """Score 92 may6 frames (local PNGs)."""
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", out_csv.name, out_csv)
        return out_csv

    paths = sorted(MAY6_RAW_DIR.glob("*.png")) + sorted(MAY6_RAW_DIR.glob("*.jpg"))
    log.info("[%s] scoring %d may6 frames", out_csv.name, len(paths))

    dataset = LocalFrameDataset(paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    n = len(paths)
    probs = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
    log.info("[%s] inference done in %.1fs", out_csv.name, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for path, prob in zip(paths, probs):
            w.writerow(["may6_falseflag", str(path), path.name, f"{prob:.6f}"])
    log.info("[%s] wrote -> %s", out_csv.name, out_csv)
    return out_csv


# ---- Comparison summary -------------------------------------------------

def video_avg_from_frames(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-frame rows to per-video avg_prob.
    rows must have video_id, prob_fake, label, identity_key.
    """
    by_v: dict[str, list[dict]] = {}
    for r in rows:
        by_v.setdefault(r["video_id"], []).append(r)
    out = {}
    for vid, lst in by_v.items():
        ps = [float(r["prob_fake"]) for r in lst]
        out[vid] = {
            "video_id": vid,
            "identity_key": lst[0]["identity_key"],
            "label": int(lst[0]["label"]),
            "avg_video_prob": float(np.mean(ps)),
            "n_frames": len(ps),
        }
    return out


def fpr_at_tau(per_video: dict[str, dict], tau: float,
               identity_filter: str | None = None) -> tuple[int, int, float]:
    """Returns (n_fired, n, fpr)."""
    records = list(per_video.values())
    records = [r for r in records if r["label"] == 0]
    if identity_filter is not None:
        records = [r for r in records if r["identity_key"].startswith(identity_filter)]
    n = len(records)
    fired = sum(1 for r in records if r["avg_video_prob"] >= tau)
    fpr = fired / n if n > 0 else float("nan")
    return fired, n, fpr


def calibrate_tau_for_target_fpr(per_video_dev: dict[str, dict],
                                  target_fpr: float = 0.07) -> tuple[float, float]:
    """Find smallest τ such that dev real_fpr ≤ target_fpr."""
    reals = [r for r in per_video_dev.values() if r["label"] == 0]
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n = len(probs)
    if n == 0:
        return 0.5, float("nan")
    # number allowed to fire under target
    allowed = int(np.floor(target_fpr * n))
    # tau such that exactly `allowed` probs are >= tau (i.e., the (allowed+1)-th highest is below tau)
    if allowed >= n:
        return 0.0, 1.0
    if allowed == 0:
        # tau = just above the max prob
        return probs[0] + 1e-9, 0.0
    tau = probs[allowed]  # the allowed+1-th highest; >=tau is fired
    # but at tau exactly, >= would count the value itself; we want strict ≤target
    # Use smallest float just above probs[allowed] when needed.
    fired = sum(1 for p in probs if p >= tau)
    fpr = fired / n
    if fpr > target_fpr:
        # bump τ to just above probs[allowed-1]
        tau = probs[allowed - 1] + 1e-9
        fired = sum(1 for p in probs if p >= tau)
        fpr = fired / n
    return tau, fpr


def load_per_frame_csv(p: Path) -> list[dict]:
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main() -> int:
    log.info("=" * 70)
    log.info("Slot 1 head-retrain step250 eval")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 70)

    # ---- 1. Load step250 model ------------------------------------------
    if not STEP250_LOCAL.exists():
        log.info("downloading step250 ckpt ...")
        # use the model_arena helper for parity
        local = _download_checkpoint(STEP250_GS)
        # move into our cache
        import shutil
        shutil.copy(local, STEP250_LOCAL)
    log.info("step250 ckpt: %s", STEP250_LOCAL)

    log.info("loading step250 model with LoRA wrap (rank=16, alpha=32, layers=[10,11])...")
    t0 = time.time()
    model = _load_model_with_lora(str(STEP250_LOCAL), DEVICE, LORA_CFG_R13)
    log.info("step250 model loaded in %.1fs", time.time() - t0)

    # ---- 2. Score may6 ---------------------------------------------------
    may6_csv = score_may6_local(model, OUT_DIR / "scores_step250_may6.csv")

    # ---- 3. Build records for lockbox + dev real -----------------------
    log.info("loading manifest ...")
    lockbox_videos = load_manifest_videos("lockbox", "teams_real_all")
    log.info("lockbox teams_real_all: %d videos", len(lockbox_videos))
    lockbox_recs = videos_to_records(lockbox_videos, label=0)
    log.info("lockbox teams_real_all: %d frames", len(lockbox_recs))

    dev_videos = load_manifest_videos("dev", "teams_real_all")
    log.info("dev teams_real_all: %d videos", len(dev_videos))
    dev_recs = videos_to_records(dev_videos, label=0)
    log.info("dev teams_real_all: %d frames", len(dev_recs))

    # ---- 4. Score lockbox real ------------------------------------------
    lockbox_csv = score_records(
        model, lockbox_recs,
        OUT_DIR / "scores_step250_lockbox_all.csv",
        label_pop="teams_real_all_lockbox",
    )

    # ---- 5. Score dev real ---------------------------------------------
    dev_csv = score_records(
        model, dev_recs,
        OUT_DIR / "scores_step250_dev_real_all.csv",
        label_pop="teams_real_all_dev",
    )

    # Release model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 6. Build summary vs as-is Slot 1 -----------------------------
    log.info("building comparison summary ...")

    may6_rows = load_per_frame_csv(may6_csv)
    lockbox_rows = load_per_frame_csv(lockbox_csv)
    dev_rows = load_per_frame_csv(dev_csv)

    # step250 per-video aggregates
    lockbox_pv_step250 = video_avg_from_frames(lockbox_rows)
    dev_pv_step250 = video_avg_from_frames(dev_rows)

    # As-is Slot 1 cached video reports
    as_is_lockbox_csv = REPO_ROOT / "analysis/r13_overnight_partial_scorecard_2026-05-13/_reports_cache/teams_real_all_lockbox_slot1_lora_p8a_top_n_step2000_videos_report.csv"
    as_is_dev_csv = REPO_ROOT / "analysis/r13_overnight_partial_scorecard_2026-05-13/_reports_cache/teams_real_all_dev_slot1_lora_p8a_top_n_step2000_videos_report.csv"

    def load_video_report(path: Path) -> dict[str, dict]:
        out = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                vid = r["video_id"]
                out[vid] = {
                    "video_id": vid,
                    "label": int(r["label"]),
                    "avg_video_prob": float(r["avg_video_prob"]),
                    # video_id like "dor_shkedi__seq1001__real" - parse identity
                    "identity_key": vid.split("__seg")[0].split("__seq")[0],
                }
        return out

    as_is_lockbox = load_video_report(as_is_lockbox_csv)
    as_is_dev = load_video_report(as_is_dev_csv)

    # The as-is identity_key parsing won't match exactly for dor_shkedi (no __seg in video_id);
    # check manifest mapping
    # The manifest has 'identity_key': 'dor_shkedi' for those video_ids
    # so we accept video_id startswith 'dor_shkedi' as the dor identity.

    # ---- τ=0.5 numbers ----------
    rows_out = []

    for label, pv in [("step250 (head-retrain)", lockbox_pv_step250),
                       ("slot1_step2000 (as-is)", as_is_lockbox)]:
        # all lockbox real
        fired, n, fpr = fpr_at_tau(pv, 0.5)
        rows_out.append({
            "ckpt": label, "tau": 0.5, "suite": "teams_real_all_lockbox", "cohort": "all",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })
        # dor_shkedi only
        fired, n, fpr = fpr_at_tau(pv, 0.5, "dor_shkedi")
        rows_out.append({
            "ckpt": label, "tau": 0.5, "suite": "teams_real_all_lockbox", "cohort": "dor_shkedi",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })

    for label, pv in [("step250 (head-retrain)", dev_pv_step250),
                       ("slot1_step2000 (as-is)", as_is_dev)]:
        fired, n, fpr = fpr_at_tau(pv, 0.5)
        rows_out.append({
            "ckpt": label, "tau": 0.5, "suite": "teams_real_all_dev", "cohort": "all",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })

    # ---- calibrated τ numbers ----
    # step250: calibrate τ on its own dev to FPR target 0.07
    tau_step250, dev_fpr_step250 = calibrate_tau_for_target_fpr(dev_pv_step250, 0.07)
    log.info("step250 calibrated τ=%.6f (dev fpr=%.4f)", tau_step250, dev_fpr_step250)

    tau_as_is, dev_fpr_as_is = calibrate_tau_for_target_fpr(as_is_dev, 0.07)
    log.info("as-is slot1 calibrated τ=%.6f (dev fpr=%.4f)", tau_as_is, dev_fpr_as_is)

    for label, pv_lock, pv_dev, tau in [
        ("step250 (head-retrain)", lockbox_pv_step250, dev_pv_step250, tau_step250),
        ("slot1_step2000 (as-is)", as_is_lockbox, as_is_dev, tau_as_is),
    ]:
        fired, n, fpr = fpr_at_tau(pv_lock, tau)
        rows_out.append({
            "ckpt": label, "tau": f"{tau:.4f}", "suite": "teams_real_all_lockbox", "cohort": "all (calibrated τ)",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })
        fired, n, fpr = fpr_at_tau(pv_lock, tau, "dor_shkedi")
        rows_out.append({
            "ckpt": label, "tau": f"{tau:.4f}", "suite": "teams_real_all_lockbox", "cohort": "dor_shkedi (calibrated τ)",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })
        fired, n, fpr = fpr_at_tau(pv_dev, tau)
        rows_out.append({
            "ckpt": label, "tau": f"{tau:.4f}", "suite": "teams_real_all_dev", "cohort": "all (calibrated τ)",
            "n_fired": fired, "n": n, "fpr": f"{100*fpr:.2f}%",
        })

    # ---- May6 stats ---------------------
    may6_probs = np.array([float(r["prob_fake"]) for r in may6_rows])
    n_fired = int((may6_probs > 0.5).sum())
    p50 = float(np.median(may6_probs))
    p90 = float(np.quantile(may6_probs, 0.9))
    p99 = float(np.quantile(may6_probs, 0.99))
    pmax = float(may6_probs.max())

    # P8A baseline from precedent harness
    prior_p8a_csv = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_P8A.csv"
    p8a_probs = None
    if prior_p8a_csv.exists():
        import pandas as pd
        df = pd.read_csv(prior_p8a_csv)
        df = df[df["population"] == "may6_falseflag"]
        p8a_probs = df["prob_fake"].to_numpy()

    # As-is Slot 1 may6 — already in this dir's sister job
    slot1_may6_csv = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/outputs/scores_SLOT1_LORA_P8A_STEP2000.csv"
    slot1_may6_probs = None
    if slot1_may6_csv.exists():
        import pandas as pd
        df = pd.read_csv(slot1_may6_csv)
        df = df[df["population"] == "may6_falseflag"]
        slot1_may6_probs = df["prob_fake"].to_numpy()

    spearman_vs_p8a = None
    if p8a_probs is not None:
        from scipy.stats import spearmanr
        spearman_vs_p8a = spearmanr(p8a_probs, may6_probs).statistic

    spearman_vs_slot1 = None
    if slot1_may6_probs is not None:
        from scipy.stats import spearmanr
        spearman_vs_slot1 = spearmanr(slot1_may6_probs, may6_probs).statistic

    may6_summary = {
        "step250_n_fired@0.5": n_fired,
        "step250_p50": p50,
        "step250_p90": p90,
        "step250_p99": p99,
        "step250_max": pmax,
        "spearman_vs_P8A": spearman_vs_p8a,
        "spearman_vs_slot1_step2000": spearman_vs_slot1,
        "slot1_step2000_n_fired@0.5": int((slot1_may6_probs > 0.5).sum()) if slot1_may6_probs is not None else None,
        "P8A_n_fired@0.5": int((p8a_probs > 0.5).sum()) if p8a_probs is not None else None,
    }

    # write CSV
    summary_csv = OUT_DIR / "summary_step250_vs_slot1.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "tau", "suite", "cohort", "n_fired", "n", "fpr"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    may6_csv_summary = OUT_DIR / "summary_step250_may6.csv"
    with open(may6_csv_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(may6_summary.keys()))
        w.writeheader()
        w.writerow(may6_summary)

    # Console output
    log.info("=" * 80)
    log.info("STEP250 vs SLOT1 SUMMARY")
    log.info("=" * 80)
    for r in rows_out:
        log.info("  %-40s τ=%-8s %-25s %-25s %s",
                 r["ckpt"], r["tau"], r["suite"], r["cohort"],
                 f"{r['n_fired']}/{r['n']} ({r['fpr']})")

    log.info("=" * 80)
    log.info("STEP250 MAY6 SUMMARY")
    log.info("=" * 80)
    for k, v in may6_summary.items():
        log.info("  %-35s %s", k, v)

    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
