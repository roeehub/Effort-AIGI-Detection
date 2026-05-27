"""Quick deployment check of Slot 1 head-retrain step1250 ckpt.

PURPOSE
-------
Step250 (same W&B run `fz84lq5k`) closed Roy_D / dor_shkedi catastrophe but
its score distribution is bimodally compressed (max prob_fake ≈ 0.79), so
no single τ satisfies both contract gates (dev_fake_recall ≥ 0.30 AND
dev_real_fpr ≤ 0.07) simultaneously.

This run checks whether step1250 (5x further training of the same head) has
SHARPENED the distribution toward P8A-style separation (max → 0.99+) or
remains compressed.

SUITES SCORED (5 most-relevant; skip viso/deeplive/dor for speed)
  1. teams_real_all_dev          (3253 videos)  — for dev-cal τ
  2. teams_real_lighting_extreme_dev (1401 videos) — robustness real
  3. teams_real_poor_quality_dev     (923 videos)  — robustness real
  4. teams_fake_all_dev              (2409 videos) — dev_fake_recall
  5. teams_real_all_lockbox          (1361 videos) — lb_real_fpr
  6. teams_fake_all_lockbox          (253 videos)  — lb_fake_recall

Re-uses _load_model_with_lora + scoring infra from the step250 sibling at
`../r13_overnight_head_retrain_step250_eval_2026-05-13/run_step250_eval.py`.

Outputs:
  outputs/scores_step1250_<suite>.csv  (per-frame, video_id+identity_key)
  outputs/score_distribution_step1250_2026-05-13.csv  (min/p50/p90/p99/max per suite)
  outputs/tau_sweep_step1250_2026-05-13.csv           (10 τ × 6 suites)
  outputs/dev_cal_readout_step1250_2026-05-13.csv     (dev-cal τ + headline numbers)

Usage:
  cd analysis/r13_overnight_head_retrain_step1250_eval_2026-05-13
  python run_step1250.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Re-use step250 sibling code for parity (LoRA wrap order + score_records)
STEP250_DIR = REPO_ROOT / "analysis/r13_overnight_head_retrain_step250_eval_2026-05-13"
sys.path.insert(0, str(STEP250_DIR))

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
log = logging.getLogger("step1250_eval")


# ---- Paths / config -----------------------------------------------------

OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = THIS_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

MANIFEST_PATH = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"

STEP1250_GS = "gs://training-job-outputs/best_checkpoints/fz84lq5k/periodic_effort_20260513_step1250_auc0.9843_eer0.0437.pth"
STEP1250_LOCAL = CACHE_DIR / "periodic_effort_20260513_step1250_auc0.9843_eer0.0437.pth"

# LoRA cfg (matches Slot 1 / head-retrain yaml)
LORA_CFG_R13 = {
    "rank": 16,
    "alpha": 32,
    "target_layers": [10, 11],
    "target_modules": ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"],
}

RESOLUTION = 224
BATCH_SIZE = 16
NUM_WORKERS = 12

# Device
try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    DEVICE = torch.device("cpu")


# 6 suites, in priority order (all manifest-based)
SUITES = [
    ("teams_real_all_dev",              "dev",     "teams_real_all",            0),
    ("teams_real_lighting_extreme_dev", "dev",     "teams_real_lighting_extreme", 0),
    ("teams_real_poor_quality_dev",     "dev",     "teams_real_poor_quality",   0),
    ("teams_fake_all_dev",              "dev",     "teams_fake_all",            1),
    ("teams_real_all_lockbox",          "lockbox", "teams_real_all",            0),
    ("teams_fake_all_lockbox",          "lockbox", "teams_fake_all",            1),
]


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

def load_manifest_videos(split: str, slice_name: str) -> list[dict]:
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    out = []
    for v in m["videos"]:
        if v["split"] != split:
            continue
        if slice_name not in v.get("slices", []):
            continue
        out.append(v)
    return out


def videos_to_records(videos: list[dict], label: int) -> list[FrameRecord]:
    recs: list[FrameRecord] = []
    for v in videos:
        for fp in v["frame_paths"]:
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


# ---- Aggregation -------------------------------------------------------

def aggregate_video(per_frame_csv: Path) -> dict[str, dict]:
    """Per-frame -> per-video avg_prob. Returns dict[video_id] -> info."""
    by_v: dict[str, list[dict]] = {}
    with open(per_frame_csv) as f:
        for r in csv.DictReader(f):
            vid = r["video_id"]
            by_v.setdefault(vid, []).append(r)
    out = {}
    for vid, rows in by_v.items():
        ps = [float(r["prob_fake"]) for r in rows]
        out[vid] = {
            "video_id": vid,
            "identity_key": rows[0]["identity_key"],
            "label": int(rows[0]["label"]),
            "avg_video_prob": float(np.mean(ps)),
            "n_frames": len(ps),
        }
    return out


def fpr_at_tau(per_video: dict[str, dict], tau: float,
               label_filter: int = 0,
               identity_filter: str | None = None) -> tuple[int, int, float]:
    records = list(per_video.values())
    records = [r for r in records if r["label"] == label_filter]
    if identity_filter is not None:
        records = [r for r in records
                   if r["identity_key"].startswith(identity_filter)
                   or r["video_id"].startswith(identity_filter)]
    n = len(records)
    fired = sum(1 for r in records if r["avg_video_prob"] >= tau)
    rate = fired / n if n > 0 else float("nan")
    return fired, n, rate


def calibrate_tau_for_target_fpr(per_video_dev: dict[str, dict],
                                  target_fpr: float = 0.07) -> tuple[float, float]:
    """Smallest τ such that real_fpr ≤ target_fpr (videos labeled 0)."""
    reals = [r for r in per_video_dev.values() if r["label"] == 0]
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n = len(probs)
    if n == 0:
        return 0.5, float("nan")
    allowed = int(np.floor(target_fpr * n))
    if allowed >= n:
        return 0.0, 1.0
    if allowed == 0:
        return probs[0] + 1e-9, 0.0
    tau = probs[allowed]
    fired = sum(1 for p in probs if p >= tau)
    fpr = fired / n
    if fpr > target_fpr:
        tau = probs[allowed - 1] + 1e-9
        fired = sum(1 for p in probs if p >= tau)
        fpr = fired / n
    return tau, fpr


def score_distribution_stats(per_video: dict[str, dict],
                              label_filter: int | None = None) -> dict:
    records = list(per_video.values())
    if label_filter is not None:
        records = [r for r in records if r["label"] == label_filter]
    probs = np.array([r["avg_video_prob"] for r in records])
    if len(probs) == 0:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "n": int(len(probs)),
        "min": float(probs.min()),
        "p50": float(np.median(probs)),
        "p90": float(np.quantile(probs, 0.9)),
        "p99": float(np.quantile(probs, 0.99)),
        "max": float(probs.max()),
    }


# ---- Main ---------------------------------------------------------------

TAU_GRID = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ROY_D_PREFIX = "Roy_D"
DOR_PREFIX = "dor_shkedi"


def main() -> int:
    log.info("=" * 70)
    log.info("Slot 1 head-retrain step1250 quick deployment eval")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 70)

    # ---- Download step1250 (if not cached) -----------------------------
    if not STEP1250_LOCAL.exists():
        log.info("downloading step1250 ckpt to local cache ...")
        local = _download_checkpoint(STEP1250_GS)
        import shutil
        shutil.copy(local, STEP1250_LOCAL)
    log.info("step1250 ckpt: %s (%.1f MB)", STEP1250_LOCAL,
             STEP1250_LOCAL.stat().st_size / 1024 / 1024)

    # ---- Load model -----------------------------------------------------
    log.info("loading step1250 model with LoRA wrap (rank=16, alpha=32, layers=[10,11])...")
    t0 = time.time()
    model = _load_model_with_lora(str(STEP1250_LOCAL), DEVICE, LORA_CFG_R13)
    log.info("step1250 model loaded in %.1fs", time.time() - t0)

    # ---- Score each suite ----------------------------------------------
    pv_by_suite: dict[str, dict[str, dict]] = {}
    t_global = time.time()
    for suite_name, split, slice_name, label in SUITES:
        videos = load_manifest_videos(split, slice_name)
        recs = videos_to_records(videos, label=label)
        log.info("[%s] %d videos -> %d frames", suite_name, len(videos), len(recs))
        out_csv = OUT_DIR / f"scores_step1250_{suite_name}.csv"
        score_records(model, recs, out_csv, label_pop=suite_name)
        pv_by_suite[suite_name] = aggregate_video(out_csv)
    log.info("All suites scored in %.1fs", time.time() - t_global)

    # Release model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- (3) Score distribution per suite ------------------------------
    dist_rows = []
    for suite_name in [s[0] for s in SUITES]:
        pv = pv_by_suite[suite_name]
        d = score_distribution_stats(pv, label_filter=None)
        d["suite"] = suite_name
        dist_rows.append(d)

    dist_csv = OUT_DIR / "score_distribution_step1250_2026-05-13.csv"
    with open(dist_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "n", "min", "p50", "p90", "p99", "max"])
        w.writeheader()
        for r in dist_rows:
            w.writerow(r)
    log.info("wrote score distribution -> %s", dist_csv)

    # ---- (4) τ-sweep ----------------------------------------------------
    sweep_rows = []
    for suite_name, _, _, label in SUITES:
        pv = pv_by_suite[suite_name]
        for tau in TAU_GRID:
            fired, n, rate = fpr_at_tau(pv, tau, label_filter=label)
            sweep_rows.append({
                "suite": suite_name,
                "label": label,                          # 0 -> rate is FPR; 1 -> recall
                "rate_kind": "fpr" if label == 0 else "recall",
                "tau": f"{tau:.4f}",
                "n_fired": fired,
                "n": n,
                "rate": f"{rate:.6f}",
            })
    sweep_csv = OUT_DIR / "tau_sweep_step1250_2026-05-13.csv"
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "label", "rate_kind", "tau", "n_fired", "n", "rate"])
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)
    log.info("wrote τ-sweep -> %s", sweep_csv)

    # ---- (5) dev-cal τ --------------------------------------------------
    dev_pv = pv_by_suite["teams_real_all_dev"]
    tau_dev_cal, dev_real_fpr = calibrate_tau_for_target_fpr(dev_pv, target_fpr=0.07)
    log.info("dev-cal τ=%.6f (dev_real_fpr=%.4f)", tau_dev_cal, dev_real_fpr)

    # ---- (6) Headline numbers at dev-cal τ -----------------------------
    # dev_fake_macro_recall: avg of fake-recall rates across teams_fake_all_dev
    # (single fake suite scored here; macro becomes single-suite recall.)
    fired, n, dev_fake_recall = fpr_at_tau(pv_by_suite["teams_fake_all_dev"], tau_dev_cal, label_filter=1)
    log.info("dev_fake_recall (teams_fake_all_dev) @ τ=%.4f: %d/%d (%.4f)",
             tau_dev_cal, fired, n, dev_fake_recall)

    lb_fired, lb_n, lb_real_fpr = fpr_at_tau(pv_by_suite["teams_real_all_lockbox"], tau_dev_cal, label_filter=0)
    log.info("lb_real_fpr (teams_real_all_lockbox) @ τ=%.4f: %d/%d (%.4f)",
             tau_dev_cal, lb_fired, lb_n, lb_real_fpr)

    lbf_fired, lbf_n, lb_fake_recall = fpr_at_tau(pv_by_suite["teams_fake_all_lockbox"], tau_dev_cal, label_filter=1)
    log.info("lb_fake_recall (teams_fake_all_lockbox) @ τ=%.4f: %d/%d (%.4f)",
             tau_dev_cal, lbf_fired, lbf_n, lb_fake_recall)

    # Roy_D pooled FPR (treat both dev + lockbox real Roy_D videos as a pooled cohort)
    roy_records = []
    for pv in [pv_by_suite["teams_real_all_lockbox"], pv_by_suite["teams_real_all_dev"]]:
        for r in pv.values():
            if r["label"] != 0:
                continue
            if r["identity_key"].startswith(ROY_D_PREFIX) or r["video_id"].startswith(ROY_D_PREFIX):
                roy_records.append(r)
    roy_n = len(roy_records)
    roy_fired = sum(1 for r in roy_records if r["avg_video_prob"] >= tau_dev_cal)
    roy_pooled_fpr = roy_fired / roy_n if roy_n > 0 else float("nan")
    log.info("Roy_D pooled FPR @ τ=%.4f: %d/%d (%.4f)", tau_dev_cal, roy_fired, roy_n, roy_pooled_fpr)

    # dor_shkedi lockbox FPR
    dor_records = [r for r in pv_by_suite["teams_real_all_lockbox"].values()
                   if r["label"] == 0 and (r["identity_key"].startswith(DOR_PREFIX)
                                            or r["video_id"].startswith(DOR_PREFIX))]
    dor_n = len(dor_records)
    dor_fired = sum(1 for r in dor_records if r["avg_video_prob"] >= tau_dev_cal)
    dor_lockbox_fpr = dor_fired / dor_n if dor_n > 0 else float("nan")
    log.info("dor_shkedi lockbox FPR @ τ=%.4f: %d/%d (%.4f)",
             tau_dev_cal, dor_fired, dor_n, dor_lockbox_fpr)

    # Lighting / poor quality robustness real-fpr
    lit_fired, lit_n, lit_fpr = fpr_at_tau(pv_by_suite["teams_real_lighting_extreme_dev"], tau_dev_cal, label_filter=0)
    pq_fired, pq_n, pq_fpr = fpr_at_tau(pv_by_suite["teams_real_poor_quality_dev"], tau_dev_cal, label_filter=0)

    # ---- Write dev-cal readout CSV -------------------------------------
    readout_rows = [
        {"metric": "tau_dev_cal",          "value": f"{tau_dev_cal:.6f}", "n_fired": "", "n": ""},
        {"metric": "dev_real_fpr",         "value": f"{dev_real_fpr:.6f}", "n_fired": "", "n": len(dev_pv)},
        {"metric": "dev_fake_recall",      "value": f"{dev_fake_recall:.6f}", "n_fired": fired, "n": n},
        {"metric": "lb_real_fpr",          "value": f"{lb_real_fpr:.6f}", "n_fired": lb_fired, "n": lb_n},
        {"metric": "lb_fake_recall",       "value": f"{lb_fake_recall:.6f}", "n_fired": lbf_fired, "n": lbf_n},
        {"metric": "roy_d_pooled_fpr",     "value": f"{roy_pooled_fpr:.6f}", "n_fired": roy_fired, "n": roy_n},
        {"metric": "dor_shkedi_lockbox_fpr","value": f"{dor_lockbox_fpr:.6f}", "n_fired": dor_fired, "n": dor_n},
        {"metric": "lighting_extreme_fpr", "value": f"{lit_fpr:.6f}", "n_fired": lit_fired, "n": lit_n},
        {"metric": "poor_quality_fpr",     "value": f"{pq_fpr:.6f}", "n_fired": pq_fired, "n": pq_n},
    ]
    readout_csv = OUT_DIR / "dev_cal_readout_step1250_2026-05-13.csv"
    with open(readout_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "n_fired", "n"])
        w.writeheader()
        for r in readout_rows:
            w.writerow(r)
    log.info("wrote dev-cal readout -> %s", readout_csv)

    # Console headline
    log.info("=" * 80)
    log.info("STEP1250 DEPLOYMENT HEADLINE @ dev-cal τ=%.4f", tau_dev_cal)
    log.info("=" * 80)
    log.info("dev_real_fpr               = %.4f  (target ≤ 0.07)", dev_real_fpr)
    log.info("dev_fake_recall (single)   = %.4f  (contract floor ≥ 0.30)", dev_fake_recall)
    log.info("lb_real_fpr                = %.4f  (want ≤ 0.025)", lb_real_fpr)
    log.info("lb_fake_recall             = %.4f", lb_fake_recall)
    log.info("Roy_D pooled FPR           = %.4f  (want < 0.30)", roy_pooled_fpr)
    log.info("dor_shkedi lockbox FPR     = %.4f  (P8A baseline 0.007)", dor_lockbox_fpr)
    log.info("lighting_extreme_dev FPR   = %.4f", lit_fpr)
    log.info("poor_quality_dev FPR       = %.4f", pq_fpr)

    log.info("SCORE DISTRIBUTION (video-level avg prob_fake):")
    for r in dist_rows:
        log.info("  %-40s n=%-5d min=%.4f p50=%.4f p90=%.4f p99=%.4f max=%.4f",
                 r["suite"], r["n"], r["min"], r["p50"], r["p90"], r["p99"], r["max"])

    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
