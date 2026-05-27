"""Roy_D probe — direct dose-response check vs BUNDLE_step500's 800× collapse.

Score Stage 2 ckpts on the 130-frame Roy_D set (the same substrate where
BUNDLE_step500 had std=0.0005 collapse and Pearson r=-0.16 with P8A).

Reference scores from `analysis/p1_pe_eval_2026-05-07/roy_d_regression/
roy_d_per_frame_scores.csv` (P8A, BUNDLE-step500/3750/4000, PAIRRANK-
step500/6000/6750, E2B).

Stage 2 ckpts: S1/S2/S3 step{500, 2500, 4500} (9 ckpts). Reused from the
dor probe download cache.

Outputs:
  outputs/roy_d_stage2_per_frame.csv   — per-frame all ckpts
  outputs/roy_d_stage2_ckpt_stats.csv  — per-ckpt stats (n, mean, p50, std,
                                          range, Pearson r vs P8A,
                                          fraction_above_0.5_FPR)
  STAGE2_ROY_D_PROBE_FACTS_2026-05-09.md (authored separately)
"""
from __future__ import annotations

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
FRAMES_DIR = THIS_DIR / "_roy_d_frames"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ROY_D_REF_CSV = (
    REPO_ROOT / "analysis" / "p1_pe_eval_2026-05-07" / "roy_d_regression" / "roy_d_per_frame_scores.csv"
)

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

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

logger = logging.getLogger("roy-d-probe")


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

    # 1. Load reference Roy_D scores (P8A + BUNDLE + PAIRRANK + E2B).
    ref = pd.read_csv(ROY_D_REF_CSV)
    # pivot to wide: one row per frame_path, columns = ckpt names with frame_prob.
    wide = ref.pivot_table(
        index=["frame_path", "video_id"],
        columns="ckpt",
        values="frame_prob",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # 2. Filter to frames present locally.
    wide["local_path"] = wide["frame_path"].apply(lambda u: FRAMES_DIR / u.split("/")[-1])
    wide["frame_present"] = wide["local_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    n_ref = len(wide)
    wide_present = wide[wide["frame_present"]].reset_index(drop=True)
    logger.info("ref frames: %d total, %d present locally", n_ref, len(wide_present))
    if len(wide_present) < n_ref / 2:
        logger.error("less than half of Roy_D frames present locally; aborting (run download first)")
        return 2

    # 3. Pick device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device: %s", device)

    # 4. Score Stage 2 ckpts.
    paths = [Path(p) for p in wide_present["local_path"].tolist()]
    stage2_scores: Dict[str, np.ndarray] = {}
    for label, ckpt_path in CKPTS_TO_SCORE.items():
        if not ckpt_path.exists():
            logger.warning("missing %s; skipping", ckpt_path)
            continue
        logger.info("[%s] loading", label)
        model = load_effort_model(ckpt_path, device)
        logger.info("[%s] scoring %d frames", label, len(paths))
        probs = score_frames(model, paths, device, batch_size=32)
        stage2_scores[label] = probs
        logger.info(
            "[%s] mean=%.3f median=%.3f std=%.4f range=%.4f",
            label,
            float(np.nanmean(probs)),
            float(np.nanmedian(probs)),
            float(np.nanstd(probs)),
            float(np.nanmax(probs) - np.nanmin(probs)),
        )
        del model

    # 5. Combine into one wide table.
    out = wide_present.drop(columns=["frame_present"]).copy()
    for label, probs in stage2_scores.items():
        out[label] = probs
    out.to_csv(OUTPUTS / "roy_d_stage2_per_frame.csv", index=False)
    logger.info("saved per-frame: %s", OUTPUTS / "roy_d_stage2_per_frame.csv")

    # 6. Per-ckpt stats.
    p8a_col = "P8A_REFERENCE_STEP5000"
    rows = []
    score_cols = [c for c in out.columns if c not in {"frame_path", "video_id", "local_path"}]
    for col in score_cols:
        s = out[col].dropna().astype(float)
        if len(s) == 0:
            continue
        # Pearson r vs P8A.
        if col == p8a_col:
            r = 1.0
        else:
            sub = out.dropna(subset=[p8a_col, col])
            if len(sub) > 1 and sub[col].std() > 1e-9 and sub[p8a_col].std() > 1e-9:
                r = float(np.corrcoef(sub[p8a_col], sub[col])[0, 1])
            else:
                r = float("nan")
        # Fraction above τ=0.5 (FPR proxy on these all-real frames).
        frac_fp_at_05 = float((s > 0.5).sum()) / len(s)
        rows.append({
            "ckpt": col,
            "n": len(s),
            "mean": float(s.mean()),
            "p10": float(np.percentile(s, 10)),
            "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
            "std": float(s.std()),
            "range": float(s.max() - s.min()),
            "min": float(s.min()),
            "max": float(s.max()),
            "pearson_r_vs_P8A": r,
            "frac_FP_at_tau_0.5": frac_fp_at_05,
        })
    stats = pd.DataFrame(rows).sort_values(by="ckpt")
    stats.to_csv(OUTPUTS / "roy_d_stage2_ckpt_stats.csv", index=False)

    # 7. Headlines.
    print("\n" + "=" * 88)
    print("Roy_D 130-frame probe — per-ckpt stats")
    print("(All frames are REAL; high frac_FP_at_tau_0.5 = collapse-to-fake; std=collapse signal)")
    print("=" * 88)
    pd.options.display.float_format = "{:.4f}".format
    cols_show = ["ckpt", "n", "mean", "p50", "std", "range", "pearson_r_vs_P8A", "frac_FP_at_tau_0.5"]
    print(stats[cols_show].to_string(index=False))

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
