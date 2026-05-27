"""CPU Diagnostic A2 RETRY — Frozen-encoder linear probe on T4 L11 features for lockbox.

Tighter scope:
- VIDEO-LEVEL ONLY: max 4 frames/video, mean → 1 feature/video.
- Lockbox reals: downsample to 200 videos (random seed=42).
- Lockbox fakes: keep all (253).
- Ckpts: T4_L1_step10500 + P8A_step5000.
- 5-fold StratifiedKFold logistic regression, n_jobs=1.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

OUT_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-11_a2_linear_probe"
CACHE_DIR = OUT_DIR / "_cache"
LOCKBOX_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local"

T4_CKPT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth"
P8A_CKPT = REPO_ROOT / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"

CKPTS = {
    "T4_LAMBDA1_TOP_N_STEP10500": T4_CKPT,
    "P8A_REFERENCE_STEP5000": P8A_CKPT,
}

LOCAL_REAL_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/real/flat")
LOCAL_FAKE_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/fake/flat")

LAYER_IX = 11
N_REAL_VIDEOS = 200
MAX_FRAMES_PER_VIDEO = 4
BATCH_SIZE = 32
RANDOM_SEED = 42

logger = logging.getLogger("a2-probe")


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
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected:
        logger.info("  dropped %d unexpected keys", len(unexpected))
    if missing:
        logger.warning("  %d missing keys (e.g. %s)", len(missing), missing[:3])
    model.eval()
    return model


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks")


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def gs_to_local(gs_path: str) -> Path | None:
    """Map gs:// path to local filesystem.
    gs://teams-faces-data-test-2914-fake-4420-real-feb-28/real/<basename>
    -> /Users/roeedar/Downloads/faces/r9_feb28_for_checker/{real,fake}/flat/<basename>
    """
    if "/real/" in gs_path:
        basename = gs_path.split("/real/")[-1]
        candidate = LOCAL_REAL_DIR / basename
    elif "/fake/" in gs_path:
        basename = gs_path.split("/fake/")[-1]
        candidate = LOCAL_FAKE_DIR / basename
    else:
        return None
    if candidate.exists():
        return candidate
    return None


def build_frame_list() -> pd.DataFrame:
    """Returns DataFrame with columns: label (int), video_id (str), frame_path (Path)."""
    rng = np.random.default_rng(RANDOM_SEED)

    real_csv = LOCKBOX_DIR / "teams_real_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv"
    fake_csv = LOCKBOX_DIR / "teams_fake_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv"
    real_df = pd.read_csv(real_csv)
    fake_df = pd.read_csv(fake_csv)

    # Map gs:// to local
    real_df["local_path"] = real_df["frame_path"].map(lambda p: gs_to_local(p))
    fake_df["local_path"] = fake_df["frame_path"].map(lambda p: gs_to_local(p))
    real_df = real_df[real_df["local_path"].notna()].reset_index(drop=True)
    fake_df = fake_df[fake_df["local_path"].notna()].reset_index(drop=True)
    logger.info("after local mapping: reals=%d frames / %d videos | fakes=%d frames / %d videos",
                len(real_df), real_df["video_id"].nunique(),
                len(fake_df), fake_df["video_id"].nunique())

    # Downsample REAL videos to N_REAL_VIDEOS. Stratify by source-identity prefix.
    real_video_ids = real_df["video_id"].unique().tolist()
    if len(real_video_ids) > N_REAL_VIDEOS:
        # group by source (split on '__' → first token)
        groups: Dict[str, List[str]] = defaultdict(list)
        for v in real_video_ids:
            src = v.split("__")[0]
            groups[src].append(v)
        # proportional allocation
        total = len(real_video_ids)
        kept = []
        for src, vids in groups.items():
            n_alloc = max(1, int(round(len(vids) * N_REAL_VIDEOS / total)))
            n_alloc = min(n_alloc, len(vids))
            idx = rng.permutation(len(vids))[:n_alloc]
            kept.extend([vids[i] for i in idx])
        # cap to N_REAL_VIDEOS exactly
        kept = list(rng.permutation(kept))[:N_REAL_VIDEOS]
        real_df = real_df[real_df["video_id"].isin(set(kept))].reset_index(drop=True)
        logger.info("downsampled reals: %d videos (%d frames)", len(set(kept)), len(real_df))

    # Cap MAX_FRAMES_PER_VIDEO per video
    def cap_frames(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for vid, grp in df.groupby("video_id"):
            idx = rng.permutation(len(grp))[:MAX_FRAMES_PER_VIDEO]
            keep.append(grp.iloc[idx])
        return pd.concat(keep, axis=0).reset_index(drop=True)

    real_df = cap_frames(real_df)
    fake_df = cap_frames(fake_df)

    real_df["label"] = 0
    fake_df["label"] = 1
    all_df = pd.concat([real_df[["label", "video_id", "local_path"]],
                        fake_df[["label", "video_id", "local_path"]]], axis=0).reset_index(drop=True)
    logger.info("final frames: %d (reals=%d, fakes=%d) across %d videos",
                len(all_df), (all_df.label == 0).sum(), (all_df.label == 1).sum(),
                all_df["video_id"].nunique())
    return all_df


def extract_features(model, frames_df: pd.DataFrame, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    resblocks = get_resblocks(model)
    captured: List[np.ndarray] = []

    def hook(_m, _i, output):
        if output.dim() == 3:
            cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[LAYER_IX].register_forward_hook(hook)
    try:
        # Pre-load all
        valid_idx = []
        tensors = []
        for i, row in frames_df.iterrows():
            t = load_and_preprocess(Path(row["local_path"]))
            if t is not None:
                tensors.append(t)
                valid_idx.append(i)
        logger.info("loaded %d/%d valid frames", len(tensors), len(frames_df))

        for j in range(0, len(tensors), BATCH_SIZE):
            chunk = tensors[j : j + BATCH_SIZE]
            batch = torch.stack(chunk).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            if (j // BATCH_SIZE) % 5 == 0:
                logger.info("  batch %d/%d", j // BATCH_SIZE + 1, (len(tensors) + BATCH_SIZE - 1) // BATCH_SIZE)
        feats = np.concatenate(captured, axis=0)
    finally:
        handle.remove()
    return feats, np.array(valid_idx, dtype=np.int64)


def aggregate_to_video(feats: np.ndarray, frames_df: pd.DataFrame, valid_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub = frames_df.iloc[valid_idx].reset_index(drop=True)
    # Group frames by video, average features
    groups = sub.groupby("video_id", sort=False)
    video_ids = []
    labels = []
    feat_rows = []
    for vid, grp in groups:
        idx = grp.index.values
        feat_rows.append(feats[idx].mean(axis=0))
        video_ids.append(vid)
        labels.append(int(grp["label"].iloc[0]))
    sources = np.array([v.split("__")[0] for v in video_ids])
    return np.stack(feat_rows, axis=0), np.array(labels, dtype=np.int64), np.array(video_ids), sources


def run_probe(feats: np.ndarray, labels: np.ndarray, label_for_log: str) -> List[Dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    logger.info("[%s] probe inputs: n_real=%d n_fake=%d dim=%d", label_for_log, n_real, n_fake, feats.shape[1])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(feats, labels)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feats[tr])
        X_te = scaler.transform(feats[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)
        clf.fit(X_tr, labels[tr])
        scores = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(labels[te], scores)
        except ValueError:
            auc = float("nan")
        n_r = int((labels[te] == 0).sum())
        n_f = int((labels[te] == 1).sum())
        rows.append({"ckpt": label_for_log, "fold": fold, "auc": float(auc), "n_real": n_r, "n_fake": n_f})
        logger.info("  fold %d AUC=%.4f (n_real=%d n_fake=%d)", fold, auc, n_r, n_f)
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    frames_df = build_frame_list()
    logger.info("frame_list built in %.1fs", time.time() - t0)

    all_results = []
    per_ckpt_feats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, ckpt_path in CKPTS.items():
        t_start = time.time()
        cache_file = CACHE_DIR / f"video_feats__{label}__L{LAYER_IX:02d}.npz"
        if cache_file.exists():
            logger.info("[%s] loading from cache", label)
            data = np.load(cache_file, allow_pickle=True)
            video_feats = data["features"]
            video_labels = data["labels"]
        else:
            logger.info("[%s] loading model from %s", label, ckpt_path)
            model = load_effort_model(ckpt_path, device)
            logger.info("[%s] extracting L%d", label, LAYER_IX)
            feats, valid_idx = extract_features(model, frames_df, device)
            video_feats, video_labels, video_ids, sources = aggregate_to_video(feats, frames_df, valid_idx)
            np.savez_compressed(cache_file, features=video_feats, labels=video_labels,
                                video_ids=video_ids, sources=sources)
            del model
        logger.info("[%s] features ready in %.1fs (shape=%s)", label, time.time() - t_start, video_feats.shape)
        per_ckpt_feats[label] = (video_feats, video_labels)

    for label, (vf, vl) in per_ckpt_feats.items():
        rows = run_probe(vf, vl, label)
        all_results.extend(rows)

    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "lockbox_probe_auc.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %s", out_csv)
    print("\n=== SUMMARY ===")
    for ckpt, grp in df.groupby("ckpt"):
        aucs = grp["auc"].values
        print(f"{ckpt}: mean_AUC={aucs.mean():.4f} ± {aucs.std():.4f} (folds={len(aucs)})")
        print(f"  per_fold: {[f'{a:.4f}' for a in aucs]}")
    logger.info("total runtime: %.1fs", time.time() - t0)


if __name__ == "__main__":
    raise SystemExit(main())
