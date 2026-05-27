"""Head retrain on frozen T4 encoder features — train on dev, eval on lockbox.

Procedure:
1. Extract L11 CLS features for T4 and P8A encoders on dev (train cohort) and lockbox (eval cohort).
2. Aggregate to video-level (mean of <=4 frames/video).
3. Train a fresh 2-layer MLP head (768 -> 128 -> 1) on dev features for 3 seeds {42, 7, 123}.
4. Hold out 20% of dev (stratified by class) for in-sample sanity AUC.
5. Evaluate head on lockbox with cohort breakdown (chronic_6, real_dor, dor_shkedi, non_chronic).
6. Identical pipeline for P8A control.

FACTS-only; no opinion language.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

OUT_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-11_head_retrain"
CACHE_DIR = OUT_DIR / "_cache"

# Dev frame sources — re-use the existing P11 dev frame reports (only the frame_path column matters; frame_prob
# is for a different ckpt but is not used here).
DEV_REAL_CSV = REPO_ROOT / "analysis/option_a_ensemble_2026-04-28/cache/teams_real_all_dev_p11_mild_step1000_frames_report.csv"
DEV_FAKE_CSV = REPO_ROOT / "analysis/option_a_ensemble_2026-04-28/cache/teams_fake_all_dev_p11_mild_step1000_frames_report.csv"

# Lockbox frame sources — use the T4 scorecard frames CSVs (these are the ones used by A2/A2-extension).
LOCKBOX_REAL_CSV = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_real_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv"
LOCKBOX_FAKE_CSV = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_fake_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv"

T4_CKPT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth"
P8A_CKPT = REPO_ROOT / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"

CKPTS = {
    "T4_LAMBDA1_TOP_N_STEP10500": T4_CKPT,
    "P8A_REFERENCE_STEP5000": P8A_CKPT,
}

LOCAL_REAL_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/real/flat")
LOCAL_FAKE_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/fake/flat")
REAL_DOR_PNG_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-11_a2_extension/_real_dor_png"

# Cap dev real to 2000 / dev fake to 1500 if larger; lockbox kept at maximum locally available.
LAYER_IX = 11
MAX_FRAMES_PER_VIDEO = 4
BATCH_SIZE = 32
RANDOM_SEED = 42
DEV_REAL_CAP = 2000
DEV_FAKE_CAP = 1500

# chronic_6 identities -- from analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json
CHRONIC_IDENTITIES = ["bla_bla_chow", "PC_Generator__s22", "PC_Generator__s45", "roy_d", "Q__s6"]
# bla_bla_chow__s2 is part of chronic_6 too, but bla_bla_chow already matches via prefix.

logger = logging.getLogger("head-retrain")


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Frame I/O
# ----------------------------------------------------------------------------

def load_and_preprocess(local_path: Path, resolution: int = 224) -> Optional[torch.Tensor]:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def gs_to_local(gs_path: str) -> Optional[Path]:
    bn = os.path.basename(gs_path)
    if bn.startswith("real_dor__") and bn.endswith(".png"):
        c = REAL_DOR_PNG_DIR / bn
        return c if c.exists() else None
    if "/real/" in gs_path:
        c = LOCAL_REAL_DIR / bn
    elif "/fake/" in gs_path:
        c = LOCAL_FAKE_DIR / bn
    else:
        return None
    return c if c.exists() else None


# ----------------------------------------------------------------------------
# Frame list builders
# ----------------------------------------------------------------------------

def build_dev_frame_list() -> pd.DataFrame:
    """Build dev frame list with local availability + proportional downsampling."""
    rng = np.random.default_rng(RANDOM_SEED)

    real_df = pd.read_csv(DEV_REAL_CSV)
    fake_df = pd.read_csv(DEV_FAKE_CSV)
    real_df["local_path"] = real_df["frame_path"].map(gs_to_local)
    fake_df["local_path"] = fake_df["frame_path"].map(gs_to_local)
    real_df = real_df[real_df["local_path"].notna()].reset_index(drop=True)
    fake_df = fake_df[fake_df["local_path"].notna()].reset_index(drop=True)
    real_df["source"] = real_df["video_id"].str.split("__").str[0]
    fake_df["source"] = fake_df["video_id"].str.split("__").str[0]
    logger.info(
        "[dev] after local mapping: reals=%d frames / %d videos | fakes=%d frames / %d videos",
        len(real_df), real_df["video_id"].nunique(),
        len(fake_df), fake_df["video_id"].nunique(),
    )

    # Cap to DEV_REAL_CAP / DEV_FAKE_CAP videos via proportional source-stratified sampling
    def downsample_videos(df: pd.DataFrame, cap: int) -> pd.DataFrame:
        vids = df["video_id"].unique().tolist()
        if len(vids) <= cap:
            return df
        groups: Dict[str, List[str]] = defaultdict(list)
        for v in vids:
            groups[v.split("__")[0]].append(v)
        total = len(vids)
        kept = []
        for src, vlist in groups.items():
            n_alloc = max(1, int(round(len(vlist) * cap / total)))
            n_alloc = min(n_alloc, len(vlist))
            idx = rng.permutation(len(vlist))[:n_alloc]
            kept.extend([vlist[i] for i in idx])
        kept = list(rng.permutation(kept))[:cap]
        return df[df["video_id"].isin(set(kept))].reset_index(drop=True)

    real_df = downsample_videos(real_df, DEV_REAL_CAP)
    fake_df = downsample_videos(fake_df, DEV_FAKE_CAP)
    logger.info(
        "[dev] after downsample (cap real=%d, fake=%d): reals=%d videos, fakes=%d videos",
        DEV_REAL_CAP, DEV_FAKE_CAP, real_df["video_id"].nunique(), fake_df["video_id"].nunique(),
    )

    # Cap MAX_FRAMES_PER_VIDEO
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

    all_df = pd.concat(
        [real_df[["label", "video_id", "local_path", "source"]],
         fake_df[["label", "video_id", "local_path", "source"]]],
        axis=0,
    ).reset_index(drop=True)
    logger.info(
        "[dev] final frames: %d (reals=%d, fakes=%d) across %d videos",
        len(all_df), (all_df.label == 0).sum(), (all_df.label == 1).sum(),
        all_df["video_id"].nunique(),
    )
    return all_df


def build_lockbox_frame_list() -> pd.DataFrame:
    """Build lockbox frame list keeping ALL local-available videos (no real-pool downsampling)."""
    rng = np.random.default_rng(RANDOM_SEED)

    real_df = pd.read_csv(LOCKBOX_REAL_CSV)
    fake_df = pd.read_csv(LOCKBOX_FAKE_CSV)
    real_df["local_path"] = real_df["frame_path"].map(gs_to_local)
    fake_df["local_path"] = fake_df["frame_path"].map(gs_to_local)
    real_df = real_df[real_df["local_path"].notna()].reset_index(drop=True)
    fake_df = fake_df[fake_df["local_path"].notna()].reset_index(drop=True)
    real_df["source"] = real_df["video_id"].str.split("__").str[0]
    fake_df["source"] = fake_df["video_id"].str.split("__").str[0]
    logger.info(
        "[lockbox] after local mapping: reals=%d frames / %d videos | fakes=%d frames / %d videos",
        len(real_df), real_df["video_id"].nunique(),
        len(fake_df), fake_df["video_id"].nunique(),
    )

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
    all_df = pd.concat(
        [real_df[["label", "video_id", "local_path", "source"]],
         fake_df[["label", "video_id", "local_path", "source"]]],
        axis=0,
    ).reset_index(drop=True)
    logger.info(
        "[lockbox] final frames: %d (reals=%d, fakes=%d) across %d videos",
        len(all_df), (all_df.label == 0).sum(), (all_df.label == 1).sum(),
        all_df["video_id"].nunique(),
    )
    return all_df


# ----------------------------------------------------------------------------
# Feature extraction (returns video-level features)
# ----------------------------------------------------------------------------

def extract_video_features(
    model: torch.nn.Module, frames_df: pd.DataFrame, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (video_features [N,768], labels [N], video_ids [N], sources [N])."""
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
        valid_idx: List[int] = []
        tensors: List[torch.Tensor] = []
        for i, row in frames_df.iterrows():
            t = load_and_preprocess(Path(row["local_path"]))
            if t is not None:
                tensors.append(t)
                valid_idx.append(i)
        logger.info("  loaded %d/%d valid frames", len(tensors), len(frames_df))

        n_batches = (len(tensors) + BATCH_SIZE - 1) // BATCH_SIZE
        for j in range(0, len(tensors), BATCH_SIZE):
            chunk = tensors[j : j + BATCH_SIZE]
            batch = torch.stack(chunk).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            if (j // BATCH_SIZE) % 20 == 0:
                logger.info("    batch %d/%d", j // BATCH_SIZE + 1, n_batches)
        feats = np.concatenate(captured, axis=0)
    finally:
        handle.remove()

    sub = frames_df.iloc[valid_idx].reset_index(drop=True)
    groups = sub.groupby("video_id", sort=False)
    feat_rows: List[np.ndarray] = []
    video_ids: List[str] = []
    labels: List[int] = []
    sources: List[str] = []
    for vid, grp in groups:
        idx = grp.index.values
        feat_rows.append(feats[idx].mean(axis=0))
        video_ids.append(vid)
        labels.append(int(grp["label"].iloc[0]))
        sources.append(str(grp["source"].iloc[0]))
    return (
        np.stack(feat_rows, axis=0),
        np.array(labels, dtype=np.int64),
        np.array(video_ids),
        np.array(sources),
    )


def load_or_extract(
    label: str,
    ckpt_path: Path,
    cache_key: str,
    frames_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_file = CACHE_DIR / f"video_feats__{label}__{cache_key}__L{LAYER_IX:02d}.npz"
    if cache_file.exists():
        logger.info("[%s/%s] loading cache %s", label, cache_key, cache_file.name)
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["features"],
            data["labels"],
            data["video_ids"],
            data["sources"],
        )
    logger.info("[%s/%s] extracting fresh", label, cache_key)
    t0 = time.time()
    model = load_effort_model(ckpt_path, device)
    feats, labels, vids, srcs = extract_video_features(model, frames_df, device)
    np.savez_compressed(cache_file, features=feats, labels=labels, video_ids=vids, sources=srcs)
    del model
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    logger.info("[%s/%s] features ready in %.1fs (shape=%s)", label, cache_key, time.time() - t0, feats.shape)
    return feats, labels, vids, srcs


# ----------------------------------------------------------------------------
# MLP Head
# ----------------------------------------------------------------------------

class TwoLayerHead(torch.nn.Module):
    def __init__(self, in_dim: int = 768, hid: int = 128):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hid)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hid, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


def train_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 5,
) -> Tuple[TwoLayerHead, Dict, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = TwoLayerHead(in_dim=X_train.shape[1], hid=128).to(device)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    Xt = torch.from_numpy(X_train.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_val.astype(np.float32)).to(device)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    batch_size = 64
    n = X_train.shape[0]

    train_log = {"epochs": [], "train_loss": [], "val_loss": [], "val_auc": []}
    for epoch in range(max_epochs):
        head.train()
        # Shuffle
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            logits = head(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        head.eval()
        with torch.inference_mode():
            val_logits = head(Xv).squeeze(-1)
            val_loss = float(criterion(val_logits, yv).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = float("nan")

        train_log["epochs"].append(epoch)
        train_log["train_loss"].append(train_loss)
        train_log["val_loss"].append(val_loss)
        train_log["val_auc"].append(val_auc)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("    [seed=%d] early stop at epoch %d (best val_loss=%.4f, val_auc=%.4f)",
                            seed, epoch, best_loss, val_auc)
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    # Score val one more time at best state
    with torch.inference_mode():
        val_logits = head(Xv).squeeze(-1)
        val_probs_best = torch.sigmoid(val_logits).cpu().numpy()
    return head, train_log, val_probs_best


def score_with_head(head: TwoLayerHead, X: np.ndarray, device: torch.device) -> np.ndarray:
    head.eval()
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.inference_mode():
        probs = torch.sigmoid(head(Xt).squeeze(-1)).cpu().numpy()
    return probs


# ----------------------------------------------------------------------------
# Cohort identification
# ----------------------------------------------------------------------------

def assign_cohort(video_id: str, source: str) -> str:
    """Returns one of: chronic_6, real_dor, dor_shkedi, non_chronic, fake."""
    vid_lower = str(video_id).lower()
    if source == "real_dor":
        return "real_dor"
    if source == "dor_shkedi":
        return "dor_shkedi"
    # Check chronic_6 list with case-insensitive substring match
    for chronic_id in CHRONIC_IDENTITIES:
        if chronic_id.lower() in vid_lower:
            return "chronic_6"
    return "non_chronic"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

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

    # --- Build frame lists ---
    dev_frames = build_dev_frame_list()
    lockbox_frames = build_lockbox_frame_list()

    feat_meta_rows = []

    # --- Extract dev + lockbox features for each ckpt ---
    per_ckpt_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for ckpt_label, ckpt_path in CKPTS.items():
        if not ckpt_path.exists():
            logger.warning("ckpt missing: %s", ckpt_path)
            continue
        per_ckpt_data[ckpt_label] = {}
        for split_label, frames_df in [("DEV", dev_frames), ("LOCKBOX", lockbox_frames)]:
            t_ck = time.time()
            feats, labels, vids, srcs = load_or_extract(
                ckpt_label, ckpt_path, split_label, frames_df, device
            )
            elapsed = time.time() - t_ck
            n_videos = feats.shape[0]
            feat_meta_rows.append({
                "ckpt": ckpt_label,
                "split": split_label,
                "n_videos": n_videos,
                "n_reals": int((labels == 0).sum()),
                "n_fakes": int((labels == 1).sum()),
                "elapsed_sec": round(elapsed, 1),
            })
            per_ckpt_data[ckpt_label][split_label] = (feats, labels, vids, srcs)

    pd.DataFrame(feat_meta_rows).to_csv(OUT_DIR / "feature_extraction_metadata.csv", index=False)
    logger.info("wrote feature_extraction_metadata.csv")

    # --- For each ckpt: split dev 80/20 stratified, train MLP head x 3 seeds ---
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    all_auc_rows: List[Dict] = []
    per_video_score_rows: List[Dict] = []
    seeds = [42, 7, 123]

    for ckpt_label in per_ckpt_data.keys():
        X_dev, y_dev, vid_dev, src_dev = per_ckpt_data[ckpt_label]["DEV"]
        X_lb, y_lb, vid_lb, src_lb = per_ckpt_data[ckpt_label]["LOCKBOX"]
        logger.info("=== [%s] dev: %d videos (%d reals, %d fakes) | lockbox: %d videos (%d reals, %d fakes)",
                    ckpt_label, X_dev.shape[0], int((y_dev == 0).sum()), int((y_dev == 1).sum()),
                    X_lb.shape[0], int((y_lb == 0).sum()), int((y_lb == 1).sum()))

        # Stratified 80/20
        idx_train, idx_val = train_test_split(
            np.arange(len(y_dev)), test_size=0.2, stratify=y_dev, random_state=RANDOM_SEED
        )

        # Cohort labels for lockbox
        lb_cohorts = np.array([assign_cohort(v, s) if y_lb[i] == 0 else "fake"
                                for i, (v, s) in enumerate(zip(vid_lb, src_lb))])
        logger.info("[%s] lockbox cohort breakdown:", ckpt_label)
        for coh in np.unique(lb_cohorts):
            n = int((lb_cohorts == coh).sum())
            logger.info("    %s: %d", coh, n)

        seed_scores_list = []  # list of (n_lb,) prob arrays, one per seed

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_dev[idx_train])
            X_va = scaler.transform(X_dev[idx_val])
            X_lb_scaled = scaler.transform(X_lb)

            head, train_log, val_probs = train_head(
                X_tr, y_dev[idx_train], X_va, y_dev[idx_val],
                seed=seed, device=device,
            )
            try:
                val_auc = float(roc_auc_score(y_dev[idx_val], val_probs))
            except Exception:
                val_auc = float("nan")

            lb_probs = score_with_head(head, X_lb_scaled, device)
            # Full lockbox AUC
            try:
                lb_auc = float(roc_auc_score(y_lb, lb_probs))
            except Exception:
                lb_auc = float("nan")

            # Per-cohort AUC (vs all fakes)
            cohort_aucs: Dict[str, float] = {}
            fake_mask = (y_lb == 1)
            for coh_name in ["chronic_6", "real_dor", "dor_shkedi", "non_chronic"]:
                coh_mask = (lb_cohorts == coh_name)
                if coh_mask.sum() < 2:
                    cohort_aucs[coh_name] = float("nan")
                    continue
                # Build (cohort_reals + all_fakes) subset
                sel = coh_mask | fake_mask
                yy = y_lb[sel]
                ss = lb_probs[sel]
                try:
                    cohort_aucs[coh_name] = float(roc_auc_score(yy, ss))
                except Exception:
                    cohort_aucs[coh_name] = float("nan")

            all_auc_rows.append({
                "ckpt": ckpt_label,
                "seed": seed,
                "dev_holdout_auc": val_auc,
                "lockbox_auc": lb_auc,
                "lockbox_n_reals": int((y_lb == 0).sum()),
                "lockbox_n_fakes": int((y_lb == 1).sum()),
                "lockbox_chronic_6_auc": cohort_aucs["chronic_6"],
                "lockbox_real_dor_auc": cohort_aucs["real_dor"],
                "lockbox_dor_shkedi_auc": cohort_aucs["dor_shkedi"],
                "lockbox_non_chronic_auc": cohort_aucs["non_chronic"],
                "n_train_videos": int(len(idx_train)),
                "n_val_videos": int(len(idx_val)),
                "train_n_reals": int((y_dev[idx_train] == 0).sum()),
                "train_n_fakes": int((y_dev[idx_train] == 1).sum()),
                "val_n_reals": int((y_dev[idx_val] == 0).sum()),
                "val_n_fakes": int((y_dev[idx_val] == 1).sum()),
                "epochs_run": len(train_log["epochs"]),
            })
            logger.info(
                "    seed=%d val_AUC=%.4f lockbox_AUC=%.4f | chronic_6=%.4f real_dor=%.4f dor_shkedi=%.4f non_chronic=%.4f",
                seed, val_auc, lb_auc,
                cohort_aucs["chronic_6"], cohort_aucs["real_dor"],
                cohort_aucs["dor_shkedi"], cohort_aucs["non_chronic"],
            )
            seed_scores_list.append(lb_probs)

        # Per-video lockbox scores — store mean across seeds for downstream
        mean_lb = np.mean(seed_scores_list, axis=0)
        for i in range(len(y_lb)):
            per_video_score_rows.append({
                "ckpt": ckpt_label,
                "video_id": str(vid_lb[i]),
                "source": str(src_lb[i]),
                "cohort": str(lb_cohorts[i]),
                "label": int(y_lb[i]),
                "mean_score": float(mean_lb[i]),
                "seed42_score": float(seed_scores_list[0][i]),
                "seed7_score": float(seed_scores_list[1][i]),
                "seed123_score": float(seed_scores_list[2][i]),
            })

    pd.DataFrame(all_auc_rows).to_csv(OUT_DIR / "head_retrain_aucs.csv", index=False)
    pd.DataFrame(per_video_score_rows).to_csv(OUT_DIR / "head_retrain_per_video_scores.csv", index=False)
    logger.info("wrote head_retrain_aucs.csv + head_retrain_per_video_scores.csv")

    # --- Summary ---
    df = pd.DataFrame(all_auc_rows)
    print("\n=== SUMMARY ===")
    for ckpt, grp in df.groupby("ckpt"):
        print(f"\n{ckpt}:")
        print(f"  lockbox_AUC mean={grp.lockbox_auc.mean():.4f} +- {grp.lockbox_auc.std():.4f} "
              f"(seeds: {grp.lockbox_auc.tolist()})")
        print(f"  dev_holdout_AUC mean={grp.dev_holdout_auc.mean():.4f} +- {grp.dev_holdout_auc.std():.4f}")
        print(f"  lockbox_chronic_6 mean={grp.lockbox_chronic_6_auc.mean():.4f}")
        print(f"  lockbox_real_dor mean={grp.lockbox_real_dor_auc.mean():.4f}")
        print(f"  lockbox_dor_shkedi mean={grp.lockbox_dor_shkedi_auc.mean():.4f}")
        print(f"  lockbox_non_chronic mean={grp.lockbox_non_chronic_auc.mean():.4f}")
    logger.info("total runtime: %.1fs", time.time() - t0)


if __name__ == "__main__":
    raise SystemExit(main())
