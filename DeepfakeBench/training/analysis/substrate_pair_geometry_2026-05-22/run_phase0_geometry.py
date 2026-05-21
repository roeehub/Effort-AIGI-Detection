"""Phase 0 A0.2 — Multi-Layer CPU/MPS Feature Geometry Probe (2026-05-22).

Builds a 12-cell table (3 checkpoints × 4 CLIP layers L0/L4/L8/L11) of
substrate-pair feature geometry metrics on the 1880-pair A0.1 inventory
(`inventory_manifest.csv`).

Per cell, 6 metrics:

    cos_pair               cosine(feat(clean_X), feat(teams_X))   mean over pairs
    cos_within_same        cosine(frame_i, frame_j) within one side, same capture
    cos_cross_id           cosine(clean_X, clean_Y), X!=Y         (2000 sampled)
    delta_pair_vs_within   cos_pair - cos_within_same             "fulcrum size"
    kliep_projection       <feat(teams)-feat(clean), w_hat>       L11 only
    score_corr             Pearson r(p_fake(clean), p_fake(teams))

Gate (applied at Slot A v2 / L11 cell only):
    delta >= -0.02 AND |kliep_proj_mean| <= 0.2     -> NO FULCRUM
    delta <= -0.05 AND |kliep_proj_mean| >= 0.4     -> STRONG FULCRUM
    middle                                          -> AMBIGUOUS

Outputs (all under analysis/substrate_pair_geometry_2026-05-22/):
    per_ckpt_layer_cosines.csv     (12 rows)
    kliep_axis_projections.csv     (per-pair L11 projections, 3 ckpts)
    RESULTS_FACTS_2026-05-22.md    (FACTS doc; no banned words)
    AGENT_PROPOSAL_2026-05-22.md   (opinion-only doc)
    feats/<ckpt_key>_L<L>_<side>.npy   (24 files; per-frame features)
    _cache_frames_<side>.pt        (single tensor cache, both sides)
    _cache_frames_meta_<side>.parquet
    scores/<ckpt_key>_<side>.npy   (per-frame head probabilities)

Usage:
    python run_phase0_geometry.py --output-dir analysis/substrate_pair_geometry_2026-05-22
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from batch_inference_gcs import load_model  # noqa: E402

# ----------------------------- Configuration -----------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RESOLUTION = 224
LAYERS = [0, 4, 8, 11]
DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"

CKPTS: Dict[str, str] = {
    "P8A_step5000": (
        "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/"
        "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
    ),
    "SlotAv2_step3500": (
        "gs://training-job-outputs/best_checkpoints/hp35c51p/"
        "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth"
    ),
    "T5C_step3500": (
        "gs://training-job-outputs/best_checkpoints/jrlldtem/"
        "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"
    ),
}

D7_CACHE = (
    REPO_ROOT
    / "analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/_clip_l11_features.npz"
)
D8_CACHE = (
    REPO_ROOT
    / "analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced"
    / "outputs/clip_frozen_l11__n4839.npz"
)
LOCKBOX_PARQUET = REPO_ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"

SEED = 42
N_FRAMES_PER_SIDE_DEFAULT = 5
BATCH_SIZE_DEFAULT = 32
N_CROSS_ID_SAMPLES = 2000
CACHE_DTYPE_DEFAULT = "float16"  # fp16 ≈ 5.5 GB at N=5, both sides (8.8 GB free disk budget)

logger = logging.getLogger("phase0_geom")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_run_phase0.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


# ----------------------------- Frame loading -----------------------------

def select_pair_frames(
    inventory_path: Path,
    n_frames: int,
    seed: int,
) -> pd.DataFrame:
    """Pick first N frames per side per pair, skipping enhanced rows (clean
    side absent). Returns a long-format frame manifest.

    Schema:
        pair_id (int)
        identity_id
        base_capture_id
        source
        side           "clean" or "teams"
        frame_idx      0..N-1
        bucket
        prefix         (with trailing slash)
    """
    df = pd.read_csv(inventory_path)
    logger.info("inventory: %d rows", len(df))
    # Skip rows missing clean count (visomaster_teams_enhanced).
    df_full = df[df["clean_real_frame_count"].notna()].reset_index(drop=True)
    logger.info("full-pair rows (clean+teams available): %d", len(df_full))

    rows = []
    for pair_id, r in df_full.iterrows():
        clean_count = int(r["clean_real_frame_count"])
        teams_count = int(r["teams_real_frame_count"])
        n_take = min(n_frames, clean_count, teams_count)
        for side, bucket, prefix in [
            ("clean", r["clean_bucket"], r["clean_prefix"]),
            ("teams", r["teams_bucket"], r["teams_prefix"]),
        ]:
            for f_idx in range(n_take):
                rows.append({
                    "pair_id": int(pair_id),
                    "identity_id": r["identity_id"],
                    "base_capture_id": r["base_capture_id"],
                    "source": r["source"],
                    "side": side,
                    "frame_idx": f_idx,
                    "bucket": bucket,
                    "prefix": prefix,
                })
    out = pd.DataFrame(rows)
    logger.info(
        "frame manifest: %d rows (clean=%d teams=%d)",
        len(out), int((out["side"] == "clean").sum()), int((out["side"] == "teams").sum()),
    )
    return out


def list_frames_for_prefix(client, bucket_obj_cache: Dict, bucket: str, prefix: str) -> List[str]:
    if bucket not in bucket_obj_cache:
        bucket_obj_cache[bucket] = client.bucket(bucket)
    b = bucket_obj_cache[bucket]
    blobs = list(b.list_blobs(prefix=prefix, max_results=64))
    return sorted([blob.name for blob in blobs if not blob.name.endswith("/")])


def cache_frames_for_side(
    manifest: pd.DataFrame,
    side: str,
    cache_path: Path,
    meta_path: Path,
    cache_dtype: str = CACHE_DTYPE_DEFAULT,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """Cache decoded RGB-normalized frames as a single tensor on disk.

    cache_dtype: "float16" (default; ≈ half disk) or "float32".
    On load, the tensor is returned in fp32 so downstream forward passes are
    numerically equivalent.
    """
    if cache_path.exists() and meta_path.exists():
        logger.info("[%s] using cached frames: %s", side, cache_path)
        t = torch.load(cache_path, map_location="cpu")
        if t.dtype != torch.float32:
            t = t.float()
        df = pd.read_parquet(meta_path)
        return t, df

    import cv2
    from google.cloud import storage
    from torchvision import transforms as T

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))]
    )

    df = manifest[manifest["side"] == side].reset_index(drop=True).copy()
    n = len(df)
    out = torch.zeros(n, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)

    # Build per-(bucket, prefix) blob lists once
    client = storage.Client()
    bucket_obj_cache: Dict[str, object] = {}

    # Group manifest by (bucket, prefix); fetch blob list once per prefix
    prefix_to_blobs: Dict[Tuple[str, str], List[str]] = {}
    unique_prefixes = df[["bucket", "prefix"]].drop_duplicates().values.tolist()
    logger.info("[%s] %d unique prefixes; listing blobs...", side, len(unique_prefixes))
    t_list = time.time()
    for i, (bucket, prefix) in enumerate(unique_prefixes):
        prefix_to_blobs[(bucket, prefix)] = list_frames_for_prefix(
            client, bucket_obj_cache, bucket, prefix
        )
        if (i + 1) % 200 == 0:
            logger.info("  [%s] listed %d/%d prefixes (%.1fs)", side, i + 1, len(unique_prefixes), time.time() - t_list)
    logger.info("[%s] blob listing done (%.1fs)", side, time.time() - t_list)

    # Pre-compute blob paths
    df["blob_path"] = ""
    for idx, r in df.iterrows():
        blobs = prefix_to_blobs[(r["bucket"], r["prefix"])]
        if r["frame_idx"] < len(blobs):
            df.at[idx, "blob_path"] = blobs[r["frame_idx"]]

    # Now decode
    fails = 0
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, r in df.iterrows():
        bp = r["blob_path"]
        if not bp:
            fails += 1
            continue
        try:
            if r["bucket"] not in bucket_obj_cache:
                bucket_obj_cache[r["bucket"]] = client.bucket(r["bucket"])
            blob = bucket_obj_cache[r["bucket"]].blob(bp)
            img_bytes = blob.download_as_bytes()
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                fails += 1
                continue
            img_bgr = cv2.resize(img_bgr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out[i] = normalize(img_rgb)
            valid[i] = True
        except Exception as e:  # noqa: BLE001
            fails += 1
            if fails <= 5:
                logger.warning("[%s] frame %d decode failed: %s", side, i, e)
        if (i + 1) % 500 == 0:
            logger.info(
                "  [%s] decoded %d/%d (%.1fs, fails=%d)",
                side, i + 1, n, time.time() - t0, fails,
            )

    n_valid = int(valid.sum())
    logger.info("[%s] decoded %d/%d valid (%.1fs, fails=%d)", side, n_valid, n, time.time() - t0, fails)
    out = out[valid]
    df = df[valid].reset_index(drop=True)

    if cache_dtype == "float16":
        out_save = out.half()
    elif cache_dtype == "float32":
        out_save = out
    else:
        raise ValueError(f"unknown cache_dtype: {cache_dtype}")
    torch.save(out_save, cache_path)
    df.to_parquet(meta_path)
    logger.info(
        "[%s] cache written: %s (%.1f MB, dtype=%s)",
        side, cache_path, cache_path.stat().st_size / 1e6, cache_dtype,
    )
    return out, df


# ----------------------------- Model wrapping with multi-layer hooks -----

def download_ckpt(gcs_path: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    fname = gcs_path.rsplit("/", 1)[-1]
    local = local_dir / fname
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    import subprocess
    logger.info("[download] %s", gcs_path)
    t0 = time.time()
    subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local)], check=True)
    logger.info("[download] -> %s (%.1fs)", local, time.time() - t0)
    return local


class MultiLayerHookManager:
    """Register forward hooks on selected resblocks to capture CLS tokens.

    Reads the captured tensor after each forward pass via :meth:`flush`.
    """

    def __init__(self, model: torch.nn.Module, layers: List[int]):
        self.model = model
        self.layers = layers
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles = []

        # OpenCLIP wrapper used by EffortDetector keeps its own hook on
        # `intermediate_layer`. We want our own — registered on the inner
        # transformer.resblocks list, which `OpenCLIPVisionModelWrapper.visual`
        # delegates to.
        visual = model.backbone.visual
        # The wrapper installs `self.visual = openclip_visual` so go one deeper
        try:
            resblocks = visual.transformer.resblocks
        except AttributeError:
            # Wrapper exposes the underlying openclip visual via .visual
            resblocks = visual.visual.transformer.resblocks  # type: ignore[attr-defined]
        self.resblocks = resblocks
        n_blocks = len(resblocks)
        for L in layers:
            assert 0 <= L < n_blocks, f"layer {L} out of [0,{n_blocks})"
            handle = resblocks[L].register_forward_hook(self._make_hook(L))
            self._handles.append(handle)
        logger.info(
            "hooks registered on layers %s of %d-block transformer",
            layers, n_blocks,
        )

    def _make_hook(self, L: int):
        def hook(module, inputs, output):  # noqa: ARG001
            # Handle both (seq, batch, dim) and (batch, seq, dim) layouts.
            if output.dim() == 3:
                if output.shape[0] >= output.shape[1]:
                    cls = output[0]      # seq-first
                else:
                    cls = output[:, 0]   # batch-first
            elif output.dim() == 2:
                cls = output
            else:
                raise RuntimeError(f"unexpected resblock {L} output shape: {output.shape}")
            self._captured[L] = cls.detach()
        return hook

    def flush(self) -> Dict[int, torch.Tensor]:
        out = self._captured
        self._captured = {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def forward_extract(
    model: torch.nn.Module,
    hooks: MultiLayerHookManager,
    tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Run inference on `tensor`. Returns (layer->features ndarray (N,D), probs (N,))."""
    model.eval()
    n = tensor.shape[0]
    feat_chunks: Dict[int, List[torch.Tensor]] = {L: [] for L in hooks.layers}
    prob_chunks: List[torch.Tensor] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tensor[start:end].to(device, non_blocking=True)
        pred = model({"image": batch}, inference=True)
        captured = hooks.flush()
        for L in hooks.layers:
            feat_chunks[L].append(captured[L].cpu())
        if isinstance(pred, dict) and "prob" in pred:
            prob = pred["prob"]
        else:
            # fall back to softmax over logits
            logits = pred.get("cls") if isinstance(pred, dict) else pred
            prob = torch.softmax(logits, dim=-1)[:, 1]
        prob_chunks.append(prob.detach().cpu().float())
        if (start // batch_size) % 20 == 0:
            logger.info("  forward %d/%d (%.1fs)", end, n, time.time() - t0)
    feats_out = {L: torch.cat(feat_chunks[L], dim=0).float().numpy() for L in hooks.layers}
    probs_out = torch.cat(prob_chunks, dim=0).numpy()
    return feats_out, probs_out


# ----------------------------- Metric computation -----------------------

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def compute_cell_metrics(
    feats_clean: np.ndarray,
    feats_teams: np.ndarray,
    meta_clean: pd.DataFrame,
    meta_teams: pd.DataFrame,
    probs_clean: np.ndarray = None,
    probs_teams: np.ndarray = None,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Compute the 5 non-KLIEP metrics for one cell (one ckpt × one layer).

    KLIEP metric handled separately because only L11 is meaningful.
    """
    assert feats_clean.shape[0] == len(meta_clean)
    assert feats_teams.shape[0] == len(meta_teams)

    Xc = l2_normalize_rows(feats_clean.astype(np.float64))
    Xt = l2_normalize_rows(feats_teams.astype(np.float64))

    # Group by pair_id
    clean_groups = meta_clean.groupby("pair_id").indices  # pair_id -> ndarray of row indices
    teams_groups = meta_teams.groupby("pair_id").indices

    common_pairs = sorted(set(clean_groups.keys()) & set(teams_groups.keys()))

    # cos_pair: per pair, mean of first-frame-to-first-frame cosine
    # (we use frame_idx==0 on each side; manifest is sorted by pair_id, side, frame_idx
    # but be explicit by taking the row with minimum frame_idx in each group)
    cos_pair_vals = []
    for pid in common_pairs:
        ci = clean_groups[pid]
        ti = teams_groups[pid]
        # pick frame_idx=0 if present else min
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        t0 = ti[np.argmin(meta_teams.iloc[ti]["frame_idx"].to_numpy())]
        cos_pair_vals.append(float(np.dot(Xc[c0], Xt[t0])))
    cos_pair_mean = float(np.mean(cos_pair_vals)) if cos_pair_vals else float("nan")
    cos_pair_std = float(np.std(cos_pair_vals, ddof=1)) if len(cos_pair_vals) > 1 else float("nan")
    n_pairs_used = len(cos_pair_vals)

    # cos_within_same: for each pair on each side, mean pairwise cosine across
    # its multiple frames; then average over (pair, side).
    within_vals = []
    for pid in common_pairs:
        for groups, X in ((clean_groups, Xc), (teams_groups, Xt)):
            idx = groups[pid]
            if len(idx) >= 2:
                sub = X[idx]
                G = sub @ sub.T  # k x k
                k = sub.shape[0]
                # off-diagonal mean
                off = (G.sum() - np.trace(G)) / (k * (k - 1))
                within_vals.append(float(off))
    cos_within = float(np.mean(within_vals)) if within_vals else float("nan")

    # cos_cross_id: sample N pairs of (clean_X first-frame, clean_Y first-frame), X != Y
    first_frame_clean = []
    for pid in common_pairs:
        ci = clean_groups[pid]
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        first_frame_clean.append(c0)
    first_frame_clean = np.array(first_frame_clean)
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_ff = len(first_frame_clean)
    if n_ff >= 2:
        sample_n = min(N_CROSS_ID_SAMPLES, n_ff * (n_ff - 1) // 2)
        cross_vals = []
        for _ in range(sample_n):
            i, j = rng.choice(n_ff, size=2, replace=False)
            v = float(np.dot(Xc[first_frame_clean[i]], Xc[first_frame_clean[j]]))
            cross_vals.append(v)
        cos_cross = float(np.mean(cross_vals))
    else:
        cos_cross = float("nan")

    delta = cos_pair_mean - cos_within

    # score_corr: pearson r over pairs of (prob_clean, prob_teams) on first frames
    if probs_clean is not None and probs_teams is not None:
        pc_vals = []
        pt_vals = []
        for pid in common_pairs:
            ci = clean_groups[pid]
            ti = teams_groups[pid]
            c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
            t0 = ti[np.argmin(meta_teams.iloc[ti]["frame_idx"].to_numpy())]
            pc_vals.append(float(probs_clean[c0]))
            pt_vals.append(float(probs_teams[t0]))
        pc_arr = np.asarray(pc_vals)
        pt_arr = np.asarray(pt_vals)
        if pc_arr.std() > 1e-12 and pt_arr.std() > 1e-12:
            score_corr = float(np.corrcoef(pc_arr, pt_arr)[0, 1])
        else:
            score_corr = float("nan")
    else:
        score_corr = float("nan")

    return {
        "cos_pair": cos_pair_mean,
        "cos_pair_std": cos_pair_std,
        "cos_within_same": cos_within,
        "cos_cross_id": cos_cross,
        "delta_pair_vs_within": delta,
        "score_corr": score_corr,
        "n_pairs": int(n_pairs_used),
    }


# ----------------------------- KLIEP at L11 -----------------------------

def fit_kliep_axis() -> Tuple[np.ndarray, float, float]:
    """Refit KLIEP discriminator on cached dev_real vs lockbox_real L11
    features. Returns (w_hat, b, accuracy).
    """
    d7 = np.load(D7_CACHE, allow_pickle=True)
    d8 = np.load(D8_CACHE, allow_pickle=True)
    feats = d8["features"].astype(np.float32)

    # Reorder per D10 convention: first 4000 dev (2000 real + 2000 fake), then
    # 839 lockbox (414 real + 425 fake). Re-derive real/fake masks.
    df_lock = pd.read_parquet(LOCKBOX_PARQUET)
    df_lock = df_lock[df_lock["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    dev_df = df_lock[df_lock["split"] == "dev"].copy()
    dev_real = dev_df[dev_df["label"] == "real"]
    dev_fake = dev_df[dev_df["label"] == "fake"]
    rng = np.random.default_rng(seed=42)
    dev_real_idx = rng.choice(len(dev_real), size=2000, replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=2000, replace=False)
    dev_sample = pd.concat(
        [dev_real.iloc[dev_real_idx], dev_fake.iloc[dev_fake_idx]]
    ).reset_index(drop=True)
    lb_df = df_lock[df_lock["split"] == "lockbox"].copy().reset_index(drop=True)

    dev_labels = (dev_sample["label"] == "fake").astype(np.int64).to_numpy()
    lb_labels = (lb_df["label"] == "fake").astype(np.int64).to_numpy()
    clip_dev = feats[:4000]
    clip_lb = feats[4000:]
    assert clip_dev.shape[0] == len(dev_sample)
    assert clip_lb.shape[0] == len(lb_df)
    dev_real_feats = clip_dev[dev_labels == 0]
    lb_real_feats = clip_lb[lb_labels == 0]

    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([dev_real_feats, lb_real_feats], axis=0).astype(np.float64)
    y = np.concatenate(
        [np.zeros(len(dev_real_feats)), np.ones(len(lb_real_feats))]
    ).astype(np.int64)
    X = l2_normalize_rows(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1, class_weight="balanced")
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    acc = float(clf.score(X, y))
    coef_norm = float(np.linalg.norm(w))
    w_hat = w / (coef_norm + 1e-12)
    logger.info(
        "KLIEP refit: acc=%.4f |w|=%.4f b=%.4f (n_dev=%d n_lb=%d)",
        acc, coef_norm, b, len(dev_real_feats), len(lb_real_feats),
    )
    return w_hat, b, acc


def compute_kliep_projections(
    feats_clean: np.ndarray,
    feats_teams: np.ndarray,
    meta_clean: pd.DataFrame,
    meta_teams: pd.DataFrame,
    w_hat: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute <feat(teams)-feat(clean), w_hat> per pair, on first frames.

    Returns (projections array, per-pair dataframe with metadata).
    """
    Xc = l2_normalize_rows(feats_clean.astype(np.float64))
    Xt = l2_normalize_rows(feats_teams.astype(np.float64))
    clean_groups = meta_clean.groupby("pair_id").indices
    teams_groups = meta_teams.groupby("pair_id").indices
    common = sorted(set(clean_groups.keys()) & set(teams_groups.keys()))

    rows = []
    proj_vals = []
    for pid in common:
        ci = clean_groups[pid]
        ti = teams_groups[pid]
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        t0 = ti[np.argmin(meta_teams.iloc[ti]["frame_idx"].to_numpy())]
        diff = Xt[t0] - Xc[c0]
        proj = float(np.dot(diff, w_hat))
        proj_vals.append(proj)
        rows.append({
            "pair_id": int(pid),
            "identity_id": meta_clean.iloc[c0]["identity_id"],
            "source": meta_clean.iloc[c0]["source"],
            "projection_value": proj,
        })
    return np.asarray(proj_vals), pd.DataFrame(rows)


# ----------------------------- Smoke test -----------------------------

def _smoke_test(ckpt_dir: Path, device: torch.device, batch_size: int) -> int:
    """Single-ckpt + 4-frame synthetic-tensor sanity check of the model load,
    hook registration, forward pass, and pred["prob"] return.
    """
    key = "SlotAv2_step3500"
    gcs_uri = CKPTS[key]
    logger.info("[smoke] downloading %s", gcs_uri)
    local_ckpt = download_ckpt(gcs_uri, ckpt_dir)
    logger.info("[smoke] loading model")
    t_load = time.time()
    model = load_model(str(local_ckpt), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    logger.info("[smoke] model loaded (%.1fs); device=%s", time.time() - t_load, device)

    # Locate transformer.resblocks list
    visual = model.backbone.visual
    try:
        resblocks = visual.transformer.resblocks
        access_path = "model.backbone.visual.transformer.resblocks"
    except AttributeError:
        resblocks = visual.visual.transformer.resblocks  # type: ignore[attr-defined]
        access_path = "model.backbone.visual.visual.transformer.resblocks"
    logger.info("[smoke] resblocks: n=%d via %s", len(resblocks), access_path)

    hooks = MultiLayerHookManager(model, LAYERS)
    rng = torch.Generator().manual_seed(SEED)
    # Build a 4-frame CLIP-normalized tensor (uniform random in [0,1], then normalize)
    raw = torch.rand((4, 3, RESOLUTION, RESOLUTION), generator=rng)
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    tensor = (raw - mean) / std
    tensor = tensor.to(device)
    t_fwd = time.time()
    with torch.no_grad():
        pred = model({"image": tensor}, inference=True)
    fwd_dt = time.time() - t_fwd
    captured = hooks.flush()
    logger.info("[smoke] forward done (%.3fs)", fwd_dt)
    for L in LAYERS:
        assert L in captured, f"missing capture for layer {L}"
        c = captured[L]
        logger.info("[smoke]  L%d capture shape=%s dtype=%s", L, tuple(c.shape), c.dtype)
        assert c.shape[0] == 4, f"L{L} batch dim {c.shape[0]} != 4"
    assert isinstance(pred, dict) and "prob" in pred, f"pred missing prob: keys={list(pred.keys()) if isinstance(pred, dict) else type(pred)}"
    prob = pred["prob"]
    logger.info("[smoke] prob shape=%s dtype=%s vals=%s", tuple(prob.shape), prob.dtype, prob.detach().cpu().tolist())
    assert prob.shape == torch.Size([4]), f"prob shape {prob.shape} != [4]"

    # Time a 32-frame batch for resource budget
    raw32 = torch.rand((batch_size, 3, RESOLUTION, RESOLUTION), generator=rng)
    tensor32 = ((raw32 - mean) / std).to(device)
    t_b = time.time()
    with torch.no_grad():
        _ = model({"image": tensor32}, inference=True)
    _ = hooks.flush()
    batch_dt = time.time() - t_b
    logger.info("[smoke] %d-frame batch fwd: %.3fs (%.1f frames/s)", batch_size, batch_dt, batch_size / batch_dt)

    hooks.remove()
    logger.info("[smoke] SUCCESS")
    return 0


# ----------------------------- Driver -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "analysis/substrate_pair_geometry_2026-05-22"))
    ap.add_argument("--inventory", default=None)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES_PER_SIDE_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-keys", default=None, help="comma-separated subset of ckpt keys to run")
    ap.add_argument("--cache-dtype", default=CACHE_DTYPE_DEFAULT, choices=["float16", "float32"])
    ap.add_argument("--smoke", action="store_true", help="smoke test only: load 1 ckpt + 4-frame synthetic tensor")
    ap.add_argument("--delete-ckpts-after", action="store_true",
                    help="delete each ckpt file from disk after its forward pass completes (saves disk)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    setup_logging(out_dir)
    feats_dir = out_dir / "feats"
    scores_dir = out_dir / "scores"
    ckpt_dir = out_dir / "ckpts"
    feats_dir.mkdir(exist_ok=True)
    scores_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    inv = Path(args.inventory) if args.inventory else (out_dir / "inventory_manifest.csv")
    assert inv.exists(), f"missing inventory: {inv}"

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("device=%s", device)

    if args.smoke:
        return _smoke_test(ckpt_dir, device, args.batch_size)

    # Frame manifests + caches
    manifest = select_pair_frames(inv, args.n_frames, SEED)
    manifest.to_parquet(out_dir / "_frame_manifest.parquet")
    cache_clean_pt = out_dir / "_cache_frames_clean.pt"
    cache_clean_meta = out_dir / "_cache_frames_meta_clean.parquet"
    cache_teams_pt = out_dir / "_cache_frames_teams.pt"
    cache_teams_meta = out_dir / "_cache_frames_meta_teams.parquet"
    tensor_clean, meta_clean = cache_frames_for_side(
        manifest, "clean", cache_clean_pt, cache_clean_meta, cache_dtype=args.cache_dtype,
    )
    tensor_teams, meta_teams = cache_frames_for_side(
        manifest, "teams", cache_teams_pt, cache_teams_meta, cache_dtype=args.cache_dtype,
    )
    logger.info(
        "tensor shapes: clean=%s teams=%s",
        tuple(tensor_clean.shape), tuple(tensor_teams.shape),
    )

    # KLIEP axis (independent of ckpts; computed on CLIP-frozen cache)
    w_hat, b, kliep_acc = fit_kliep_axis()
    np.save(out_dir / "_kliep_w_hat.npy", w_hat)
    with open(out_dir / "_kliep_metadata.json", "w") as f:
        json.dump({"b": b, "accuracy": kliep_acc, "w_norm": 1.0}, f, indent=2)

    # Iterate over checkpoints
    ckpt_keys = (
        [k.strip() for k in args.ckpt_keys.split(",")] if args.ckpt_keys else list(CKPTS.keys())
    )

    cell_rows: List[Dict] = []
    kliep_rows: List[Dict] = []

    for ckpt_key in ckpt_keys:
        gcs_uri = CKPTS[ckpt_key]
        logger.info("=" * 60)
        logger.info("CKPT %s -> %s", ckpt_key, gcs_uri)
        try:
            local_ckpt = download_ckpt(gcs_uri, ckpt_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("download failed for %s: %s", ckpt_key, e)
            continue

        try:
            model = load_model(str(local_ckpt), str(DETECTOR_CFG), str(TRAIN_CFG), device)
        except Exception as e:  # noqa: BLE001
            logger.error("model load failed for %s: %s", ckpt_key, e)
            continue

        # Register multi-layer hooks (4 hooks, single forward per batch)
        hooks = MultiLayerHookManager(model, LAYERS)

        # Determine whether per-side feature caches already exist.
        all_layer_caches_exist = lambda side: all(
            (feats_dir / f"{ckpt_key}_L{L}_{side}.npy").exists() for L in LAYERS
        )
        scores_cache_exists = lambda side: (scores_dir / f"{ckpt_key}_{side}.npy").exists()

        feats_clean_per_L: Dict[int, np.ndarray] = {}
        feats_teams_per_L: Dict[int, np.ndarray] = {}
        probs_clean: np.ndarray
        probs_teams: np.ndarray

        if all_layer_caches_exist("clean") and scores_cache_exists("clean"):
            for L in LAYERS:
                feats_clean_per_L[L] = np.load(feats_dir / f"{ckpt_key}_L{L}_clean.npy")
            probs_clean = np.load(scores_dir / f"{ckpt_key}_clean.npy")
            logger.info("[%s] clean cache hit", ckpt_key)
        else:
            t0 = time.time()
            feats_per_L, probs = forward_extract(model, hooks, tensor_clean, device, args.batch_size)
            logger.info("[%s] clean forward done (%.1fs)", ckpt_key, time.time() - t0)
            for L, arr in feats_per_L.items():
                np.save(feats_dir / f"{ckpt_key}_L{L}_clean.npy", arr)
                feats_clean_per_L[L] = arr
            np.save(scores_dir / f"{ckpt_key}_clean.npy", probs)
            probs_clean = probs

        if all_layer_caches_exist("teams") and scores_cache_exists("teams"):
            for L in LAYERS:
                feats_teams_per_L[L] = np.load(feats_dir / f"{ckpt_key}_L{L}_teams.npy")
            probs_teams = np.load(scores_dir / f"{ckpt_key}_teams.npy")
            logger.info("[%s] teams cache hit", ckpt_key)
        else:
            t0 = time.time()
            feats_per_L, probs = forward_extract(model, hooks, tensor_teams, device, args.batch_size)
            logger.info("[%s] teams forward done (%.1fs)", ckpt_key, time.time() - t0)
            for L, arr in feats_per_L.items():
                np.save(feats_dir / f"{ckpt_key}_L{L}_teams.npy", arr)
                feats_teams_per_L[L] = arr
            np.save(scores_dir / f"{ckpt_key}_teams.npy", probs)
            probs_teams = probs

        # Compute metrics per layer
        rng = np.random.default_rng(SEED)
        for L in LAYERS:
            m = compute_cell_metrics(
                feats_clean_per_L[L],
                feats_teams_per_L[L],
                meta_clean,
                meta_teams,
                probs_clean,
                probs_teams,
                rng=rng,
            )

            if L == 11:
                proj_vals, per_pair_df = compute_kliep_projections(
                    feats_clean_per_L[L], feats_teams_per_L[L],
                    meta_clean, meta_teams, w_hat,
                )
                m["kliep_projection_mean"] = float(np.mean(proj_vals))
                m["kliep_projection_std"] = float(np.std(proj_vals, ddof=1))
                per_pair_df.insert(0, "ckpt_key", ckpt_key)
                kliep_rows.extend(per_pair_df.to_dict(orient="records"))
            else:
                m["kliep_projection_mean"] = None
                m["kliep_projection_std"] = None

            row = {
                "ckpt_key": ckpt_key,
                "layer": L,
                **m,
                "n_frames_per_side": args.n_frames,
            }
            cell_rows.append(row)
            logger.info(
                "  L%-2d  cos_pair=%.4f  cos_within=%.4f  cos_cross=%.4f  Δ=%.4f  score_r=%.4f  KLIEPμ=%s",
                L, m["cos_pair"], m["cos_within_same"], m["cos_cross_id"], m["delta_pair_vs_within"],
                m["score_corr"], "%.4f" % m["kliep_projection_mean"] if m["kliep_projection_mean"] is not None else "—",
            )

        hooks.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.delete_ckpts_after:
            try:
                local_ckpt.unlink(missing_ok=True)
                logger.info("[%s] ckpt deleted from disk: %s", ckpt_key, local_ckpt)
            except OSError as e:  # noqa: BLE001
                logger.warning("[%s] ckpt unlink failed: %s", ckpt_key, e)

    # ---- Write CSVs ----
    cells_csv = out_dir / "per_ckpt_layer_cosines.csv"
    fieldnames = [
        "ckpt_key", "layer",
        "cos_pair", "cos_pair_std", "cos_within_same", "cos_cross_id",
        "delta_pair_vs_within", "kliep_projection_mean", "kliep_projection_std",
        "score_corr", "n_pairs", "n_frames_per_side",
    ]
    with open(cells_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cell_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    logger.info("wrote %s (%d rows)", cells_csv, len(cell_rows))

    kliep_csv = out_dir / "kliep_axis_projections.csv"
    pd.DataFrame(kliep_rows).to_csv(kliep_csv, index=False)
    logger.info("wrote %s (%d rows)", kliep_csv, len(kliep_rows))

    # ---- Gate verdict ----
    sa_l11 = next(
        (r for r in cell_rows if r["ckpt_key"] == "SlotAv2_step3500" and r["layer"] == 11),
        None,
    )
    verdict = "UNKNOWN"
    verdict_reason = ""
    if sa_l11 is not None and sa_l11.get("kliep_projection_mean") is not None:
        delta = sa_l11["delta_pair_vs_within"]
        kliep_abs = abs(sa_l11["kliep_projection_mean"])
        if delta >= -0.02 and kliep_abs <= 0.2:
            verdict = "NO FULCRUM"
        elif delta <= -0.05 and kliep_abs >= 0.4:
            verdict = "STRONG FULCRUM"
        else:
            verdict = "AMBIGUOUS"
        verdict_reason = f"delta={delta:+.4f}, |kliep_proj_mean|={kliep_abs:.4f}"
    logger.info("Phase 0 gate verdict: %s   (%s)", verdict, verdict_reason)

    # Persist gate
    with open(out_dir / "_gate_verdict.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "reason": verdict_reason,
            "slot_a_v2_l11": sa_l11,
            "kliep_axis_acc": kliep_acc,
        }, f, indent=2, default=str)

    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
