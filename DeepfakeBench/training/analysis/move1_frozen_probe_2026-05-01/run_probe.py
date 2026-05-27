"""Move 1 — frozen-feature linear probe.

Question: is the visomaster bucket gap (eval-bucket viso recall << train-bucket
viso AUC for P8A) reflective of a real bucket-distribution gap, or is it
dominantly identity-confounded?

Method:
  1. Sample ~150 fake + ~150 real frames from training-bucket viso
     (gs://live-deepfake-methods-real-and-fake-frames-cropped) — one frame
     per sample to maximize identity diversity.
  2. Reuse the existing eval-bucket cache (sampled_frames.csv at
     analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/
     and matching P8A features at
     analysis/_features_cache_2026-04-30/triptych_features__P8A__n800.npz).
     The eval bucket frames are exclusively from
     gs://teams-faces-data-test-2914-fake-4420-real-feb-28.
  3. Extract P8A backbone features for the new training-bucket samples using
     the same model (final-layer `feat` from outputs of model({...},
     inference=True)) and the same checkpoint
     (value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth).
  4. Train logistic regression to predict (training_bucket vs eval_bucket)
     using GroupShuffleSplit by (identity, session_id). Report AUC and
     balanced accuracy.
  5. Train identity-only-control LR (multi-class, OVR macro-average AUC) on
     the same features.
  6. Per-class fake-vs-real AUC within each bucket (separately).
  7. Save outputs/probe_results.json + outputs/probe_results.csv.

CPU/MPS only. Hard n_jobs=1 on every sklearn call. No git commits.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
EVAL_FEATS_NPZ = CACHE_DIR / "triptych_features__P8A__n800.npz"
P8A_CKPT = (
    CACHE_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

OUTPUT_DIR = REPO_ROOT / "analysis" / "move1_frozen_probe_2026-05-01" / "outputs"
FRAME_CACHE = REPO_ROOT / "analysis" / "move1_frozen_probe_2026-05-01" / "_frame_cache"

logger = logging.getLogger("move1-probe")


# -----------------------------------------------------------------------------
# Sample selection from training-bucket visomaster.
# -----------------------------------------------------------------------------
def select_train_bucket_samples(
    n_per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return list of dicts with sample_id, swap_model, frame_index, and a
    GCS URI for ONE frame from each selected sample (for both real and fake).

    Strategy: list all visomaster_* sample_ids in
    gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/, group
    by swap_model, sample uniformly across swap_models. Pull one randomly-
    chosen frame index per sample per (real,fake).
    """
    from google.cloud import storage

    client = storage.Client(project="train-cvit2")
    bucket = client.bucket("live-deepfake-methods-real-and-fake-frames-cropped")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # List manifests (one per sample). Use prefix listing.
    sample_ids: List[str] = []
    swap_model_of: Dict[str, str] = {}
    logger.info("Listing manifests under samples/visomaster_* ...")
    t0 = time.time()
    for blob in bucket.list_blobs(prefix="samples/visomaster_"):
        if not blob.name.endswith("manifest.json"):
            continue
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        sid = parts[1]
        sample_ids.append(sid)
        # extract swap model: visomaster_<model>_<num>
        rem = sid[len("visomaster_"):]
        first_us = rem.find("_")
        sm = rem[:first_us] if first_us > 0 else "UNKNOWN"
        swap_model_of[sid] = sm
        if len(sample_ids) % 1000 == 0:
            logger.info("  listed %d sample manifests so far (%.1fs)", len(sample_ids), time.time() - t0)

    logger.info("Total visomaster sample manifests: %d (%.1fs)",
                len(sample_ids), time.time() - t0)

    # Stratify by swap model.
    by_sm: Dict[str, List[str]] = {}
    for sid in sample_ids:
        by_sm.setdefault(swap_model_of[sid], []).append(sid)
    swap_models = sorted(by_sm.keys())
    logger.info("Swap models: %s", {sm: len(by_sm[sm]) for sm in swap_models})

    # We want n_per_class samples for fake AND for real. Each sample has a
    # paired real and fake side, so we can use the SAME sample for both. But
    # we want identity diversity, so we'll sample 2*n_per_class distinct
    # samples, half giving fake frames and half real frames.
    n_total = 2 * n_per_class
    per_sm = max(1, n_total // len(swap_models))
    chosen: List[str] = []
    for sm in swap_models:
        pool = sorted(by_sm[sm])
        rng.shuffle(pool)
        chosen.extend(pool[:per_sm])
    rng.shuffle(chosen)
    chosen = chosen[:n_total]
    logger.info("Selected %d distinct samples", len(chosen))

    # Half become fake, half real.
    half = len(chosen) // 2
    fake_sids = chosen[:half]
    real_sids = chosen[half:]
    rows: List[Dict[str, Any]] = []
    for sid in fake_sids:
        # Pick one frame index 0..15.
        fi = int(np_rng.integers(0, 16))
        rows.append(
            {
                "sample_id": sid,
                "swap_model": swap_model_of[sid],
                "side": "fake",
                "frame_index": fi,
                "gcs_uri": (
                    f"gs://live-deepfake-methods-real-and-fake-frames-cropped/"
                    f"samples/{sid}/frames/fake/frame_{fi:04d}.png"
                ),
            }
        )
    for sid in real_sids:
        fi = int(np_rng.integers(0, 16))
        rows.append(
            {
                "sample_id": sid,
                "swap_model": swap_model_of[sid],
                "side": "real",
                "frame_index": fi,
                "gcs_uri": (
                    f"gs://live-deepfake-methods-real-and-fake-frames-cropped/"
                    f"samples/{sid}/frames/real/frame_{fi:04d}.png"
                ),
            }
        )
    return rows


def _identity_from_sample_id(sample_id: str) -> str:
    """Sample_ids look like visomaster_<SwapModel>_<NUM>. The NUM segment is
    the original-video numeric ID (each NUM corresponds to one source clip
    per the pipeline), so it's a coarse 'identity' proxy across swap models.
    """
    rem = sample_id[len("visomaster_"):] if sample_id.startswith("visomaster_") else sample_id
    last_us = rem.rfind("_")
    if last_us > 0:
        return rem[last_us + 1:]
    return sample_id


def fetch_frame(uri: str, cache_dir: Path) -> Optional[Path]:
    """Download a single GCS frame to local cache and return path. Returns
    None on missing/failed download.
    """
    from google.cloud import storage

    if not uri.startswith("gs://"):
        return None
    bucket_name, blob_name = uri[5:].split("/", 1)
    h = "".join(c if c.isalnum() else "_" for c in blob_name)[-200:]
    local = cache_dir / h
    if local.exists():
        return local
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project="train-cvit2")
    blob = client.bucket(bucket_name).blob(blob_name)
    try:
        if not blob.exists(client=client):
            logger.warning("Missing blob: %s", uri)
            return None
        blob.download_to_filename(str(local))
        return local
    except Exception as exc:
        logger.warning("Download failed %s: %s", uri, exc)
        return None


# -----------------------------------------------------------------------------
# Model loading + feature extraction.
# -----------------------------------------------------------------------------
def load_p8a_model(device: torch.device) -> torch.nn.Module:
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)
    ckpt = torch.load(str(P8A_CKPT), map_location=device, weights_only=False)
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


def load_and_preprocess(local_path: Path, resolution: int = 224) -> Optional[torch.Tensor]:
    import cv2

    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_feats_from_paths(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract P8A `feat` outputs (512-dim per docs). Returns (feats,
    valid_idx_into_paths)."""
    feats: List[np.ndarray] = []
    valid: List[int] = []
    pending: List[Tuple[int, torch.Tensor]] = []
    for i, p in enumerate(paths):
        t = load_and_preprocess(p) if p is not None else None
        if t is None:
            continue
        pending.append((i, t))
    for j in range(0, len(pending), batch_size):
        chunk = pending[j : j + batch_size]
        batch_idx = [c[0] for c in chunk]
        batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model({"image": batch}, inference=True)
            f = outputs["feat"].detach().cpu().to(torch.float32).numpy()
        feats.append(f)
        valid.extend(batch_idx)
    feats_arr = (
        np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)
    )
    return feats_arr, np.array(valid, dtype=np.int64)


# -----------------------------------------------------------------------------
# Probe.
# -----------------------------------------------------------------------------
def run_grouped_lr_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """K-fold-style grouped split. Returns per-split metrics + overall.
    Hard n_jobs=1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    splits = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)

    rows: List[Dict[str, Any]] = []
    for split_idx, (tr, te) in enumerate(splits.split(X, y, groups=groups)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        # Verify zero group leakage.
        tr_groups = set(groups[tr].tolist())
        te_groups = set(groups[te].tolist())
        assert not (tr_groups & te_groups), "GROUP LEAK"
        clf = LogisticRegression(
            max_iter=2000, C=1.0, n_jobs=1, solver="liblinear", random_state=seed
        )
        clf.fit(Xtr, y[tr])
        pr = clf.decision_function(Xte)
        try:
            auc = float(roc_auc_score(y[te], pr))
        except Exception:
            auc = float("nan")
        bac = float(balanced_accuracy_score(y[te], (pr > 0).astype(int)))
        rows.append(
            {
                "split": split_idx,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_train_groups": len(tr_groups),
                "n_test_groups": len(te_groups),
                "auc": auc,
                "bal_acc": bac,
                "pos_frac_train": float(y[tr].mean()),
                "pos_frac_test": float(y[te].mean()),
            }
        )

    aucs = [r["auc"] for r in rows if not np.isnan(r["auc"])]
    bacs = [r["bal_acc"] for r in rows]
    return {
        "splits": rows,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "bal_acc_mean": float(np.mean(bacs)),
        "bal_acc_std": float(np.std(bacs)),
    }


def run_identity_control(
    X: np.ndarray,
    identity_labels: np.ndarray,
    seed: int = 0,
    min_per_class: int = 4,
) -> Dict[str, Any]:
    """Multi-class LR predicting identity. Reports macro-average OVR AUC on a
    held-out 30% split (NOT grouped — this IS the identity probe).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.preprocessing import label_binarize

    counts = Counter(identity_labels.tolist())
    keep_classes = [c for c, n in counts.items() if n >= min_per_class]
    mask = np.isin(identity_labels, keep_classes)
    if mask.sum() < 8 or len(keep_classes) < 2:
        return {
            "n_classes_kept": len(keep_classes),
            "n_samples_kept": int(mask.sum()),
            "auc_macro_ovr": None,
            "note": "insufficient identities with >= min_per_class samples",
        }
    Xs = X[mask]
    ys = identity_labels[mask]
    le = LabelEncoder().fit(ys)
    ye = le.transform(ys)
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, ye, test_size=0.3, stratify=ye, random_state=seed
        )
    except ValueError:
        # not enough per class to stratify
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, ye, test_size=0.3, random_state=seed
        )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, n_jobs=1, solver="lbfgs", multi_class="ovr",
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(Xte)
    else:
        scores = clf.predict_proba(Xte)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    try:
        Y_onehot = label_binarize(yte, classes=np.arange(len(le.classes_)))
        if Y_onehot.shape[1] == 1:
            # binary case
            auc = float(roc_auc_score(yte, scores[:, -1]))
        else:
            auc = float(
                roc_auc_score(Y_onehot, scores, average="macro", multi_class="ovr")
            )
    except Exception as exc:
        logger.warning("identity-OVR AUC failed: %s", exc)
        auc = None
    return {
        "n_classes_kept": int(len(le.classes_)),
        "n_samples_kept": int(len(ys)),
        "auc_macro_ovr": auc,
        "min_per_class": int(min_per_class),
    }


def run_within_bucket_fake_real_auc(
    X: np.ndarray,
    label_fake: np.ndarray,  # 1=fake, 0=real
    bucket_label: np.ndarray,  # 0=eval, 1=train
    groups: np.ndarray,
    seed: int = 0,
) -> Dict[str, Any]:
    """For each bucket, fit grouped-CV LR fake-vs-real and report mean AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    out: Dict[str, Any] = {}
    for b_name, b_val in (("eval", 0), ("train", 1)):
        m = bucket_label == b_val
        if m.sum() < 20:
            out[b_name] = {"auc_mean": None, "note": "too few samples"}
            continue
        Xb = X[m]
        yb = label_fake[m]
        gb = groups[m]
        if yb.sum() < 4 or (yb == 0).sum() < 4:
            out[b_name] = {"auc_mean": None, "note": "imbalanced"}
            continue
        try:
            splits = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
            aucs = []
            for tr, te in splits.split(Xb, yb, groups=gb):
                if len(set(yb[te].tolist())) < 2:
                    continue
                scaler = StandardScaler().fit(Xb[tr])
                Xtr = scaler.transform(Xb[tr])
                Xte = scaler.transform(Xb[te])
                clf = LogisticRegression(
                    max_iter=2000, C=1.0, n_jobs=1, solver="liblinear",
                    random_state=seed,
                )
                clf.fit(Xtr, yb[tr])
                aucs.append(float(roc_auc_score(yb[te], clf.decision_function(Xte))))
            out[b_name] = {
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_std": float(np.std(aucs)) if aucs else None,
                "n_splits": len(aucs),
                "n": int(m.sum()),
                "n_fake": int(yb.sum()),
                "n_real": int((yb == 0).sum()),
            }
        except Exception as exc:
            out[b_name] = {"auc_mean": None, "note": f"error: {exc}"}
    return out


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=150,
                    help="Num training-bucket fake AND real samples (each).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_download_workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="Force re-extract feats")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --------------------------------------------------------------------
    # 1. Load eval-bucket cached features + metadata.
    # --------------------------------------------------------------------
    logger.info("Loading eval-bucket features from %s", EVAL_FEATS_NPZ)
    eval_npz = np.load(EVAL_FEATS_NPZ, allow_pickle=False)
    eval_feats = eval_npz["features"].astype(np.float32)
    eval_valid = eval_npz["valid_idx"]
    logger.info("eval_feats: %s", eval_feats.shape)
    df_eval = pd.read_csv(SAMPLED_CSV, low_memory=False)
    df_eval_valid = df_eval.iloc[eval_valid].reset_index(drop=True)
    assert len(df_eval_valid) == len(eval_feats), \
        f"eval mismatch: {len(df_eval_valid)} vs {len(eval_feats)}"
    logger.info("eval rows: %d, identities: %d, sessions: %d",
                len(df_eval_valid),
                df_eval_valid["identity_key"].nunique(),
                df_eval_valid["session_id"].nunique())

    # --------------------------------------------------------------------
    # 2. Pick training-bucket samples (and pre-cached metadata if exists).
    # --------------------------------------------------------------------
    train_meta_csv = OUTPUT_DIR / "train_bucket_samples.csv"
    if train_meta_csv.exists() and not args.rebuild:
        logger.info("Loading train-bucket sample list from cache: %s", train_meta_csv)
        df_train_meta = pd.read_csv(train_meta_csv)
    else:
        rows = select_train_bucket_samples(
            n_per_class=args.n_per_class, seed=args.seed,
        )
        df_train_meta = pd.DataFrame(rows)
        df_train_meta.to_csv(train_meta_csv, index=False)
    logger.info("train-bucket sample rows: %d", len(df_train_meta))

    # --------------------------------------------------------------------
    # 3. Download frames (parallel).
    # --------------------------------------------------------------------
    train_feats_npz = OUTPUT_DIR / "train_bucket_feats.npz"
    if train_feats_npz.exists() and not args.rebuild:
        logger.info("Loading cached train-bucket feats: %s", train_feats_npz)
        cached = np.load(train_feats_npz, allow_pickle=False)
        train_feats = cached["features"].astype(np.float32)
        train_valid = cached["valid_idx"]
        df_train_meta_valid = df_train_meta.iloc[train_valid].reset_index(drop=True)
    else:
        from concurrent.futures import ThreadPoolExecutor
        logger.info("Downloading %d frames to %s", len(df_train_meta), FRAME_CACHE)
        local_paths: List[Optional[Path]] = [None] * len(df_train_meta)
        with ThreadPoolExecutor(max_workers=args.max_download_workers) as pool:
            futures = {
                pool.submit(fetch_frame, row.gcs_uri, FRAME_CACHE): i
                for i, row in enumerate(df_train_meta.itertuples(index=False))
            }
            n_done = 0
            for fut in futures:
                i = futures[fut]
                local_paths[i] = fut.result()
                n_done += 1
                if n_done % 50 == 0 or n_done == len(df_train_meta):
                    logger.info("  downloaded %d/%d", n_done, len(df_train_meta))
        df_train_meta["local_path"] = [str(p) if p else "" for p in local_paths]
        df_train_meta.to_csv(train_meta_csv, index=False)
        valid_paths = [p for p in local_paths]
        # 4. Load model + extract feats.
        logger.info("Loading P8A model from %s", P8A_CKPT)
        model = load_p8a_model(device)
        train_feats, train_valid = extract_feats_from_paths(
            model, valid_paths, device, batch_size=args.batch_size,
        )
        np.savez_compressed(train_feats_npz, features=train_feats, valid_idx=train_valid)
        df_train_meta_valid = df_train_meta.iloc[train_valid].reset_index(drop=True)
        del model
    logger.info("train_feats: %s", train_feats.shape)

    # --------------------------------------------------------------------
    # 5. Build the unified probe arrays.
    # --------------------------------------------------------------------
    # Bucket label: 0 = eval-bucket (teams-faces-data-test-2914-...),
    #               1 = train-bucket viso (live-deepfake-methods-...-cropped).
    # Identity:    eval -> identity_key column; train -> sample_id NUM segment.
    # Session:     eval -> session_id; train -> sample_id (each unique).
    # Group key:   (identity_key, session_id) for eval. For train-bucket, group
    #              key = identity (NUM) since each visomaster sample has its
    #              own session by construction.
    eval_X = eval_feats
    eval_y = np.zeros(len(eval_X), dtype=np.int32)  # bucket label 0
    eval_id = df_eval_valid["identity_key"].astype(str).to_numpy()
    eval_sess = df_eval_valid["session_id"].astype(str).to_numpy()
    eval_method = df_eval_valid["method"].astype(str).to_numpy()
    eval_label = df_eval_valid["label"].astype(str).to_numpy()  # real / fake
    # group is identity__session
    eval_group = np.array(
        [f"E_{i}__{s}" for i, s in zip(eval_id, eval_sess)], dtype=object
    )

    train_X = train_feats
    train_y = np.ones(len(train_X), dtype=np.int32)  # bucket label 1
    df_train_meta_valid["identity"] = df_train_meta_valid["sample_id"].apply(
        _identity_from_sample_id
    )
    train_id = df_train_meta_valid["identity"].astype(str).to_numpy()
    train_sm = df_train_meta_valid["swap_model"].astype(str).to_numpy()
    train_side = df_train_meta_valid["side"].astype(str).to_numpy()
    train_method = np.where(
        train_side == "real",
        np.array(["viso_real"] * len(train_side), dtype=object),
        np.array([f"viso_fake_{m}" for m in train_sm], dtype=object),
    )
    train_label = np.where(train_side == "real", "real", "fake")
    train_group = np.array(
        [f"T_{i}" for i in train_id], dtype=object
    )

    X = np.concatenate([eval_X, train_X], axis=0)
    y = np.concatenate([eval_y, train_y], axis=0)
    groups = np.concatenate([eval_group, train_group], axis=0)
    method = np.concatenate([eval_method, train_method], axis=0)
    label_str = np.concatenate([eval_label, train_label], axis=0)
    label_fake = (label_str == "fake").astype(np.int32)
    identity = np.concatenate([eval_id, train_id], axis=0)
    bucket_str = np.array(
        ["eval"] * len(eval_X) + ["train"] * len(train_X), dtype=object
    )

    logger.info("Combined: X=%s, y=%s; eval=%d, train=%d",
                X.shape, y.shape, int((y == 0).sum()), int((y == 1).sum()))
    logger.info("Eval identities: %d (%d sessions)",
                int(pd.Series(eval_id).nunique()),
                int(pd.Series(eval_sess).nunique()))
    logger.info("Train identities: %d (%d swap_models, sides=%s)",
                int(pd.Series(train_id).nunique()),
                int(pd.Series(train_sm).nunique()),
                Counter(train_side))

    # --------------------------------------------------------------------
    # 6. Bucket-discrimination LR (the headline question).
    # --------------------------------------------------------------------
    logger.info("Running grouped LR for bucket discrimination ...")
    bucket_probe = run_grouped_lr_probe(X, y, groups, n_splits=5, seed=args.seed)
    logger.info("Bucket-LR AUC mean=%.4f ± %.4f",
                bucket_probe["auc_mean"], bucket_probe["auc_std"])

    # --------------------------------------------------------------------
    # 7. Identity-only control LR.
    # --------------------------------------------------------------------
    logger.info("Running identity-only control LR (multi-class OVR) ...")
    id_probe = run_identity_control(X, identity, seed=args.seed, min_per_class=4)
    logger.info("Identity OVR AUC: %s (n_classes_kept=%s n_samples=%s)",
                id_probe.get("auc_macro_ovr"),
                id_probe.get("n_classes_kept"),
                id_probe.get("n_samples_kept"))

    # Identity probe variant filtered to n>=2 (broader coverage):
    id_probe_n2 = run_identity_control(X, identity, seed=args.seed, min_per_class=2)
    logger.info("Identity OVR AUC (min_per_class=2): %s (kept_classes=%s, kept_n=%s)",
                id_probe_n2.get("auc_macro_ovr"),
                id_probe_n2.get("n_classes_kept"),
                id_probe_n2.get("n_samples_kept"))

    # --------------------------------------------------------------------
    # 8. Per-bucket fake-vs-real AUC.
    # --------------------------------------------------------------------
    logger.info("Running within-bucket fake/real LR (grouped) ...")
    within = run_within_bucket_fake_real_auc(
        X, label_fake, y, groups, seed=args.seed,
    )
    for b, r in within.items():
        logger.info("  bucket=%s: %s", b, r)

    # --------------------------------------------------------------------
    # 9. Verdict per PLAN.md §9 outcome ladder.
    # --------------------------------------------------------------------
    bucket_auc = bucket_probe["auc_mean"]
    id_auc = id_probe.get("auc_macro_ovr")

    # Outcome ladder:
    #   POSITIVE: bucket-AUC > 0.80 AND id-AUC < 0.70
    #   AMBIGUOUS: id-AUC >= 0.70
    #   NEGATIVE: bucket-AUC < 0.65
    #   else: AMBIGUOUS-MIDLINE
    if id_auc is not None and id_auc >= 0.70:
        verdict = "AMBIGUOUS"
        verdict_reason = (
            f"identity-only control AUC={id_auc:.3f} >= 0.70 "
            f"=> bucket probe is plausibly identity-confounded"
        )
    elif bucket_auc > 0.80:
        verdict = "POSITIVE"
        verdict_reason = (
            f"bucket-AUC={bucket_auc:.3f} > 0.80 AND identity AUC={id_auc} < 0.70"
        )
    elif bucket_auc < 0.65:
        verdict = "NEGATIVE"
        verdict_reason = (
            f"bucket-AUC={bucket_auc:.3f} < 0.65 => bucket gap not dominant in features"
        )
    else:
        verdict = "AMBIGUOUS-MIDLINE"
        verdict_reason = (
            f"bucket-AUC={bucket_auc:.3f} in [0.65, 0.80] and id_auc={id_auc}"
        )

    summary: Dict[str, Any] = {
        "date": "2026-05-01",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "ckpt": str(P8A_CKPT.relative_to(REPO_ROOT)),
        "device": str(device),
        "feature_dim": int(X.shape[1]),
        "n_eval_frames": int((y == 0).sum()),
        "n_train_frames": int((y == 1).sum()),
        "n_eval_identities": int(pd.Series(eval_id).nunique()),
        "n_train_identities": int(pd.Series(train_id).nunique()),
        "n_eval_groups": int(pd.Series(eval_group).nunique()),
        "n_train_groups": int(pd.Series(train_group).nunique()),
        "bucket_probe": bucket_probe,
        "identity_probe_min4": id_probe,
        "identity_probe_min2": id_probe_n2,
        "within_bucket_fake_vs_real": within,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": {
            "bucket_auc_mean": bucket_auc,
            "bucket_auc_std": bucket_probe["auc_std"],
            "identity_ovr_auc_min4": id_probe.get("auc_macro_ovr"),
            "identity_ovr_auc_min2": id_probe_n2.get("auc_macro_ovr"),
            "fake_vs_real_eval": within.get("eval", {}).get("auc_mean"),
            "fake_vs_real_train": within.get("train", {}).get("auc_mean"),
        },
        "method_breakdown": {
            "eval": dict(Counter(eval_method.tolist())),
            "train": dict(Counter(train_method.tolist())),
        },
    }

    # Save outputs.
    out_json = OUTPUT_DIR / "probe_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    out_csv = OUTPUT_DIR / "probe_results.csv"
    rows: List[Dict[str, Any]] = []
    for r in bucket_probe["splits"]:
        rows.append({"probe": "bucket_grouped_LR", **r})
    for b, info in within.items():
        if isinstance(info, dict) and "auc_mean" in info:
            rows.append({"probe": f"fake_vs_real_within_{b}", **info})
    rows.append({"probe": "identity_min4", **id_probe})
    rows.append({"probe": "identity_min2", **id_probe_n2})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print()
    print("=" * 78)
    print(f"MOVE 1 — frozen-feature linear probe — VERDICT: {verdict}")
    print(f"  reason: {verdict_reason}")
    print(f"  bucket-disc AUC: {bucket_probe['auc_mean']:.4f} ± {bucket_probe['auc_std']:.4f}  (n=5 grouped splits)")
    if id_probe.get("auc_macro_ovr") is not None:
        print(f"  identity-control OVR AUC (min4): {id_probe['auc_macro_ovr']:.4f}  ({id_probe['n_classes_kept']} classes)")
    if id_probe_n2.get("auc_macro_ovr") is not None:
        print(f"  identity-control OVR AUC (min2): {id_probe_n2['auc_macro_ovr']:.4f}  ({id_probe_n2['n_classes_kept']} classes)")
    for b, info in within.items():
        if isinstance(info, dict) and info.get("auc_mean") is not None:
            print(f"  within-bucket fake-vs-real ({b}): {info['auc_mean']:.4f} ± {info['auc_std']:.4f}  (n={info.get('n')}, fake={info.get('n_fake')}, real={info.get('n_real')})")
    print("=" * 78)
    print(f"results: {out_json}")
    print(f"csv:     {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
