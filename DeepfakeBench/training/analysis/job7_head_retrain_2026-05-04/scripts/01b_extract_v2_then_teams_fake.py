#!/usr/bin/env python3
"""Job 7 — Stage 1 (resume B): extract v2 held-out FIRST, then teams_fake_all_dev.

This is a thin wrapper around 01_extract_features.py helpers that runs the
held-out v2 extraction (visomaster_enhanced_v2 manifest) and then
teams_fake_all_dev. Per task priority, v2 must come before teams_fake_all_dev
in case wall-clock truncates again.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
HERE = REPO_ROOT / "analysis" / "job7_head_retrain_2026-05-04"
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_features",
    str(HERE / "scripts" / "01_extract_features.py"),
)
ef = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(ef)  # type: ignore

FROZEN = HERE / "frozen_features"
LOG_PATH = HERE / "extract_v2_then_teamsfake.log"
V2_MANIFEST = REPO_ROOT / "arena" / "manifests" / "visomaster_enhanced_v2_manifest_2026-04-13.json"

logger = logging.getLogger("job7-extract-b")


def setup_logging():
    fh = logging.FileHandler(LOG_PATH); sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]; logger.setLevel(logging.INFO)
    # Also pipe ef's logger.
    ef.logger.handlers = [fh, sh]; ef.logger.setLevel(logging.INFO)


def extract_v2(model, device):
    out_path = FROZEN / "visomaster_enhanced_v2_p8a_features.npz"
    if out_path.exists():
        logger.info("v2 already extracted at %s — skipping", out_path)
        return
    logger.info("=" * 80)
    logger.info("Suite: visomaster_enhanced_v2 (held-out)")
    with open(V2_MANIFEST) as f:
        mani = json.load(f)
    rows = []
    for v in mani["videos"]:
        method = v.get("method", "")
        for fp in v.get("frame_paths", []):
            rows.append({
                "frame_path": fp,
                "video_id": v.get("video_id", ""),
                "label": 1,
                "method": method,
                "family_key": method,
                "frame_prob": 0.0,
            })
    df_v2 = pd.DataFrame(rows)
    logger.info("  v2 manifest: %d frames across %d videos / %d methods",
                len(df_v2), df_v2["video_id"].nunique(), df_v2["method"].nunique())
    res = ef.extract_for_suite(model, device, "visomaster_enhanced_v2", df_v2, batch_size=32)
    np.savez_compressed(
        out_path,
        features=res["features"], iq_features=res["iq_features"],
        iq_cols=np.array(ef.IQ_COLS),
        labels=res["labels"], frame_paths=res["frame_paths"],
        video_ids=res["video_ids"], identities=res["identities"],
        local_paths=res["local_paths"], family_keys=res["family_keys"],
        scores_p8a_existing=res["scores_p8a_existing"],
        suite=np.array("visomaster_enhanced_v2"),
    )
    logger.info("[visomaster_enhanced_v2] saved %s features=%s",
                out_path, res["features"].shape)


def extract_teams_fake_all_dev(model, device):
    suite = "teams_fake_all_dev"
    out_path = FROZEN / f"{suite}_p8a_features.npz"
    if out_path.exists():
        logger.info("%s already extracted at %s — skipping", suite, out_path)
        return
    logger.info("=" * 80)
    logger.info("Suite: %s", suite)
    report = ef.SUITE_REPORTS[suite]
    df = pd.read_csv(report, low_memory=False)
    logger.info("  %d rows, label counts=%s", len(df), df["label"].value_counts().to_dict())

    cached_p8a = ef.maybe_load_existing_p8a_cache()
    df["from_cache"] = df["frame_path"].isin(cached_p8a)
    n_cache_hits = int(df["from_cache"].sum())
    logger.info("  cache hits (P8A feats already extracted): %d", n_cache_hits)

    df_to_extract = df[~df["from_cache"]].reset_index(drop=True)
    if len(df_to_extract) > 0:
        t0 = time.time()
        res = ef.extract_for_suite(model, device, suite, df_to_extract, batch_size=32)
        logger.info("  forward elapsed: %.1fs", time.time() - t0)
    else:
        res = None

    df_full = df.copy()
    df_full["identity"] = [ef.identity_for(v, p) for v, p in zip(df_full["video_id"], df_full["frame_path"])]
    if n_cache_hits > 0:
        cache_uris = df_full[df_full["from_cache"]]["frame_path"].tolist()
        cache_locals = ef.download_frames(cache_uris)
        df_cache = df_full[df_full["from_cache"]].reset_index(drop=True)
        df_cache["local_path"] = [str(p) if p.exists() else "" for p in cache_locals]
        df_cache = df_cache[df_cache["local_path"] != ""].reset_index(drop=True)
        X_cache = np.stack([cached_p8a[fp] for fp in df_cache["frame_path"]], axis=0)
        iq_cache = []
        import cv2
        for lp in df_cache["local_path"]:
            img = cv2.imread(lp, cv2.IMREAD_COLOR)
            iq_cache.append(ef.compute_iq_features(img) if img is not None else np.zeros(7, dtype=np.float32))
        iq_cache = np.stack(iq_cache, axis=0)
    else:
        X_cache = np.zeros((0, 512), dtype=np.float32)
        iq_cache = np.zeros((0, 7), dtype=np.float32)
        df_cache = df_full.iloc[:0].assign(local_path="").reset_index(drop=True)

    if res is not None:
        X = np.concatenate([X_cache, res["features"]], axis=0).astype(np.float32)
        IQ = np.concatenate([iq_cache, res["iq_features"]], axis=0).astype(np.float32)
        labels = np.concatenate([
            df_cache["label"].astype(np.int32).to_numpy(), res["labels"]], axis=0)
        frame_paths = np.concatenate([
            df_cache["frame_path"].astype(str).to_numpy(), res["frame_paths"]], axis=0)
        video_ids = np.concatenate([
            df_cache["video_id"].astype(str).to_numpy(), res["video_ids"]], axis=0)
        identities = np.concatenate([
            df_cache["identity"].astype(str).to_numpy(), res["identities"]], axis=0)
        local_paths = np.concatenate([
            df_cache["local_path"].astype(str).to_numpy(), res["local_paths"]], axis=0)
        family_keys = np.concatenate([
            df_cache.get("family_key", pd.Series([""] * len(df_cache))).astype(str).to_numpy(),
            res["family_keys"]], axis=0)
        scores_p8a = np.concatenate([
            df_cache["frame_prob"].astype(np.float32).to_numpy(),
            res["scores_p8a_existing"]], axis=0)
    else:
        X = X_cache; IQ = iq_cache
        labels = df_cache["label"].astype(np.int32).to_numpy()
        frame_paths = df_cache["frame_path"].astype(str).to_numpy()
        video_ids = df_cache["video_id"].astype(str).to_numpy()
        identities = df_cache["identity"].astype(str).to_numpy()
        local_paths = df_cache["local_path"].astype(str).to_numpy()
        family_keys = df_cache.get("family_key", pd.Series([""] * len(df_cache))).astype(str).to_numpy()
        scores_p8a = df_cache["frame_prob"].astype(np.float32).to_numpy()

    np.savez_compressed(
        out_path,
        features=X, iq_features=IQ, iq_cols=np.array(ef.IQ_COLS),
        labels=labels, frame_paths=frame_paths, video_ids=video_ids,
        identities=identities, local_paths=local_paths,
        family_keys=family_keys, scores_p8a_existing=scores_p8a,
        suite=np.array(suite),
    )
    logger.info("[%s] saved %s features=%s", suite, out_path, X.shape)


def main():
    setup_logging()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("device=%s", device)
    model = ef.load_p8a_model(device)
    extract_v2(model, device)
    extract_teams_fake_all_dev(model, device)


if __name__ == "__main__":
    raise SystemExit(main())
