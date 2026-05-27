"""Score T4_LAMBDA1_TOP_N_STEP10500 on a HDTF subsample (8 load-bearing cells).

Strategy: deterministically sample N_VIDEOS_PER_CELL videos from each cell's
cached P8A scoring (using the existing local `scores_cache/` CSVs), download
those frames from GCS in parallel, run T4 inference, write per-frame results.

The subsample is deterministic (seed=42 on sorted video_ids), so we can
recompute P8A/E2B/T4 AUC on the identical subset.

Output: t4_hdtf_per_frame.csv (and also writes a copy of the matched
P8A/E2B scores for the same subsample, for easy joint AUC computation).
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD, load_model  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("t4_hdtf")

THIS_DIR = Path(__file__).resolve().parent
SCORES_CACHE = REPO_ROOT / "analysis" / "iq_shortcut_decomp_2026-05-08" / "scores_cache"
RAW_REPORTS = REPO_ROOT / "analysis" / "p2_d_hdtf_2026-05-08" / "raw_reports"
T4_LOCAL = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-10" / "_ckpts_t4" / "top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

OUT_T4 = THIS_DIR / "t4_hdtf_per_frame.csv"
OUT_REF = THIS_DIR / "hdtf_subsample_p8a_e2b_match.csv"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
DOWNLOAD_THREADS = 16
N_VIDEOS_PER_CELL = int(os.environ.get("N_VIDEOS_PER_CELL", "50"))
SEED = 42

# Cells from the cache that are load-bearing per the task instructions.
# (cell_key, label, source_kind)
#   source_kind ∈ {"cache","raw_reports"} indicates which folder to read.
CELLS = [
    # (suite_key, label, source, p8a_filename_stem)
    ("hdtf_real_clean_dev", 0, "cache", "P8A_REFERENCE_STEP5000__hdtf_real_clean_dev.csv"),
    ("hdtf_fake_clean_dev", 1, "cache", "P8A_REFERENCE_STEP5000__hdtf_fake_clean_dev.csv"),
    ("hdtf_real_teams_dev", 0, "cache", "P8A_REFERENCE_STEP5000__hdtf_real_teams_dev.csv"),
    ("hdtf_fake_teams_dev", 1, "cache", "P8A_REFERENCE_STEP5000__hdtf_fake_teams_dev.csv"),
    ("hdtf_real_teams_lockbox", 0, "cache", "P8A_REFERENCE_STEP5000__hdtf_real_teams_lockbox.csv"),
    ("hdtf_fake_teams_lockbox", 1, "cache", "P8A_REFERENCE_STEP5000__hdtf_fake_teams_lockbox.csv"),
    # visomaster_enhanced_teams cells live in raw_reports (HDTF run 2026-05-08).
    # Frame paths in those reports are gs:// URIs that include hdtf_visomaster_cropped_frames_teams,
    # so they're identical-substrate to the cache ones; we add them for fake-recall context.
    ("hdtf_viso_enh_teams_dev", 1, "raw_reports", "proper_visomaster_enhanced_teams_dev_p8a_reference_step5000_frames_report.csv"),
    ("hdtf_viso_enh_teams_lockbox", 1, "raw_reports", "proper_visomaster_enhanced_teams_lockbox_p8a_reference_step5000_frames_report.csv"),
]

# Companion E2B + P2D paths used to extract matching reference scores for the subsample.
E2B_FILE_FOR_CELL = {
    "hdtf_real_clean_dev": ("cache", "E2B_TOP_N_STEP3200__hdtf_real_clean_dev.csv"),
    "hdtf_fake_clean_dev": ("cache", "E2B_TOP_N_STEP3200__hdtf_fake_clean_dev.csv"),
    "hdtf_real_teams_dev": ("cache", "E2B_TOP_N_STEP3200__hdtf_real_teams_dev.csv"),
    "hdtf_fake_teams_dev": ("cache", "E2B_TOP_N_STEP3200__hdtf_fake_teams_dev.csv"),
    "hdtf_real_teams_lockbox": ("cache", "E2B_TOP_N_STEP3200__hdtf_real_teams_lockbox.csv"),
    "hdtf_fake_teams_lockbox": ("cache", "E2B_TOP_N_STEP3200__hdtf_fake_teams_lockbox.csv"),
    "hdtf_viso_enh_teams_dev": ("raw_reports", "proper_visomaster_enhanced_teams_dev_e2b_top_n_step3200_frames_report.csv"),
    "hdtf_viso_enh_teams_lockbox": ("raw_reports", "proper_visomaster_enhanced_teams_lockbox_e2b_top_n_step3200_frames_report.csv"),
}


def load_cell_table(source: str, fname: str) -> pd.DataFrame:
    if source == "cache":
        path = SCORES_CACHE / fname
    elif source == "raw_reports":
        path = RAW_REPORTS / fname
    else:
        raise ValueError(source)
    df = pd.read_csv(path)
    return df


def pick_subsample(df: pd.DataFrame, n_videos: int, seed: int) -> pd.DataFrame:
    """Sample n_videos by video_id deterministically; return all rows for those videos."""
    if "video_id" not in df.columns:
        raise ValueError(f"missing video_id; cols={df.columns.tolist()}")
    vids = sorted(df["video_id"].dropna().unique().tolist())
    rng = random.Random(seed)
    chosen = set(rng.sample(vids, min(n_videos, len(vids))))
    out = df[df["video_id"].isin(chosen)].copy()
    return out


def download_one(gcs_uri: str, local_path: Path) -> bool:
    from google.cloud import storage
    if local_path.exists():
        return True
    no_scheme = gcs_uri[5:]
    bucket_name, blob_name = no_scheme.split("/", 1)
    try:
        client = _get_thread_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        return True
    except Exception as e:
        log.warning("download-fail %s: %s", gcs_uri, e)
        return False


_thread_local = None


def _get_thread_client():
    """Per-thread GCS client (anonymous storage.Client is thread-safe but
    we still want one per thread for connection pooling)."""
    global _thread_local
    if _thread_local is None:
        import threading
        _thread_local = threading.local()
    cli = getattr(_thread_local, "cli", None)
    if cli is None:
        from google.cloud import storage
        _thread_local.cli = storage.Client()
        cli = _thread_local.cli
    return cli


def download_many(rows: list[dict], tmpdir: Path) -> list[dict]:
    """Download all gs:// frame_paths in rows to tmpdir; return rows with new 'local_path' key
    for those that succeeded."""
    tasks = []
    for r in rows:
        uri = r["frame_path"]
        if not uri.startswith("gs://"):
            r["local_path"] = uri
            tasks.append(r)
            continue
        # Use a hash of the URI to avoid collisions (cells share frame_0006.png etc.)
        suffix = uri.split("/")[-1]
        hsh = abs(hash(uri)) % (10**10)
        local_path = tmpdir / f"{hsh}_{suffix}"
        r["local_path"] = str(local_path)
        tasks.append(r)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        futs = {ex.submit(download_one, r["frame_path"], Path(r["local_path"])): r for r in tasks if r["frame_path"].startswith("gs://")}
        done = 0
        ok = 0
        for f in as_completed(futs):
            done += 1
            if f.result():
                ok += 1
            if done % 200 == 0:
                log.info("download: %d/%d ok=%d (%.1f f/s)", done, len(futs), ok,
                         done / max(time.time() - t0, 1e-3))
    log.info("download complete in %.1fs (n=%d)", time.time() - t0, len(futs))
    return [r for r in tasks if Path(r["local_path"]).exists()]


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_bgr = cv2.imread(r["local_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning("decode-fail: %s", r["local_path"])
            return torch.zeros(3, RESOLUTION, RESOLUTION), idx
        img_bgr = cv2.resize(img_bgr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def main():
    log.info("=" * 70)
    log.info("T4 on HDTF subsample (N=%d videos per cell, seed=%d)", N_VIDEOS_PER_CELL, SEED)
    log.info("device=%s", DEVICE)
    log.info("=" * 70)

    # 1. Build the subsample manifest from cached P8A files
    manifest_rows = []
    p8a_match_rows = []
    e2b_match_rows = []
    for cell_key, label, source, fname in CELLS:
        df_p8a = load_cell_table(source, fname)
        sub = pick_subsample(df_p8a, N_VIDEOS_PER_CELL, SEED)
        log.info("cell=%s n_videos_in_cell=%d sampled=%d frames=%d",
                 cell_key, df_p8a["video_id"].nunique(),
                 sub["video_id"].nunique(), len(sub))
        for _, row in sub.iterrows():
            manifest_rows.append({
                "cell_key": cell_key,
                "label": label,
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "p8a_frame_prob": row["frame_prob"],
            })
            p8a_match_rows.append({
                "cell_key": cell_key,
                "label": label,
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "p8a_frame_prob": row["frame_prob"],
            })

        # Pull matching E2B scores
        e2b_source, e2b_fname = E2B_FILE_FOR_CELL[cell_key]
        df_e2b = load_cell_table(e2b_source, e2b_fname)
        merge_keys = ["video_id", "frame_path"]
        sub_e2b = df_e2b[df_e2b["video_id"].isin(sub["video_id"].unique())]
        # Make sure we get the same (video_id, frame_path) pairs
        sub_e2b = sub_e2b[sub_e2b["frame_path"].isin(sub["frame_path"])]
        for _, row in sub_e2b.iterrows():
            e2b_match_rows.append({
                "cell_key": cell_key,
                "label": label,
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "e2b_frame_prob": row["frame_prob"],
            })

    log.info("total manifest rows: %d", len(manifest_rows))

    # 2. Download frames into tmpdir
    tmpdir = Path(tempfile.mkdtemp(prefix="t4_hdtf_"))
    log.info("tmpdir=%s", tmpdir)
    downloaded = download_many(manifest_rows, tmpdir)
    log.info("downloaded %d/%d frames", len(downloaded), len(manifest_rows))

    # 3. Run T4 inference
    log.info("loading T4 model from %s", T4_LOCAL)
    t0 = time.time()
    model = load_model(str(T4_LOCAL), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("model loaded in %.1fs", time.time() - t0)

    dataset = FrameDataset(downloaded)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
    n = len(downloaded)
    probs = np.zeros(n, dtype=np.float32)
    log.info("inference: %d frames, batch=%d", n, BATCH_SIZE)
    t0 = time.time()
    last_log = t0
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        if time.time() - last_log > 30:
            done = (batch_idx + 1) * BATCH_SIZE
            elapsed = time.time() - t0
            log.info("inf: batch=%d ~%d/%d elapsed=%.0fs (%.1f fps)",
                     batch_idx + 1, min(done, n), n, elapsed, min(done, n) / max(elapsed, 1e-3))
            last_log = time.time()
    log.info("inference done in %.1fs (%.1f fps)", time.time() - t0, n / max(time.time() - t0, 1e-3))

    # 4. Write T4 per-frame CSV
    with OUT_T4.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_key", "label", "video_id", "frame_path", "t4_frame_prob"])
        for r, p in zip(downloaded, probs):
            w.writerow([r["cell_key"], r["label"], r["video_id"], r["frame_path"], f"{float(p):.6f}"])
    log.info("wrote -> %s", OUT_T4)

    # 5. Write companion reference CSV (joins P8A + E2B for the same subsample)
    df_p8a_match = pd.DataFrame(p8a_match_rows)
    df_e2b_match = pd.DataFrame(e2b_match_rows)
    df_join = df_p8a_match.merge(df_e2b_match[["cell_key", "video_id", "frame_path", "e2b_frame_prob"]],
                                  on=["cell_key", "video_id", "frame_path"], how="left")
    df_join.to_csv(OUT_REF, index=False)
    log.info("wrote -> %s (n=%d)", OUT_REF, len(df_join))

    # 6. Cleanup tmpdir
    try:
        import shutil
        shutil.rmtree(tmpdir)
        log.info("cleaned tmpdir %s", tmpdir)
    except Exception as e:
        log.warning("tmpdir cleanup failed: %s", e)


if __name__ == "__main__":
    main()
