"""Probe 2 — Face-region attention pool CPU probe (Fallback 1).

For each of the 3 ckpts (P8A_step5000, SlotAv2_step3500, T5C_step3500):
  1) Download ckpt; load model via batch_inference_gcs.load_model.
  2) Register a forward hook on model.backbone.visual.transformer.resblocks[11]
     that captures the FULL token sequence (CLS + 196 patches in a 14×14 grid).
  3) Re-use the cached frame tensors (5475 clean + 5478 teams) and forward in
     fp32 batches; pool ONLY the face-region patches (mean-pool a 7×7 central
     subgrid of the 14×14 → 49 face patches; documented fallback).
  4) Compute the same 6 metrics from A0.2 (cos_pair, cos_within_same,
     cos_cross_id, delta, kliep_projection on frozen-CLIP axis, score_corr)
     using these face-region pooled features.
  5) Save:
       per_ckpt_face_region_cosines.csv         (3 rows, 1 per ckpt at L11)
       face_region_kliep_projections.csv        (3 × 1825 = 5475 rows)
       _probe_complete.json                     (sentinel marker, written LAST)

FALLBACK NOTE: this probe uses the centered 7×7 patch subgrid because the
cached frame metadata does not carry a face_bbox column. Documented in
`FALLBACK1_PROBE1_FACTS_2026-05-22.md`. Crops in the source bucket
`live-deepfake-methods-real-and-fake-frames-cropped/...` are already
face-cropped, so the centered subgrid is a reasonable proxy for face-region
patches; it is NOT a per-frame face-localized mask.

Usage:
    nohup python run_probe2_face_region.py \\
        > _probe2.log 2>&1 &
    disown
"""

from __future__ import annotations

import csv
import json
import logging
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

OUT_DIR = Path(__file__).resolve().parent
LAYER = 11
RESOLUTION = 224
PATCH_GRID = 14  # 224 / 16 (CLIP-B/16)
FACE_SUBGRID_RADIUS = 3  # center 7x7 of 14x14 -> 49 patches (~ center 75%)
BATCH_SIZE = 16  # reduced from 32 because we hold the full 197-token sequence
SEED = 42
N_CROSS_ID_SAMPLES = 2000

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "_probe2.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("probe2")


# -------------------- Helpers --------------------

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def download_ckpt(gcs_path: str, local_dir: Path) -> Path:
    import subprocess
    local_dir.mkdir(parents=True, exist_ok=True)
    fname = gcs_path.rsplit("/", 1)[-1]
    local = local_dir / fname
    if local.exists() and local.stat().st_size > 1_000_000:
        logger.info("[download] cached: %s", local)
        return local
    logger.info("[download] %s", gcs_path)
    t0 = time.time()
    subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local)], check=True)
    logger.info("[download] -> %s (%.1fs)", local, time.time() - t0)
    return local


class FullSequenceHook:
    """Forward hook on resblocks[L] that captures the FULL token sequence
    (CLS + 196 patches) per batch, not just CLS.

    Handles both (seq, batch, dim) and (batch, seq, dim) layouts.
    """

    def __init__(self, model: torch.nn.Module, layer: int):
        self.layer = layer
        visual = model.backbone.visual
        try:
            resblocks = visual.transformer.resblocks
        except AttributeError:
            resblocks = visual.visual.transformer.resblocks  # type: ignore[attr-defined]
        n_blocks = len(resblocks)
        assert 0 <= layer < n_blocks, f"layer {layer} out of range [0,{n_blocks})"
        self._handle = resblocks[layer].register_forward_hook(self._hook)
        self._captured: torch.Tensor = None
        logger.info("FullSequenceHook registered on layer %d (n_blocks=%d)", layer, n_blocks)

    def _hook(self, module, inputs, output):  # noqa: ARG002
        if output.dim() != 3:
            raise RuntimeError(f"unexpected resblock output dim: {output.shape}")
        # Identify layout: seq-first if output.shape[0] >= output.shape[1]
        # (197 tokens vs batch ~16, so seq-first will satisfy)
        if output.shape[0] >= output.shape[1]:
            # (seq, batch, dim) -> permute to (batch, seq, dim)
            tokens = output.permute(1, 0, 2)
        else:
            tokens = output  # already (batch, seq, dim)
        self._captured = tokens.detach()

    def flush(self) -> torch.Tensor:
        t = self._captured
        self._captured = None
        return t

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def face_region_mask_7x7(grid_size: int = PATCH_GRID, radius: int = FACE_SUBGRID_RADIUS) -> np.ndarray:
    """Return a flat boolean mask of length grid_size**2 for the centered
    (2*radius+1)^2 subgrid. Default: 7x7 = 49 patches at the center of 14x14.
    """
    center = grid_size // 2
    # 14x14 -> center indices {6, 7}. For symmetric 7x7, take [center-radius-1: center+radius]?
    # We want 7 rows: indices center-3, center-2, ..., center+3.
    # 14 has no exact center; use rows {center - radius - 1, ..., center + radius - 1}
    # i.e. rows in 14 -> {3, 4, 5, 6, 7, 8, 9} for radius=3 -> 7 rows
    lo = center - radius - 1
    hi = center + radius
    assert hi - lo == 2 * radius + 1
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask.reshape(-1)


@torch.no_grad()
def forward_extract_face_pool(
    model: torch.nn.Module,
    hook: FullSequenceHook,
    tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    face_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward each batch. For each frame, mean-pool patch tokens inside
    face_mask (which has length 196). Returns:
       face_feats   (N, 768) float32 np.ndarray
       probs        (N,) float32 np.ndarray
    """
    model.eval()
    n = tensor.shape[0]
    face_idx_torch = torch.from_numpy(np.where(face_mask)[0].astype(np.int64))
    n_face = int(face_mask.sum())
    feats: List[torch.Tensor] = []
    probs: List[torch.Tensor] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tensor[start:end].to(device, non_blocking=True)
        pred = model({"image": batch}, inference=True)
        tokens = hook.flush()  # (B, 1+196, D)
        if tokens.shape[1] != 1 + PATCH_GRID * PATCH_GRID:
            raise RuntimeError(
                f"unexpected token count {tokens.shape[1]}; expected {1+PATCH_GRID*PATCH_GRID}"
            )
        # Patches are tokens[:, 1:, :]
        patches = tokens[:, 1:, :]  # (B, 196, D)
        # Mean-pool the face_idx patches
        face_patches = patches.index_select(1, face_idx_torch.to(tokens.device))  # (B, n_face, D)
        face_feat = face_patches.mean(dim=1)  # (B, D)
        feats.append(face_feat.detach().cpu().float())

        if isinstance(pred, dict) and "prob" in pred:
            prob = pred["prob"]
        else:
            logits = pred.get("cls") if isinstance(pred, dict) else pred
            prob = torch.softmax(logits, dim=-1)[:, 1]
        probs.append(prob.detach().cpu().float())

        if (start // batch_size) % 20 == 0:
            logger.info(
                "  forward %d/%d (%.1fs, batches_done=%d)",
                end, n, time.time() - t0, start // batch_size + 1,
            )

    feats_out = torch.cat(feats, dim=0).numpy().astype(np.float32)
    probs_out = torch.cat(probs, dim=0).numpy()
    return feats_out, probs_out


# -------------------- Metric computation --------------------

def compute_cell_metrics(
    feats_clean: np.ndarray,
    feats_teams: np.ndarray,
    meta_clean: pd.DataFrame,
    meta_teams: pd.DataFrame,
    probs_clean: np.ndarray,
    probs_teams: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute the 5 non-KLIEP metrics on one cell.

    Identical structure to run_phase0_geometry.py:compute_cell_metrics.
    """
    Xc = l2_normalize_rows(feats_clean.astype(np.float64))
    Xt = l2_normalize_rows(feats_teams.astype(np.float64))

    clean_groups = meta_clean.groupby("pair_id").indices
    teams_groups = meta_teams.groupby("pair_id").indices
    common_pairs = sorted(set(clean_groups.keys()) & set(teams_groups.keys()))

    cos_pair_vals = []
    pc_vals = []
    pt_vals = []
    for pid in common_pairs:
        ci = clean_groups[pid]
        ti = teams_groups[pid]
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        t0 = ti[np.argmin(meta_teams.iloc[ti]["frame_idx"].to_numpy())]
        cos_pair_vals.append(float(np.dot(Xc[c0], Xt[t0])))
        pc_vals.append(float(probs_clean[c0]))
        pt_vals.append(float(probs_teams[t0]))
    cos_pair_mean = float(np.mean(cos_pair_vals))
    cos_pair_std = float(np.std(cos_pair_vals, ddof=1))
    n_pairs_used = len(cos_pair_vals)

    within_vals = []
    for pid in common_pairs:
        for groups, X in ((clean_groups, Xc), (teams_groups, Xt)):
            idx = groups[pid]
            if len(idx) >= 2:
                sub = X[idx]
                G = sub @ sub.T
                k = sub.shape[0]
                off = (G.sum() - np.trace(G)) / (k * (k - 1))
                within_vals.append(float(off))
    cos_within = float(np.mean(within_vals))

    first_frame_clean = []
    for pid in common_pairs:
        ci = clean_groups[pid]
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        first_frame_clean.append(c0)
    first_frame_clean = np.array(first_frame_clean)
    n_ff = len(first_frame_clean)
    sample_n = min(N_CROSS_ID_SAMPLES, n_ff * (n_ff - 1) // 2)
    cross_vals = []
    for _ in range(sample_n):
        i, j = rng.choice(n_ff, size=2, replace=False)
        v = float(np.dot(Xc[first_frame_clean[i]], Xc[first_frame_clean[j]]))
        cross_vals.append(v)
    cos_cross = float(np.mean(cross_vals))

    delta = cos_pair_mean - cos_within

    pc_arr = np.asarray(pc_vals)
    pt_arr = np.asarray(pt_vals)
    if pc_arr.std() > 1e-12 and pt_arr.std() > 1e-12:
        score_corr = float(np.corrcoef(pc_arr, pt_arr)[0, 1])
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


def compute_kliep_projections_face(
    feats_clean: np.ndarray,
    feats_teams: np.ndarray,
    meta_clean: pd.DataFrame,
    meta_teams: pd.DataFrame,
    w_hat: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame]:
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


# -------------------- Driver --------------------

def write_sentinel(probe1_status: str, probe2_status: str, t0_start: float, verdict: str) -> None:
    """Write _probe_complete.json sentinel; idempotent."""
    sentinel = OUT_DIR / "_probe_complete.json"
    payload = {
        "probe1": probe1_status,
        "probe2": probe2_status,
        "wall_seconds": int(time.time() - t0_start),
        "verdict_summary": verdict,
    }
    with open(sentinel, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("sentinel written -> %s : %s", sentinel, payload)


def main() -> int:
    t0_total = time.time()
    feats_dir = OUT_DIR / "feats_face_region"
    ckpt_dir = OUT_DIR / "ckpts"
    feats_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s batch_size=%d", device, BATCH_SIZE)

    # Load cached frame tensors (fp16 on disk -> fp32 for forward)
    cache_clean_pt = OUT_DIR / "_cache_frames_clean.pt"
    cache_teams_pt = OUT_DIR / "_cache_frames_teams.pt"
    meta_clean = pd.read_parquet(OUT_DIR / "_cache_frames_meta_clean.parquet")
    meta_teams = pd.read_parquet(OUT_DIR / "_cache_frames_meta_teams.parquet")
    logger.info("loading cached frame tensors")
    t_load = time.time()
    tensor_clean = torch.load(cache_clean_pt, map_location="cpu")
    if tensor_clean.dtype != torch.float32:
        tensor_clean = tensor_clean.float()
    tensor_teams = torch.load(cache_teams_pt, map_location="cpu")
    if tensor_teams.dtype != torch.float32:
        tensor_teams = tensor_teams.float()
    logger.info(
        "tensors loaded: clean=%s teams=%s (%.1fs)",
        tuple(tensor_clean.shape), tuple(tensor_teams.shape), time.time() - t_load,
    )

    # Face mask (centered 7x7 of 14x14 = 49 patches)
    face_mask = face_region_mask_7x7(grid_size=PATCH_GRID, radius=FACE_SUBGRID_RADIUS)
    logger.info(
        "face mask: shape=%s n_active=%d (centered 7x7 of 14x14)",
        face_mask.shape, int(face_mask.sum()),
    )

    # Frozen-CLIP KLIEP axis (re-used)
    w_hat_frozen = np.load(OUT_DIR / "_kliep_w_hat.npy").astype(np.float64)

    cell_rows: List[Dict] = []
    kliep_rows: List[Dict] = []

    try:
        for ckpt_key, gcs_uri in CKPTS.items():
            t0_ckpt = time.time()
            logger.info("=" * 60)
            logger.info("CKPT %s -> %s", ckpt_key, gcs_uri)
            try:
                local_ckpt = download_ckpt(gcs_uri, ckpt_dir)
            except Exception as e:
                logger.error("download failed: %s", e)
                continue

            try:
                model = load_model(str(local_ckpt), str(DETECTOR_CFG), str(TRAIN_CFG), device)
            except Exception as e:
                logger.error("model load failed: %s", e)
                continue

            hook = FullSequenceHook(model, LAYER)

            # Check for cached face-region features
            clean_cache_path = feats_dir / f"{ckpt_key}_L11_face_clean.npy"
            teams_cache_path = feats_dir / f"{ckpt_key}_L11_face_teams.npy"
            probs_clean_path = feats_dir / f"{ckpt_key}_probs_clean.npy"
            probs_teams_path = feats_dir / f"{ckpt_key}_probs_teams.npy"

            if clean_cache_path.exists() and probs_clean_path.exists():
                feats_clean = np.load(clean_cache_path)
                probs_clean = np.load(probs_clean_path)
                logger.info("[%s] clean face-pool cache hit", ckpt_key)
            else:
                logger.info("[%s] forward clean (%d frames)", ckpt_key, tensor_clean.shape[0])
                feats_clean, probs_clean = forward_extract_face_pool(
                    model, hook, tensor_clean, device, BATCH_SIZE, face_mask,
                )
                np.save(clean_cache_path, feats_clean)
                np.save(probs_clean_path, probs_clean)
                logger.info("[%s] clean face-pool feats=%s probs=%s",
                            ckpt_key, feats_clean.shape, probs_clean.shape)

            if teams_cache_path.exists() and probs_teams_path.exists():
                feats_teams = np.load(teams_cache_path)
                probs_teams = np.load(probs_teams_path)
                logger.info("[%s] teams face-pool cache hit", ckpt_key)
            else:
                logger.info("[%s] forward teams (%d frames)", ckpt_key, tensor_teams.shape[0])
                feats_teams, probs_teams = forward_extract_face_pool(
                    model, hook, tensor_teams, device, BATCH_SIZE, face_mask,
                )
                np.save(teams_cache_path, feats_teams)
                np.save(probs_teams_path, probs_teams)
                logger.info("[%s] teams face-pool feats=%s probs=%s",
                            ckpt_key, feats_teams.shape, probs_teams.shape)

            # Metrics
            rng = np.random.default_rng(SEED)
            m = compute_cell_metrics(
                feats_clean, feats_teams, meta_clean, meta_teams,
                probs_clean, probs_teams, rng=rng,
            )
            proj_vals, per_pair_df = compute_kliep_projections_face(
                feats_clean, feats_teams, meta_clean, meta_teams, w_hat_frozen,
            )
            m["kliep_projection_mean"] = float(np.mean(proj_vals))
            m["kliep_projection_std"] = float(np.std(proj_vals, ddof=1))
            per_pair_df.insert(0, "ckpt_key", ckpt_key)
            kliep_rows.extend(per_pair_df.to_dict(orient="records"))

            row = {
                "ckpt_key": ckpt_key,
                "layer": LAYER,
                "pool": "face_region_centered_7x7",
                **m,
                "n_face_patches": int(face_mask.sum()),
                "n_total_patches": int(PATCH_GRID * PATCH_GRID),
            }
            cell_rows.append(row)
            logger.info(
                "[%s] L11 face-pool: cos_pair=%.4f cos_within=%.4f cos_cross=%.4f "
                "delta=%.4f kliep_mean=%+.4f score_r=%.4f (%.1fs)",
                ckpt_key, m["cos_pair"], m["cos_within_same"], m["cos_cross_id"],
                m["delta_pair_vs_within"], m["kliep_projection_mean"], m["score_corr"],
                time.time() - t0_ckpt,
            )

            hook.remove()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Delete ckpt to free disk
            try:
                local_ckpt.unlink(missing_ok=True)
                logger.info("[%s] ckpt deleted: %s", ckpt_key, local_ckpt)
            except OSError as e:
                logger.warning("[%s] ckpt unlink failed: %s", ckpt_key, e)

        # Write CSVs
        cells_csv = OUT_DIR / "per_ckpt_face_region_cosines.csv"
        if cell_rows:
            fieldnames = list(cell_rows[0].keys())
            with open(cells_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in cell_rows:
                    w.writerow(r)
            logger.info("wrote %s (%d rows)", cells_csv, len(cell_rows))

        kliep_csv = OUT_DIR / "face_region_kliep_projections.csv"
        if kliep_rows:
            pd.DataFrame(kliep_rows).to_csv(kliep_csv, index=False)
            logger.info("wrote %s (%d rows)", kliep_csv, len(kliep_rows))

        # Build verdict summary
        verdict_parts = []
        for r in cell_rows:
            verdict_parts.append(
                f"{r['ckpt_key']}: cos_pair={r['cos_pair']:.4f} delta={r['delta_pair_vs_within']:+.4f} kliep_mu={r['kliep_projection_mean']:+.4f}"
            )
        verdict_summary = " | ".join(verdict_parts) if verdict_parts else "no cells computed"

        write_sentinel("done", "done", t0_total, verdict_summary)
        logger.info("Probe 2 DONE wall=%.1fs", time.time() - t0_total)
        return 0

    except Exception as e:  # noqa: BLE001
        logger.exception("Probe 2 aborted: %s", e)
        write_sentinel("done", "aborted", t0_total, f"aborted: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
