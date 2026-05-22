"""CPU-1 Viso-fake signature localization (Phase 1, 2026-05-23).

Goal: locate the visomaster_enhanced_macro_dev signature in face vs non-face
patches at L11 on Slot A v2 step3500. Compute three readouts per frame:
  (a) CLS pool (baseline) — 1 token
  (b) face pool — centered 7x7 of 14x14 patch grid (49 patches)
  (c) non-face pool — complement (147 patches)
Then per-patch ablation: for each of 196 patch positions, replace the patch
with the mean of all patches and recompute the CLS-pool prob_fake. Aggregate
Delta into a 14x14 saliency map per suite (viso + deeplive control).

Inputs:
  - Slot A v2 step3500 ckpt (already cached locally)
  - Visomaster_enhanced_macro_dev frames manifest (550 fakes)
  - Deeplive_enhanced_dev frames manifest (545 fakes, control)

Outputs:
  RESULTS_FACTS_2026-05-23.md
  viso_patch_saliency_14x14.npy   (14x14)
  deeplive_patch_saliency_14x14.npy (14x14)
  viso_readout_probs.npy / deeplive_readout_probs.npy (N x 3)
  _cpu1_complete.json (sentinel marker)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from batch_inference_gcs import CLIP_MEAN, CLIP_STD, load_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"

# Slot A v2 step3500 — already cached on 2026-05-20 per the canary script
CKPT_LOCAL = (
    REPO_ROOT / "analysis/manual_canary_2026-05-20/ckpts/"
    "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth"
)

# Manifests — same as face-pool full scorecard
VISO_MANIFEST = (
    REPO_ROOT / "analysis/face_pool_scorecard_2026-05-22/_tmp/"
    "visomaster_enhanced_macro_dev_frames.csv"
)
DEEPLIVE_MANIFEST = (
    REPO_ROOT / "analysis/face_pool_scorecard_2026-05-22/_tmp/"
    "deeplive_enhanced_dev_frames.csv"
)

L11_LAYER = 11
PATCH_GRID = 14
FACE_SUBGRID_RADIUS = 3  # centered 7x7 of 14x14 -> 49 face patches
RESOLUTION = 224
N_PATCHES = PATCH_GRID * PATCH_GRID  # 196
N_TOKENS = 1 + N_PATCHES  # CLS + 196

# Calibrated τ from face-pool scorecard 2026-05-22 selected_threshold_scorecard.csv
TAU_CLS = 0.787956     # Slot A v2 step3500 CLS-pool video-level
TAU_FACE = 0.736775    # Slot A v2 step3500 face-pool video-level
# Non-face pool: no calibrated τ exists; we report fake-recall at multiple τ
# and ALSO at the CLS τ as a per-frame readout for apples-to-apples.
TAU_NONFACE_GRID = [TAU_CLS, TAU_FACE, 0.5]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "_cpu1.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cpu1")


# -------------------- Frame loading --------------------

def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    s = uri.removeprefix("gs://")
    bucket, _, blob = s.partition("/")
    return bucket, blob


def load_frames_into_tensor(manifest_path: Path, cache_path: Path) -> Tuple[torch.Tensor, pd.DataFrame]:
    """Stream frames from GCS into a single fp16 cached tensor."""
    df = pd.read_csv(manifest_path)
    n = len(df)
    if cache_path.exists():
        logger.info("[cache hit] %s", cache_path)
        t = torch.load(cache_path, map_location="cpu")
        if t.dtype != torch.float32:
            t = t.float()
        return t, df

    logger.info("[cache miss] streaming %d frames from %s", n, manifest_path.name)
    from google.cloud import storage
    from torchvision import transforms as T

    normalize = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    client = storage.Client()
    bucket_cache: Dict[str, object] = {}

    out = torch.zeros(n, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)
    fails = 0
    t0 = time.time()
    for i in range(n):
        fp = str(df["frame_path"].iloc[i])
        bucket_name, blob_path = _parse_gs_uri(fp)
        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = client.bucket(bucket_name)
        try:
            blob = bucket_cache[bucket_name].blob(blob_path)
            img_bytes = blob.download_as_bytes()
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                fails += 1
                continue
            img_bgr = cv2.resize(img_bgr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out[i] = normalize(img_rgb)
        except Exception as e:
            fails += 1
            if fails <= 3:
                logger.warning("frame %d failed: %s", i, e)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d streamed (%.1fs, %d fails)", i + 1, n, time.time() - t0, fails)

    nonzero = out.abs().sum(dim=(1, 2, 3)) > 1e-6
    n_valid = int(nonzero.sum().item())
    logger.info("[cache built] %d/%d valid frames in %.1fs", n_valid, n, time.time() - t0)
    out = out[nonzero]
    df = df.iloc[nonzero.cpu().numpy()].reset_index(drop=True)
    # Save as fp16 to save disk
    torch.save(out.half(), cache_path)
    return out, df


# -------------------- Face mask helpers --------------------

def face_region_mask_7x7(grid_size: int = PATCH_GRID, radius: int = FACE_SUBGRID_RADIUS) -> np.ndarray:
    center = grid_size // 2
    lo = center - radius - 1
    hi = center + radius
    assert hi - lo == 2 * radius + 1
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask.reshape(-1)


# -------------------- Encoder forward with full token capture --------------------

class TokenCaptureHook:
    """Register a forward hook on resblocks[L11] that captures the full
    (CLS + 196) token sequence each forward. Used by both the readout
    scorer (single forward per frame) AND the ablation loop (single
    forward per frame; ablation done downstream via token substitution).
    """

    def __init__(self, model: torch.nn.Module, layer: int):
        visual = model.backbone.visual
        try:
            resblocks = visual.transformer.resblocks
        except AttributeError:
            resblocks = visual.visual.transformer.resblocks
        self._handle = resblocks[layer].register_forward_hook(self._hook)
        self._captured: torch.Tensor = None

    def _hook(self, module, inputs, output):  # noqa: ARG002
        if output.dim() != 3:
            raise RuntimeError(f"unexpected output dim: {output.shape}")
        if output.shape[0] >= output.shape[1]:
            tokens = output.permute(1, 0, 2)
        else:
            tokens = output
        self._captured = tokens.detach()

    def flush(self) -> torch.Tensor:
        t = self._captured
        self._captured = None
        return t

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def get_visual_head_components(model: torch.nn.Module):
    """Locate ln_post + proj for both OpenCLIP and HF CLIP backbones."""
    visual = model.backbone.visual
    try:
        # OpenCLIP path
        resblocks = visual.transformer.resblocks  # noqa: F841
        ln_post = getattr(visual, "ln_post", None)
        proj = getattr(visual, "proj", None)
        return ln_post, proj
    except AttributeError:
        resblocks = visual.visual.transformer.resblocks  # noqa: F841
        ln_post = getattr(visual.visual, "post_layernorm", None)
        return ln_post, None


def get_classifier_head(model: torch.nn.Module):
    """Return a callable head(features_512) -> prob_fake.

    The Effort detector uses an ArcFace-style head; for inference we want the
    softmax fake-class probability. We mimic the model's standard forward
    path by calling the head directly with our face/non-face/CLS pooled
    representation.
    """
    return model


@torch.no_grad()
def score_all_three_readouts(
    model: torch.nn.Module,
    hook: TokenCaptureHook,
    tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    ln_post,
    proj,
    face_idx: torch.Tensor,
    nonface_idx: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward each frame ONCE. Capture L11 tokens. Compute 3 pooled
    representations: CLS, face-mean, non-face-mean. Apply ln_post + proj.
    Feed into the unchanged head. Returns 4 prob arrays:
        probs_cls_baseline (from the model's own forward — sanity check)
        probs_cls          (recomputed via head from captured CLS token)
        probs_face         (head on face-pool 49 patches)
        probs_nonface      (head on non-face-pool 147 patches)
    """
    model.eval()
    n = tensor.shape[0]
    probs_baseline: List[float] = []
    probs_cls: List[float] = []
    probs_face: List[float] = []
    probs_nonface: List[float] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tensor[start:end].to(device, non_blocking=True)
        pred = model({"image": batch}, inference=True)
        # baseline prob from the model's own (CLS-based) forward
        if isinstance(pred, dict):
            if "prob" in pred:
                p_base = pred["prob"]
            else:
                logits = None
                for k in ("cls", "raw_logits", "logits", "classifier_logits", "pred_logits"):
                    if k in pred:
                        logits = pred[k]
                        break
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)
                p_base = torch.softmax(logits, dim=-1)[:, 1]
        else:
            p_base = torch.softmax(pred, dim=-1)[:, 1]
        probs_baseline.extend(p_base.float().cpu().tolist())

        tokens = hook.flush()  # (B, 197, 768)
        if tokens.shape[1] != N_TOKENS:
            raise RuntimeError(f"unexpected token count {tokens.shape[1]}; want {N_TOKENS}")
        # Patches: tokens[:, 1:, :]
        patches = tokens[:, 1:, :]  # (B, 196, 768)
        face_patches = patches.index_select(1, face_idx.to(tokens.device))  # (B, 49, 768)
        nonface_patches = patches.index_select(1, nonface_idx.to(tokens.device))  # (B, 147, 768)
        face_pool = face_patches.mean(dim=1)  # (B, 768)
        nonface_pool = nonface_patches.mean(dim=1)
        cls_pool = tokens[:, 0, :]  # (B, 768)

        # Apply post-block ops
        face_pool = ln_post(face_pool)
        nonface_pool = ln_post(nonface_pool)
        cls_pool = ln_post(cls_pool)
        if proj is not None:
            face_pool = face_pool @ proj
            nonface_pool = nonface_pool @ proj
            cls_pool = cls_pool @ proj

        # Run each through the head via _head_prob (mirrors detector's
        # inference forward path: normalize_features_before_head flag,
        # use_arcface_head dispatch, label=None for inference, softmax index 1).
        for x, dst in ((cls_pool, probs_cls), (face_pool, probs_face), (nonface_pool, probs_nonface)):
            p = _head_prob(model, x)
            dst.extend(p.float().cpu().tolist())

        if (start // batch_size) % 5 == 0:
            logger.info(
                "  readouts %d/%d  elapsed=%.1fs", end, n, time.time() - t0,
            )

    return (
        np.asarray(probs_baseline, dtype=np.float64),
        np.asarray(probs_cls, dtype=np.float64),
        np.asarray(probs_face, dtype=np.float64),
        np.asarray(probs_nonface, dtype=np.float64),
    )


@torch.no_grad()
def compute_patch_saliency(
    model: torch.nn.Module,
    hook: TokenCaptureHook,
    tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    ln_post,
    proj,
    baseline_cls_probs: np.ndarray,
    n_perturbations_per_frame: int = N_PATCHES,
) -> np.ndarray:
    """Per-patch soft ablation. For each frame, capture tokens once, then
    for each of the 196 patch positions: substitute the patch with the
    mean of all 196 patches, recompute CLS-pool head output, log
    Delta = baseline_prob - ablated_prob. Returns saliency_map (14x14)
    averaged over all frames.

    Optimization: we don't re-run the backbone for each ablation. The
    ablation modifies only the CLS-token-equivalent (we ablate at the
    POST-L11 stage), so we can:
      1. Capture tokens once per frame.
      2. For each ablation: replace patch i with mean(all_patches), then
         re-pool CLS. (CLS isn't directly modified; instead, we use the
         mean-of-tokens-as-CLS-replacement approximation.)

    Actually, the standard interpretation for "ablate patch i" at L11
    AFTER the encoder is to remove patch i from the pool. But the model
    uses ONLY the CLS token at L11, not a pool of patches. So patch
    ablation at L11 has no effect on the CLS-only head.

    Correct interpretation per the plan: replace patch i token with the
    mean of all patches, then re-pool *as a face-region pool* (or CLS
    pool with the substituted token). Since CLS doesn't see patches in
    the L11 readout, we use the face-pool readout for ablation: each
    ablation substitutes one patch in the 49-patch face pool, and we
    measure Delta on the face-pool head output.

    Implementation: substitute, then take the mean over the 49 face
    patches. If the ablated patch is OUTSIDE the face region (147
    positions), the substitution doesn't affect the face pool — so its
    Delta will be ~0 by construction. To make non-face Delta meaningful,
    we ALSO measure Delta on the full-patch pool (all 196 averaged).

    To preserve plan intent: we use FULL-patch pool as the "all-patches"
    readout — this is sensitive to all 196 positions. baseline_full_prob
    is recorded; saliency_map[r,c] = baseline_full_prob - prob_after_ablation.
    """
    model.eval()
    n = tensor.shape[0]
    saliency = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float64)
    n_processed = 0
    t0 = time.time()
    # Patch index grid (flat 0..195 -> (row,col))
    rows, cols = np.divmod(np.arange(N_PATCHES), PATCH_GRID)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tensor[start:end].to(device, non_blocking=True)
        _ = model({"image": batch}, inference=True)
        tokens = hook.flush()  # (B, 197, 768)
        patches = tokens[:, 1:, :]  # (B, 196, 768)
        B = patches.shape[0]
        # Baseline: full-patch pool (mean of 196 patches)
        full_pool = patches.mean(dim=1)  # (B, 768)
        full_pool = ln_post(full_pool)
        if proj is not None:
            full_pool = full_pool @ proj
        baseline = _head_prob(model, full_pool)  # (B,)
        baseline_np = baseline.cpu().float().numpy()  # (B,)

        # Per-patch ablation. patches_mean: (B, 1, 768) replacement value.
        patches_mean = patches.mean(dim=1, keepdim=True)  # (B, 1, 768)
        # We need to evaluate 196 ablations per frame. To avoid OOM, do it
        # in groups of `chunk` patches at once: build a (B*chunk, 768)
        # ablated-pool tensor by computing pool_with_patch_replaced =
        # baseline_sum - patches[:, i, :] + patches_mean (then / 196).
        # baseline_sum across patches (B, 768)
        sum_patches = patches.sum(dim=1)  # (B, 768)
        # For each patch position p:
        #   new_pool = (sum_patches - patches[:, p, :] + patches_mean[:, 0, :]) / 196
        # Do it in chunks of 14 to bound memory ((B*14, 768) ~ 32*14*768*4=1.4MB per chunk)
        chunk = 14
        for c_start in range(0, N_PATCHES, chunk):
            c_end = min(c_start + chunk, N_PATCHES)
            c_n = c_end - c_start
            # patches_slice (B, c_n, 768)
            patches_slice = patches[:, c_start:c_end, :]
            # ablated_sum: (B, c_n, 768) = sum_patches[None,:].expand - patches_slice + patches_mean
            ablated_sum = sum_patches.unsqueeze(1) - patches_slice + patches_mean
            ablated_pool = ablated_sum / float(N_PATCHES)  # (B, c_n, 768)
            # Apply ln_post + proj along the last dim
            ablated_pool_flat = ablated_pool.reshape(-1, ablated_pool.shape[-1])  # (B*c_n, 768)
            ablated_pool_flat = ln_post(ablated_pool_flat)
            if proj is not None:
                ablated_pool_flat = ablated_pool_flat @ proj
            ablated_probs = _head_prob(model, ablated_pool_flat).cpu().float().numpy().reshape(B, c_n)
            # Delta = baseline - ablated  (positive Delta = ablation REDUCED prob_fake = patch was supporting fake)
            delta = baseline_np[:, None] - ablated_probs  # (B, c_n)
            # Aggregate sum over frames for each patch position
            for ci in range(c_n):
                p_idx = c_start + ci
                r, col = rows[p_idx], cols[p_idx]
                saliency[r, col] += float(delta[:, ci].sum())
            del ablated_sum, ablated_pool, ablated_pool_flat
        n_processed += B
        if (start // batch_size) % 5 == 0:
            logger.info(
                "  saliency batch %d/%d  frames_done=%d  elapsed=%.1fs",
                start // batch_size + 1, (n + batch_size - 1) // batch_size,
                n_processed, time.time() - t0,
            )
    saliency /= max(n_processed, 1)
    return saliency


def _head_prob(model: torch.nn.Module, features_512: torch.Tensor) -> torch.Tensor:
    """Mirror the detector's inference forward path from line 1799-1821 of
    detectors/effort_detector.py: normalize (if configured) -> head(features,
    label=None) for ArcFace, head(features) for linear, then softmax index 1.
    """
    # Match the detector's normalize_features_before_head flag
    norm_flag = getattr(model, "normalize_features_before_head", False)
    if norm_flag:
        features_512 = torch.nn.functional.normalize(features_512, p=2, dim=1)
    use_arc = getattr(model, "use_arcface_head", True)
    if use_arc:
        raw_logits = model.head(features_512, label=None)
    else:
        raw_logits = model.head(features_512)
    if raw_logits.dim() == 3:
        raw_logits = raw_logits.mean(dim=1)
    return torch.softmax(raw_logits, dim=-1)[:, 1]


# -------------------- Verdict --------------------

def compute_verdict(viso_recall: Dict[str, float], deeplive_recall: Dict[str, float],
                    viso_saliency: np.ndarray, face_mask_2d: np.ndarray) -> Tuple[str, str]:
    """Apply alpha/beta/gamma close criterion to saved metrics.

    alpha: viso non-face-pool recall >= face-pool recall + 0.05 AND non-face
           saliency mass > 70% -> viso signature in non-face patches
    beta:  viso non-face-pool recall < CLS recall - 0.05 AND face saliency
           mass > 50% -> viso signature spans face + non-face symmetrically
           (HEAD ALT = dual-readout)
    gamma: in-between
    """
    cls_r = viso_recall["recall_at_tau_cls"]
    face_r = viso_recall["recall_at_tau_face"]
    nonface_r_cls = viso_recall["recall_nonface_at_tau_cls"]

    face_mass = float(np.abs(viso_saliency)[face_mask_2d].sum())
    nonface_mass = float(np.abs(viso_saliency)[~face_mask_2d].sum())
    total_mass = face_mass + nonface_mass
    face_frac = face_mass / max(total_mass, 1e-12)
    nonface_frac = nonface_mass / max(total_mass, 1e-12)

    # alpha test
    if (nonface_r_cls >= face_r + 0.05) and (nonface_frac > 0.70):
        verdict = "alpha"
        summary = (
            f"viso signature in non-face patches; nonface_recall={nonface_r_cls:.3f} "
            f">= face_recall+0.05={face_r + 0.05:.3f}, nonface_saliency_frac={nonface_frac:.3f}"
        )
        return verdict, summary
    # beta test
    if (nonface_r_cls < cls_r - 0.05) and (face_frac > 0.50):
        verdict = "beta"
        summary = (
            f"viso signature spans face+non-face symmetric; nonface_recall={nonface_r_cls:.3f} "
            f"< cls_recall-0.05={cls_r - 0.05:.3f}, face_saliency_frac={face_frac:.3f}"
        )
        return verdict, summary
    # gamma fallback
    verdict = "gamma"
    summary = (
        f"in-between: cls_recall={cls_r:.3f} face_recall={face_r:.3f} "
        f"nonface_recall_at_tau_cls={nonface_r_cls:.3f}; "
        f"face_saliency_frac={face_frac:.3f} nonface_saliency_frac={nonface_frac:.3f}"
    )
    return verdict, summary


def write_facts_doc(
    viso_recall: Dict[str, float],
    deeplive_recall: Dict[str, float],
    viso_saliency: np.ndarray,
    deeplive_saliency: np.ndarray,
    face_mask_2d: np.ndarray,
    n_viso: int,
    n_deeplive: int,
    wall_seconds: float,
    verdict: str,
    summary: str,
    out_path: Path,
) -> None:
    """Write FACTS doc. Banned words: succeeds, fails, wins, loses, promotes,
    deployment-grade, ship, kill, best, worst, unfortunately, remarkably,
    lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.
    """
    face_mass_viso = float(np.abs(viso_saliency)[face_mask_2d].sum())
    nonface_mass_viso = float(np.abs(viso_saliency)[~face_mask_2d].sum())
    total_viso = face_mass_viso + nonface_mass_viso
    face_frac_viso = face_mass_viso / max(total_viso, 1e-12)
    nonface_frac_viso = nonface_mass_viso / max(total_viso, 1e-12)

    face_mass_dl = float(np.abs(deeplive_saliency)[face_mask_2d].sum())
    nonface_mass_dl = float(np.abs(deeplive_saliency)[~face_mask_2d].sum())
    total_dl = face_mass_dl + nonface_mass_dl
    face_frac_dl = face_mass_dl / max(total_dl, 1e-12)
    nonface_frac_dl = nonface_mass_dl / max(total_dl, 1e-12)

    # Top-5 patches by absolute saliency
    flat_viso = viso_saliency.flatten()
    top5_viso = np.argsort(-np.abs(flat_viso))[:5]
    top5_viso_rc = [(int(i // PATCH_GRID), int(i % PATCH_GRID), float(flat_viso[i])) for i in top5_viso]
    flat_dl = deeplive_saliency.flatten()
    top5_dl = np.argsort(-np.abs(flat_dl))[:5]
    top5_dl_rc = [(int(i // PATCH_GRID), int(i % PATCH_GRID), float(flat_dl[i])) for i in top5_dl]

    lines = []
    lines.append("# CPU-1 Viso-Fake Signature Localization — FACTS (2026-05-23)")
    lines.append("")
    lines.append("> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.")
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append(f"- Checkpoint: Slot A v2 step3500 (`{CKPT_LOCAL.name}`).")
    lines.append(f"- Forward hook on resblock[11] capturing the full 197-token sequence (CLS + 196 patches in 14x14 grid).")
    lines.append(f"- Three readouts per frame computed by routing each pooled vector through the unchanged classifier head:")
    lines.append(f"  - CLS-pool (1 token).")
    lines.append(f"  - Face-pool: mean of centered 7x7 patch subgrid (49 patches).")
    lines.append(f"  - Non-face-pool: mean of complement (147 patches).")
    lines.append(f"- Per-patch ablation (saliency map): for each of 196 patch positions, replace that patch token with the mean of all 196 patches in the full-patch pool, recompute the head output, Delta = baseline - ablated. Aggregate to a 14x14 saliency map.")
    lines.append(f"- N(viso) = {n_viso} frames; N(deeplive control) = {n_deeplive} frames.")
    lines.append(f"- Wall time: {wall_seconds:.0f}s.")
    lines.append("")
    lines.append("## 2. Per-readout recall table")
    lines.append("")
    lines.append("Calibrated thresholds from the 2026-05-22 face-pool scorecard's `selected_threshold_scorecard.csv`:")
    lines.append(f"- CLS-pool τ = {TAU_CLS:.6f}")
    lines.append(f"- Face-pool τ = {TAU_FACE:.6f}")
    lines.append("")
    lines.append("| Suite | CLS @ τ_CLS | Face-pool @ τ_face | Non-face @ τ_CLS | Non-face @ τ_face | Non-face @ τ=0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, recall in (("visomaster_enhanced_macro_dev (n="+str(n_viso)+")", viso_recall),
                         ("deeplive_enhanced_dev (control, n="+str(n_deeplive)+")", deeplive_recall)):
        lines.append(
            f"| {label} | {recall['recall_at_tau_cls']:.4f} | "
            f"{recall['recall_at_tau_face']:.4f} | "
            f"{recall['recall_nonface_at_tau_cls']:.4f} | "
            f"{recall['recall_nonface_at_tau_face']:.4f} | "
            f"{recall['recall_nonface_at_tau_05']:.4f} |"
        )
    lines.append("")
    lines.append("## 3. Saliency-map summary statistics")
    lines.append("")
    lines.append("Saliency map values: mean(baseline_prob - ablated_prob) across frames per patch position. Higher absolute value = larger Delta = patch contributed more to the head's fake/real decision.")
    lines.append("")
    lines.append("| Suite | mean(saliency) | std | max | min | face-region mass | non-face-region mass | face-frac | non-face-frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| visomaster | {viso_saliency.mean():+.6f} | {viso_saliency.std():.6f} | "
        f"{viso_saliency.max():+.6f} | {viso_saliency.min():+.6f} | "
        f"{face_mass_viso:.4f} | {nonface_mass_viso:.4f} | "
        f"{face_frac_viso:.4f} | {nonface_frac_viso:.4f} |"
    )
    lines.append(
        f"| deeplive (control) | {deeplive_saliency.mean():+.6f} | {deeplive_saliency.std():.6f} | "
        f"{deeplive_saliency.max():+.6f} | {deeplive_saliency.min():+.6f} | "
        f"{face_mass_dl:.4f} | {nonface_mass_dl:.4f} | "
        f"{face_frac_dl:.4f} | {nonface_frac_dl:.4f} |"
    )
    lines.append("")
    lines.append("## 4. Top-5 most-salient patches (by absolute Delta)")
    lines.append("")
    lines.append("Patch coordinates as (row, col), with row 0 = top of frame, col 0 = left.")
    lines.append("")
    lines.append("### visomaster")
    lines.append("| rank | row | col | mean Delta | in-face-region? |")
    lines.append("|---|---:|---:|---:|---|")
    for rank, (r, c, v) in enumerate(top5_viso_rc):
        in_face = face_mask_2d[r, c]
        lines.append(f"| {rank+1} | {r} | {c} | {v:+.6f} | {in_face} |")
    lines.append("")
    lines.append("### deeplive (control)")
    lines.append("| rank | row | col | mean Delta | in-face-region? |")
    lines.append("|---|---:|---:|---:|---|")
    for rank, (r, c, v) in enumerate(top5_dl_rc):
        in_face = face_mask_2d[r, c]
        lines.append(f"| {rank+1} | {r} | {c} | {v:+.6f} | {in_face} |")
    lines.append("")
    lines.append("## 5. Close criterion verdict")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"Summary: {summary}")
    lines.append("")
    lines.append("Decision rule (verbatim from plan Phase 1):")
    lines.append("- alpha: viso non-face-pool recall >= face-pool recall + 0.05 AND non-face saliency mass > 70%")
    lines.append("- beta:  viso non-face-pool recall < CLS recall - 0.05 AND face saliency mass > 50%")
    lines.append("- gamma: in-between")
    lines.append("")
    lines.append("## 6. Output artifacts")
    lines.append("")
    lines.append(f"- `viso_patch_saliency_14x14.npy` — 14x14 float64")
    lines.append(f"- `deeplive_patch_saliency_14x14.npy` — 14x14 float64")
    lines.append(f"- `viso_readout_probs.npy` — (n_viso, 4) float64: baseline, cls, face, nonface")
    lines.append(f"- `deeplive_readout_probs.npy` — (n_deeplive, 4) float64")
    lines.append(f"- `_cpu1_complete.json` — sentinel")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    t0_total = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("device=%s batch_size=%d", device, args.batch_size)

    # Load frames
    viso_cache = OUT_DIR / "_cache_viso_frames.pt"
    dl_cache = OUT_DIR / "_cache_deeplive_frames.pt"
    logger.info("[1/4] streaming/loading visomaster frames")
    viso_tensor, viso_meta = load_frames_into_tensor(VISO_MANIFEST, viso_cache)
    logger.info("  viso tensor: %s", tuple(viso_tensor.shape))
    logger.info("[1.5/4] streaming/loading deeplive frames")
    dl_tensor, dl_meta = load_frames_into_tensor(DEEPLIVE_MANIFEST, dl_cache)
    logger.info("  deeplive tensor: %s", tuple(dl_tensor.shape))

    # Load model
    logger.info("[2/4] loading model from %s", CKPT_LOCAL)
    t_m = time.time()
    model = load_model(str(CKPT_LOCAL), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    logger.info("  model loaded in %.1fs", time.time() - t_m)
    ln_post, proj = get_visual_head_components(model)
    logger.info("  ln_post: %s   proj.shape: %s",
                type(ln_post).__name__,
                tuple(proj.shape) if proj is not None else "None")

    # Face / non-face masks
    face_mask = face_region_mask_7x7()  # (196,) bool
    nonface_mask = ~face_mask
    face_idx = torch.from_numpy(np.where(face_mask)[0].astype(np.int64))
    nonface_idx = torch.from_numpy(np.where(nonface_mask)[0].astype(np.int64))
    face_mask_2d = face_mask.reshape(PATCH_GRID, PATCH_GRID)
    logger.info("  face patches: %d, non-face patches: %d", int(face_mask.sum()), int(nonface_mask.sum()))

    hook = TokenCaptureHook(model, L11_LAYER)

    # Three-readout scoring
    logger.info("[3/4] three-readout scoring on visomaster (%d frames)", viso_tensor.shape[0])
    viso_baseline, viso_cls, viso_face_pool, viso_nonface_pool = score_all_three_readouts(
        model, hook, viso_tensor, device, args.batch_size, ln_post, proj, face_idx, nonface_idx,
    )
    logger.info("[3/4] three-readout scoring on deeplive (%d frames)", dl_tensor.shape[0])
    dl_baseline, dl_cls, dl_face_pool, dl_nonface_pool = score_all_three_readouts(
        model, hook, dl_tensor, device, args.batch_size, ln_post, proj, face_idx, nonface_idx,
    )

    # Per-frame ablation -> saliency map
    logger.info("[4/4] saliency map on visomaster")
    # Recompute baseline_full_prob inside; reuse hook
    viso_saliency = compute_patch_saliency(
        model, hook, viso_tensor, device, args.batch_size, ln_post, proj,
        baseline_cls_probs=viso_cls,
    )
    logger.info("[4/4] saliency map on deeplive (control)")
    dl_saliency = compute_patch_saliency(
        model, hook, dl_tensor, device, args.batch_size, ln_post, proj,
        baseline_cls_probs=dl_cls,
    )

    hook.remove()

    # Compute recall at calibrated tau (fakes only -> recall = mean(prob > tau))
    def recall_at_tau(probs: np.ndarray, tau: float) -> float:
        return float((probs > tau).mean())

    viso_recall = {
        "recall_at_tau_cls": recall_at_tau(viso_cls, TAU_CLS),
        "recall_at_tau_face": recall_at_tau(viso_face_pool, TAU_FACE),
        "recall_nonface_at_tau_cls": recall_at_tau(viso_nonface_pool, TAU_CLS),
        "recall_nonface_at_tau_face": recall_at_tau(viso_nonface_pool, TAU_FACE),
        "recall_nonface_at_tau_05": recall_at_tau(viso_nonface_pool, 0.5),
    }
    dl_recall = {
        "recall_at_tau_cls": recall_at_tau(dl_cls, TAU_CLS),
        "recall_at_tau_face": recall_at_tau(dl_face_pool, TAU_FACE),
        "recall_nonface_at_tau_cls": recall_at_tau(dl_nonface_pool, TAU_CLS),
        "recall_nonface_at_tau_face": recall_at_tau(dl_nonface_pool, TAU_FACE),
        "recall_nonface_at_tau_05": recall_at_tau(dl_nonface_pool, 0.5),
    }

    # Verdict
    verdict, summary = compute_verdict(viso_recall, dl_recall, viso_saliency, face_mask_2d)
    logger.info("VERDICT: %s — %s", verdict, summary)

    # Save artifacts
    np.save(OUT_DIR / "viso_patch_saliency_14x14.npy", viso_saliency)
    np.save(OUT_DIR / "deeplive_patch_saliency_14x14.npy", dl_saliency)
    viso_probs_stack = np.stack([viso_baseline, viso_cls, viso_face_pool, viso_nonface_pool], axis=1)
    dl_probs_stack = np.stack([dl_baseline, dl_cls, dl_face_pool, dl_nonface_pool], axis=1)
    np.save(OUT_DIR / "viso_readout_probs.npy", viso_probs_stack)
    np.save(OUT_DIR / "deeplive_readout_probs.npy", dl_probs_stack)

    wall = time.time() - t0_total
    write_facts_doc(
        viso_recall, dl_recall, viso_saliency, dl_saliency, face_mask_2d,
        viso_tensor.shape[0], dl_tensor.shape[0],
        wall, verdict, summary,
        OUT_DIR / "RESULTS_FACTS_2026-05-23.md",
    )
    sentinel = OUT_DIR / "_cpu1_complete.json"
    with open(sentinel, "w") as f:
        json.dump({
            "status": "done",
            "wall_seconds": int(wall),
            "verdict": verdict,
            "verdict_summary": summary,
            "viso_recall": viso_recall,
            "deeplive_recall": dl_recall,
            "n_viso": int(viso_tensor.shape[0]),
            "n_deeplive": int(dl_tensor.shape[0]),
        }, f, indent=2)
    logger.info("DONE — sentinel %s wrote in %.1fs total", sentinel, wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
