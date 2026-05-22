"""Face-pool canary scoring on Slot A v2 step3500 (2026-05-22).

Follow-up to Phase 0 Probe 2 (face-region geometry probe). Probe 2 showed
that face-region patch features are MORE substrate-invariant than CLS at L11:
- cos_pair: 0.87 (CLS) -> 0.96 (face)
- delta_pair_vs_within: -0.077 (CLS) -> -0.016 (face)

This script tests whether that representation-level invariance translates to a
lockbox-FPR improvement on the existing 800-frame canary. We do NOT modify the
classifier weights — only swap the pooling op feeding the head:

    CLS-pool (baseline): pooler_output = tokens[CLS]                 # 1 token
    Face-pool (new):     pooler_output = mean(tokens[face_idx])      # 49 tokens

The 49 face indices are the centered 7x7 of the 14x14 patch grid, same as in
Probe 2 (`run_probe2_face_region.py`).

The classifier head was trained on CLS-pool features, so absolute scores will
shift — but `aggregate_metrics` uses real-score percentiles as the FPR-5%/10%
threshold grid, so calibrated rank-order is what matters.

Usage:
    python analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py \\
        --output-dir analysis/face_pool_canary_2026-05-22/outputs \\
        --ckpt analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Make project modules importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from batch_inference_gcs import load_model  # noqa: E402

# Re-use aggregate_metrics from baseline scorer
sys.path.insert(0, str(REPO_ROOT / "analysis/manual_canary_2026-05-20"))
from score_canary import aggregate_metrics  # noqa: E402

DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"
CANARY_FRAMES = REPO_ROOT / "analysis/manual_canary_2026-05-20/frames.pt"
CANARY_META = REPO_ROOT / "analysis/manual_canary_2026-05-20/frames_meta.parquet"

L11_LAYER = 11
PATCH_GRID = 14
FACE_SUBGRID_RADIUS = 3  # centered 7x7 of 14x14 -> 49 face patches


def face_region_mask_7x7(grid_size: int = PATCH_GRID, radius: int = FACE_SUBGRID_RADIUS) -> np.ndarray:
    """Identical to Probe 2's mask. Returns flat boolean mask of length grid_size**2.

    Centered 7x7 of 14x14: rows {3..9}, cols {3..9} = 49 patches at center ~75%.
    """
    center = grid_size // 2
    lo = center - radius - 1
    hi = center + radius
    assert hi - lo == 2 * radius + 1, f"hi-lo={hi-lo} expected {2*radius+1}"
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask.reshape(-1)


class FacePoolMonkeyPatch:
    """Monkey-patch model.backbone.forward to substitute face-region pool
    for CLS at L11, feeding the existing classifier head.

    On __enter__: install forward hook on resblocks[L11] to capture full
    token sequence, then wrap model.backbone.forward so it overrides
    'pooler_output' with the face-region mean (49 patch tokens).

    On __exit__: restore original forward and remove hook.
    """

    def __init__(self, model: torch.nn.Module, layer: int = L11_LAYER):
        self.model = model
        self.layer = layer
        self._handle = None
        self._captured_tokens = None
        self._orig_forward = None
        self._face_idx_cache = None
        self._n_patches_expected = 1 + PATCH_GRID * PATCH_GRID  # CLS + 196

    def _hook(self, module, inputs, output):  # noqa: ARG002
        if output.dim() != 3:
            raise RuntimeError(f"unexpected resblock output dim: {output.shape}")
        if output.shape[0] >= output.shape[1]:
            # seq-first (seq, batch, dim) -> (batch, seq, dim)
            tokens = output.permute(1, 0, 2)
        else:
            tokens = output
        self._captured_tokens = tokens  # KEEP requires_grad path; do not detach here

    def __enter__(self):
        visual = self.model.backbone.visual
        # OpenCLIP path: visual is the OpenCLIP visual encoder (has .transformer.resblocks)
        # HF CLIP path:  visual would have .visual.transformer.resblocks (extra layer)
        try:
            resblocks = visual.transformer.resblocks
            self._is_openclip = True
            # Capture post-block heads
            self._ln_post = getattr(visual, "ln_post", None)
            self._proj = getattr(visual, "proj", None)
        except AttributeError:
            resblocks = visual.visual.transformer.resblocks
            self._is_openclip = False
            self._ln_post = getattr(visual.visual, "post_layernorm", None)
            self._proj = None  # HF CLIPVisionModel: pooler_output is post-LN, no proj here
        if self._ln_post is None:
            raise RuntimeError("could not locate ln_post / post_layernorm on visual encoder")
        # `proj` may be None for HF CLIPVisionModel — that path returns 768-dim
        # CLS already, no projection. For OpenCLIP visual.proj is a Parameter
        # (Tensor, not Module): 768 x 512.

        self._handle = resblocks[self.layer].register_forward_hook(self._hook)

        # Pre-compute face index tensor (used at every forward; lives on hook device)
        face_mask = face_region_mask_7x7()
        self._face_idx_cache = torch.from_numpy(np.where(face_mask)[0].astype(np.int64))

        # Save & replace backbone.forward
        self._orig_forward = self.model.backbone.forward
        face_idx = self._face_idx_cache
        ln_post = self._ln_post
        proj = self._proj

        def wrapped_forward(pixel_values, **kwargs):
            # Trigger upstream forward (which fires our hook to populate
            # self._captured_tokens). The upstream forward computes the
            # 512-dim CLS pooler which we will DISCARD.
            _ = self._orig_forward(pixel_values, **kwargs)
            tokens = self._captured_tokens
            if tokens is None:
                raise RuntimeError("face-pool hook did not capture tokens")
            if tokens.shape[1] != self._n_patches_expected:
                raise RuntimeError(
                    f"unexpected token count {tokens.shape[1]}; "
                    f"expected {self._n_patches_expected}"
                )
            patches = tokens[:, 1:, :]  # drop CLS -> (B, 196, 768)
            face_idx_dev = face_idx.to(patches.device)
            face_patches = patches.index_select(1, face_idx_dev)  # (B, 49, 768)
            face_pool = face_patches.mean(dim=1)  # (B, 768)
            # Apply same post-block ops as the CLS path:
            #  OpenCLIP: ln_post then @ visual.proj (768 -> 512)
            #  HF CLIP : post_layernorm only (returns 768)
            face_pool = ln_post(face_pool)
            if proj is not None:
                face_pool = face_pool @ proj
            return {"pooler_output": face_pool}

        self.model.backbone.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._orig_forward is not None:
            self.model.backbone.forward = self._orig_forward
            self._orig_forward = None
        self._captured_tokens = None
        return False


@torch.no_grad()
def score_frames_face_pool(model, tensor: torch.Tensor, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Forward all canary frames with face-pool substituted for CLS-pool."""
    model.eval()
    scores: List[float] = []
    n = tensor.shape[0]
    t0 = time.time()
    with FacePoolMonkeyPatch(model, layer=L11_LAYER):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tensor[start:end].to(device, non_blocking=True)
            pred = model({"image": batch}, inference=True)
            if isinstance(pred, dict):
                if "prob" in pred:
                    probs = pred["prob"]
                else:
                    logits = pred.get("cls")
                    if logits is None:
                        for k in ("raw_logits", "logits", "classifier_logits", "pred_logits"):
                            if k in pred:
                                logits = pred[k]
                                break
                    if logits is None:
                        raise RuntimeError("could not extract logits")
                    if logits.dim() == 3:
                        logits = logits.mean(dim=1)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.softmax(pred, dim=-1)[:, 1]
            scores.extend(probs.float().cpu().tolist())
            if (start // batch_size) % 5 == 0:
                print(f"  [face-pool score] {end}/{n} ({time.time() - t0:.1f}s)", flush=True)
    return np.asarray(scores, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="Local ckpt path (.pth)")
    ap.add_argument("--key", default="SLOT_A_V2_STEP3500_face_pool")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[main] device={device}", flush=True)

    # Load cached canary frames
    print(f"[main] loading canary tensor from {CANARY_FRAMES}", flush=True)
    tensor = torch.load(CANARY_FRAMES, map_location="cpu")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    meta = pd.read_parquet(CANARY_META)
    print(
        f"[main] tensor={tuple(tensor.shape)} "
        f"reals={int((meta['label']==0).sum())} fakes={int((meta['label']==1).sum())}",
        flush=True,
    )

    # Load model from local ckpt
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] ckpt not found: {ckpt_path}", flush=True)
        return 1
    print(f"[main] loading model from {ckpt_path}", flush=True)
    t0 = time.time()
    model = load_model(str(ckpt_path), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    print(f"[main] model loaded in {time.time()-t0:.1f}s", flush=True)

    # Score with face-pool
    t0 = time.time()
    scores = score_frames_face_pool(model, tensor, device, args.batch_size)
    print(f"[main] face-pool scoring done in {time.time()-t0:.1f}s", flush=True)

    # Metrics
    metrics = aggregate_metrics(scores, meta)
    metrics["_meta/ckpt_key"] = args.key
    metrics["_meta/ckpt_path"] = str(ckpt_path)
    metrics["_meta/pool_variant"] = "face_region_centered_7x7"
    metrics["_meta/layer"] = L11_LAYER

    out_json = output_dir / f"{args.key}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(output_dir / f"{args.key}.scores.npy", scores)
    print(f"[ok] wrote {out_json}", flush=True)

    # Write sentinel
    sentinel = output_dir / "_done.json"
    with open(sentinel, "w") as f:
        json.dump({
            "status": "done",
            "output_json": str(out_json),
            "scores_npy": str(output_dir / f"{args.key}.scores.npy"),
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
