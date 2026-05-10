"""Pre-test 3 — CLS attention vs face/swap-region IoU.

Question: does the model's L11 CLS-to-patch attention concentrate on the
face region (where forgery lives) or scatter across background (where IQ
shortcut signal lives)?

If attention IoU with face bbox is HIGH (>0.5) for P8A → model already does
local-forgery-attention; forgery-localization auxiliary loss adds little.
If LOW (<0.3) → there's substantial room. Path δ has structural promise.
If T3 step1500 IoU > P8A IoU → T3's mechanism implicitly added local attention.

Method:
  1. Load each ckpt model
  2. Hook L11 attention layer to capture CLS-to-patch attention weights
  3. Forward sample of 50-100 frames (mix of fakes and reals)
  4. Reshape attention to 14×14 patch grid (B16 ViT)
  5. Convert face bbox to 14×14 patch coverage
  6. Compute IoU between top-K attended patches and face-coverage patches
  7. Aggregate per ckpt × label (real/fake)

Output: outputs/pretest3_attention_iou.csv
        outputs/pretest3_attention_iou_PROFILE.md
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import DETECTOR
import yaml

OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
TRIPTYCH_CSV = (REPO_ROOT / "analysis/embedding_triptych_2026-04-30/outputs"
                / "triptych_p8a_slot2_slot3/sampled_frames.csv")

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config/train_config.yaml"

CKPTS = {
    "P8A": REPO_ROOT / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "T3_S1_step1500": REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_ckpts_t3/T3_SLOT1_step1500.pth",
    "T3_S1_step2500": REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_ckpts_t3/T3_SLOT1_step2500.pth",
    "E2B": REPO_ROOT / "analysis/iq_perlayer_probe_2026-05-08/_cache/e2b_top_n_step3200.pth",
}

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
PATCH_SIZE = 16
INPUT_SIZE = 224
GRID = INPUT_SIZE // PATCH_SIZE  # 14

N_SAMPLE = 100  # frames to compute attention on


def load_model(ckpt_path: Path, device: torch.device):
    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        sd = ckpt
        model_config = {}
    model = DETECTOR[cfg["model_name"]](cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model


def get_resblocks(model):
    if hasattr(model.backbone, "visual"):
        v = model.backbone.visual
    else:
        v = model.backbone
    return v.transformer.resblocks


def load_preprocess(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    img_rs = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
    return tensor, (h, w)


def face_to_patch_mask(face_x, face_y, face_w, face_h, src_h, src_w):
    """Convert face bbox in source image coordinates to 14x14 patch mask in 224 input."""
    if any(pd_isna(v) for v in [face_x, face_y, face_w, face_h]):
        return None
    # Scale to input coords
    sx = INPUT_SIZE / src_w
    sy = INPUT_SIZE / src_h
    bx = face_x * sx
    by = face_y * sy
    bw = face_w * sx
    bh = face_h * sy
    # Convert to patch grid (14x14)
    px0 = int(np.floor(bx / PATCH_SIZE))
    py0 = int(np.floor(by / PATCH_SIZE))
    px1 = int(np.ceil((bx + bw) / PATCH_SIZE))
    py1 = int(np.ceil((by + bh) / PATCH_SIZE))
    px0 = max(0, min(GRID, px0)); py0 = max(0, min(GRID, py0))
    px1 = max(0, min(GRID, px1)); py1 = max(0, min(GRID, py1))
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[py0:py1, px0:px1] = True
    return mask


def pd_isna(v):
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def compute_cls_attention(model, tensor, device):
    """Forward and capture L11 attention from CLS to patch tokens.
    Returns (14x14) attention map averaged across heads.
    """
    resblocks = get_resblocks(model)
    target = resblocks[-1]  # L11

    # Patch attention: hook on the inner attn module's QK.
    # OpenCLIP B16 uses MultiheadAttention; we need to extract attn weights
    # by running with need_weights=True. Easier: monkey-patch the forward
    # to set need_weights.
    captured = {}
    orig_forward = target.attention if hasattr(target, "attention") else None

    # Find MultiheadAttention module
    attn_module = None
    if hasattr(target, "attn"):
        attn_module = target.attn
    elif hasattr(target, "attention"):
        attn_module = target.attention

    if attn_module is None:
        return None

    # Hook into the attn module's forward to capture attention weights
    orig_attn_forward = attn_module.forward

    def patched_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        out = orig_attn_forward(*args, **kwargs)
        # MultiheadAttention returns (output, attn_weights) or just output
        if isinstance(out, tuple) and len(out) >= 2:
            captured["attn"] = out[1]
        return out

    attn_module.forward = patched_forward
    try:
        with torch.inference_mode():
            x = tensor.unsqueeze(0).to(device)
            _ = model.backbone(x)
    finally:
        attn_module.forward = orig_attn_forward

    if "attn" not in captured:
        return None
    attn = captured["attn"]  # (batch, seq, seq) or (batch, heads, seq, seq)
    if attn.dim() == 4:
        attn = attn.mean(dim=1)  # average heads
    # CLS is token 0; we want CLS-to-patches: row 0 of (seq, seq)
    if attn.dim() == 3:
        cls_attn = attn[0, 0, 1:]  # skip self-attention to CLS
    else:
        cls_attn = attn[0, 1:]
    cls_attn = cls_attn.cpu().numpy()
    n_patches = GRID * GRID
    if len(cls_attn) < n_patches:
        return None
    grid = cls_attn[:n_patches].reshape(GRID, GRID)
    return grid


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    print("Loading triptych...")
    df = pd.read_csv(TRIPTYCH_CSV).iloc[:N_SAMPLE].reset_index(drop=True)
    print(f"  {len(df)} frames; labels: {df['label'].value_counts().to_dict()}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    md = ["# Pre-test 3 — CLS attention vs face IoU", "",
          "For each ckpt, compute L11 CLS attention to patches on first 100 triptych frames; convert to 14x14 grid; compare to face bbox.", "",
          "If attention concentrates on face → model uses local face features.",
          "If attention scatters / on background → forgery-localization aux loss could redirect.",
          ""]

    for ckpt_label, ckpt_path in CKPTS.items():
        if not ckpt_path.exists():
            print(f"  SKIP {ckpt_label}: {ckpt_path} not found")
            continue
        print(f"\n=== {ckpt_label} ===")
        model = load_model(ckpt_path, device)

        ious_real, ious_fake = [], []
        attn_centroid_in_face_real, attn_centroid_in_face_fake = [], []
        attn_face_mass_real, attn_face_mass_fake = [], []
        n_done, n_skip = 0, 0
        for i, r in df.iterrows():
            local_path = Path(r["local_path"])
            if not local_path.exists():
                n_skip += 1; continue
            tensor, src_hw = load_preprocess(local_path)
            if tensor is None:
                n_skip += 1; continue
            face_mask = face_to_patch_mask(r.get("face_bbox_x"), r.get("face_bbox_y"),
                                            r.get("face_bbox_w"), r.get("face_bbox_h"),
                                            src_hw[0], src_hw[1])
            if face_mask is None or face_mask.sum() == 0:
                n_skip += 1; continue
            attn_grid = compute_cls_attention(model, tensor, device)
            if attn_grid is None:
                n_skip += 1; continue
            # Normalize attention to sum=1
            attn_grid = attn_grid / (attn_grid.sum() + 1e-12)
            # Top-K patches: K = number of face patches
            K = int(face_mask.sum())
            topK_idx = np.argsort(attn_grid.flatten())[-K:]
            topK_mask = np.zeros_like(attn_grid, dtype=bool).flatten()
            topK_mask[topK_idx] = True
            topK_mask = topK_mask.reshape(GRID, GRID)
            this_iou = iou(topK_mask, face_mask)
            # Centroid in face check
            yy, xx = np.indices(attn_grid.shape).astype(np.float32)
            cy = (attn_grid * yy).sum() / (attn_grid.sum() + 1e-12)
            cx = (attn_grid * xx).sum() / (attn_grid.sum() + 1e-12)
            cy_int, cx_int = int(round(cy)), int(round(cx))
            cy_int = min(max(0, cy_int), GRID - 1)
            cx_int = min(max(0, cx_int), GRID - 1)
            centroid_in_face = bool(face_mask[cy_int, cx_int])
            face_attn_mass = float(attn_grid[face_mask].sum())  # how much attn lands in face?

            if r["label"] == "fake":
                ious_fake.append(this_iou)
                attn_centroid_in_face_fake.append(centroid_in_face)
                attn_face_mass_fake.append(face_attn_mass)
            else:
                ious_real.append(this_iou)
                attn_centroid_in_face_real.append(centroid_in_face)
                attn_face_mass_real.append(face_attn_mass)
            n_done += 1
        print(f"  computed for {n_done} frames ({n_skip} skipped)")
        if ious_real:
            print(f"  REAL: mean IoU={np.mean(ious_real):.3f}, p25={np.percentile(ious_real,25):.3f}, p75={np.percentile(ious_real,75):.3f}, centroid_in_face={np.mean(attn_centroid_in_face_real):.2f}, face_attn_mass={np.mean(attn_face_mass_real):.3f}")
        if ious_fake:
            print(f"  FAKE: mean IoU={np.mean(ious_fake):.3f}, p25={np.percentile(ious_fake,25):.3f}, p75={np.percentile(ious_fake,75):.3f}, centroid_in_face={np.mean(attn_centroid_in_face_fake):.2f}, face_attn_mass={np.mean(attn_face_mass_fake):.3f}")
        rows.append({
            "ckpt": ckpt_label,
            "n_real": len(ious_real), "n_fake": len(ious_fake),
            "real_iou_mean": np.mean(ious_real) if ious_real else None,
            "fake_iou_mean": np.mean(ious_fake) if ious_fake else None,
            "real_centroid_in_face_pct": np.mean(attn_centroid_in_face_real) if attn_centroid_in_face_real else None,
            "fake_centroid_in_face_pct": np.mean(attn_centroid_in_face_fake) if attn_centroid_in_face_fake else None,
            "real_face_attn_mass_mean": np.mean(attn_face_mass_real) if attn_face_mass_real else None,
            "fake_face_attn_mass_mean": np.mean(attn_face_mass_fake) if attn_face_mass_fake else None,
        })
        del model

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "pretest3_attention_iou.csv", index=False)
    md.append("")
    md.append("| ckpt | n_real | n_fake | real IoU | fake IoU | real face_attn_mass | fake face_attn_mass |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df_out.iterrows():
        md.append(f"| {r['ckpt']} | {int(r['n_real'])} | {int(r['n_fake'])} | "
                  f"{r['real_iou_mean']:.3f} | {r['fake_iou_mean']:.3f} | "
                  f"{r['real_face_attn_mass_mean']:.3f} | {r['fake_face_attn_mass_mean']:.3f} |")
    (OUT / "pretest3_PROFILE.md").write_text("\n".join(md))
    print("\nDone.")

if __name__ == "__main__":
    main()
