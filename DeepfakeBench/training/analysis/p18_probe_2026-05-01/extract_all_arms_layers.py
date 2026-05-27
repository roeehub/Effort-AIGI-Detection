"""Extract L3, L6, L9, L11 + final [CLS] features for all 3 arms (P8A, T, C).

Used to back the bootstrap (Diagnostic A), paired-test (B), s-scaling (C), and
multi-layer probe (E). Outputs go to the existing _features_cache_2026-04-30/
folder using the same naming conventions as intermediate_layer_probe.py:

    intermediate__{ARM}__layer{LL}__n800.npz  (per-layer, dim 768)
    final_cls__{ARM}__n800.npz                (final pooled, dim 512)

Skips already-cached files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "p18_probe_2026-05-01"))

from corrective_probes import (  # noqa: E402
    build_effort_detector_from_ckpt,
    detect_device,
    load_image_clip_normalize,
)

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"

ARMS = {
    "P8A": REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "P18T": Path("/tmp/p18_ckpts/xpbvc1e4__periodic_step4000.pth"),
    "P18C": Path("/tmp/p18_ckpts/rgt4kw2u__periodic_step4000.pth"),
}

LAYERS = [3, 6, 9, 11]  # 0 we already have via cache; not load-bearing for Phase 1A


def get_resblocks(model):
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("no transformer.resblocks under backbone(.visual)")


def extract_all(model, df_valid: pd.DataFrame, device, batch_size: int = 16) -> Dict:
    """Single-pass per frame, capture L3/L6/L9/L11 [CLS] + final [CLS]."""
    resblocks = get_resblocks(model)
    captured: Dict[int, list] = {ix: [] for ix in LAYERS}

    def make_hook(ix: int):
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                # corrective_probes shape detection: batch-first if dim0 == batch
                # We don't know batch from inside hook, so use heuristic: seq_len=197
                # for ViT-B-16 224×224. The shorter dim is batch.
                if output.shape[0] >= output.shape[1]:  # seq-first (197, B, D)
                    cls = output[0]  # (B, D)
                else:  # batch-first (B, 197, D)
                    cls = output[:, 0]  # (B, D)
            elif output.dim() == 2:
                cls = output
            else:
                raise RuntimeError(f"unexpected shape {tuple(output.shape)}")
            captured[ix].append(cls.detach().cpu().to(torch.float32).numpy())
        return hook

    handles = [resblocks[ix].register_forward_hook(make_hook(ix)) for ix in LAYERS]

    feats_cls = np.zeros((len(df_valid), 512), dtype=np.float32)
    valid_idx: List[int] = []
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        with torch.inference_mode():
            out = model.backbone(x)
        if isinstance(out, dict):
            out = out["pooler_output"]
        feats_cls[batch_idx] = out.detach().cpu().float().numpy()
        valid_idx.extend(batch_idx)
        batch_imgs.clear()
        batch_idx.clear()

    try:
        for i, row in df_valid.iterrows():
            img = load_image_clip_normalize(row["local_path"])
            if img is None:
                continue
            batch_imgs.append(img)
            batch_idx.append(i)
            if len(batch_imgs) >= batch_size:
                flush()
        flush()
    finally:
        for h in handles:
            h.remove()

    per_layer = {}
    for ix in LAYERS:
        per_layer[ix] = np.concatenate(captured[ix], axis=0)
    return {"cls": feats_cls, "per_layer": per_layer, "valid_idx": np.array(valid_idx, dtype=np.int64)}


def cache_layer(arm: str, layer: int, n: int) -> Path:
    return CACHE_DIR / f"intermediate__{arm}__layer{layer:02d}__n{n}.npz"


def cache_cls(arm: str, n: int) -> Path:
    return CACHE_DIR / f"final_cls__{arm}__n{n}.npz"


def main():
    device = detect_device()
    print(f"Device: {device}")

    df = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:800]
    n = len(df_valid)
    print(f"df_valid: {n} rows")

    for arm, ckpt_path in ARMS.items():
        print()
        print(f"=== Arm: {arm} ===")
        print(f"  ckpt: {ckpt_path}")

        # Determine what's missing
        layer_caches = {ix: cache_layer(arm, ix, n) for ix in LAYERS}
        cls_cache_p = cache_cls(arm, n)
        missing_layers = [ix for ix, p in layer_caches.items() if not p.exists()]
        missing_cls = not cls_cache_p.exists()

        # P8A's intermediate layers are already cached under "P8A" prefix in
        # the existing intermediate_layer_probe.py format. Use those instead.
        if arm == "P8A":
            for ix in LAYERS:
                old = CACHE_DIR / f"intermediate__P8A__layer{ix:02d}__n{n}.npz"
                if old.exists():
                    layer_caches[ix] = old
                    if ix in missing_layers:
                        missing_layers.remove(ix)
                    print(f"  L{ix:02d}: reuse existing {old.name}")

        if not missing_layers and not missing_cls:
            print("  all cached, skip.")
            continue

        # Build model + extract
        model, _ = build_effort_detector_from_ckpt(ckpt_path, device)
        print(f"  extracting layers {missing_layers if missing_layers else '[]'} + cls={'YES' if missing_cls else 'no'}")
        out = extract_all(model, df_valid, device, batch_size=16)

        for ix in missing_layers:
            np.savez_compressed(layer_caches[ix], features=out["per_layer"][ix].astype(np.float32), valid_idx=out["valid_idx"])
            print(f"  saved L{ix:02d}: {layer_caches[ix].name} shape={out['per_layer'][ix].shape}")

        if missing_cls:
            np.savez_compressed(cls_cache_p, features=out["cls"].astype(np.float32), valid_idx=out["valid_idx"])
            print(f"  saved CLS: {cls_cache_p.name} shape={out['cls'].shape}")

        del model
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
