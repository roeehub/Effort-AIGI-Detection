"""Diagnostic G: verify corrective_probes.py L3 hook matches cached P8A features.

The cached features at analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz
were produced by intermediate_layer_probe.py using a different code path (different
hook setup, different module-traversal). If the corrective_probes.py L3 hook captures
a different tensor at the wrong spot, all corrective-probe results are contaminated.

This script runs the corrective_probes.py L3 extraction on a small subset of the
800-frame substrate using P8A, and compares numerically to the cached features
for those same frames. Pass criterion: per-frame max abs diff < 1e-3 on >95% of
frames. (Tiny floating-point variation is OK; structural mismatch is not.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Re-use functions from corrective_probes.py — folder name has dashes, can't `import` it
# directly; add its directory to sys.path and import the module file by name.
sys.path.insert(0, str(REPO_ROOT / "analysis" / "p18_probe_2026-05-01"))
from corrective_probes import (  # noqa: E402
    build_effort_detector_from_ckpt,
    extract_l3_features,
    detect_device,
)

P8A_CKPT = REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
P8A_L3_CACHE = REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "intermediate__P8A__layer03__n800.npz"
SAMPLED_CSV = REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"

N_VERIFY = 64  # Small sample is enough for a structural check.


def main():
    device = detect_device()
    print(f"Device: {device}")
    print(f"P8A ckpt: {P8A_CKPT}")
    print(f"Cache: {P8A_L3_CACHE}")

    # 1. Load P8A model via corrective_probes.py's path
    model, ck = build_effort_detector_from_ckpt(P8A_CKPT, device)

    # 2. Same df_valid construction as corrective_probes.py
    df = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:800]
    print(f"df_valid: {len(df_valid)} rows")

    # 3. Run extract_l3_features on first N_VERIFY frames only
    df_subset = df_valid.iloc[:N_VERIFY].reset_index(drop=True)
    print(f"Extracting L3 on first {N_VERIFY} frames via corrective_probes hook...")
    feats_via_hook = extract_l3_features(model, df_subset, device, batch_size=16)
    print(f"  hook features shape: {feats_via_hook.shape}")

    # 4. Load cached features
    cache = np.load(P8A_L3_CACHE, allow_pickle=True)
    feats_cached = cache["features"]  # (800, 768)
    valid_idx_cached = cache["valid_idx"]  # which positions are valid
    print(f"  cache features shape: {feats_cached.shape}, valid_idx range: {valid_idx_cached.min()}..{valid_idx_cached.max()}")

    # 5. The cache was produced over the full 800-frame df_valid. The first
    #    N_VERIFY rows of corrective_probes.py's df_valid should map to the
    #    first ~N_VERIFY rows of the cache (modulo dropouts of invalid paths).
    #    Compare row-by-row.
    cached_subset = feats_cached[:N_VERIFY]
    print(f"  cached subset shape: {cached_subset.shape}")
    print(f"  hook subset stats: min={feats_via_hook.min():.4f} max={feats_via_hook.max():.4f} mean={feats_via_hook.mean():.4f}")
    print(f"  cache subset stats: min={cached_subset.min():.4f} max={cached_subset.max():.4f} mean={cached_subset.mean():.4f}")

    diff = np.abs(feats_via_hook - cached_subset)
    per_frame_max = diff.max(axis=1)
    per_frame_mean = diff.mean(axis=1)
    print()
    print("=== Per-frame |hook - cache| ===")
    print(f"  max-of-max:  {per_frame_max.max():.6e}")
    print(f"  mean-of-max: {per_frame_max.mean():.6e}")
    print(f"  median-of-max: {np.median(per_frame_max):.6e}")
    print(f"  fraction with max < 1e-3: {(per_frame_max < 1e-3).mean():.4f}")
    print(f"  fraction with max < 1e-4: {(per_frame_max < 1e-4).mean():.4f}")

    # Also report cosine similarity (more robust to global scaling)
    def cosrow(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (an * bn).sum(axis=1)

    cs = cosrow(feats_via_hook, cached_subset)
    print()
    print("=== Per-frame cosine(hook, cache) ===")
    print(f"  min: {cs.min():.6f}, mean: {cs.mean():.6f}, max: {cs.max():.6f}")
    print(f"  fraction > 0.999: {(cs > 0.999).mean():.4f}")
    print(f"  fraction > 0.99: {(cs > 0.99).mean():.4f}")

    if (per_frame_max < 1e-3).mean() >= 0.95:
        print()
        print("VERDICT: PASS — hook captures the same tensor as the cached extraction.")
    elif (cs > 0.999).mean() >= 0.95:
        print()
        print("VERDICT: PASS-COSINE — features differ by a global affine but represent same direction.")
    else:
        print()
        print("VERDICT: FAIL — hook captures a different tensor. Investigate before trusting probe results.")


if __name__ == "__main__":
    main()
