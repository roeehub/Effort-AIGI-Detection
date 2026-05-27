"""Decompose lockbox AUC by dor-shkedi vs non-dor reals.

Question: P18T regresses 12pp lockbox AUC vs P8A. Is this driven by dor-shkedi
specifically, or is it a generalized regression?

Build two AUC views:
1. Lockbox with ALL real identities (status quo).
2. Lockbox with dor_shkedi reals removed (test if non-dor is fine).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
CKPTS = {
    "P8A": REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "P18T": Path("/tmp/p18_ckpts/xpbvc1e4__periodic_step4000.pth"),
    "P18C": Path("/tmp/p18_ckpts/rgt4kw2u__periodic_step4000.pth"),
}
ARCFACE_S = {"P8A": 9.749250411987305, "P18T": 11.998499870300293, "P18C": 11.998499870300293}


def l2norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_head_weight(arm: str) -> np.ndarray:
    ck = torch.load(str(CKPTS[arm]), map_location="cpu", weights_only=False)
    state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    for key in ("head.weight", "module.head.weight"):
        if key in state:
            return state[key].cpu().numpy().astype(np.float64)
    raise RuntimeError("no head.weight")


def main():
    n = 800
    df = pd.read_csv(SAMPLED_CSV).iloc[:n].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:n]
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    is_dor = np.array([("dor_shkedi" in i) for i in identity])

    cls = {arm: np.load(CACHE_DIR / f"final_cls__{arm}__n800.npz", allow_pickle=True)["features"] for arm in CKPTS}
    head_w = {arm: load_head_weight(arm) for arm in CKPTS}

    probs = {}
    for arm, s in ARCFACE_S.items():
        fn = l2norm_rows(cls[arm].astype(np.float64))
        wn = l2norm_rows(head_w[arm])
        cos_logits = fn @ wn.T
        margin = cos_logits[:, 1] - cos_logits[:, 0]
        probs[arm] = 1.0 / (1.0 + np.exp(-s * margin))

    # AUC views
    print("=== Lockbox AUC views (full vs dor-excluded) ===")
    masks = {
        "lockbox_all": is_lockbox,
        "lockbox_minus_dor_reals": is_lockbox & ~(is_dor & (label == 0)),
        "lockbox_only_dor_reals_plus_all_fakes": is_lockbox & ((is_dor & (label == 0)) | (label == 1)),
        "lockbox_only_nondor_reals_plus_all_fakes": is_lockbox & ((~is_dor & (label == 0)) | (label == 1)),
    }
    for name, mask in masks.items():
        n_real = int((mask & (label == 0)).sum())
        n_fake = int((mask & (label == 1)).sum())
        print(f"\n  {name}  (n_real={n_real}, n_fake={n_fake})")
        if n_real == 0 or n_fake == 0:
            print("    skip (one class missing)")
            continue
        for arm in CKPTS:
            try:
                auc = roc_auc_score(label[mask], probs[arm][mask])
                print(f"    {arm} AUC = {auc:.4f}")
            except ValueError as e:
                print(f"    {arm} AUC error: {e}")


if __name__ == "__main__":
    main()
