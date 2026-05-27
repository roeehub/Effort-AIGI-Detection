"""Per-arm ROC curve on lockbox subset of the 800-frame substrate.

Tells us: at FIXED lockbox-real FPR (e.g., P8A's 0.12), what's each arm's
lockbox-fake recall? This isolates whether the trade is necessarily worse,
or whether different operating points reveal different rankings.

Cheap CPU. Uses cached CLS features + ArcFace s-scaling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
OUTPUT_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs"

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
    n = len(df_valid)
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    method = df_valid["method"].astype(str).to_numpy()

    cls = {arm: np.load(CACHE_DIR / f"final_cls__{arm}__n800.npz", allow_pickle=True)["features"] for arm in CKPTS}
    head_w = {arm: load_head_weight(arm) for arm in CKPTS}
    probs: Dict[str, np.ndarray] = {}
    for arm, s in ARCFACE_S.items():
        fn = l2norm_rows(cls[arm].astype(np.float64))
        wn = l2norm_rows(head_w[arm])
        cos_logits = fn @ wn.T
        margin = cos_logits[:, 1] - cos_logits[:, 0]
        probs[arm] = 1.0 / (1.0 + np.exp(-s * margin))

    # ============================================================
    # Lockbox-only ROC analysis
    # ============================================================
    print("=== Lockbox-only ROC analysis ===")
    print(f"  n_lockbox_real={int((is_lockbox & (label==0)).sum())}, n_lockbox_fake={int((is_lockbox & (label==1)).sum())}")

    lb_mask = is_lockbox
    lb_y = label[lb_mask]
    out: Dict = {}

    for arm in CKPTS:
        scores = probs[arm][lb_mask]
        fpr, tpr, taus = roc_curve(lb_y, scores)
        auc = roc_auc_score(lb_y, scores)
        out[arm] = {"auc_lockbox": float(auc), "n_lockbox": int(lb_y.size)}
        print(f"\n  {arm} lockbox AUC = {auc:.4f}")

        # Find recall at fixed FPR points
        for tgt_fpr in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            ix = np.searchsorted(fpr, tgt_fpr, side="right") - 1
            if 0 <= ix < len(fpr):
                r = float(tpr[ix])
                t = float(taus[ix]) if ix < len(taus) else None
                print(f"    FPR ≤ {tgt_fpr:.2f}: recall = {r:.4f}, τ ≈ {t:.4f}")
                out[arm].setdefault("recall_at_fpr", {})[f"{tgt_fpr:.2f}"] = {"recall": r, "tau": t}

    # ============================================================
    # Per-arm: at FIXED tau, recall on each lockbox fake bucket
    # ============================================================
    print("\n=== Lockbox-fake bucket recall at fixed τ ===")
    for tgt_fpr in [0.05, 0.10]:
        print(f"\n  -- Operating point: aggregate FPR ≤ {tgt_fpr:.2f} --")
        for arm in CKPTS:
            scores = probs[arm][lb_mask]
            fpr, tpr, taus = roc_curve(lb_y, scores)
            ix = np.searchsorted(fpr, tgt_fpr, side="right") - 1
            tau = float(taus[ix]) if 0 <= ix < len(taus) else None
            if tau is None:
                continue
            print(f"    {arm} (aggregate τ={tau:.4f}, FPR={float(fpr[ix]):.3f}, recall={float(tpr[ix]):.3f}):")
            for m in pd.Series(method[is_lockbox & (label == 1)]).value_counts().index:
                m_mask = is_lockbox & (label == 1) & (method == m)
                if m_mask.sum() == 0:
                    continue
                rec = float((probs[arm][m_mask] >= tau).mean())
                print(f"      {m:<35} n={int(m_mask.sum()):>3}  recall={rec:.4f}")
            for c in ["webcam", "phone_screen", "normal_photo"]:
                from numpy import asarray
                cap_array = df_valid["clip_capture_mode"].astype(str).to_numpy()
                c_mask = is_lockbox & (label == 0) & (cap_array == c)
                if c_mask.sum() < 3:
                    continue
                fp = float((probs[arm][c_mask] >= tau).mean())
                print(f"      [real] {c:<25} n={int(c_mask.sum()):>3}  FPR={fp:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / "roc_curves_2026-05-02.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
