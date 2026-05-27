"""Mini-scorecard on the 800-frame substrate.

Uses cached CLS features + ArcFace s-scaling to compute deployment-relevant
metrics on the full eval substrate, not just the 40-frame dor subset:
- Per-bucket FPR @ τ ∈ {0.5, 0.92, 0.974} on lockbox reals
- Per-bucket fake-recall @ same τ on lockbox fakes
- Per-bucket sample sizes

Decides whether the dor-specific finding generalizes. If P18T improves on P8A
across many buckets, the case for running the full Vertex contract scorecard
strengthens; if P18T's improvements are dor-specific and elsewhere it
matches/regresses from P8A, dor is an outlier and a Vertex run might mislead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

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
    capture = df_valid["clip_capture_mode"].astype(str).to_numpy()
    print(f"df_valid: n={n}")

    # Inventory
    print("\n=== Substrate inventory ===")
    print(f"  total: {n}")
    print(f"  is_lockbox: {is_lockbox.sum()} (~dev: {(~is_lockbox).sum()})")
    print(f"  fake: {(label == 1).sum()}, real: {(label == 0).sum()}")
    print(f"  lockbox fakes: {(is_lockbox & (label==1)).sum()}")
    print(f"  lockbox reals: {(is_lockbox & (label==0)).sum()}")
    print("  by method:")
    for m in pd.Series(method).value_counts().head(15).index:
        n_m = int((method == m).sum())
        n_lb = int((is_lockbox & (method == m)).sum())
        print(f"    {m:<35} n={n_m:>4}  (lockbox: {n_lb})")
    print("  by capture_mode (where applicable):")
    for c in pd.Series(capture).value_counts().head(8).index:
        n_c = int((capture == c).sum())
        print(f"    {c:<25} n={n_c}")

    # Score each arm
    print("\n=== Scoring all arms (with ArcFace s-scaling) ===")
    cls = {arm: np.load(CACHE_DIR / f"final_cls__{arm}__n800.npz", allow_pickle=True)["features"] for arm in CKPTS}
    head_w = {arm: load_head_weight(arm) for arm in CKPTS}
    probs: Dict[str, np.ndarray] = {}
    for arm, s in ARCFACE_S.items():
        fn = l2norm_rows(cls[arm].astype(np.float64))
        wn = l2norm_rows(head_w[arm])
        cos_logits = fn @ wn.T
        margin = cos_logits[:, 1] - cos_logits[:, 0]
        probs[arm] = 1.0 / (1.0 + np.exp(-s * margin))
        print(f"  {arm} (s={s:.2f}): mean prob_fake={probs[arm].mean():.4f}")

    # Mini-scorecard: FPR on each lockbox-real bucket; recall on each lockbox-fake bucket
    print("\n=== Mini-scorecard at deployment τ values ===")
    out: Dict = {}
    # Lockbox real buckets: by method (real_pool_real, teams_real, etc.)
    print("\n-- LOCKBOX REALS by method, FPR @ deployment τ --")
    print(f"  {'method':<30}  n   τ=0.5    τ=0.92   τ=0.974   ::  P8A | P18T | P18C")
    for m in pd.Series(method[is_lockbox & (label == 0)]).value_counts().index:
        m_mask = is_lockbox & (label == 0) & (method == m)
        if m_mask.sum() == 0:
            continue
        n_m = int(m_mask.sum())
        line = f"  {m:<30}  {n_m:>3}  "
        bucket_data = {}
        for tau in [0.5, 0.92, 0.974]:
            fprs = {arm: float((probs[arm][m_mask] >= tau).mean()) for arm in CKPTS}
            bucket_data[f"tau_{tau}"] = fprs
            triple = f"{fprs['P8A']:.2f}|{fprs['P18T']:.2f}|{fprs['P18C']:.2f}"
            line += f"  {triple:>16}"
        out.setdefault("lockbox_reals_by_method", {})[m] = bucket_data
        print(line)

    # Lockbox real bonus: by capture_mode within real-pool
    print("\n-- LOCKBOX REALS by capture_mode (when applicable), FPR @ deployment τ --")
    print(f"  {'capture_mode':<25}  n   τ=0.5    τ=0.92   τ=0.974   ::  P8A | P18T | P18C")
    for c in pd.Series(capture[is_lockbox & (label == 0)]).value_counts().index[:6]:
        c_mask = is_lockbox & (label == 0) & (capture == c)
        if c_mask.sum() < 5:
            continue
        n_c = int(c_mask.sum())
        line = f"  {c:<25}  {n_c:>3}  "
        bucket_data = {}
        for tau in [0.5, 0.92, 0.974]:
            fprs = {arm: float((probs[arm][c_mask] >= tau).mean()) for arm in CKPTS}
            bucket_data[f"tau_{tau}"] = fprs
            triple = f"{fprs['P8A']:.2f}|{fprs['P18T']:.2f}|{fprs['P18C']:.2f}"
            line += f"  {triple:>16}"
        out.setdefault("lockbox_reals_by_capture_mode", {})[c] = bucket_data
        print(line)

    # Lockbox fakes — recall (1 - prob of <τ)
    print("\n-- LOCKBOX FAKES by method, RECALL (frac with prob_fake >= τ) @ deployment τ --")
    print(f"  {'method':<30}  n   τ=0.5    τ=0.92   τ=0.974   ::  P8A | P18T | P18C")
    for m in pd.Series(method[is_lockbox & (label == 1)]).value_counts().index:
        m_mask = is_lockbox & (label == 1) & (method == m)
        if m_mask.sum() == 0:
            continue
        n_m = int(m_mask.sum())
        line = f"  {m:<30}  {n_m:>3}  "
        bucket_data = {}
        for tau in [0.5, 0.92, 0.974]:
            recalls = {arm: float((probs[arm][m_mask] >= tau).mean()) for arm in CKPTS}
            bucket_data[f"tau_{tau}"] = recalls
            triple = f"{recalls['P8A']:.2f}|{recalls['P18T']:.2f}|{recalls['P18C']:.2f}"
            line += f"  {triple:>16}"
        out.setdefault("lockbox_fakes_recall_by_method", {})[m] = bucket_data
        print(line)

    # Aggregate FPR / recall over all lockbox reals/fakes
    print("\n-- AGGREGATE on lockbox --")
    for label_str, mask in [("all_lockbox_reals_FPR", is_lockbox & (label == 0)),
                              ("all_lockbox_fakes_RECALL", is_lockbox & (label == 1))]:
        n_m = int(mask.sum())
        line = f"  {label_str:<30}  {n_m:>3}  "
        for tau in [0.5, 0.92, 0.974]:
            vals = {arm: float((probs[arm][mask] >= tau).mean()) for arm in CKPTS}
            triple = f"{vals['P8A']:.2f}|{vals['P18T']:.2f}|{vals['P18C']:.2f}"
            line += f"  {triple:>16}"
        out.setdefault("aggregate_lockbox", {})[label_str] = vals
        print(line)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / "mini_scorecard_2026-05-02.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
