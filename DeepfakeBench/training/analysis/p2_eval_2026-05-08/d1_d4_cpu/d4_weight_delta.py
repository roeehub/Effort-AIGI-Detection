#!/usr/bin/env python3
"""D4 — Weight-delta probe across P2 slots and P8A.

Compares state_dicts of {Slot A, Slot B, Slot C, Slot D-step6000, Slot D-step19000, P8A_step5000}
pairwise. Reports per-tensor L2 distance + cosine similarity grouped by transformer block,
so we can see WHERE in the encoder each slot moved differently.

P8A_step5000 is downloaded once if not present, mirroring the GCS path used in the recent
P1 / RLP / PD eval folders.

Output:
- ``weight_delta_summary.csv`` — pairwise per-block summary statistics
- ``weight_delta_per_tensor.csv`` — full per-tensor data (large)
- ``weight_delta_FINDINGS.md`` — short factual narrative
"""
from __future__ import annotations

import logging
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("d4_weight_delta")

THIS = Path(__file__).resolve().parent
CKPT_DIR = THIS / "ckpts"
OUT_DIR = THIS / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# P8A reference path per arena/checkpoint_maps/teams_target_domain.*.yaml
P8A_GCS = ("gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/"
           "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth")
P8A_LOCAL = CKPT_DIR / "p8a_step5000.pth"

CKPTS_LOCAL: Dict[str, Path] = {
    "slotA_top_n_step500": CKPT_DIR / "slotA_top_n_step500.pth",
    "slotB_top_n_step500": CKPT_DIR / "slotB_top_n_step500.pth",
    "slotC_top_n_step7000": CKPT_DIR / "slotC_top_n_step7000.pth",
    "slotD_top_n_step6000": CKPT_DIR / "slotD_top_n_step6000.pth",
    "slotD_top_n_step19000": CKPT_DIR / "slotD_top_n_step19000.pth",
    "p8a_step5000": P8A_LOCAL,
}


def ensure_p8a() -> bool:
    if P8A_LOCAL.exists():
        return True
    log.info("downloading P8A reference -> %s", P8A_LOCAL)
    cp = subprocess.run(["gcloud", "storage", "cp", P8A_GCS, str(P8A_LOCAL)],
                        capture_output=True, text=True, timeout=600)
    if cp.returncode != 0:
        log.warning("P8A download failed (rc=%d):\n%s", cp.returncode, cp.stderr[:400])
        return False
    return True


def load_state(path: Path) -> OrderedDict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()
                     if isinstance(v, torch.Tensor))
    return sd


def block_of(name: str) -> str:
    """Tag a parameter name with a coarse block label for grouping."""
    if "patch_embed" in name or "conv1.weight" in name or "class_embedding" in name or "positional_embedding" in name:
        return "embed"
    if "ln_pre" in name or "ln_post" in name:
        return "layernorm_outer"
    if "transformer.resblocks" in name or "blocks." in name:
        # extract block index
        for tok in name.split("."):
            if tok.isdigit():
                return f"resblock_{int(tok):02d}"
    if "proj" in name and "transformer" not in name:
        return "visual_proj"
    if "head" in name or "classifier" in name or "fc" in name:
        return "head"
    return "other"


def pairwise(sdA: OrderedDict, sdB: OrderedDict, label_a: str, label_b: str) -> Tuple[List[dict], List[dict]]:
    common = [k for k in sdA if k in sdB and sdA[k].shape == sdB[k].shape and sdA[k].dtype == sdB[k].dtype]
    rows_per_tensor: List[dict] = []
    for k in common:
        a = sdA[k].float().flatten()
        b = sdB[k].float().flatten()
        delta = (a - b)
        l2 = delta.norm().item()
        denom = max(a.norm().item(), 1e-12)
        rel = l2 / denom
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()) \
            if a.numel() > 0 else float("nan")
        rows_per_tensor.append({"a": label_a, "b": label_b, "tensor": k,
                                "block": block_of(k), "numel": a.numel(),
                                "l2_delta": l2, "rel_l2_delta": rel, "cosine": cos})
    df = pd.DataFrame(rows_per_tensor)
    if df.empty:
        return rows_per_tensor, []
    summary: List[dict] = []
    for blk, sub in df.groupby("block"):
        summary.append({
            "a": label_a, "b": label_b, "block": blk,
            "n_tensors": len(sub),
            "total_params": int(sub["numel"].sum()),
            "mean_rel_l2": float(sub["rel_l2_delta"].mean()),
            "max_rel_l2": float(sub["rel_l2_delta"].max()),
            "mean_cosine": float(sub["cosine"].mean()),
            "min_cosine": float(sub["cosine"].min()),
        })
    return rows_per_tensor, summary


def main():
    log.info("=" * 70)
    log.info("D4 — weight-delta probe across P2 slots + P8A")
    log.info("=" * 70)
    have_p8a = ensure_p8a()
    states: Dict[str, OrderedDict] = {}
    for label, path in CKPTS_LOCAL.items():
        if not path.exists():
            log.warning("missing %s -> skip", path)
            continue
        log.info("loading %s ...", label)
        states[label] = load_state(path)
        log.info("  %d tensors", len(states[label]))
    if not have_p8a or "p8a_step5000" not in states:
        log.warning("P8A not available — pairs vs P8A will be skipped")
    pairs: List[Tuple[str, str]] = []
    p2_labels = [k for k in states if k != "p8a_step5000"]
    for i, a in enumerate(p2_labels):
        for b in p2_labels[i + 1:]:
            pairs.append((a, b))
    if "p8a_step5000" in states:
        for a in p2_labels:
            pairs.append((a, "p8a_step5000"))
    log.info("computing %d pairs ...", len(pairs))
    all_per_tensor: List[dict] = []
    all_summary: List[dict] = []
    for a, b in pairs:
        log.info("  %s  vs  %s", a, b)
        per_t, summ = pairwise(states[a], states[b], a, b)
        all_per_tensor.extend(per_t)
        all_summary.extend(summ)
    df_pt = pd.DataFrame(all_per_tensor)
    df_sm = pd.DataFrame(all_summary)
    pt_csv = OUT_DIR / "weight_delta_per_tensor.csv"
    sm_csv = OUT_DIR / "weight_delta_summary.csv"
    df_pt.to_csv(pt_csv, index=False)
    df_sm.to_csv(sm_csv, index=False)
    log.info("wrote %s (%d rows)", pt_csv, len(df_pt))
    log.info("wrote %s (%d rows)", sm_csv, len(df_sm))
    # Console summary preview
    print()
    print("=== summary preview (mean_rel_l2 per block, sorted by block) ===")
    if not df_sm.empty:
        block_order = sorted(df_sm["block"].unique(), key=lambda s: (s != "embed", s))
        for blk in block_order:
            sub = df_sm[df_sm["block"] == blk]
            print(f"\n[{blk}]  n_tensors_per_pair={int(sub['n_tensors'].iloc[0]) if len(sub) else 0}")
            print(sub[["a", "b", "mean_rel_l2", "min_cosine"]].to_string(index=False))


if __name__ == "__main__":
    main()
