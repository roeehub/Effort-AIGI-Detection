"""FROZEN_PAIR_HEAD_PROBE_2026-05-06 — runnable probe.

State: BLUEPRINT-READY but BLOCKED at the data-coverage step.

This script runs end-to-end IF a feature cache that covers the paired
frames already exists on disk. Run with:

    python analysis/frozen_pair_head_probe_2026-05-06/run_probe.py \
        --features analysis/<NEW_CACHE>/p8a_paired_features.npz \
        --pair_gaps_csv analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv \
        --output_dir analysis/frozen_pair_head_probe_2026-05-06/outputs/

The required cache schema (`.npz`):

    features:   float32 (N, D)  — frozen final-layer features.
    frame_path: object (N,)     — gs:// or local path matching pair_gaps_csv.
    label:      int (N,)        — 0=real, 1=fake.

If --features is missing or covers <200 same-source pairs, the script exits
with a coverage-only report (already written to outputs/coverage_report.md
when the parent agent ran).

The pair-rank loss expects per-batch index alignment between paired
real/fake items; we build pair-id arrays and a custom Dataset to enforce that.

CPU-only PyTorch is sufficient for a 1- or 2-layer head on 512-dim features
with ~5000 pairs; expected training time per head: 30-90 seconds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def load_features(features_npz: Path) -> Dict[str, np.ndarray]:
    d = np.load(features_npz, allow_pickle=True)
    keys = list(d.keys())
    if "frame_path" not in keys:
        raise RuntimeError(
            f"Cache {features_npz} does not store frame_path; cannot map to "
            "pair structure. Use a cache produced with a frame_path-aware "
            "extractor (see analysis/clip_vs_p8a_viso_2026-05-03/extract_*.py "
            "for a working pattern)."
        )
    return {
        "features": d["features"],
        "frame_path": np.asarray(d["frame_path"]),
        "label": np.asarray(d["label"]) if "label" in keys else None,
    }


def map_pair_features(
    feats: Dict[str, np.ndarray],
    pair_gaps_csv: Path,
) -> pd.DataFrame:
    """Returns a DataFrame with one row per cross-product pair where BOTH the
    real and fake feature vectors are present in the cache."""
    pg = pd.read_csv(pair_gaps_csv)
    path_to_idx = {p: i for i, p in enumerate(feats["frame_path"].tolist())}
    pg["real_idx"] = pg["real_path"].map(path_to_idx)
    pg["fake_idx"] = pg["fake_path"].map(path_to_idx)
    paired = pg.dropna(subset=["real_idx", "fake_idx"]).copy()
    paired["real_idx"] = paired["real_idx"].astype(int)
    paired["fake_idx"] = paired["fake_idx"].astype(int)
    return paired


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 0):
        super().__init__()
        if d_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 1),
            )
        else:
            self.net = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_head(
    feats_real: np.ndarray, feats_fake: np.ndarray,
    pair_real_idx: np.ndarray, pair_fake_idx: np.ndarray,
    group_id: np.ndarray | None,
    head_kind: str,                  # "ce" / "ce_pair" / "ce_pair_group"
    margin: float = 0.5,
    lambda_pair: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size_pairs: int = 32,
    seed: int = 737,
):
    """CPU-only.  Feeds CE on real+fake stacked, plus pair-rank if requested.

    Returns: head, history dict.
    """
    assert TORCH_OK, "PyTorch required"
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_pairs = len(pair_real_idx)
    d = feats_real.shape[1]
    head = LinearHead(d, d_hidden=0)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    feats_real_t = torch.from_numpy(feats_real).float()
    feats_fake_t = torch.from_numpy(feats_fake).float()
    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n_pairs)
        epoch_ce, epoch_pr = 0.0, 0.0
        n_batches = (n_pairs + batch_size_pairs - 1) // batch_size_pairs
        for b in range(n_batches):
            idx = perm[b * batch_size_pairs : (b + 1) * batch_size_pairs]
            r_idx = pair_real_idx[idx]
            f_idx = pair_fake_idx[idx]
            r_feat = feats_real_t[r_idx]
            f_feat = feats_fake_t[f_idx]
            r_score = head(r_feat)
            f_score = head(f_feat)
            ce = F.binary_cross_entropy_with_logits(
                torch.cat([r_score, f_score]),
                torch.cat([torch.zeros_like(r_score), torch.ones_like(f_score)]),
            )
            loss = ce
            pair_loss = torch.tensor(0.0)
            if head_kind in ("ce_pair", "ce_pair_group"):
                gap = f_score - r_score
                pair_loss = F.softplus(margin - gap).mean()
                if head_kind == "ce_pair_group" and group_id is not None:
                    g = torch.from_numpy(group_id[idx]).long()
                    losses_per = F.softplus(margin - gap)
                    g_loss = []
                    for gv in g.unique():
                        m = g == gv
                        if m.any():
                            g_loss.append(losses_per[m].mean())
                    if g_loss:
                        pair_loss = torch.stack(g_loss).mean()
                loss = ce + lambda_pair * pair_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_ce += ce.item()
            epoch_pr += pair_loss.item() if isinstance(pair_loss, torch.Tensor) else 0.0
        history.append({"epoch": epoch, "ce": epoch_ce / n_batches, "pair": epoch_pr / n_batches})
    return head, history


def evaluate_head(head, feats_real, feats_fake, pair_real_idx, pair_fake_idx) -> Dict:
    """Returns AUC, FPR@0.5, recall@0.5, P(fake>real), pair_gap distribution."""
    head.eval()
    with torch.no_grad():
        r_score = head(torch.from_numpy(feats_real).float()).numpy()
        f_score = head(torch.from_numpy(feats_fake).float()).numpy()
    # AUC over all
    y = np.concatenate([np.zeros(len(r_score)), np.ones(len(f_score))])
    s = np.concatenate([r_score, f_score])
    auc = float(roc_auc_score(y, s)) if SKLEARN_OK else float("nan")
    # τ=0.5 on sigmoid; equivalent to logit threshold = 0.0
    fp_real = float((r_score > 0.0).mean())
    rec_fake = float((f_score > 0.0).mean())
    # Pair-rank
    r_sub = r_score[pair_real_idx]
    f_sub = f_score[pair_fake_idx]
    p_fake_gt_real = float((f_sub > r_sub).mean())
    pair_gap = (f_sub - r_sub)
    return {
        "auc": auc,
        "fpr_at_0_5_logit": fp_real,
        "recall_at_0_5_logit": rec_fake,
        "p_fake_gt_real_on_pairs": p_fake_gt_real,
        "pair_gap_mean": float(pair_gap.mean()),
        "pair_gap_median": float(np.median(pair_gap)),
        "pair_gap_p_le_0": float((pair_gap <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=None,
                    help=".npz with features+frame_path+label, covering paired frames")
    ap.add_argument("--pair_gaps_csv", type=str,
                    default="analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv")
    ap.add_argument("--output_dir", type=str,
                    default="analysis/frozen_pair_head_probe_2026-05-06/outputs/")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lambda_pair", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Coverage check
    if args.features is None or not Path(args.features).exists():
        msg = (
            f"--features is required and must exist on disk. Provided: "
            f"{args.features!r}. See outputs/coverage_report.md for the list "
            "of caches that exist locally and the reason they fail to cover "
            "same-source pairs. See FINDINGS.md for the extraction blueprint."
        )
        print(msg)
        return 0

    feats = load_features(Path(args.features))
    if not TORCH_OK or not SKLEARN_OK:
        print("PyTorch and sklearn required for end-to-end probe.")
        return 1
    pairs = map_pair_features(feats, Path(args.pair_gaps_csv))
    n_pairs = len(pairs)
    print(f"Pairs covered by cache: {n_pairs}")
    if n_pairs < 200:
        print("Insufficient pair coverage (<200). Exiting.")
        with open(out / "summary.json", "w") as f:
            json.dump({"verdict": "INSUFFICIENT_COVERAGE",
                       "n_pairs_covered_by_cache": int(n_pairs)}, f, indent=2)
        return 0

    # Build feature arrays per real / fake side
    feat_arr = feats["features"]
    real_idx_arr = pairs["real_idx"].to_numpy()
    fake_idx_arr = pairs["fake_idx"].to_numpy()
    # group_id from method × transport
    pairs["group"] = pairs["method"].astype(str) + "__" + pairs.get("fake_transport", "raw").astype(str)
    g_map = {g: i for i, g in enumerate(sorted(pairs["group"].unique()))}
    group_id = pairs["group"].map(g_map).to_numpy()

    # Train/test split at canonical_subject level for substrate-out
    rng = np.random.RandomState(737)
    subjects = sorted(pairs["canonical_subject"].unique())
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * 0.2))
    test_subjects = set(subjects[:n_test])
    test_mask = pairs["canonical_subject"].isin(test_subjects).to_numpy()
    train_mask = ~test_mask

    results = {}
    for kind in ["ce", "ce_pair", "ce_pair_group"]:
        head, hist = train_head(
            feat_arr, feat_arr,
            real_idx_arr[train_mask], fake_idx_arr[train_mask],
            group_id[train_mask] if kind == "ce_pair_group" else None,
            head_kind=kind,
            margin=args.margin,
            lambda_pair=args.lambda_pair,
            epochs=args.epochs,
        )
        m = evaluate_head(
            head, feat_arr, feat_arr,
            real_idx_arr[test_mask], fake_idx_arr[test_mask],
        )
        results[kind] = m
        print(f"{kind}: {m}")

    pd.DataFrame.from_dict(results, orient="index").to_csv(out / "head_metrics.csv")
    with open(out / "summary.json", "w") as f:
        json.dump({"verdict": "RAN", "n_pairs": int(n_pairs), "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
