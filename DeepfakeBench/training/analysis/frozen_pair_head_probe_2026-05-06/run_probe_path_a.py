"""FROZEN_PAIR_HEAD_PROBE_2026-05-06 — Phase 0h Path-A wrapper.

Trains 3 heads (CE / CE+pair-rank / CE+pair-rank+GroupDRO) on frozen features
extracted by Vertex job 3077166152858730496. Filters to ok==1 rows before
training so failed-download placeholders (zero features) don't poison the head.

Wraps the existing run_probe.py logic but adds the ok-filter step.

CPU-only. n_jobs=1 enforced for any sklearn calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def load_features_filtered(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    ok = d["ok"] == 1
    return {
        "features": d["features"][ok],
        "frame_path": np.asarray(d["frame_path"])[ok],
        "label": np.asarray(d["label"])[ok],
        "scores": d["scores"][ok],
        "n_total": int(len(d["ok"])),
        "n_ok": int(ok.sum()),
    }


def map_pair_features(feats, pair_gaps_csv: Path) -> pd.DataFrame:
    pg = pd.read_csv(pair_gaps_csv)
    path_to_idx = {p: i for i, p in enumerate(feats["frame_path"].tolist())}
    pg["real_idx"] = pg["real_path"].map(path_to_idx)
    pg["fake_idx"] = pg["fake_path"].map(path_to_idx)
    paired = pg.dropna(subset=["real_idx", "fake_idx"]).copy()
    paired["real_idx"] = paired["real_idx"].astype(int)
    paired["fake_idx"] = paired["fake_idx"].astype(int)
    return paired


class LinearHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(
    feat_arr: np.ndarray,
    pair_real_idx: np.ndarray, pair_fake_idx: np.ndarray,
    group_id,
    head_kind: str,            # "ce" / "ce_pair" / "ce_pair_group"
    margin: float = 0.5,
    lambda_pair: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size_pairs: int = 32,
    seed: int = 737,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_pairs = len(pair_real_idx)
    d = feat_arr.shape[1]
    head = LinearHead(d)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    feat_t = torch.from_numpy(feat_arr).float()
    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n_pairs)
        epoch_ce, epoch_pr = 0.0, 0.0
        n_batches = (n_pairs + batch_size_pairs - 1) // batch_size_pairs
        for b in range(n_batches):
            idx = perm[b * batch_size_pairs : (b + 1) * batch_size_pairs]
            r_idx = pair_real_idx[idx]
            f_idx = pair_fake_idx[idx]
            r_feat = feat_t[r_idx]
            f_feat = feat_t[f_idx]
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
        history.append({"epoch": epoch, "ce": epoch_ce / max(1, n_batches), "pair": epoch_pr / max(1, n_batches)})
    return head, history


def evaluate_head(head, feat_arr, pair_real_idx, pair_fake_idx, threshold_logit: float = 0.0) -> dict:
    head.eval()
    feat_t = torch.from_numpy(feat_arr).float()
    with torch.no_grad():
        all_scores = head(feat_t).numpy()
    r_score = all_scores[pair_real_idx]
    f_score = all_scores[pair_fake_idx]
    y = np.concatenate([np.zeros(len(r_score)), np.ones(len(f_score))])
    s = np.concatenate([r_score, f_score])
    auc = float(roc_auc_score(y, s))
    fp_real = float((r_score > threshold_logit).mean())
    rec_fake = float((f_score > threshold_logit).mean())
    accuracy = float(((s > threshold_logit) == y).mean())
    p_fake_gt_real = float((f_score > r_score).mean())
    pair_gap = (f_score - r_score)
    return {
        "auc": auc,
        "fpr_at_tau_logit_0": fp_real,
        "recall_at_tau_logit_0": rec_fake,
        "accuracy_at_tau_logit_0": accuracy,
        "p_fake_gt_real_on_pairs": p_fake_gt_real,
        "pair_gap_mean": float(pair_gap.mean()),
        "pair_gap_median": float(np.median(pair_gap)),
        "pair_gap_p_le_0": float((pair_gap <= 0).mean()),
        "n_pairs": int(len(pair_real_idx)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--pair_gaps_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lambda_pair", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=737)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feats = load_features_filtered(Path(args.features))
    print(f"Loaded {feats['n_ok']} ok features (out of {feats['n_total']} total)")

    pairs = map_pair_features(feats, Path(args.pair_gaps_csv))
    print(f"Pairs covered: {len(pairs)}")

    if len(pairs) < 200:
        with open(out / "summary.json", "w") as f:
            json.dump({"verdict": "INSUFFICIENT_COVERAGE", "n_pairs": int(len(pairs))}, f, indent=2)
        print("Insufficient pair coverage. Exiting.")
        return

    feat_arr = feats["features"]
    real_idx_arr = pairs["real_idx"].to_numpy()
    fake_idx_arr = pairs["fake_idx"].to_numpy()

    pairs["group"] = pairs["method"].astype(str) + "__" + pairs.get("fake_transport", "raw").astype(str)
    g_map = {g: i for i, g in enumerate(sorted(pairs["group"].unique()))}
    group_id = pairs["group"].map(g_map).to_numpy()

    # Train/test split at canonical_subject level for substrate-out
    rng = np.random.RandomState(args.seed)
    subjects = sorted(pairs["canonical_subject"].unique())
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * 0.2))
    test_subjects = set(subjects[:n_test])
    test_mask = pairs["canonical_subject"].isin(test_subjects).to_numpy()
    train_mask = ~test_mask
    print(f"Subjects: {len(subjects)} total, {len(test_subjects)} test ({sorted(test_subjects)})")
    print(f"Train pairs: {train_mask.sum()}, test pairs: {test_mask.sum()}")

    results = {}
    for kind in ["ce", "ce_pair", "ce_pair_group"]:
        head, hist = train_head(
            feat_arr,
            real_idx_arr[train_mask], fake_idx_arr[train_mask],
            group_id[train_mask] if kind == "ce_pair_group" else None,
            head_kind=kind,
            margin=args.margin,
            lambda_pair=args.lambda_pair,
            epochs=args.epochs,
            seed=args.seed,
        )
        m_train = evaluate_head(head, feat_arr, real_idx_arr[train_mask], fake_idx_arr[train_mask])
        m_test = evaluate_head(head, feat_arr, real_idx_arr[test_mask], fake_idx_arr[test_mask])
        results[kind] = {"train": m_train, "test": m_test}
        print(f"=== {kind} ===")
        print(f"  test  AUC={m_test['auc']:.4f} acc={m_test['accuracy_at_tau_logit_0']:.4f} p_fake>real={m_test['p_fake_gt_real_on_pairs']:.4f} gap_p_le_0={m_test['pair_gap_p_le_0']:.4f}")

    pd.DataFrame([
        {"head": k, "split": s, **v[s]}
        for k, v in results.items() for s in ["train", "test"]
    ]).to_csv(out / "head_metrics.csv", index=False)

    # Lift table (B vs A, C vs A) on test split
    a_test = results["ce"]["test"]
    b_test = results["ce_pair"]["test"]
    c_test = results["ce_pair_group"]["test"]
    lift_table = {
        "head_a_ce_test": a_test,
        "head_b_ce_pair_test": b_test,
        "head_c_ce_pair_group_test": c_test,
        "lift_b_minus_a_pair_gap_pp": (b_test["p_fake_gt_real_on_pairs"] - a_test["p_fake_gt_real_on_pairs"]) * 100,
        "lift_b_minus_a_accuracy_pp": (b_test["accuracy_at_tau_logit_0"] - a_test["accuracy_at_tau_logit_0"]) * 100,
        "lift_b_minus_a_auc_pp": (b_test["auc"] - a_test["auc"]) * 100,
        "lift_c_minus_a_pair_gap_pp": (c_test["p_fake_gt_real_on_pairs"] - a_test["p_fake_gt_real_on_pairs"]) * 100,
        "lift_c_minus_a_accuracy_pp": (c_test["accuracy_at_tau_logit_0"] - a_test["accuracy_at_tau_logit_0"]) * 100,
        "lift_c_minus_a_auc_pp": (c_test["auc"] - a_test["auc"]) * 100,
    }

    # Promotion gate verdict
    b_lift = lift_table["lift_b_minus_a_pair_gap_pp"]
    if b_lift >= 3:
        gate_verdict = "HEAD_SIDE_SIGNAL"
        recommendation = "Demote P1 in favour of head-only retrain (~$0-1)"
    elif b_lift >= 0:
        gate_verdict = "BORDERLINE"
        recommendation = "P1 viable but expected ROI lower than headline"
    else:
        gate_verdict = "NO_HEAD_SIDE_SIGNAL"
        recommendation = "P1 only justified if encoder change is strictly necessary"

    summary = {
        "phase": "0h",
        "source_npz": str(args.features),
        "n_features_ok": feats["n_ok"],
        "n_features_total": feats["n_total"],
        "n_pairs_covered": len(pairs),
        "n_train_pairs": int(train_mask.sum()),
        "n_test_pairs": int(test_mask.sum()),
        "test_subjects": sorted(list(test_subjects)),
        "results": results,
        "lift_table": lift_table,
        "promotion_gate": {
            "lift_b_minus_a_pair_gap_pp": b_lift,
            "verdict": gate_verdict,
            "recommendation": recommendation,
        },
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nLift B-A pair_gap (pp): {b_lift:+.2f}")
    print(f"Lift B-A accuracy (pp): {lift_table['lift_b_minus_a_accuracy_pp']:+.2f}")
    print(f"Verdict: {gate_verdict}")
    print(f"Wrote: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
