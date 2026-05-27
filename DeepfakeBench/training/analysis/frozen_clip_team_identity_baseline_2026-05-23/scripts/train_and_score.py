"""Train a linear (and MLP) head on frozen-CLIP-L11 features and score the team-identity cohort.

Inputs:
- D8's cached CLIP-frozen L11 features for 2000 dev_real + 2000 dev_fake + 414 lockbox_real + 425 lockbox_fake
  at analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz
  (already in the same OpenCLIP B16 DataComp.XL space at L11, INTER_LINEAR preprocessing, CLS pool).
- D8's dev/lockbox sample order can be re-derived from the same parquet + seed=42
  (we re-derive to get labels; reuse the D8 features array directly).
- Team-identity CLIP features extracted in scripts/extract_clip_features.py:
  outputs/clip_frozen_l11__team_identity_n{N}.npz with arrays (features, frame_paths, labels, humans, roles, deploy_relevant).

Heads:
- LR: LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver='lbfgs') — per D8.
- MLP: MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True,
       n_iter_no_change=10, validation_fraction=0.15, random_state=42, n_jobs=1) — single hidden layer 256u.

Training corpora (Option A first):
- "DEV" = D8's 2000 dev_real + 2000 dev_fake
- "DEV+LB" = "DEV" + 414 lockbox_real + 425 lockbox_fake (includes the lockbox pool we use for τ comparison)
  This is an "informational" head that has seen lockbox-domain reals; used as an upper-bound bound on what frozen-CLIP can do.

τ-modes (cross-ckpt constants from project_deployment_three_modes_slot_a_v2_2026-05-21):
- A: 0.535 (recall-leaning)
- B: 0.78  (contract-compliant)
- C: 0.87  (FPR-leaning)
Plus dev-calibrated 5% FPR τ as a methodological cross-check.

Output:
- outputs/per_frame_baseline.csv — (frame_path, human, role, label, prob_lr_dev, prob_mlp_dev, prob_lr_devlb, prob_mlp_devlb)
- outputs/per_human_baseline.csv — (head, human, role, n, mean_prob, metric_mode_A, _B, _C, _devcal5)
- outputs/training_summary.csv — (head, corpus, n_train, train_auc, lockbox_auc, lockbox_fpr_devcal5, lockbox_recall_devcal5)
- outputs/comparison_table.csv — full T5 table (rows = ckpts, cols = team-aggregate FPR @B / min-fake-recall @B / per-human FPR @B)
- outputs/comparison_table.md — same in markdown form for the FACTS doc.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPANDED_READOUT = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23"
D8_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced"
THIS_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

LOCKBOX_PARQUET = REPO_ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
D8_CLIP_CACHE = D8_DIR / "outputs/clip_frozen_l11__n4839.npz"
TEAM_FEAT_CACHE_GLOB = "clip_frozen_l11__team_identity_n*.npz"
PER_HUMAN_SUMMARY_FT = EXPANDED_READOUT / "outputs/per_human_summary.csv"
PER_FRAME_FULL = EXPANDED_READOUT / "outputs/per_frame_full.csv"

SEED = 42

# τ-modes (cross-ckpt constants per project_deployment_three_modes_slot_a_v2_2026-05-21)
MODES = {
    "mode_A_tau_0_535": 0.535,
    "mode_B_tau_0_78": 0.78,
    "mode_C_tau_0_87": 0.87,
}
FPR_TARGET = 0.05

logger = logging.getLogger("train_score")


# ---------- Loading D8 + dev/lockbox labels ------------------------------

def replay_dev_lockbox() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate D8's dev/lockbox sampling for label alignment to the cached features."""
    df = pd.read_parquet(LOCKBOX_PARQUET)
    df = df[df["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].copy()
    dev_real = dev_df[dev_df["label"] == "real"]
    dev_fake = dev_df[dev_df["label"] == "fake"]
    rng = np.random.default_rng(seed=42)
    dev_real_idx = rng.choice(len(dev_real), size=2000, replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=2000, replace=False)
    dev_sample = pd.concat(
        [dev_real.iloc[dev_real_idx], dev_fake.iloc[dev_fake_idx]]
    ).reset_index(drop=True)
    lb_df = df[df["split"] == "lockbox"].copy().reset_index(drop=True)
    return dev_sample, lb_df


def load_d8_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dev_feats, dev_labels_int, lb_feats, lb_labels_int).
    Labels: 0=real, 1=fake.
    """
    blob = np.load(D8_CLIP_CACHE)
    clip_feats = blob["features"].astype(np.float32)  # (4839, 768)
    assert clip_feats.shape == (4839, 768)
    dev_clip = clip_feats[:4000]
    lb_clip = clip_feats[4000:]
    dev_df, lb_df = replay_dev_lockbox()
    dev_labels = (dev_df["label"] == "fake").astype(np.int64).to_numpy()
    lb_labels = (lb_df["label"] == "fake").astype(np.int64).to_numpy()
    assert len(dev_labels) == 4000 and len(lb_labels) == 839
    return dev_clip, dev_labels, lb_clip, lb_labels


def load_team_features() -> Tuple[np.ndarray, pd.DataFrame]:
    """Load team-identity CLIP features extracted by extract_clip_features.py."""
    matches = sorted(OUTPUTS.glob(TEAM_FEAT_CACHE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No team-identity CLIP feature cache found in {OUTPUTS} matching {TEAM_FEAT_CACHE_GLOB}. "
            "Run extract_clip_features.py first."
        )
    npz_path = matches[-1]
    blob = np.load(npz_path, allow_pickle=True)
    feats = blob["features"].astype(np.float32)  # (N, 768)
    frame_paths = blob["frame_paths"]
    labels = blob["labels"].astype(np.int64)
    humans = blob["humans"]
    roles = blob["roles"]
    deploy_flags = blob["deploy_relevant"].astype(bool)

    df = pd.DataFrame({
        "frame_path": frame_paths,
        "label": labels,
        "human": humans,
        "role": roles,
        "deploy_relevant": deploy_flags,
    })
    logger.info("loaded team features %s shape=%s; df rows=%d", npz_path.name, feats.shape, len(df))
    return feats, df


# ---------- Head fitting ------------------------------------------------

def fit_lr(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.15,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def score(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


# ---------- Calibration / scoring helpers --------------------------------

def calibrate_tau_fpr(real_scores: np.ndarray, fpr_target: float = FPR_TARGET) -> float:
    sorted_scores = np.sort(real_scores)
    n = len(sorted_scores)
    k = int(np.ceil((1.0 - fpr_target) * n))
    if k >= n:
        return float(sorted_scores[-1] + 1e-9)
    return float(sorted_scores[k])


def fpr_at(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    real = probs[labels == 0]
    if len(real) == 0:
        return float("nan")
    return float((real >= tau).mean())


def recall_at(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    fake = probs[labels == 1]
    if len(fake) == 0:
        return float("nan")
    return float((fake >= tau).mean())


def auc_safe(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


# ---------- Main pipeline ------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(OUTPUTS / "_train_score.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    t0 = time.time()

    # 1) Load D8 cached features + labels
    dev_X, dev_y, lb_X, lb_y = load_d8_features()
    logger.info("DEV (%d real, %d fake), LOCKBOX (%d real, %d fake)",
                int((dev_y == 0).sum()), int((dev_y == 1).sum()),
                int((lb_y == 0).sum()), int((lb_y == 1).sum()))

    # 2) Load team-identity features
    team_X, team_df = load_team_features()
    n_team = len(team_df)
    # Mask out frames that failed to load (NaN rows)
    good_mask = ~np.isnan(team_X).any(axis=1)
    n_good = int(good_mask.sum())
    logger.info("team features: %d total, %d successfully extracted", n_team, n_good)

    # 3) Train heads on DEV (4000)
    logger.info("fitting LR(DEV) ...")
    lr_dev = fit_lr(dev_X, dev_y)
    logger.info("fitting MLP(DEV) ...")
    mlp_dev = fit_mlp(dev_X, dev_y)
    # 4) Train heads on DEV+LB (informational upper bound)
    devlb_X = np.concatenate([dev_X, lb_X], axis=0)
    devlb_y = np.concatenate([dev_y, lb_y], axis=0)
    logger.info("fitting LR(DEV+LB) ...")
    lr_devlb = fit_lr(devlb_X, devlb_y)
    logger.info("fitting MLP(DEV+LB) ...")
    mlp_devlb = fit_mlp(devlb_X, devlb_y)

    # 5) Diagnostics: training AUCs and transfer AUCs
    diag_rows = []
    for head_name, clf, X_tr, y_tr in [
        ("LR_DEV", lr_dev, dev_X, dev_y),
        ("MLP_DEV", mlp_dev, dev_X, dev_y),
        ("LR_DEV_LB", lr_devlb, devlb_X, devlb_y),
        ("MLP_DEV_LB", mlp_devlb, devlb_X, devlb_y),
    ]:
        probs_train = score(clf, X_tr)
        train_auc = auc_safe(y_tr, probs_train)
        # On lockbox alone (whether or not it was in training)
        probs_lb = score(clf, lb_X)
        lb_auc = auc_safe(lb_y, probs_lb)
        # Dev-calibrated 5% τ + lockbox FPR/recall at that τ
        probs_dev = score(clf, dev_X)
        tau5 = calibrate_tau_fpr(probs_dev[dev_y == 0], 0.05)
        lb_fpr_at_dev5 = fpr_at(probs_lb, lb_y, tau5)
        lb_recall_at_dev5 = recall_at(probs_lb, lb_y, tau5)
        diag_rows.append({
            "head": head_name,
            "n_train": int(len(y_tr)),
            "n_train_real": int((y_tr == 0).sum()),
            "n_train_fake": int((y_tr == 1).sum()),
            "train_auc": train_auc,
            "lockbox_auc": lb_auc,
            "tau_dev5pct": tau5,
            "lockbox_fpr_at_dev5pct": lb_fpr_at_dev5,
            "lockbox_recall_at_dev5pct": lb_recall_at_dev5,
        })
        logger.info(
            "  %s: train_auc=%.4f  lockbox_auc=%.4f  tau5=%.4f  lb_fpr@tau5=%.4f  lb_recall@tau5=%.4f",
            head_name, train_auc, lb_auc, tau5, lb_fpr_at_dev5, lb_recall_at_dev5,
        )
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUTPUTS / "training_summary.csv", index=False)

    # 6) Score the team-identity cohort with all 4 heads.
    team_probs = {}
    for head_name, clf in [
        ("prob_LR_DEV", lr_dev),
        ("prob_MLP_DEV", mlp_dev),
        ("prob_LR_DEV_LB", lr_devlb),
        ("prob_MLP_DEV_LB", mlp_devlb),
    ]:
        probs = np.full(n_team, np.nan, dtype=np.float32)
        probs[good_mask] = score(clf, team_X[good_mask])
        team_df[head_name] = probs
        team_probs[head_name] = probs

    # Save per-frame CSV
    team_df.to_csv(OUTPUTS / "per_frame_baseline.csv", index=False)
    logger.info("wrote per-frame baseline scores: %s", OUTPUTS / "per_frame_baseline.csv")

    # 7) Per-human metrics at each mode
    per_human_rows = []
    deploy_team = team_df[team_df.deploy_relevant].reset_index(drop=True)
    for head_col in ["prob_LR_DEV", "prob_MLP_DEV", "prob_LR_DEV_LB", "prob_MLP_DEV_LB"]:
        for human in sorted(deploy_team.human.unique()):
            sub = deploy_team[deploy_team.human == human]
            for role in sorted(sub.role.unique()):
                ss = sub[sub.role == role]
                probs = ss[head_col].to_numpy()
                # Drop NaNs (frames that failed to extract)
                pmask = ~np.isnan(probs)
                p = probs[pmask]
                n_total = len(probs)
                n_scored = int(pmask.sum())
                if n_scored == 0:
                    continue
                # For "real" role, FPR = fraction(p >= tau); for fake roles, recall = fraction(p >= tau)
                row = {
                    "head": head_col,
                    "human": human,
                    "role": role,
                    "n_frames": n_total,
                    "n_scored": n_scored,
                    "mean_prob": float(p.mean()),
                }
                for mode_name, tau in MODES.items():
                    metric = float((p >= tau).mean())
                    row[f"metric_{mode_name}"] = metric
                per_human_rows.append(row)
    per_human_df = pd.DataFrame(per_human_rows)
    per_human_df.to_csv(OUTPUTS / "per_human_baseline.csv", index=False)
    logger.info("wrote per-human baseline metrics: %s", OUTPUTS / "per_human_baseline.csv")

    # 8) Build the comparison table vs FT'd ckpts (per_human_summary.csv)
    ft_per_human = pd.read_csv(PER_HUMAN_SUMMARY_FT)
    # Add the new baseline rows by mapping head -> ckpt name
    baseline_ckpts = {
        "prob_LR_DEV":     "FrozenCLIP_LR_DEV",
        "prob_MLP_DEV":    "FrozenCLIP_MLP_DEV",
        "prob_LR_DEV_LB":  "FrozenCLIP_LR_DEV_LB",
        "prob_MLP_DEV_LB": "FrozenCLIP_MLP_DEV_LB",
    }
    aug_rows = []
    for _, r in per_human_df.iterrows():
        aug_rows.append({
            "ckpt": baseline_ckpts[r["head"]],
            "human": r["human"],
            "role": r["role"],
            "n_frames": r["n_frames"],
            "n_scored": r["n_scored"],
            "mean_prob": r["mean_prob"],
            # Add a tau_0_5 column too for parity with FT table
            "metric_tau_0_5": float(((team_df.loc[(team_df.human == r["human"]) & (team_df.role == r["role"]), r["head"]].dropna()) >= 0.5).mean()),
            "metric_mode_A_tau_0_535": r["metric_mode_A_tau_0_535"],
            "metric_mode_B_tau_0_78": r["metric_mode_B_tau_0_78"],
            "metric_mode_C_tau_0_87": r["metric_mode_C_tau_0_87"],
        })
    combined_per_human = pd.concat([ft_per_human, pd.DataFrame(aug_rows)], ignore_index=True)
    combined_per_human.to_csv(OUTPUTS / "per_human_combined_with_baselines.csv", index=False)
    logger.info("wrote combined per-human (FT + baselines): %s", OUTPUTS / "per_human_combined_with_baselines.csv")

    # 9) The T5 comparison table — focus on mode B
    summary_rows = []
    all_ckpts = sorted(combined_per_human.ckpt.unique())
    for ck in all_ckpts:
        sub = combined_per_human[combined_per_human.ckpt == ck]
        real = sub[sub.role == "real"]
        fakes = sub[sub.role.str.startswith("fake_target_")]
        # Team-aggregate FPR @B (sample-weighted)
        weighted = (real.metric_mode_B_tau_0_78 * real.n_scored).sum() / max(real.n_scored.sum(), 1)
        max_fpr_B = real.metric_mode_B_tau_0_78.max()
        min_fake_recall_B = fakes.metric_mode_B_tau_0_78.min() if len(fakes) else float("nan")
        weighted_A = (real.metric_mode_A_tau_0_535 * real.n_scored).sum() / max(real.n_scored.sum(), 1)
        max_fpr_A = real.metric_mode_A_tau_0_535.max()
        min_fake_recall_A = fakes.metric_mode_A_tau_0_535.min() if len(fakes) else float("nan")
        # Per-human FPR @B
        per_human_fpr_B = {}
        for human in sorted(real.human.unique()):
            v = real[real.human == human].metric_mode_B_tau_0_78
            per_human_fpr_B[f"fpr_B__{human}"] = float(v.iloc[0]) if len(v) else float("nan")
        per_human_fake_B = {}
        for human in sorted(fakes.human.unique()):
            v = fakes[fakes.human == human].metric_mode_B_tau_0_78
            per_human_fake_B[f"fake_recall_B__{human}"] = float(v.iloc[0]) if len(v) else float("nan")
        summary_rows.append({
            "ckpt": ck,
            "team_aggregate_fpr_B": float(weighted),
            "team_max_fpr_B": float(max_fpr_B) if len(real) else float("nan"),
            "team_min_fake_recall_B": float(min_fake_recall_B) if len(fakes) else float("nan"),
            "team_aggregate_fpr_A": float(weighted_A),
            "team_max_fpr_A": float(max_fpr_A) if len(real) else float("nan"),
            "team_min_fake_recall_A": float(min_fake_recall_A) if len(fakes) else float("nan"),
            **per_human_fpr_B,
            **per_human_fake_B,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS / "comparison_table.csv", index=False)
    logger.info("wrote comparison_table.csv:\n%s", summary_df.to_string())

    # 10) Markdown comparison
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if not np.isnan(v) else "nan"
        return str(v)
    lines = ["| Ckpt | Team-agg FPR @B | Team-max FPR @B | Team-min fake recall @B | Roee_Win FPR | dor FPR | Xinhe FPR | Xiang FPR | Noyn FPR | dor fake recall | Xinhe fake recall | Xiang fake recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r.ckpt} | {fmt(r.team_aggregate_fpr_B)} | {fmt(r.team_max_fpr_B)} | {fmt(r.team_min_fake_recall_B)} | "
            f"{fmt(r.get('fpr_B__Roee_Windows', float('nan')))} | {fmt(r.get('fpr_B__dor', float('nan')))} | "
            f"{fmt(r.get('fpr_B__Xinhe', float('nan')))} | {fmt(r.get('fpr_B__Xiang', float('nan')))} | "
            f"{fmt(r.get('fpr_B__Noyn', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__dor', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__Xinhe', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__Xiang', float('nan')))} |"
        )
    md = "\n".join(lines)
    (OUTPUTS / "comparison_table.md").write_text(md, encoding="utf-8")

    logger.info("DONE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
