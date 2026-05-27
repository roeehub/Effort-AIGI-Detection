"""V2: train_and_score.py + Option B (training-corpus head) + per-head FPR-calibrated metrics.

V2 adds, on top of v1:
- Option B heads: LR + MLP trained on a 3K real + 3K fake stratified sample of the actual
  training corpus (frame_properties.parquet). Inputs at
  outputs/clip_frozen_l11__training_optionB_n*.npz (extracted by extract_option_b_training.py).
- Per-head τ calibrated to team-aggregate real FPR ∈ {0.02, 0.05, 0.10} on the team-identity
  cohort; per-human fake recall at that τ. Same calibration applied to the FT'd ckpts.
- Combined table where every row carries: (head, mode, team-agg-FPR, per-human-fake-recall).
- Saves a "joint_calibrated_summary.csv" that's directly comparable across encoders.

Output files (in addition to v1's outputs):
- outputs/per_human_baseline_v2.csv (with Option B columns where available)
- outputs/training_summary_v2.csv (with Option B rows)
- outputs/joint_calibrated_summary.csv — per-head fake-recall at team-FPR ∈ {2,5,10}%
- outputs/comparison_table_v2.md
"""
from __future__ import annotations

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
TEAM_FEAT_GLOB = "clip_frozen_l11__team_identity_n*.npz"
OPTIONB_FEAT_GLOB = "clip_frozen_l11__training_optionB_n*.npz"
PER_HUMAN_SUMMARY_FT = EXPANDED_READOUT / "outputs/per_human_summary.csv"
PER_FRAME_FULL = EXPANDED_READOUT / "outputs/per_frame_full.csv"

SEED = 42
TEAM_FPR_TARGETS = [0.02, 0.05, 0.10]

logger = logging.getLogger("train_score_v2")


# ----- helpers ------------------------------------------------------------

def replay_dev_lockbox() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def load_d8_features():
    blob = np.load(D8_CLIP_CACHE)
    clip_feats = blob["features"].astype(np.float32)
    assert clip_feats.shape == (4839, 768)
    dev_clip = clip_feats[:4000]
    lb_clip = clip_feats[4000:]
    dev_df, lb_df = replay_dev_lockbox()
    dev_labels = (dev_df["label"] == "fake").astype(np.int64).to_numpy()
    lb_labels = (lb_df["label"] == "fake").astype(np.int64).to_numpy()
    assert len(dev_labels) == 4000 and len(lb_labels) == 839
    return dev_clip, dev_labels, lb_clip, lb_labels


def load_team_features():
    matches = sorted(OUTPUTS.glob(TEAM_FEAT_GLOB))
    blob = np.load(matches[-1], allow_pickle=True)
    feats = blob["features"].astype(np.float32)
    df = pd.DataFrame({
        "frame_path": blob["frame_paths"],
        "label": blob["labels"].astype(np.int64),
        "human": blob["humans"],
        "role": blob["roles"],
        "deploy_relevant": blob["deploy_relevant"].astype(bool),
    })
    return feats, df


def load_optionb_features():
    matches = sorted(OUTPUTS.glob(OPTIONB_FEAT_GLOB))
    if not matches:
        return None, None
    blob = np.load(matches[-1], allow_pickle=True)
    feats = blob["features"].astype(np.float32)
    labels = blob["labels"].astype(np.int64)
    # Drop NaN rows
    good = ~np.isnan(feats).any(axis=1)
    return feats[good], labels[good]


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    clf.fit(X, y)
    return clf


def fit_mlp(X, y):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.15,
        random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def score(clf, X):
    return clf.predict_proba(X)[:, 1]


def auc_safe(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ----- main pipeline -------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(OUTPUTS / "_train_score_v2.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    t0 = time.time()

    dev_X, dev_y, lb_X, lb_y = load_d8_features()
    logger.info("DEV: real=%d fake=%d, LOCKBOX: real=%d fake=%d",
                int((dev_y == 0).sum()), int((dev_y == 1).sum()),
                int((lb_y == 0).sum()), int((lb_y == 1).sum()))

    team_X, team_df = load_team_features()
    good_mask = ~np.isnan(team_X).any(axis=1)
    logger.info("team features: total=%d good=%d", len(team_df), int(good_mask.sum()))

    # Try to load Option B features
    optb_X, optb_y = load_optionb_features()
    have_optb = optb_X is not None
    if have_optb:
        logger.info("OptionB: %d frames (real=%d, fake=%d)",
                    len(optb_y), int((optb_y == 0).sum()), int((optb_y == 1).sum()))

    # ---- train heads ----
    heads_def = []
    logger.info("fitting LR(DEV) ...")
    heads_def.append(("LR_DEV", fit_lr(dev_X, dev_y), dev_X, dev_y, "DEV"))
    logger.info("fitting MLP(DEV) ...")
    heads_def.append(("MLP_DEV", fit_mlp(dev_X, dev_y), dev_X, dev_y, "DEV"))

    devlb_X = np.concatenate([dev_X, lb_X], axis=0)
    devlb_y = np.concatenate([dev_y, lb_y], axis=0)
    logger.info("fitting LR(DEV+LB) ...")
    heads_def.append(("LR_DEV_LB", fit_lr(devlb_X, devlb_y), devlb_X, devlb_y, "DEV+LB"))
    logger.info("fitting MLP(DEV+LB) ...")
    heads_def.append(("MLP_DEV_LB", fit_mlp(devlb_X, devlb_y), devlb_X, devlb_y, "DEV+LB"))

    if have_optb:
        logger.info("fitting LR(OPTB) ...")
        heads_def.append(("LR_OPTB", fit_lr(optb_X, optb_y), optb_X, optb_y, "OPTB"))
        logger.info("fitting MLP(OPTB) ...")
        heads_def.append(("MLP_OPTB", fit_mlp(optb_X, optb_y), optb_X, optb_y, "OPTB"))

        # Also "OPTB+DEV" combined: large diverse training corpus + DEV-domain teams substrate
        optbdev_X = np.concatenate([optb_X, dev_X], axis=0)
        optbdev_y = np.concatenate([optb_y, dev_y], axis=0)
        logger.info("fitting LR(OPTB+DEV) ...")
        heads_def.append(("LR_OPTB_DEV", fit_lr(optbdev_X, optbdev_y), optbdev_X, optbdev_y, "OPTB+DEV"))
        logger.info("fitting MLP(OPTB+DEV) ...")
        heads_def.append(("MLP_OPTB_DEV", fit_mlp(optbdev_X, optbdev_y), optbdev_X, optbdev_y, "OPTB+DEV"))

    # ---- diagnostics: train AUC + transfer AUCs ----
    diag_rows = []
    for hn, clf, X_tr, y_tr, corpus in heads_def:
        p_tr = score(clf, X_tr)
        train_auc = auc_safe(y_tr, p_tr)
        p_lb = score(clf, lb_X)
        lb_auc = auc_safe(lb_y, p_lb)
        p_dev = score(clf, dev_X)
        dev_auc = auc_safe(dev_y, p_dev)
        # τ at dev 5% FPR
        sorted_dev_real = np.sort(p_dev[dev_y == 0])
        k = int(np.ceil(0.95 * len(sorted_dev_real)))
        tau5 = float(sorted_dev_real[k]) if k < len(sorted_dev_real) else float(sorted_dev_real[-1] + 1e-9)
        lb_fpr5 = float((p_lb[lb_y == 0] >= tau5).mean()) if (lb_y == 0).any() else float("nan")
        lb_rec5 = float((p_lb[lb_y == 1] >= tau5).mean()) if (lb_y == 1).any() else float("nan")
        diag_rows.append({
            "head": hn,
            "corpus": corpus,
            "n_train": int(len(y_tr)),
            "n_train_real": int((y_tr == 0).sum()),
            "n_train_fake": int((y_tr == 1).sum()),
            "train_auc": train_auc,
            "dev_auc": dev_auc,
            "lockbox_auc": lb_auc,
            "tau_dev5pct": tau5,
            "lockbox_fpr_at_dev5pct": lb_fpr5,
            "lockbox_recall_at_dev5pct": lb_rec5,
        })
        logger.info("  %s (%s): train_auc=%.4f lockbox_auc=%.4f lb_fpr@dev5=%.4f lb_rec@dev5=%.4f",
                    hn, corpus, train_auc, lb_auc, lb_fpr5, lb_rec5)
    pd.DataFrame(diag_rows).to_csv(OUTPUTS / "training_summary_v2.csv", index=False)

    # ---- score team-identity frames with all heads ----
    n_team = len(team_df)
    for hn, clf, _, _, _ in heads_def:
        probs = np.full(n_team, np.nan, dtype=np.float32)
        probs[good_mask] = score(clf, team_X[good_mask])
        team_df[f"prob_{hn}"] = probs
    team_df.to_csv(OUTPUTS / "per_frame_baseline_v2.csv", index=False)

    # ---- Apples-to-apples calibration: per-head τ such that team-real-FPR ∈ targets ----
    deploy = team_df[team_df.deploy_relevant].reset_index(drop=True)
    team_real = deploy[deploy.role == "real"]
    fake_subs = {h: deploy[(deploy.human == h) & (deploy.role == f"fake_target_{h}")] for h in ["dor", "Xinhe", "Xiang"]}

    # Combined list of "ckpts": FT'd + baselines (each baseline = one head)
    ft_per_frame = pd.read_csv(PER_FRAME_FULL, low_memory=False)
    ft_per_frame_deploy = ft_per_frame[ft_per_frame.deploy_relevant.astype(bool)].reset_index(drop=True)
    ckpt_cols_ft = ["prob_P8A", "prob_E2B", "prob_T5C", "prob_SlotAv2_CLS", "prob_SlotAv2_FACE"]
    baseline_cols = [f"prob_{hn}" for hn, _, _, _, _ in heads_def]
    # Build a per-frame combined table by aligning on frame_path
    ft_per_frame_deploy = ft_per_frame_deploy[["frame_path", "human", "role"] + ckpt_cols_ft].copy()
    deploy_local = deploy[["frame_path", "human", "role"] + baseline_cols].copy()
    combined = ft_per_frame_deploy.merge(deploy_local, on=["frame_path", "human", "role"], how="inner")
    logger.info("combined per-frame table: %d rows", len(combined))

    # Compute per-head τ at team-aggregate real FPR target and per-human fake recall
    summary_rows = []
    real_mask = combined.role == "real"
    real_n = int(real_mask.sum())
    for head_col in ckpt_cols_ft + baseline_cols:
        head_short = head_col[len("prob_"):]
        p_team_real = combined.loc[real_mask, head_col].dropna().to_numpy()
        if len(p_team_real) == 0:
            continue
        # The mean prob can help diagnose calibration drift
        mean_real = float(p_team_real.mean())
        for fpr_t in TEAM_FPR_TARGETS:
            tau = float(np.quantile(p_team_real, 1 - fpr_t))
            row = {
                "head": head_short,
                "team_real_n": real_n,
                "team_real_n_scored": int(len(p_team_real)),
                "team_real_mean_prob": mean_real,
                "team_fpr_target": fpr_t,
                "tau_at_team_fpr": tau,
            }
            for h in ["dor", "Xinhe", "Xiang"]:
                sub = combined[(combined.human == h) & (combined.role == f"fake_target_{h}")]
                p_fake = sub[head_col].dropna().to_numpy()
                if len(p_fake) == 0:
                    row[f"recall_{h}"] = float("nan")
                    continue
                row[f"recall_{h}"] = float((p_fake >= tau).mean())
            row["min_recall"] = float(min(row.get(f"recall_{h}", float("nan")) for h in ["dor", "Xinhe", "Xiang"]))
            row["mean_recall"] = float(np.mean([row.get(f"recall_{h}", float("nan")) for h in ["dor", "Xinhe", "Xiang"]]))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS / "joint_calibrated_summary.csv", index=False)
    logger.info("joint_calibrated_summary.csv head:\n%s", summary_df.head(20).to_string())

    # ---- Build markdown comparison table at team-FPR 5% ----
    summary_5 = summary_df[summary_df.team_fpr_target == 0.05].copy()
    summary_5 = summary_5.sort_values("min_recall", ascending=False).reset_index(drop=True)
    lines = ["# Joint-calibrated comparison @ team-real-FPR = 5%",
             "",
             "Each head τ set so that fraction(team-real >= τ) = 0.05 on the 1,821 deploy-relevant real frames.",
             "Fake recall reported per-human on fake-attack cohorts. Higher = better.",
             "",
             "| Head | Mean real prob | τ | dor recall | Xinhe recall | Xiang recall | min recall | mean recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary_5.iterrows():
        lines.append(
            f"| {r['head']} | {r['team_real_mean_prob']:.3f} | {r['tau_at_team_fpr']:.4f} | "
            f"{r['recall_dor']:.3f} | {r['recall_Xinhe']:.3f} | {r['recall_Xiang']:.3f} | "
            f"{r['min_recall']:.3f} | {r['mean_recall']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    (OUTPUTS / "comparison_table_v2.md").write_text(md, encoding="utf-8")
    logger.info("wrote %s", OUTPUTS / "comparison_table_v2.md")

    logger.info("DONE v2 in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
