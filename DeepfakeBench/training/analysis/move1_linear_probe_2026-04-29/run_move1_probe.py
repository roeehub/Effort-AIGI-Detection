#!/usr/bin/env python3
"""Move 1 — frozen-feature linear probe (per master plan §9 Priority 1).

Tests: is P8A's eval-bucket viso AUC materially lower than train-bucket viso AUC?
A material gap (>0.10) with train-bucket > 0.80 vs eval-bucket < 0.70 means the
bucket gap is dispositive and licenses the $70 P14_DATA_FIX bet.

Inputs: cached P8A features.npz from probe_battery_2026-04-26.
Outputs: JSON + CSV summary in this directory.

Pools (source_name selectors):
  - TRAIN viso       : dl_bucket_visomaster (fake+real, training distribution)
  - EVAL viso clean+teams (no enhancers): proper_visomaster_*
  - EVAL viso enhanced (clean+teams)    : proper_visomaster_enhanced_* + proper_real_*__paired
  - EVAL viso enhanced v2               : visomaster_enhanced_v2_fake + tv2_visomaster_real
  - EVAL viso tv2 (Teams-v2 transport)  : tv2_visomaster_fake + tv2_visomaster_real

Probe: 5-fold stratified CV LogisticRegression on frozen 512-d ViT features,
n_jobs=1 per local-Mac discipline (memory feedback_sklearn_njobs.md).
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("move1-probe")

SEED = 737
HERE = Path(__file__).resolve().parent
FEATURES_NPZ = HERE / "p8a_features.npz"


def load_features(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["features"].astype(np.float32)
    src_name = np.array([str(v) for v in data["source_name"]])
    src_bucket = np.array([str(v) for v in data["source_bucket"]])
    labels = np.array([str(v) for v in data["label"]])
    probs = data["probs"].astype(np.float32)
    if probs.ndim == 1:
        prob_fake = probs
    elif probs.shape[1] == 1:
        prob_fake = probs[:, 0]
    else:
        prob_fake = probs[:, -1]
    return X, src_name, src_bucket, labels, prob_fake


def evaluate_pool(X, y, prob_fake_p8a, name, seed=SEED, n_splits=5):
    p8a_auc = roc_auc_score(y, prob_fake_p8a) if len(set(y.tolist())) > 1 else float("nan")
    p8a_acc = accuracy_score(y, (prob_fake_p8a >= 0.5).astype(int))

    n_min = min((y == 0).sum(), (y == 1).sum())
    n_splits = max(2, min(n_splits, n_min))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs: List[float] = []
    fold_accs: List[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        pred_prob = clf.predict_proba(X[te])[:, 1]
        pred_label = clf.predict(X[te])
        fold_aucs.append(float(roc_auc_score(y[te], pred_prob)))
        fold_accs.append(float(accuracy_score(y[te], pred_label)))

    return {
        "name": name,
        "n_total": int(len(y)),
        "n_fake": int((y == 1).sum()),
        "n_real": int((y == 0).sum()),
        "p8a_classifier_auc": float(p8a_auc),
        "p8a_classifier_acc_at_0.5": float(p8a_acc),
        "probe_auc_mean": float(np.mean(fold_aucs)),
        "probe_auc_std": float(np.std(fold_aucs)),
        "probe_acc_mean": float(np.mean(fold_accs)),
        "probe_fold_aucs": [float(a) for a in fold_aucs],
        "mean_prob_fake_on_fakes": float(prob_fake_p8a[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_prob_fake_on_reals": float(prob_fake_p8a[y == 0].mean()) if (y == 0).any() else float("nan"),
        "n_splits_used": n_splits,
    }


def make_pools(src_name: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    def has(names):
        return np.isin(src_name, names)

    pools = {
        "TRAIN_viso (dl_bucket_visomaster, fake+real same-bucket)": {
            "fake": has(["dl_bucket_visomaster_fake"]),
            "real": has(["dl_bucket_visomaster_real"]),
        },
        "EVAL_viso_clean_teams (proper_visomaster, no enhancers, paired reals)": {
            "fake": has(["proper_visomaster_clean_fake", "proper_visomaster_teams_fake"]),
            "real": has(["proper_real_clean__paired", "proper_real_teams__paired"]),
        },
        "EVAL_viso_enhanced (proper_visomaster_enhanced + paired reals)": {
            "fake": has(["proper_visomaster_enhanced_clean_fake", "proper_visomaster_enhanced_teams_fake"]),
            "real": has(["proper_real_clean__paired", "proper_real_teams__paired"]),
        },
        "EVAL_viso_enhanced_v2 (visomaster_enhanced_v2 + tv2 reals)": {
            "fake": has(["visomaster_enhanced_v2_fake"]),
            "real": has(["tv2_visomaster_real"]),
        },
        "EVAL_viso_tv2 (tv2_visomaster Teams-v2 transport, same-bucket reals)": {
            "fake": has(["tv2_visomaster_fake"]),
            "real": has(["tv2_visomaster_real"]),
        },
        # Deployment-realistic real pool (external VCD webcam) — tests whether the
        # camera-signature shortcut on reals is what drives the eval-suite AUC
        # down to 0.7527 (per master plan §1).
        "DEPLOY_train_viso_vs_VCD (dl_bucket_visomaster_fake + external_vcd_real)": {
            "fake": has(["dl_bucket_visomaster_fake"]),
            "real": has(["external_vcd_real"]),
        },
        "DEPLOY_eval_viso_clean_teams_vs_VCD (proper_visomaster_fakes + external_vcd_real)": {
            "fake": has(["proper_visomaster_clean_fake", "proper_visomaster_teams_fake"]),
            "real": has(["external_vcd_real"]),
        },
        "DEPLOY_eval_viso_enhanced_vs_VCD (proper_visomaster_enhanced_fakes + external_vcd_real)": {
            "fake": has(["proper_visomaster_enhanced_clean_fake", "proper_visomaster_enhanced_teams_fake"]),
            "real": has(["external_vcd_real"]),
        },
        "DEPLOY_eval_viso_v2_vs_VCD (visomaster_enhanced_v2_fake + external_vcd_real)": {
            "fake": has(["visomaster_enhanced_v2_fake"]),
            "real": has(["external_vcd_real"]),
        },
        "DEPLOY_eval_viso_tv2_vs_VCD (tv2_visomaster_fake + external_vcd_real)": {
            "fake": has(["tv2_visomaster_fake"]),
            "real": has(["external_vcd_real"]),
        },
    }
    return pools


def verdict_for_gap(train_auc: float, eval_auc: float) -> str:
    gap = train_auc - eval_auc
    if train_auc > 0.80 and eval_auc < 0.70 and gap > 0.10:
        return "DISPOSITIVE_bucket_gap"
    if gap > 0.10:
        return "STRONG_gap"
    if gap > 0.05:
        return "MODERATE_gap"
    if gap >= -0.05:
        return "NO_gap (inconclusive)"
    return "REVERSED (eval > train)"


def main() -> int:
    if not FEATURES_NPZ.exists():
        logger.error("Features file not found at %s", FEATURES_NPZ)
        return 1

    logger.info("Loading P8A features from %s", FEATURES_NPZ)
    X, src_name, src_bucket, labels, prob_fake = load_features(FEATURES_NPZ)
    y_all = (labels == "fake").astype(int)
    logger.info("Loaded %d frames, feature dim %d", X.shape[0], X.shape[1])
    logger.info("Source-bucket distribution: %s", dict(Counter(src_bucket.tolist())))

    pools = make_pools(src_name)

    results: List[Dict] = []
    for name, sel in pools.items():
        mask = sel["fake"] | sel["real"]
        if mask.sum() == 0:
            logger.warning("Pool %s has 0 frames — skipping", name)
            continue
        X_p = X[mask]
        y_p = y_all[mask]
        prob_p = prob_fake[mask]
        if len(set(y_p.tolist())) < 2:
            logger.warning("Pool %s has only one class (n=%d) — skipping", name, mask.sum())
            continue
        logger.info("Evaluating pool %s (n=%d, fake=%d, real=%d)",
                    name, mask.sum(), int((y_p == 1).sum()), int((y_p == 0).sum()))
        r = evaluate_pool(X_p, y_p, prob_p, name)
        results.append(r)

    train_r = next((r for r in results if r["name"].startswith("TRAIN_")), None)
    eval_rs = [r for r in results if r["name"].startswith("EVAL_")]
    deploy_train_r = next((r for r in results if r["name"].startswith("DEPLOY_train_")), None)
    deploy_eval_rs = [r for r in results if r["name"].startswith("DEPLOY_eval_")]

    summary = {
        "checkpoint": "P8A_REFERENCE_STEP5000",
        "features_uri": "gs://training-job-outputs/probe_battery_2026-04-26/p8a_reference_step5000/features.npz",
        "seed": SEED,
        "method": "frozen-feature linear probe (LogisticRegression, 5-fold stratified CV)",
        "pools": results,
        "verdicts": [],
    }

    if train_r is not None:
        train_auc = train_r["probe_auc_mean"]
        for e in eval_rs:
            gap = train_auc - e["probe_auc_mean"]
            v = verdict_for_gap(train_auc, e["probe_auc_mean"])
            summary["verdicts"].append({
                "context": "same-substrate-real-pairing",
                "eval_pool": e["name"],
                "train_probe_auc": train_auc,
                "eval_probe_auc": e["probe_auc_mean"],
                "gap": float(gap),
                "verdict": v,
            })
    if deploy_train_r is not None:
        d_train_auc = deploy_train_r["probe_auc_mean"]
        for e in deploy_eval_rs:
            gap = d_train_auc - e["probe_auc_mean"]
            v = verdict_for_gap(d_train_auc, e["probe_auc_mean"])
            summary["verdicts"].append({
                "context": "deployment-real-pool (external_vcd_real)",
                "eval_pool": e["name"],
                "train_probe_auc": d_train_auc,
                "eval_probe_auc": e["probe_auc_mean"],
                "gap": float(gap),
                "verdict": v,
            })

    out_json = HERE / "move1_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out_json)

    csv_lines = ["pool,n_total,n_fake,n_real,p8a_classifier_auc,probe_auc_mean,probe_auc_std,probe_acc_mean,mean_prob_fake_on_fakes,mean_prob_fake_on_reals"]
    for r in results:
        csv_lines.append(
            f"\"{r['name']}\",{r['n_total']},{r['n_fake']},{r['n_real']},"
            f"{r['p8a_classifier_auc']:.4f},{r['probe_auc_mean']:.4f},"
            f"{r['probe_auc_std']:.4f},{r['probe_acc_mean']:.4f},"
            f"{r['mean_prob_fake_on_fakes']:.4f},{r['mean_prob_fake_on_reals']:.4f}"
        )
    out_csv = HERE / "move1_results.csv"
    with open(out_csv, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    logger.info("Wrote %s", out_csv)

    print()
    print("=" * 100)
    print("MOVE 1 — Frozen-feature linear probe of P8A_REFERENCE_STEP5000")
    print("=" * 100)
    print(f"{'pool':<70} {'n':>5} {'P8A_AUC':>9} {'Probe_AUC':>11}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<70} {r['n_total']:>5} "
              f"{r['p8a_classifier_auc']:>9.4f} "
              f"{r['probe_auc_mean']:>7.4f} ±{r['probe_auc_std']:.3f}")
    print()
    if summary["verdicts"]:
        for ctx in ("same-substrate-real-pairing", "deployment-real-pool (external_vcd_real)"):
            ctx_verdicts = [v for v in summary["verdicts"] if v["context"] == ctx]
            if not ctx_verdicts:
                continue
            print(f"VERDICTS — context: {ctx}")
            print("-" * 100)
            for v in ctx_verdicts:
                print(f"  {v['eval_pool']}")
                print(f"    train_probe_auc={v['train_probe_auc']:.4f}  "
                      f"eval_probe_auc={v['eval_probe_auc']:.4f}  "
                      f"gap={v['gap']:+.4f}  →  {v['verdict']}")
            print()

        all_v = summary["verdicts"]
        decisive = [v for v in all_v if v["verdict"] == "DISPOSITIVE_bucket_gap"]
        strong = [v for v in all_v if v["verdict"] == "STRONG_gap"]
        moderate = [v for v in all_v if v["verdict"] == "MODERATE_gap"]
        inconclusive = [v for v in all_v if v["verdict"] == "NO_gap (inconclusive)"]
        deploy_v = [v for v in all_v if v["context"].startswith("deployment-real-pool")]
        same_v = [v for v in all_v if v["context"] == "same-substrate-real-pairing"]
        print("OVERALL READ:")
        print(f"  Same-substrate context  : {len(same_v)} comparisons; "
              f"dispositive={sum(1 for v in same_v if v['verdict']=='DISPOSITIVE_bucket_gap')}, "
              f"strong={sum(1 for v in same_v if v['verdict']=='STRONG_gap')}, "
              f"inconclusive={sum(1 for v in same_v if v['verdict']=='NO_gap (inconclusive)')}")
        print(f"  Deployment-real context : {len(deploy_v)} comparisons; "
              f"dispositive={sum(1 for v in deploy_v if v['verdict']=='DISPOSITIVE_bucket_gap')}, "
              f"strong={sum(1 for v in deploy_v if v['verdict']=='STRONG_gap')}, "
              f"inconclusive={sum(1 for v in deploy_v if v['verdict']=='NO_gap (inconclusive)')}")
        if decisive:
            print(f"  → BUCKET GAP DISPOSITIVE on {len(decisive)} comparison(s).")
            print(f"  → Recommendation: LAUNCH P14_DATA_FIX (the $70 bet is licensed).")
        elif strong:
            print(f"  → STRONG gap on {len(strong)} comparison(s) but not strict-dispositive.")
            print(f"  → Recommendation: P14_DATA_FIX likely worthwhile; review per-pool gaps before commit.")
        elif inconclusive and len(inconclusive) >= len(all_v) // 2:
            print(f"  → INCONCLUSIVE on {len(inconclusive)}/{len(all_v)} comparisons — bucket gap is not the dominant feature-level signal.")
            print(f"  → Recommendation: DOWNGRADE P14_DATA_FIX priority; redirect spend to P15 GRL + face-scale-jitter.")
        else:
            print(f"  → MIXED signal — review per-pool details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
