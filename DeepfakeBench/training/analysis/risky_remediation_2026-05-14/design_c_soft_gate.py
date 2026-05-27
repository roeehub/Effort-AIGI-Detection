"""Design C — soft gate via cheap risk score.

Idea: train a tiny logistic regression on (iq_lap_var, iq_luma, iq_lab_a_dev,
iq_lab_b_dev, iq_edge_density) features → predicts P(model_will_err) per frame.

Then at inference: instead of remediating, downweight or abstain on frames with
high predicted error probability. The per-identity majority vote already
aggregates many frames, so abstaining on a fraction of risky frames is natural.

We need to be careful about train/test split — risk model must be fit on a
held-out partition not used for the headline AUC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent / "outputs"
# Use G2-filtered pool (production-eligible)
df = pd.read_csv(OUT / "g2_pass_pool.csv")
print(f"loaded {len(df)} G2-pass frames")

# Build pools
POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}


def assign_pool(suite):
    for pool, suites in POOL_DEFS.items():
        if suite in suites:
            return pool
    return "other"


df["pool"] = df["suite"].map(assign_pool)
two_label = df[df["pool"] != "other"].copy()
print(f"two-label production-eligible: {len(two_label)}")
print(two_label.groupby(["pool", "label"]).size().to_string())
print()

FEATS = ["iq_lap_var", "iq_luma", "iq_lab_a_dev", "iq_lab_b_dev", "iq_edge_density"]


def make_risk_label(df_sub, ckpt, tau=0.49):
    """A frame is 'risky' if the model would err on it at threshold tau.

    For label=0 (real): error = score > tau (false positive)
    For label=1 (fake): error = score < tau (false negative)
    """
    pred = df_sub[f"{ckpt}_orig"].values
    label = df_sub["label"].values
    pred_label = (pred > tau).astype(int)
    return (pred_label != label).astype(int)


# ============================================================================
# Train risk model on a held-out partition; evaluate on remaining cohorts.
# ============================================================================
# Split: train risk model on teams_real_all_dev + teams_fake_all_dev,
# test on lockbox + cross-substrate cohorts.
DEV_SUITES = ["teams_real_all_dev", "teams_fake_all_dev"]
train = two_label[two_label["suite"].isin(DEV_SUITES)].copy()
test = two_label[~two_label["suite"].isin(DEV_SUITES)].copy()
print(f"train pool (dev suites): {len(train)}")
print(f"test pool (lockbox + others): {len(test)}")
print(f"  test suite distribution: {dict(test.groupby('suite').size())}")
print()


def eval_strategy(df_test, ckpt, strategy_name, score_col, abstain_mask=None,
                  abstain_action="drop"):
    """Eval AUC + recall@5% FPR on test data.

    abstain_mask: boolean array same length as df_test; True = abstain on this frame.
    abstain_action:
      - 'drop'      : compute AUC/recall on only the kept frames
      - 'downweight': replace abstained scores with 0.5 (neutral) -> useful for majority vote scenario
    """
    s = df_test[score_col].values.copy()
    labels = df_test["label"].values
    if abstain_mask is not None:
        if abstain_action == "drop":
            keep = ~abstain_mask
            s_eval = s[keep]
            labels_eval = labels[keep]
        elif abstain_action == "downweight":
            s_eval = s.copy()
            s_eval[abstain_mask] = 0.5
            labels_eval = labels
        else:
            raise ValueError(abstain_action)
    else:
        s_eval = s
        labels_eval = labels
    if len(np.unique(labels_eval)) < 2:
        return {
            "strategy": strategy_name, "n": len(s_eval),
            "AUC": float("nan"), "rec@5": float("nan"),
            "n_dropped": int(abstain_mask.sum()) if abstain_mask is not None else 0,
        }
    auc = roc_auc_score(labels_eval, s_eval)
    fpr, tpr, _ = roc_curve(labels_eval, s_eval)
    ok = fpr <= 0.05
    rec = float(tpr[np.where(ok)[0][-1]]) if ok.sum() > 0 else float("nan")
    return {
        "strategy": strategy_name, "n": len(s_eval),
        "AUC": auc, "rec@5": rec,
        "n_dropped": int(abstain_mask.sum()) if abstain_mask is not None else 0,
        "frac_dropped": float(abstain_mask.mean()) if abstain_mask is not None else 0.0,
    }


# ============================================================================
# Build risk model from train pool, eval strategies on test pool.
# ============================================================================
for ckpt in ["T5C", "P8A"]:
    print("=" * 80)
    print(f"### {ckpt} — soft-gate evaluation")
    print("=" * 80)

    y_train_risk = make_risk_label(train, ckpt, tau=0.49)
    print(f"train pool: {len(train)} frames, {y_train_risk.sum()} errors at tau=0.49 ({y_train_risk.mean():.4f})")
    X_train = train[FEATS].values

    # Train logistic — class weight balanced because errors are rare
    risk_clf = LogisticRegression(class_weight="balanced", max_iter=500)
    risk_clf.fit(X_train, y_train_risk)

    # Diagnostic: AUC of risk model on train (sanity)
    risk_p_train = risk_clf.predict_proba(X_train)[:, 1]
    risk_auc_train = roc_auc_score(y_train_risk, risk_p_train) if y_train_risk.sum() > 0 else float("nan")
    print(f"  risk model train-AUC: {risk_auc_train:.4f}")
    print(f"  coefficients (on standardized features):")
    for f, c in zip(FEATS, risk_clf.coef_[0]):
        print(f"    {f:<22} {c:+.4f}")

    # Apply to test pool
    risk_p_test = risk_clf.predict_proba(test[FEATS].values)[:, 1]
    print(f"  risk_p_test distribution: p10={np.quantile(risk_p_test, 0.1):.3f} "
          f"p50={np.quantile(risk_p_test, 0.5):.3f} p90={np.quantile(risk_p_test, 0.9):.3f}")

    # Strategies:
    rows = []
    # 1) Baseline: orig, no remediation, no abstain
    rows.append(eval_strategy(test, ckpt, "orig (no remediation)", f"{ckpt}_orig"))
    # 2) Universal blend, no abstain
    rows.append(eval_strategy(test, ckpt, "universal blend@0.50", f"{ckpt}_blend_050"))

    # 3) Abstain on top-X% riskiest frames using orig scores
    for abstain_frac in [0.05, 0.10, 0.20, 0.30]:
        thresh = np.quantile(risk_p_test, 1.0 - abstain_frac)
        mask = risk_p_test >= thresh
        rows.append(eval_strategy(test, ckpt, f"orig + abstain top {abstain_frac*100:.0f}%",
                                  f"{ckpt}_orig", abstain_mask=mask, abstain_action="drop"))

    # 4) Universal blend + abstain on top-X%
    for abstain_frac in [0.05, 0.10, 0.20]:
        thresh = np.quantile(risk_p_test, 1.0 - abstain_frac)
        mask = risk_p_test >= thresh
        rows.append(eval_strategy(test, ckpt, f"blend + abstain top {abstain_frac*100:.0f}%",
                                  f"{ckpt}_blend_050", abstain_mask=mask, abstain_action="drop"))

    # 5) Hybrid — apply blend only to high-risk frames (use orig for low-risk)
    for risk_thresh_quantile in [0.50, 0.75, 0.90]:
        thresh = np.quantile(risk_p_test, risk_thresh_quantile)
        is_risky = risk_p_test >= thresh
        s_hybrid = np.where(is_risky, test[f"{ckpt}_blend_050"].values, test[f"{ckpt}_orig"].values)
        test["_hybrid_tmp"] = s_hybrid
        rows.append(eval_strategy(test, ckpt, f"hybrid: blend top {(1-risk_thresh_quantile)*100:.0f}%",
                                  "_hybrid_tmp"))
        test = test.drop(columns=["_hybrid_tmp"])

    rd = pd.DataFrame(rows).round(4)
    print()
    print(rd.to_string(index=False))
    rd.to_csv(OUT / f"design_c_soft_gate_{ckpt}.csv", index=False)
    print()
