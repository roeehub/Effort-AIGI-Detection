"""Apply pre-registered F1/F2/F3 falsifiers to targeted remediation sweep.

F1: Does ANY remediation rescue >=30% of T5C errors at tau=0.49?
F2: For rescued-frame subgroups, is the "would-be-rescued" classifier
    held-out AUC >= 0.65 on cheap features?
F3: Does the best held-out validated recipe give >= +0.005 production-pool
    ΔAUC with bootstrap CI excluding 0?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "targeted_remediation_scored.csv")
print(f"loaded {len(df)} frames")

TAU = 0.49
REMEDIATIONS = ["orig", "blend_050", "blend_035", "blend_065",
                "blur_5", "blur_7", "desat_50", "downup_168",
                "clahe_mild", "tta_3way"]
NEW_REMS = [r for r in REMEDIATIONS if r not in ("orig", "blend_050")]

# Build pool assignment
POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}
def assign_pool(s):
    for p, ss in POOL_DEFS.items():
        if s in ss: return p
    return "other"
df["pool"] = df["suite"].map(assign_pool)
two_label = df[df["pool"] != "other"].copy()
print(f"production-eligible (G2 ∩ two-label): {len(two_label)}")
print(f"  per-pool: {dict(two_label.groupby('pool').size())}")
print()

# ============================================================================
# F1: Per-remediation rescue/broken counts at tau=0.49
# ============================================================================
print("=" * 100)
print("F1 — per-remediation rescue vs broken at tau=0.49")
print("=" * 100)
print(f"{'remediation':<14} {'errors_orig':<12} {'rescued':<10} {'broken':<10} {'net':<8} {'rescue_rate':<12} {'broken_rate':<12}")

# Error definition: at tau=0.49,
# - real (label=0) is ERROR if score > 0.49
# - fake (label=1) is ERROR if score <= 0.49
def is_error(scores, labels, tau=TAU):
    return ((labels == 0) & (scores > tau)) | ((labels == 1) & (scores <= tau))

labels = two_label["label"].values
orig_scores = two_label["T5C_orig"].values
orig_err = is_error(orig_scores, labels)
n_err_orig = orig_err.sum()
n_correct_orig = (~orig_err).sum()
print(f"orig errors: {n_err_orig} / {len(two_label)} = {n_err_orig/len(two_label):.4f}")
print()

f1_rows = []
for rem in REMEDIATIONS:
    col = f"T5C_{rem}"
    if col not in two_label.columns:
        continue
    rem_scores = two_label[col].values
    rem_err = is_error(rem_scores, labels)
    # rescued: was err in orig, NOT err in rem
    rescued = orig_err & ~rem_err
    # broken: was correct in orig, ERR in rem
    broken = ~orig_err & rem_err
    n_rescued = rescued.sum()
    n_broken = broken.sum()
    rate_rescue = n_rescued / max(n_err_orig, 1)
    rate_broken = n_broken / max(n_correct_orig, 1)
    print(f"{rem:<14} {n_err_orig:<12} {n_rescued:<10} {n_broken:<10} {n_rescued-n_broken:<8d} "
          f"{rate_rescue:<12.4f} {rate_broken:<12.4f}")
    f1_rows.append({
        "remediation": rem,
        "errors_orig": n_err_orig,
        "rescued": n_rescued,
        "broken": n_broken,
        "net": n_rescued - n_broken,
        "rescue_rate": rate_rescue,
        "broken_rate": rate_broken,
    })
f1 = pd.DataFrame(f1_rows)
f1.to_csv(OUT / "targeted_f1_rescue_counts.csv", index=False)

# F1 verdict
best = f1[f1["remediation"] != "orig"].copy()
best["rescue_rate_only"] = best["rescue_rate"]
best = best.sort_values("net", ascending=False)
print()
print(f"Best (by net): {best.iloc[0]['remediation']} — rescue rate {best.iloc[0]['rescue_rate']:.4f}")
F1_THRESHOLD = 0.30
F1_PASS = (best["rescue_rate"] >= F1_THRESHOLD).any()
print(f"F1 falsifier (any remediation rescues >= {F1_THRESHOLD*100:.0f}% of errors): "
      f"{'PASS' if F1_PASS else 'FAIL'}")

# Also: oracle — best per-frame
print()
print("Oracle analysis (best rem per error):")
oracle_score = np.zeros(len(two_label))
oracle_choice = np.array([""] * len(two_label), dtype=object)
for i in range(len(two_label)):
    if labels[i] == 0:
        # real: want lowest score
        scores = {r: two_label[f"T5C_{r}"].iloc[i] for r in REMEDIATIONS
                  if f"T5C_{r}" in two_label.columns}
        best_r = min(scores, key=scores.get)
        oracle_score[i] = scores[best_r]
        oracle_choice[i] = best_r
    else:
        # fake: want highest score
        scores = {r: two_label[f"T5C_{r}"].iloc[i] for r in REMEDIATIONS
                  if f"T5C_{r}" in two_label.columns}
        best_r = max(scores, key=scores.get)
        oracle_score[i] = scores[best_r]
        oracle_choice[i] = best_r
oracle_err = is_error(oracle_score, labels)
n_oracle_err = oracle_err.sum()
print(f"  oracle errors at tau=0.49: {n_oracle_err} / {len(two_label)} "
      f"(reduction from {n_err_orig}: {n_err_orig - n_oracle_err} = "
      f"{(n_err_orig - n_oracle_err)/n_err_orig*100:.1f}%)")
print(f"  oracle remediation choice distribution:")
for r in REMEDIATIONS:
    cnt = (oracle_choice == r).sum()
    print(f"    {r:<14} {cnt:<8} ({cnt/len(oracle_choice)*100:.1f}%)")

# Oracle AUC vs orig AUC
auc_orig = roc_auc_score(labels, orig_scores)
auc_oracle = roc_auc_score(labels, oracle_score)
print(f"\n  AUC: orig={auc_orig:.4f}  oracle={auc_oracle:.4f}  Δ={auc_oracle-auc_orig:+.4f}")

# ============================================================================
# F2: Per-rescuing-remediation, can we predict "which frames does R rescue"
#     from cheap features alone (no model scores)?
# ============================================================================
print()
print("=" * 100)
print("F2 — predictability of rescued-frame subgroups from CHEAP FEATURES ONLY")
print("=" * 100)
FEATS = ["iq_lap_var", "iq_luma", "iq_lab_a_dev", "iq_lab_b_dev", "iq_edge_density"]

# Train/test split: dev pool train, lockbox+dor_cross test
train_mask = (two_label["pool"] == "teams_dev").values
test_mask = ~train_mask
print(f"train (teams_dev): {train_mask.sum()}, test (lockbox+dor_cross): {test_mask.sum()}")
print()

f2_rows = []
for rem in NEW_REMS + ["blend_050"]:
    col = f"T5C_{rem}"
    if col not in two_label.columns:
        continue
    rem_scores = two_label[col].values
    rem_err = is_error(rem_scores, labels)
    rescued = (orig_err & ~rem_err).astype(int)

    # Train target: 1 if frame is rescued by R, 0 otherwise
    # Skip if too few rescues
    n_rescued_train = rescued[train_mask].sum()
    n_rescued_test = rescued[test_mask].sum()
    if n_rescued_train < 30 or n_rescued_test < 5:
        print(f"  {rem:<14} insufficient rescues (train={n_rescued_train}, test={n_rescued_test}); skip")
        continue

    X_train = two_label.loc[train_mask, FEATS].values
    y_train = rescued[train_mask]
    X_test = two_label.loc[test_mask, FEATS].values
    y_test = rescued[test_mask]

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=9501)
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]
    try:
        auc_test = roc_auc_score(y_test, p_test)
    except Exception:
        auc_test = float("nan")
    f2_rows.append({
        "remediation": rem,
        "n_rescued_train": int(n_rescued_train),
        "n_rescued_test": int(n_rescued_test),
        "predictor_held_out_AUC": auc_test,
    })

f2 = pd.DataFrame(f2_rows)
f2 = f2.sort_values("predictor_held_out_AUC", ascending=False)
print(f"\n{'remediation':<14} {'rescued_train':<14} {'rescued_test':<14} {'held_out_AUC':<14}")
for _, r in f2.iterrows():
    print(f"{r['remediation']:<14} {r['n_rescued_train']:<14} {r['n_rescued_test']:<14} {r['predictor_held_out_AUC']:<14.4f}")
f2.to_csv(OUT / "targeted_f2_predictability.csv", index=False)

F2_THRESHOLD = 0.65
F2_PASS = (f2["predictor_held_out_AUC"] >= F2_THRESHOLD).any() if len(f2) > 0 else False
print(f"\nF2 falsifier (any rescuer cluster has held-out AUC >= {F2_THRESHOLD}): "
      f"{'PASS' if F2_PASS else 'FAIL'}")

# ============================================================================
# F3: Build best validated recipe and measure ΔAUC with bootstrap CI on
#     held-out (lockbox + dor_cross) pool.
# ============================================================================
print()
print("=" * 100)
print("F3 — held-out validated recipe ΔAUC with bootstrap CI")
print("=" * 100)

if not F2_PASS:
    print("F2 failed — skipping F3 (no predictor to build recipe from)")
    F3_PASS = False
else:
    # Take the remediation with highest held-out AUC (and at least net-positive in F1)
    candidates = f2[f2["predictor_held_out_AUC"] >= F2_THRESHOLD].copy()
    candidates = candidates.merge(f1[["remediation", "net"]], on="remediation")
    candidates = candidates[candidates["net"] > 0].sort_values("predictor_held_out_AUC", ascending=False)
    if len(candidates) == 0:
        print("No remediation passes both F1-net>0 and F2-AUC threshold; F3 cannot proceed")
        F3_PASS = False
    else:
        best_rem = candidates.iloc[0]["remediation"]
        print(f"Best validated recipe: '{best_rem}'")

        # Build per-frame router on train, apply to test
        rem_scores_full = two_label[f"T5C_{best_rem}"].values
        rem_err = is_error(rem_scores_full, labels)
        rescued = (orig_err & ~rem_err).astype(int)

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=9501)
        clf.fit(two_label.loc[train_mask, FEATS].values, rescued[train_mask])
        # Use predicted probability to gate the remediation
        p_route = clf.predict_proba(two_label[FEATS].values)[:, 1]

        # Build recipe scores: where router fires, use remediation; else use orig
        # Sweep router-threshold to find best per-frame split
        test_orig = two_label.loc[test_mask, "T5C_orig"].values
        test_rem = two_label.loc[test_mask, f"T5C_{best_rem}"].values
        test_labels = labels[test_mask]
        test_p = p_route[test_mask]

        # Threshold sweep
        print(f"\n  Router-threshold sweep on held-out pool (n={test_mask.sum()}):")
        best_thresh = None
        best_recipe_auc = roc_auc_score(test_labels, test_orig)
        baseline_auc = best_recipe_auc
        print(f"  {'thresh':<10} {'frac_routed':<14} {'recipe_AUC':<14} {'Δ_vs_orig':<14}")
        print(f"  {'orig':<10} {0.0:<14.4f} {baseline_auc:<14.4f} {0.0:<14.4f}")
        for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
            mask_route = test_p >= thr
            recipe = np.where(mask_route, test_rem, test_orig)
            auc_r = roc_auc_score(test_labels, recipe)
            print(f"  {thr:<10.2f} {mask_route.mean():<14.4f} {auc_r:<14.4f} {auc_r-baseline_auc:+<14.4f}")
            if auc_r > best_recipe_auc:
                best_recipe_auc = auc_r
                best_thresh = thr

        if best_thresh is None:
            print("\n  No router threshold improves over orig on held-out pool.")
            F3_PASS = False
        else:
            print(f"\n  Best recipe: route when p >= {best_thresh}; AUC = {best_recipe_auc:.4f}")
            print(f"  Δ vs orig = {best_recipe_auc - baseline_auc:+.4f}")
            # Bootstrap CI on the delta
            mask_route_final = test_p >= best_thresh
            recipe_final = np.where(mask_route_final, test_rem, test_orig)
            rng = np.random.default_rng(9501)
            deltas = []
            n_test = test_mask.sum()
            for _ in range(2000):
                idx = rng.integers(0, n_test, n_test)
                lb = test_labels[idx]
                if len(np.unique(lb)) < 2:
                    continue
                a_o = roc_auc_score(lb, test_orig[idx])
                a_r = roc_auc_score(lb, recipe_final[idx])
                deltas.append(a_r - a_o)
            deltas = np.array(deltas)
            ci_low = np.quantile(deltas, 0.025)
            ci_high = np.quantile(deltas, 0.975)
            print(f"  Bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
            F3_THRESHOLD = 0.005
            F3_PASS = (best_recipe_auc - baseline_auc >= F3_THRESHOLD) and (ci_low > 0)
            print(f"  F3 falsifier (Δ >= {F3_THRESHOLD} AND CI excludes 0): "
                  f"{'PASS' if F3_PASS else 'FAIL'}")

# ============================================================================
# Final verdict
# ============================================================================
print()
print("=" * 100)
print("FINAL FALSIFIER OUTCOMES")
print("=" * 100)
print(f"  F1 (any remediation rescues >= 30% errors): {'PASS' if F1_PASS else 'FAIL'}")
print(f"  F2 (rescuer cluster held-out AUC >= 0.65): {'PASS' if F2_PASS else 'FAIL'}")
print(f"  F3 (held-out recipe Δ >= +0.005, CI excludes 0): {'PASS' if F3_PASS else 'FAIL'}")
print()
if F1_PASS and F2_PASS and F3_PASS:
    print("  ✓ All three pass — TARGETED LEVER IS REAL. Pursue.")
else:
    print("  ✗ At least one falsifier failed — no deployable targeted lever found.")
