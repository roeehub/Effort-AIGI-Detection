"""Phase C: per-method scorecard-style readout for the layer-3 head.

Using the cached scaled-validation features (4000 dev + 839 lockbox at
layer 3), compute per-method fake recall on the same dev/lockbox sample.

This is the eval-suite-equivalent readout for the layer-3 LR head, mapped
to the suites used by the v3 promotion contract scorecard:
  - teams_fake_all_dev   = method.startswith('teams_capture')|'teams_flat'|'visomaster_enhanced_macro'
  - visomaster_enhanced_macro_dev = method == 'visomaster_enhanced_macro'
  - deeplive_enhanced_dev = method == 'deeplive_enhanced'
  - teams_real_all_dev   = method == 'teams_real' & split=='dev'
  - teams_real_all_lockbox = method == 'teams_real' & split=='lockbox'
  - teams_fake_all_lockbox = label=='fake' & split=='lockbox'

Compares head's lockbox FPR=5% calibrated τ against P8A baseline τ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"
PARQUET = REPO_ROOT / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load parquet to get method labels for dev + lockbox.
    df = pd.read_parquet(PARQUET)
    print(f"parquet: dev={int((df['split']=='dev').sum())}, lockbox={int((df['split']=='lockbox').sum())}")

    # Reconstruct the dev sample order used by scaled_layer3_validation.
    # That script used seed=42 and stratified 50/50 real/fake on dev.
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    dev_real = dev_df[dev_df["label"] == "real"].reset_index(drop=True)
    dev_fake = dev_df[dev_df["label"] == "fake"].reset_index(drop=True)
    rng = np.random.default_rng(seed=42)
    n_dev = 4000
    dev_real_idx = rng.choice(len(dev_real), size=min(n_dev // 2, len(dev_real)), replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=min(n_dev // 2, len(dev_fake)), replace=False)
    dev_sample = pd.concat([dev_real.iloc[dev_real_idx], dev_fake.iloc[dev_fake_idx]]).reset_index(drop=True)
    lb_df = df[df["split"] == "lockbox"].reset_index(drop=True)

    # Load cached features.
    cache = np.load(CACHE_DIR / "layer_validation__P8A__4000_839__layers_3_6_11.npz")
    print("Cache files:", cache.files)
    dev_labels = cache["dev_labels"]
    lb_labels = cache["lb_labels"]
    assert len(dev_labels) == len(dev_sample), f"dev mismatch: cache {len(dev_labels)} vs sample {len(dev_sample)}"
    assert len(lb_labels) == len(lb_df), f"lb mismatch: cache {len(lb_labels)} vs df {len(lb_df)}"

    # Add prediction columns to the dataframes.
    summary = {"per_layer": {}}

    # Per-suite metric definitions:
    suite_filters = {
        "teams_real_all_dev":          lambda r: (r["method"] == "teams_real"),
        "teams_real_all_lockbox":      lambda r: (r["method"] == "teams_real"),
        "teams_fake_all_dev":          lambda r: (r["label"] == "fake"),
        "teams_fake_all_lockbox":      lambda r: (r["label"] == "fake"),
        "visomaster_enhanced_macro_dev": lambda r: (r["method"] == "visomaster_enhanced_macro"),
        "deeplive_enhanced_dev":       lambda r: (r["method"] == "deeplive_enhanced"),
    }

    for layer in (3, 6, 11):
        F_dev = cache[f"dev_layer_{layer}_feats"]
        F_lb = cache[f"lb_layer_{layer}_feats"]
        F_dev_n = F_dev / (np.linalg.norm(F_dev, axis=1, keepdims=True) + 1e-12)
        F_lb_n = F_lb / (np.linalg.norm(F_lb, axis=1, keepdims=True) + 1e-12)
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(F_dev_n, dev_labels)
        dev_proba = clf.predict_proba(F_dev_n)[:, 1]
        lb_proba = clf.predict_proba(F_lb_n)[:, 1]

        # Calibrate τ at lockbox real FPR=5%.
        lb_real_idx = (lb_labels == 0)
        lb_fake_idx = (lb_labels == 1)
        fpr, tpr, thr = roc_curve(lb_labels, lb_proba)
        target_fpr = 0.05
        eligible = np.where(fpr <= target_fpr + 1e-12)[0]
        if len(eligible):
            best = eligible[np.argmax(tpr[eligible])]
            tau = float(thr[best])
        else:
            tau = 0.5
        # ALSO calibrate at FPR=2% (deployment-relevant tighter target) and FPR=10%.
        taus = {}
        for tgt in (0.02, 0.05, 0.10):
            eligible = np.where(fpr <= tgt + 1e-12)[0]
            if len(eligible):
                best = eligible[np.argmax(tpr[eligible])]
                taus[f"tau_at_lb_fpr_{tgt:.2f}"] = {
                    "tau": float(thr[best]),
                    "lb_fpr": float(fpr[best]),
                    "lb_recall": float(tpr[best]),
                }
            else:
                taus[f"tau_at_lb_fpr_{tgt:.2f}"] = {"tau": 0.5, "lb_fpr": 0.0, "lb_recall": 0.0}

        # Add scores to dataframes.
        dev_sample[f"L{layer}_prob_fake"] = dev_proba
        lb_df[f"L{layer}_prob_fake"] = lb_proba

        # Per-suite metrics at the FPR=5% τ.
        per_suite = {}
        for tgt in (0.02, 0.05, 0.10):
            tau_t = taus[f"tau_at_lb_fpr_{tgt:.2f}"]["tau"]
            per_suite[f"@fpr_{tgt:.2f}"] = {}
            for suite_name, filt in suite_filters.items():
                use_dev = "dev" in suite_name
                source = dev_sample if use_dev else lb_df
                proba_col = f"L{layer}_prob_fake"
                # Apply suite filter.
                mask = source.apply(filt, axis=1).to_numpy()
                if mask.sum() == 0:
                    per_suite[f"@fpr_{tgt:.2f}"][suite_name] = {"n": 0}
                    continue
                sub_proba = source.loc[mask, proba_col].to_numpy()
                # Determine if this is a real-suite (compute FPR) or fake-suite (compute recall).
                if "real" in suite_name:
                    fpr_val = float((sub_proba >= tau_t).mean())
                    per_suite[f"@fpr_{tgt:.2f}"][suite_name] = {
                        "n": int(mask.sum()),
                        "fpr": fpr_val,
                    }
                elif "fake" in suite_name or "enhanced" in suite_name or "deeplive" in suite_name:
                    recall_val = float((sub_proba >= tau_t).mean())
                    per_suite[f"@fpr_{tgt:.2f}"][suite_name] = {
                        "n": int(mask.sum()),
                        "recall": recall_val,
                    }
                else:
                    per_suite[f"@fpr_{tgt:.2f}"][suite_name] = {"n": int(mask.sum())}

        summary["per_layer"][f"layer_{layer}"] = {
            "lockbox_AUC": float(roc_auc_score(lb_labels, lb_proba)),
            "taus": taus,
            "suites": per_suite,
        }

        # Pretty print.
        print(f"\n=== Layer {layer} (lb_AUC={summary['per_layer'][f'layer_{layer}']['lockbox_AUC']:.4f}) ===")
        for tgt in (0.02, 0.05, 0.10):
            print(f"  -- @ τ at lb_fpr={tgt:.2f} (τ={taus[f'tau_at_lb_fpr_{tgt:.2f}']['tau']:.4f}, "
                  f"lb_fpr={taus[f'tau_at_lb_fpr_{tgt:.2f}']['lb_fpr']:.4f}, "
                  f"lb_recall={taus[f'tau_at_lb_fpr_{tgt:.2f}']['lb_recall']:.4f}) --")
            for suite_name, m in per_suite[f"@fpr_{tgt:.2f}"].items():
                if m.get("n", 0) == 0:
                    continue
                if "fpr" in m:
                    print(f"     {suite_name:<35s} n={m['n']:5d}  FPR={m['fpr']:.4f}")
                elif "recall" in m:
                    print(f"     {suite_name:<35s} n={m['n']:5d}  recall={m['recall']:.4f}")

    out_json = OUTPUT_DIR / "per_method_layer3.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\noutputs → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
