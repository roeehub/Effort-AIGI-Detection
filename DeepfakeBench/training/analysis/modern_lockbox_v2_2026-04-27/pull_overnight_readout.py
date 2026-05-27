"""Tomorrow morning's first read: pull W&B metrics for the 4 overnight P11 runs.

Identifies runs by unique p11-* tag in the enhanced-aug-test project.

Outputs:
- decision-grade comparison table (anchor_mean@1500, val_dev_recall per family @5000, lockbox FPR).
- CSV alongside this script.

Usage:
  WANDB_API_KEY=... WANDB_ENTITY=dtect-vision python -m analysis.modern_lockbox_v2_2026-04-27.pull_overnight_readout

If a run hasn't reached a step yet, the corresponding column shows '—'.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"

ENTITY = os.environ.get("WANDB_ENTITY", "dtect-vision")
PROJECT = os.environ.get("WANDB_PROJECT", "enhanced-aug-test")

# 2026-04-27 night: tags didn't propagate from yaml to wandb — match by name prefix.
# Known W&B IDs (locked at 23:25 CEST after launches):
KNOWN_IDS = {
    "MILD": "u0whbpg5",
    "HEAVY": "lr916mpq",
    "HEAVY_DEEPLIVE": "h7eqzw1j",
    "WEBCAM_HARDEN": "89ohe6l8",
}
# Fallback: match by run.name prefix if KNOWN_IDS lookup fails (e.g. relaunch).
NAME_PREFIXES = {
    "MILD": "R13_P11_MILD_",
    "HEAVY": "R13_P11_HEAVY_",
    "HEAVY_DEEPLIVE": "R13_P11_HEAVY_DEEPLIVE_",
    "WEBCAM_HARDEN": "R13_P11_WEBCAM_HARDEN_",
}

# P8A baseline (from prior LOG entries) for delta column.
P8A_BASELINE = {
    "viso_enhanced_macro_dev_recall": 0.356,
    "deeplive_enhanced_dev_recall": 0.530,
    "teams_fake_all_dev_recall": 0.702,
    "lockbox_fpr_baseline_tau_0.5": 0.2198,
    "lockbox_fpr_v2_tau_0.5": 0.0676,
    "lockbox_fpr_baseline_tau_0.9741": 0.0459,
    "lockbox_fpr_v2_tau_0.9741": 0.0071,
}

# W&B metric keys (verified from smoke run vh6fn11b summary 2026-04-27 23:30 CEST).
# val_holdout/method/<method_name>/acc carries per-method accuracy — must aggregate to family-level.
# anchor_mean/spread track the real-pool false-flag rate (kill-gate metric).
METRIC_KEYS = [
    "anchor/anchor_mean",
    "anchor/spread_mean",
    "val_holdout/at_indist/recall",
    "ood/at_indist/recall",
    "ood/at_indist/fpr",
    "best/auc",
    "best/tpr_at_fpr5pct",
    "best/tpr_at_fpr1pct",
    "best_value_composite/metric",
]
# Family-level recall from val_holdout requires aggregating method-level keys.
# Tomorrow morning: glob `val_holdout/method/<method>/acc` from history, bucket per family.
VISO_METHOD_PREFIX = "proper_visomaster"
DEEPLIVE_METHOD_PREFIX = "deeplive"
TEAMS_METHOD_PREFIX = "teams_capture"


def fetch_run_history(api, run, keys: list[str]) -> pd.DataFrame:
    """Pull a run's full history for the given metric keys."""
    try:
        hist = run.history(keys=keys, samples=10000, pandas=True)
    except Exception as exc:
        print(f"  warn: history fetch failed for {run.name}: {exc}")
        return pd.DataFrame()
    return hist


def value_at_step(hist: pd.DataFrame, key: str, step: int) -> float | None:
    if hist is None or hist.empty or "_step" not in hist.columns or key not in hist.columns:
        return None
    near = hist[(hist["_step"] >= step - 250) & (hist["_step"] <= step + 250)]
    if near.empty:
        return None
    return float(near[key].dropna().iloc[-1]) if near[key].dropna().size else None


def main() -> None:
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed in current env")
    api = wandb.Api(timeout=60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve runs: try KNOWN_IDS first, fallback to name-prefix scan.
    runs_by_label = {}
    for label, run_id in KNOWN_IDS.items():
        try:
            runs_by_label[label] = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        except Exception:
            runs_by_label[label] = None
    if any(r is None for r in runs_by_label.values()):
        project_runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=200))
        for label, prefix in NAME_PREFIXES.items():
            if runs_by_label.get(label) is not None:
                continue
            cands = [r for r in project_runs if r.name.startswith(prefix)]
            # HEAVY needs to exclude HEAVY_DEEPLIVE
            if label == "HEAVY":
                cands = [r for r in cands if not r.name.startswith("R13_P11_HEAVY_DEEPLIVE_")]
            cands = sorted(cands, key=lambda r: r.created_at, reverse=True)
            runs_by_label[label] = cands[0] if cands else None

    print(f"\n=== overnight P11 readout — {ENTITY}/{PROJECT} ===\n")
    rows = []
    for label in KNOWN_IDS:
        run = runs_by_label.get(label)
        if run is None:
            print(f"{label:<18s} — NOT FOUND in project")
            rows.append({"run": label, "status": "not_found"})
            continue
        hist = fetch_run_history(api, run, METRIC_KEYS)
        summary = run.summary._json_dict if hasattr(run, "summary") else {}
        max_step = int(hist["_step"].max()) if (not hist.empty and "_step" in hist.columns) else 0

        # Aggregate val_holdout per-method acc into family-level recall (cur summary).
        viso_acc = [v for k, v in summary.items()
                    if k.startswith(f"val_holdout/method/{VISO_METHOD_PREFIX}") and k.endswith("/acc") and isinstance(v, (int, float))]
        dl_acc = [v for k, v in summary.items()
                  if k.startswith(f"val_holdout/method/{DEEPLIVE_METHOD_PREFIX}") and k.endswith("/acc") and isinstance(v, (int, float))]
        tf_acc = [v for k, v in summary.items()
                  if k.startswith(f"val_holdout/method/{TEAMS_METHOD_PREFIX}") and k.endswith("/acc") and isinstance(v, (int, float))]
        viso_macro = sum(viso_acc) / len(viso_acc) if viso_acc else None
        dl_macro = sum(dl_acc) / len(dl_acc) if dl_acc else None
        tf_macro = sum(tf_acc) / len(tf_acc) if tf_acc else None

        rows.append({
            "run": label,
            "wandb_id": run.id,
            "wandb_name": run.name,
            "state": run.state,
            "max_step": max_step,
            "anchor_mean_cur": summary.get("anchor/anchor_mean"),
            "best_auc": summary.get("best/auc"),
            "best_tpr@fpr5pct": summary.get("best/tpr_at_fpr5pct"),
            "viso_macro_acc_cur": viso_macro,
            "deeplive_macro_acc_cur": dl_macro,
            "teams_capture_macro_acc_cur": tf_macro,
            "n_viso_methods": len(viso_acc),
            "n_deeplive_methods": len(dl_acc),
            "n_teams_capture_methods": len(tf_acc),
            "ood_recall_cur": summary.get("ood/at_indist/recall"),
            "ood_fpr_cur": summary.get("ood/at_indist/fpr"),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        for col in ["anchor_mean@1500", "viso_recall@5000", "deeplive_recall@5000",
                    "teams_fake_all_recall@5000", "teams_real_all_fpr@5000", "teams_real_dor_dev_fpr@5000"]:
            if col in table.columns:
                table[col] = table[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    print(table.to_string(index=False))
    print()
    print(f"P8A baseline reference: {json.dumps(P8A_BASELINE, indent=2)}")

    out_csv = OUT_DIR / "overnight_readout.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # Decision-grade verdict (uses current macro acc — proxy for per-family recall)
    if not table.empty and "viso_macro_acc_cur" in table.columns:
        verdict = []
        for _, r in table.iterrows():
            v = r.get("viso_macro_acc_cur")
            d = r.get("deeplive_macro_acc_cur")
            step = r.get("max_step", 0)
            if v is None or d is None:
                verdict.append((r["run"], f"INCOMPLETE — no eval yet (step={step})"))
                continue
            qual = "STRONG" if (v >= 0.75 and d >= 0.80) else ("PARTIAL" if (v >= 0.50 and d >= 0.65) else "WEAK")
            verdict.append((r["run"], f"{qual} — viso {v:.2%}, deeplive {d:.2%} (step {step})"))
        print("\nDecision-grade verdict (per Plan v3 §5 Day-2 gate, val_holdout-macro proxy):")
        for run, v in verdict:
            print(f"  {run}: {v}")


if __name__ == "__main__":
    main()
