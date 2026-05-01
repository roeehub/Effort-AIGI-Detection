"""Synthesize the P18 verdict from the trajectory.csv produced by repeated
probe_p18_ckpt.py runs.

Reads:
    analysis/p18_probe_2026-05-01/outputs/trajectory.csv

Writes:
    analysis/p18_probe_2026-05-01/outputs/verdict.json
    analysis/p18_probe_2026-05-01/outputs/trajectory_plot.png
    docs/relaunch_handoffs/P18_VERDICT_2026-05-01.md  (the canonical record)

Verdict logic per HANDOFF.md outcome lattice:
    α (passes): treatment final cos(head, dor_shkedi) ≤ 0.05 AND
                12-class method-LR macro-AUC drops materially (< 0.85) AND
                cos(head, fresh_LR) preserved (≥ 0.50).
                → next: launch promotion-contract scorecards.

    β (partial): treatment trajectory shows GRL bit something but final
                doesn't fully clear. → propose ramped-λ trainer patch + retry.

    γ (no bite): treatment trajectory looks like control trajectory; both
                stay aligned with dor_shkedi axis and AUC stays ~0.99.
                → 12-class GRL also wrong axis. Pivot to Move 4 paired
                same-identity contrastive (Option II in synthesis).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAJ_CSV = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs" / "trajectory.csv"
VERDICT_JSON = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs" / "verdict.json"
PLOT_PNG = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs" / "trajectory_plot.png"
VERDICT_MD = REPO_ROOT / "docs" / "relaunch_handoffs" / "P18_VERDICT_2026-05-01.md"


def extract_step(label: str) -> int:
    """Pull the training step from a probe label (best-effort)."""
    for tok in str(label).split("_"):
        if tok.startswith("step"):
            try:
                return int(tok[4:])
            except (ValueError, TypeError):
                continue
        if tok.startswith("ep") and tok[2:].isdigit():
            return -1 * int(tok[2:])  # epoch ckpt; negative = early
    return 0


def main():
    if not TRAJ_CSV.exists():
        print(f"trajectory.csv not found at {TRAJ_CSV}; nothing to synthesize.")
        return 1

    df = pd.read_csv(TRAJ_CSV)
    df["step"] = df["label"].apply(extract_step)
    df = df.sort_values(["arm", "step"]).reset_index(drop=True)

    treatment = df[df["arm"] == "treatment"].copy()
    control = df[df["arm"] == "control"].copy()

    print(f"Treatment ckpts probed: {len(treatment)}")
    print(f"Control ckpts probed:   {len(control)}")
    if len(treatment) == 0 or len(control) == 0:
        print("Cannot synthesize verdict: missing one arm.")
        return 1

    # Final ckpt = highest step per arm
    t_final = treatment.iloc[-1]
    c_final = control.iloc[-1]

    print("\n=== TREATMENT trajectory ===")
    cols = ["step", "cos_is_dor_shkedi", "cos_is_deeplive_enhanced", "cos_fresh_LR_fake_vs_real",
            "method_lr_macro_auc", "verdict_preliminary"]
    print(treatment[cols].to_string(index=False))

    print("\n=== CONTROL trajectory ===")
    print(control[cols].to_string(index=False))

    # Apply verdict thresholds to treatment final
    cos_dor_t = float(t_final.get("cos_is_dor_shkedi", 0.0) or 0.0)
    cos_fresh_t = float(t_final.get("cos_fresh_LR_fake_vs_real", 0.0) or 0.0)
    auc_t = float(t_final.get("method_lr_macro_auc", 0.0) or 0.0)
    cos_dor_c = float(c_final.get("cos_is_dor_shkedi", 0.0) or 0.0)
    auc_c = float(c_final.get("method_lr_macro_auc", 0.0) or 0.0)

    if abs(cos_dor_t) <= 0.05 and auc_t < 0.85 and cos_fresh_t >= 0.50:
        verdict = "α"
        verdict_text = (
            "GRL BITES at the 12-class method-conditional level. Treatment final "
            "head direction moved off the dor_shkedi axis AND encoder's [CLS] "
            "no longer linearly separates the 12 method buckets. Fresh-LR "
            "alignment preserved. Next: launch promotion-contract scorecards "
            "on treatment + control to verify deployment-grade lift."
        )
    elif abs(cos_dor_t) <= 0.05 or auc_t < 0.85:
        verdict = "β"
        verdict_text = (
            f"PARTIAL bite. cos(head, dor_shkedi)={cos_dor_t:+.3f} "
            f"(threshold ≤0.05), method-LR AUC={auc_t:.3f} (threshold <0.85). "
            f"GRL moved one axis but not both. Next: propose ramped-λ trainer "
            f"patch (set_lambda 0→target over warmup) + production-tight crop "
            f"work + re-launch as P19."
        )
    else:
        verdict = "γ"
        verdict_text = (
            f"NO BITE. cos(head, dor_shkedi)={cos_dor_t:+.3f} (treatment) vs "
            f"{cos_dor_c:+.3f} (control). method-LR AUC={auc_t:.3f} (treatment) "
            f"vs {auc_c:.3f} (control). 12-class method-conditional GRL also "
            f"failed to bite. Pivot to Move 4 (paired same-identity contrastive) "
            f"as the next architectural address."
        )

    print(f"\n=== VERDICT: {verdict} ===")
    print(verdict_text)

    # Save artifacts
    VERDICT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "treatment_final": t_final.to_dict(),
        "control_final": c_final.to_dict(),
        "thresholds": {
            "alpha_cos_dor_max": 0.05,
            "alpha_method_auc_max": 0.85,
            "alpha_cos_fresh_min": 0.50,
        },
        "n_treatment_ckpts": int(len(treatment)),
        "n_control_ckpts": int(len(control)),
    }
    with open(VERDICT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot trajectories
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax_idx, metric in enumerate(["cos_is_dor_shkedi", "cos_fresh_LR_fake_vs_real",
                                          "cos_is_deeplive_enhanced", "method_lr_macro_auc"]):
            ax = axes[ax_idx // 2, ax_idx % 2]
            ax.plot(treatment["step"], pd.to_numeric(treatment[metric], errors="coerce"),
                    "o-", label="P18 treatment (12-class GRL)", color="C0")
            ax.plot(control["step"], pd.to_numeric(control[metric], errors="coerce"),
                    "s-", label="P18 control (no GRL)", color="C1")
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.suptitle(f"P18 trajectory — verdict: {verdict}", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=120)
        print(f"\nSaved plot: {PLOT_PNG}")
    except ImportError:
        print("\n(matplotlib not available; skipping plot)")

    # Write the verdict markdown
    md = f"""# P18 verdict — {verdict}

**Date**: 2026-05-01 (synthesis after both Vertex jobs terminal)
**Treatment**: R13_P18_METHOD_DOMAIN_GRL (W&B run xpbvc1e4)
**Control**: R13_P18_NO_GRL_CONTROL (W&B run rgt4kw2u)

## Verdict

**{verdict}** — {verdict_text}

## Treatment vs control summary

| Metric | Treatment final | Control final | Threshold (α) |
|---|---:|---:|---|
| cos(head_dir, is_dor_shkedi) | {cos_dor_t:+.4f} | {cos_dor_c:+.4f} | ≤ 0.05 |
| cos(head_dir, fresh_LR_fake_vs_real) | {cos_fresh_t:+.4f} | {float(c_final.get('cos_fresh_LR_fake_vs_real', 0) or 0):+.4f} | ≥ 0.50 |
| method-LR macro-OVR AUC (P8A feature space) | {auc_t:.4f} | {auc_c:.4f} | < 0.85 |

## Trajectories

See `analysis/p18_probe_2026-05-01/outputs/trajectory.csv` for the full
per-checkpoint table and `trajectory_plot.png` for the 4-panel comparison.

## What's next

{
"Phase 5: launch promotion-contract scorecards on treatment + control finals (~$10 each, ~3h Vertex). If scorecard recall lifts ≥ 5 pp on visomaster_enhanced_macro_dev vs P8A baseline (13.6% under v3 default), ship P18." if verdict == "α"
else "Phase 5: propose P19 with ramped λ (0 → λ_target over warmup_steps; ~10-line trainer patch wiring set_lambda) + production-tight training crops (RFA=0.85 deterministic; new bbox-aware crop transform). ~$120 / 1 day Vertex." if verdict == "β"
else "Phase 5: pivot to Move 4 paired same-identity contrastive. proper-data wave (705 paired identities) + companion_bucket plumbing exists. New code: paired sampler integration into combined_paired's combined_paired strategy + contrastive-loss head exposure on effort_detector + trainer forward-pass changes. ~2-3 days new code, then ~$60 Vertex."
}
"""
    VERDICT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(VERDICT_MD, "w") as f:
        f.write(md)
    print(f"Saved verdict doc: {VERDICT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
