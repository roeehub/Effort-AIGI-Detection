"""Post-process tau_sweep_table.csv to produce calibrated-τ summary using the
correct suite names from the 800-frame canary panel (proper_real_clean_lockbox
NOT teams_real_all_lockbox — the panel composition is different from the 29-suite
contract substrate).

Outputs:
  outputs/calibrated_tau_summary.csv
  outputs/operating_point_comparison.md (markdown table for FACTS doc)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "outputs"
df = pd.read_csv(OUT_DIR / "tau_sweep_table.csv")

# Suite names that ARE in the 800-frame canary panel
DEV_REAL_SUITE = "teams_real_all_dev"
LOCKBOX_REAL_SUITE = "proper_real_clean_lockbox"  # HDTF-style; the Teams lockbox real is NOT in this panel
LOCKBOX_FAKE_SUITE = "teams_fake_all_lockbox"
DEV_FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]

CKPT_ORDER = [
    "P8A_REFERENCE_STEP5000",
    "T5C_PERIODIC_STEP3500",
    "SLOT_A_V2_STEP3500",
    "SLOT_1_6AXIS_ANCHOR_STEP1500",
    "SLOT_1_6AXIS_ANCHOR_STEP2500",
    "SLOT_1_6AXIS_ANCHOR_STEP3500",
    "SLOT_2_LORA_L8_L9_STEP2500",
    "SLOT_3_5AXIS_NOLUMA_STEP3500",
]


def summary_at_target_dev_fpr(target_fpr: float):
    rows = []
    for ckpt in CKPT_ORDER:
        dev = df[(df["ckpt"] == ckpt) & (df["suite"] == DEV_REAL_SUITE)].sort_values("tau")
        if dev.empty:
            continue
        below = dev[dev["real_fpr"] <= target_fpr]
        if below.empty:
            tau_cal = float(dev["tau"].max())
            dev_fpr = float(dev["real_fpr"].iloc[-1])
            reached = False
        else:
            tau_cal = float(below["tau"].iloc[0])
            dev_fpr = float(below["real_fpr"].iloc[0])
            reached = True

        def metric_at(suite, col):
            sub = df[(df["ckpt"] == ckpt) & (df["suite"] == suite)
                     & (df["tau"].between(tau_cal - 1e-4, tau_cal + 1e-4))]
            return float(sub[col].iloc[0]) if not sub.empty else np.nan

        rows.append({
            "ckpt": ckpt,
            "target_dev_fpr": target_fpr,
            "tau_cal": tau_cal,
            "dev_real_fpr_achieved": dev_fpr,
            "reached_target": reached,
            "lockbox_fake_recall": metric_at(LOCKBOX_FAKE_SUITE, "fake_recall"),
            "lockbox_real_fpr_proper_clean": metric_at(LOCKBOX_REAL_SUITE, "real_fpr"),
            "viso_recall_dev": metric_at("visomaster_enhanced_macro_dev", "fake_recall"),
            "deeplive_recall_dev": metric_at("deeplive_enhanced_dev", "fake_recall"),
        })
    return pd.DataFrame(rows)


# Compute summary at 3 FPR targets
results = {}
for target in [0.05, 0.10, 0.20]:
    results[target] = summary_at_target_dev_fpr(target)

# Concat & save
combined = pd.concat([results[t] for t in [0.05, 0.10, 0.20]], ignore_index=True)
combined.to_csv(OUT_DIR / "calibrated_tau_summary.csv", index=False)

# Print markdown
md_lines = []
md_lines.append("# Operating-point comparison — calibrated-τ summary\n")
md_lines.append(
    f"Panel: 800-frame manual canary (`analysis/manual_canary_2026-05-20/frames_meta.parquet`)\n"
)
md_lines.append(
    f"Calibration: τ s.t. real_fpr on `{DEV_REAL_SUITE}` ≤ target_dev_fpr.\n"
)
md_lines.append(
    f"NOTE: `lockbox_real_fpr_proper_clean` is on `{LOCKBOX_REAL_SUITE}` (HDTF-style cleans), "
    f"not the production Teams lockbox. The 29-suite scorecard uses `teams_real_all_lockbox` "
    f"which is NOT in this panel.\n"
)

for target, sub in results.items():
    md_lines.append(f"\n## At target_dev_fpr ≤ {target:.2f}\n")
    md_lines.append("| ckpt | τ_cal | dev_fpr | dev_target_reached | lockbox_fake_recall | proper_clean_lockbox_fpr | viso_recall_dev | deeplive_recall_dev |\n")
    md_lines.append("|---|---:|---:|:---:|---:|---:|---:|---:|\n")
    for _, r in sub.iterrows():
        reached = "✅" if r["reached_target"] else "❌"
        md_lines.append(
            f"| {r['ckpt']} | {r['tau_cal']:.4f} | {r['dev_real_fpr_achieved']:.4f} | {reached} | "
            f"{r['lockbox_fake_recall']:.4f} | {r['lockbox_real_fpr_proper_clean']:.4f} | "
            f"{r['viso_recall_dev']:.4f} | {r['deeplive_recall_dev']:.4f} |\n"
        )

with open(OUT_DIR / "operating_point_comparison.md", "w") as f:
    f.writelines(md_lines)

print("".join(md_lines))
