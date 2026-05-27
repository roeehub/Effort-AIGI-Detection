"""Per-mode tau on LOCKBOX reals — does the deployment policy generalize?

Three tests:
1. Per-mode tau distribution on lockbox reals vs dev reals — same mode tau values?
2. Apply DEV-calibrated per-mode tau to LOCKBOX fakes — recall + actual FPR (cross-substrate)
3. Apply LOCKBOX-calibrated per-mode tau to LOCKBOX fakes — within-substrate ceiling

If dev-calibrated per-mode tau gives ≤10% FPR on lockbox AND lockbox-fake recall close to within-substrate ceiling, the policy generalizes; ship.
If dev-calibrated tau under-controls FPR on lockbox, the policy needs lockbox-shaped re-calibration at deployment time.
"""

import os
import json
import pandas as pd
import numpy as np

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT = f"{ROOT}/analysis/lockbox_per_mode_tau_2026-05-05"

os.makedirs(OUT, exist_ok=True)

REAL_DEV = {
    "P8A":      f"{RAW}/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/teams_real_all_dev_e3_top_n_step6600_frames_report.csv",
}
REAL_LOCKBOX = {
    "P8A":      f"{RAW}/teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/teams_real_all_lockbox_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/teams_real_all_lockbox_e3_top_n_step6600_frames_report.csv",
}
FAKE_LOCKBOX = {
    "P8A":      f"{RAW}/teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/teams_fake_all_lockbox_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/teams_fake_all_lockbox_e3_top_n_step6600_frames_report.csv",
}
TAU_F0_DEV = {"P8A": 0.70503, "E2B_3200": 0.50644, "E3_6600": 0.85196}
MODES = ["normal_photo", "webcam", "phone_screen", "screen", "screen_recording"]

# ----------------------------------------------------------------------
# Load tags
# ----------------------------------------------------------------------
print("Loading tags...")
tags = pd.read_parquet(TAGS)[["gcs_uri", "clip_capture_mode", "split"]].rename(columns={"gcs_uri": "frame_path"})
print(f"  total tagged: {len(tags)}")
print(f"  splits: {tags['split'].value_counts().to_dict()}")

# ----------------------------------------------------------------------
# Helper: load + merge per-frame reports across 3 ckpts + tags
# ----------------------------------------------------------------------
def load_merged(file_dict):
    dfs = {}
    for ck, path in file_dict.items():
        d = pd.read_csv(path).rename(columns={"frame_prob": f"score_{ck}"})
        dfs[ck] = d[["frame_path", "label", f"score_{ck}"]]
    m = dfs["P8A"]
    for ck in ["E2B_3200", "E3_6600"]:
        m = m.merge(dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")
    m = m.merge(tags[["frame_path", "clip_capture_mode"]], on="frame_path", how="left")
    return m

print("Loading reals (dev + lockbox) and lockbox fakes...")
real_dev = load_merged(REAL_DEV)
real_lock = load_merged(REAL_LOCKBOX)
fake_lock = load_merged(FAKE_LOCKBOX)

print(f"  real_dev: {len(real_dev)}, with mode tag: {real_dev['clip_capture_mode'].notna().sum()}")
print(f"  real_lock: {len(real_lock)}, with mode tag: {real_lock['clip_capture_mode'].notna().sum()}")
print(f"  fake_lock: {len(fake_lock)}, with mode tag: {fake_lock['clip_capture_mode'].notna().sum()}")

print("\nLockbox real per-mode counts:")
print(real_lock["clip_capture_mode"].value_counts(dropna=False))
print("\nLockbox fake per-mode counts:")
print(fake_lock["clip_capture_mode"].value_counts(dropna=False))

# ----------------------------------------------------------------------
# 1. Per-mode tau on dev vs lockbox — are they the same?
# ----------------------------------------------------------------------
def compute_per_mode_tau(real_df, ckpts):
    out = {ck: {} for ck in ckpts}
    for ck in ckpts:
        score_col = f"score_{ck}"
        for mode in MODES:
            sub = real_df[real_df["clip_capture_mode"] == mode]
            if len(sub) == 0:
                out[ck][mode] = None
            else:
                out[ck][mode] = float(sub[score_col].quantile(0.90))
    return out

per_mode_tau_dev = compute_per_mode_tau(real_dev, REAL_DEV.keys())
per_mode_tau_lock = compute_per_mode_tau(real_lock, REAL_LOCKBOX.keys())

print("\nPer-mode tau (FPR=10% within mode) — DEV reals:")
print(pd.DataFrame(per_mode_tau_dev).T.to_string())
print("\nPer-mode tau (FPR=10% within mode) — LOCKBOX reals:")
print(pd.DataFrame(per_mode_tau_lock).T.to_string())

# Save comparison table
comp_rows = []
for ck in REAL_DEV:
    for mode in MODES:
        comp_rows.append({
            "ckpt": ck,
            "capture_mode": mode,
            "tau_dev_calibrated": per_mode_tau_dev[ck].get(mode),
            "tau_lockbox_calibrated": per_mode_tau_lock[ck].get(mode),
        })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(f"{OUT}/per_mode_tau_dev_vs_lockbox.csv", index=False)
print("\nDev vs lockbox per-mode tau comparison saved.")

# ----------------------------------------------------------------------
# 2. Apply dev-calibrated per-mode tau to LOCKBOX reals — what's the FPR?
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("CROSS-SUBSTRATE: dev-calibrated tau applied to lockbox")
print("="*60)

lock_with_mode = real_lock[real_lock["clip_capture_mode"].notna()].copy()
xrows = []
for ck in REAL_DEV:
    score_col = f"score_{ck}"
    # Global tau
    global_fpr = (real_lock[score_col] >= TAU_F0_DEV[ck]).mean()
    global_fpr_covered = (lock_with_mode[score_col] >= TAU_F0_DEV[ck]).mean()
    # Dev-calibrated per-mode tau, applied to lockbox covered reals
    lock_with_mode["tau_dev_per_mode"] = lock_with_mode["clip_capture_mode"].map(lambda m: per_mode_tau_dev[ck].get(m))
    valid = lock_with_mode["tau_dev_per_mode"].notna()
    fpr_dev_per_mode_on_lockbox = (lock_with_mode.loc[valid, score_col] >= lock_with_mode.loc[valid, "tau_dev_per_mode"]).mean()
    # Lockbox-calibrated per-mode tau, applied to lockbox covered reals
    lock_with_mode["tau_lock_per_mode"] = lock_with_mode["clip_capture_mode"].map(lambda m: per_mode_tau_lock[ck].get(m))
    valid_l = lock_with_mode["tau_lock_per_mode"].notna()
    fpr_lock_per_mode_on_lockbox = (lock_with_mode.loc[valid_l, score_col] >= lock_with_mode.loc[valid_l, "tau_lock_per_mode"]).mean()
    xrows.append({
        "ckpt": ck,
        "n_lockbox_reals_total": len(real_lock),
        "n_lockbox_reals_covered": valid.sum(),
        "fpr_dev_global_tau_pct": 100 * global_fpr,
        "fpr_dev_global_tau_covered_pct": 100 * global_fpr_covered,
        "fpr_dev_per_mode_tau_on_lockbox_pct": 100 * fpr_dev_per_mode_on_lockbox,
        "fpr_lock_per_mode_tau_on_lockbox_pct": 100 * fpr_lock_per_mode_on_lockbox,
    })
x_df = pd.DataFrame(xrows)
print("\nFPR on lockbox reals under various tau choices:")
print(x_df.to_string(index=False))
x_df.to_csv(f"{OUT}/lockbox_real_fpr_under_each_tau.csv", index=False)

# ----------------------------------------------------------------------
# 3. Recall on LOCKBOX FAKES under each tau choice
# ----------------------------------------------------------------------
print("\nRecall on LOCKBOX FAKES under each tau choice:")
fake_with_mode = fake_lock[fake_lock["clip_capture_mode"].notna()].copy()
recall_rows = []
for ck in REAL_DEV:
    score_col = f"score_{ck}"
    # Global tau
    global_recall = (fake_lock[score_col] >= TAU_F0_DEV[ck]).mean()
    global_recall_covered = (fake_with_mode[score_col] >= TAU_F0_DEV[ck]).mean()
    # Dev-calibrated per-mode
    fake_with_mode["tau_dev_per_mode"] = fake_with_mode["clip_capture_mode"].map(lambda m: per_mode_tau_dev[ck].get(m))
    valid = fake_with_mode["tau_dev_per_mode"].notna()
    recall_dev_per_mode = (fake_with_mode.loc[valid, score_col] >= fake_with_mode.loc[valid, "tau_dev_per_mode"]).mean()
    # Lockbox-calibrated per-mode
    fake_with_mode["tau_lock_per_mode"] = fake_with_mode["clip_capture_mode"].map(lambda m: per_mode_tau_lock[ck].get(m))
    valid_l = fake_with_mode["tau_lock_per_mode"].notna()
    recall_lock_per_mode = (fake_with_mode.loc[valid_l, score_col] >= fake_with_mode.loc[valid_l, "tau_lock_per_mode"]).mean()
    recall_rows.append({
        "ckpt": ck,
        "n_lockbox_fakes_total": len(fake_lock),
        "n_lockbox_fakes_covered": valid.sum(),
        "recall_dev_global_tau_pct": 100 * global_recall,
        "recall_dev_global_tau_covered_pct": 100 * global_recall_covered,
        "recall_dev_per_mode_tau_pct": 100 * recall_dev_per_mode,
        "recall_lock_per_mode_tau_pct": 100 * recall_lock_per_mode,
    })
recall_df = pd.DataFrame(recall_rows)
print(recall_df.to_string(index=False))
recall_df.to_csv(f"{OUT}/lockbox_fake_recall_under_each_tau.csv", index=False)

# ----------------------------------------------------------------------
# 4. Per-mode breakdown (which mode of the lockbox-fake suite is hardest?)
# ----------------------------------------------------------------------
print("\nPer-mode lockbox FAKE recall under DEV per-mode tau:")
breakdown_rows = []
for ck in REAL_DEV:
    score_col = f"score_{ck}"
    for mode in MODES:
        sub = fake_lock[fake_lock["clip_capture_mode"] == mode]
        if len(sub) == 0:
            continue
        tau = per_mode_tau_dev[ck].get(mode)
        if tau is None:
            continue
        rec = (sub[score_col] >= tau).mean()
        breakdown_rows.append({
            "ckpt": ck,
            "capture_mode": mode,
            "n_fakes_in_mode": len(sub),
            "tau_dev": tau,
            "recall_pct": 100 * rec,
        })
breakdown_df = pd.DataFrame(breakdown_rows)
print(breakdown_df.to_string(index=False))
breakdown_df.to_csv(f"{OUT}/lockbox_fake_recall_per_mode.csv", index=False)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
summary = {
    "tau_dev_vs_lockbox": {
        ck: {mode: {"dev": per_mode_tau_dev[ck].get(mode),
                    "lockbox": per_mode_tau_lock[ck].get(mode)}
             for mode in MODES} for ck in REAL_DEV
    },
    "lockbox_real_fpr_under_dev_per_mode_tau": {
        r["ckpt"]: r["fpr_dev_per_mode_tau_on_lockbox_pct"] for r in xrows
    },
    "lockbox_fake_recall_under_dev_per_mode_tau": {
        r["ckpt"]: r["recall_dev_per_mode_tau_pct"] for r in recall_rows
    },
    "interpretation": (
        "If lockbox FPR under dev-calibrated per-mode tau is close to 10%, the policy generalizes. "
        "If much higher, deployment needs lockbox-shaped re-calibration. "
        "Compare lockbox-fake recall under dev vs lockbox per-mode tau — gap is the cost of using dev-calibrated tau."
    ),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDone. Outputs in {OUT}")
