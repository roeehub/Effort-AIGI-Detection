"""Analyze the team-identity deploy readout: merge scores, compute per-human metrics.

Steps:
1. Load master_inventory.csv.
2. For each of the 5 ckpts: load per-frame scores CSV; merge into inventory by frame_path.
3. For cached ckpts (P8A, E2B, T5C): backfill from inventory's score_P8A/score_E2B/score_T5C
   wherever the fresh-score CSV doesn't cover (these are the deploy-relevant frames already
   in grouped_manifest_v2).
4. Compute per-(ckpt × human) FPR (real-side) and fake-recall (fake-side) at the 4 τ modes.
5. Per-cohort breakdown, per-human aggregate, ship verdict.

Output:
- outputs/per_frame_full.csv (all frames × all 5 ckpt scores)
- outputs/per_cohort_summary.csv (one row per (ckpt × base_identity × mode), FPR or recall)
- outputs/per_human_summary.csv (one row per (ckpt × human), aggregated)
- outputs/ship_verdict.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs"
INV_CSV = OUTPUT_DIR / "master_inventory.csv"

CKPT_KEYS = [
    "P8A_REFERENCE_STEP5000",
    "E2B_TOP_N_STEP3200",
    "T5C_PERIODIC_STEP3500",
    "SLOT_A_ANCHOR_AWARE_STEP3500",
    "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL",
]

# Map fresh-score-CSV ckpt_key -> short name for output columns
CKPT_SHORTNAME = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "T5C_PERIODIC_STEP3500": "T5C",
    "SLOT_A_ANCHOR_AWARE_STEP3500": "SlotAv2_CLS",
    "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL": "SlotAv2_FACE",
}

# Map cached score column -> ckpt_key
CACHED_COLS = {
    "P8A_REFERENCE_STEP5000": "score_P8A",
    "E2B_TOP_N_STEP3200": "score_E2B",
    "T5C_PERIODIC_STEP3500": "score_T5C",
}

# τ modes — same as prior readout, per project_deployment_three_modes_slot_a_v2_2026-05-21
TAU_MODES = {
    "tau_0_5": 0.5,
    "mode_A_tau_0_535": 0.535,
    "mode_B_tau_0_78": 0.78,
    "mode_C_tau_0_87": 0.87,
}


def load_and_merge() -> pd.DataFrame:
    inv = pd.read_csv(INV_CSV, low_memory=False)
    print(f"[merge] inventory: {len(inv)} frames", flush=True)

    # Make sure frame_path is normalized to gs:// uri for join
    if "frame_path" not in inv.columns:
        raise SystemExit("inventory missing frame_path")
    inv["join_uri"] = inv["frame_path"]

    # Backfill SCORE columns: start from cached
    for ck, cached_col in CACHED_COLS.items():
        out_col = f"prob_{CKPT_SHORTNAME[ck]}"
        if cached_col in inv.columns:
            inv[out_col] = inv[cached_col].astype(float)
        else:
            inv[out_col] = np.nan

    # Initialize SlotAv2 cols (no cached source)
    for ck in ["SLOT_A_ANCHOR_AWARE_STEP3500", "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL"]:
        out_col = f"prob_{CKPT_SHORTNAME[ck]}"
        if out_col not in inv.columns:
            inv[out_col] = np.nan

    # Merge fresh scores: each per-frame CSV has cols: base_identity, frame_path, prob_fake, status
    for ck in CKPT_KEYS:
        out_col = f"prob_{CKPT_SHORTNAME[ck]}"
        csv_path = OUTPUT_DIR / f"{ck}_scores.per_frame.csv"
        if not csv_path.exists():
            print(f"[merge] {ck}: CSV not found — {csv_path}", flush=True)
            continue
        sc = pd.read_csv(csv_path)
        sc = sc.rename(columns={"prob_fake": f"fresh_{ck}"})
        # Set 0/failed_decode rows to NaN (so they don't count as FPR=0)
        sc.loc[sc.status != "ok", f"fresh_{ck}"] = np.nan
        # Merge on frame_path  (left join keeping inv order)
        sc_to_merge = sc[["frame_path", f"fresh_{ck}"]].drop_duplicates("frame_path")
        inv = inv.merge(sc_to_merge, left_on="join_uri", right_on="frame_path",
                        how="left", suffixes=("", "_right"))
        # Drop the right-side frame_path
        if "frame_path_right" in inv.columns:
            inv = inv.drop(columns=["frame_path_right"])
        # Backfill fresh score onto out_col, preferring fresh over cached
        fresh_col = f"fresh_{ck}"
        if fresh_col in inv.columns:
            n_fresh = inv[fresh_col].notna().sum()
            print(f"[merge] {ck}: {n_fresh} fresh scores merged", flush=True)
            # Where fresh is non-null, use it; else keep cached/prior
            inv[out_col] = np.where(inv[fresh_col].notna(), inv[fresh_col], inv[out_col])
            inv = inv.drop(columns=[fresh_col])

    # Final coverage report
    for ck in CKPT_KEYS:
        out_col = f"prob_{CKPT_SHORTNAME[ck]}"
        n_have = inv[out_col].notna().sum()
        print(f"[merge] final coverage {ck}: {n_have}/{len(inv)} ({100*n_have/len(inv):.1f}%)",
              flush=True)

    out_csv = OUTPUT_DIR / "per_frame_full.csv"
    keep_cols = [
        "base_identity", "suite", "bucket", "frame_path", "label",
        "human", "device", "deploy_relevant", "role",
        "needs_fresh_score",
        "prob_P8A", "prob_E2B", "prob_T5C", "prob_SlotAv2_CLS", "prob_SlotAv2_FACE",
    ]
    keep_cols = [c for c in keep_cols if c in inv.columns]
    inv_out = inv[keep_cols]
    inv_out.to_csv(out_csv, index=False)
    print(f"[merge] wrote {out_csv}", flush=True)
    return inv


def compute_per_cohort(inv: pd.DataFrame) -> pd.DataFrame:
    """One row per (ckpt × base_identity × role × mode) with FPR or recall."""
    rows = []
    for ck in CKPT_KEYS:
        prob_col = f"prob_{CKPT_SHORTNAME[ck]}"
        for bi, sub in inv.groupby("base_identity"):
            # split by role
            for role, sub_role in sub.groupby("role"):
                n = len(sub_role)
                n_scored = sub_role[prob_col].notna().sum()
                human = sub_role["human"].iloc[0]
                device = sub_role["device"].iloc[0]
                deploy = sub_role["deploy_relevant"].iloc[0]
                row = {
                    "ckpt": CKPT_SHORTNAME[ck],
                    "base_identity": bi,
                    "human": human,
                    "device": device,
                    "deploy_relevant": deploy,
                    "role": role,
                    "n_frames": n,
                    "n_scored": n_scored,
                    "mean_prob": sub_role[prob_col].mean() if n_scored else np.nan,
                    "median_prob": sub_role[prob_col].median() if n_scored else np.nan,
                }
                for mode_name, tau in TAU_MODES.items():
                    if n_scored == 0:
                        row[f"metric_{mode_name}"] = np.nan
                        continue
                    if role == "real":
                        # FPR
                        fpr = (sub_role[prob_col] >= tau).sum() / n_scored
                        row[f"metric_{mode_name}"] = float(fpr)
                    else:
                        # fake_recall
                        recall = (sub_role[prob_col] >= tau).sum() / n_scored
                        row[f"metric_{mode_name}"] = float(recall)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "per_cohort_summary.csv", index=False)
    return out


def compute_per_human(inv: pd.DataFrame) -> pd.DataFrame:
    """One row per (ckpt × human × role) with frame-weighted aggregate FPR or recall.

    Only includes deploy-relevant frames (deploy_relevant=True).
    """
    rows = []
    for ck in CKPT_KEYS:
        prob_col = f"prob_{CKPT_SHORTNAME[ck]}"
        for human, sub_h in inv[inv.deploy_relevant].groupby("human"):
            for role, sub in sub_h.groupby("role"):
                n = len(sub)
                n_scored = sub[prob_col].notna().sum()
                row = {
                    "ckpt": CKPT_SHORTNAME[ck],
                    "human": human,
                    "role": role,
                    "n_frames": n,
                    "n_scored": n_scored,
                    "mean_prob": sub[prob_col].mean() if n_scored else np.nan,
                }
                for mode_name, tau in TAU_MODES.items():
                    if n_scored == 0:
                        row[f"metric_{mode_name}"] = np.nan
                        continue
                    metric = (sub[prob_col] >= tau).sum() / n_scored
                    row[f"metric_{mode_name}"] = float(metric)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "per_human_summary.csv", index=False)
    return out


def compute_per_human_mac_info(inv: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_per_human but for Mac-Roee (info only — out of scope for ship verdict)."""
    rows = []
    for ck in CKPT_KEYS:
        prob_col = f"prob_{CKPT_SHORTNAME[ck]}"
        for human, sub_h in inv[~inv.deploy_relevant].groupby("human"):
            if human != "Roee_Mac":
                continue
            for role, sub in sub_h.groupby("role"):
                n = len(sub)
                n_scored = sub[prob_col].notna().sum()
                row = {
                    "ckpt": CKPT_SHORTNAME[ck],
                    "human": human,
                    "role": role,
                    "n_frames": n,
                    "n_scored": n_scored,
                    "mean_prob": sub[prob_col].mean() if n_scored else np.nan,
                }
                for mode_name, tau in TAU_MODES.items():
                    if n_scored == 0:
                        row[f"metric_{mode_name}"] = np.nan
                        continue
                    metric = (sub[prob_col] >= tau).sum() / n_scored
                    row[f"metric_{mode_name}"] = float(metric)
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(OUTPUT_DIR / "per_human_summary_mac_info.csv", index=False)
    return out


def compute_ship_verdict(per_human: pd.DataFrame) -> pd.DataFrame:
    """Apply gates: real-side per-human FPR ≤ 5%, fake-side per-human fake_recall ≥ 50%."""
    REAL_FLOOR = 0.05
    FAKE_FLOOR = 0.50

    rows = []
    for ck in CKPT_KEYS:
        short = CKPT_SHORTNAME[ck]
        sub = per_human[per_human.ckpt == short]
        for mode_name in TAU_MODES:
            mcol = f"metric_{mode_name}"
            # Real side: every deploy-relevant human's FPR ≤ REAL_FLOOR
            real = sub[sub.role == "real"]
            real_fail = []
            for _, r in real.iterrows():
                if r["n_scored"] == 0:
                    continue
                if r[mcol] > REAL_FLOOR:
                    real_fail.append(f"{r['human']}({r[mcol]:.3f})")
            real_pass = len(real_fail) == 0

            # Fake side: every human's fake_recall ≥ FAKE_FLOOR (for humans with fake-attack cohorts)
            fake = sub[sub.role.str.startswith("fake_target_")]
            fake_fail = []
            for _, r in fake.iterrows():
                if r["n_scored"] == 0:
                    continue
                if r[mcol] < FAKE_FLOOR:
                    fake_fail.append(f"{r['human']}({r[mcol]:.3f})")
            fake_pass = len(fake_fail) == 0

            row = {
                "ckpt": short,
                "mode": mode_name,
                "tau": TAU_MODES[mode_name],
                "real_floor": REAL_FLOOR,
                "fake_floor": FAKE_FLOOR,
                "real_pass": real_pass,
                "fake_pass": fake_pass,
                "both_pass": real_pass and fake_pass,
                "real_failing_humans": ",".join(real_fail) if real_fail else "",
                "fake_failing_humans": ",".join(fake_fail) if fake_fail else "",
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "ship_verdict.csv", index=False)
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = load_and_merge()
    per_cohort = compute_per_cohort(inv)
    per_human = compute_per_human(inv)
    per_human_mac = compute_per_human_mac_info(inv)
    verdict = compute_ship_verdict(per_human)

    print("\n=== PER-HUMAN SUMMARY (deploy-relevant) ===")
    print(per_human.to_string(index=False))

    if len(per_human_mac):
        print("\n=== PER-HUMAN SUMMARY (Mac-Roee — informational only) ===")
        print(per_human_mac.to_string(index=False))

    print("\n=== SHIP VERDICT ===")
    print(verdict.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
