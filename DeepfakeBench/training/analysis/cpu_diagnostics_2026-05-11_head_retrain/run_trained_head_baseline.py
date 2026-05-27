"""Compute trained-head baseline AUC on the LOCAL-AVAILABLE lockbox subset (same 494 reals + 253 fakes)
used by the head retrain. Reads per-frame `frame_prob` from the existing T4 scorecard CSVs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-11_head_retrain"
LOCAL_REAL_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/real/flat")
LOCAL_FAKE_DIR = Path("/Users/roeedar/Downloads/faces/r9_feb28_for_checker/fake/flat")
REAL_DOR_PNG_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-11_a2_extension/_real_dor_png"


def gs_to_local(gs_path: str):
    bn = os.path.basename(gs_path)
    if bn.startswith("real_dor__") and bn.endswith(".png"):
        c = REAL_DOR_PNG_DIR / bn
        return c if c.exists() else None
    if "/real/" in gs_path:
        c = LOCAL_REAL_DIR / bn
        return c if c.exists() else None
    if "/fake/" in gs_path:
        c = LOCAL_FAKE_DIR / bn
        return c if c.exists() else None
    return None


CHRONIC = ["bla_bla_chow", "PC_Generator__s22", "PC_Generator__s45", "roy_d", "Q__s6"]


def assign_cohort(vid: str, src: str) -> str:
    vl = str(vid).lower()
    if src == "real_dor":
        return "real_dor"
    if src == "dor_shkedi":
        return "dor_shkedi"
    for c in CHRONIC:
        if c.lower() in vl:
            return "chronic_6"
    return "non_chronic"


def main():
    rows = []
    for ckpt_label, ckpt_csv in [
        ("T4_LAMBDA1_TOP_N_STEP10500", "t4_lambda1_top_n_step10500"),
        ("P8A_REFERENCE_STEP5000", "p8a_reference_step5000"),
    ]:
        real_csv = REPO_ROOT / f"analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_real_all_lockbox_{ckpt_csv}_frames_report.csv"
        fake_csv = REPO_ROOT / f"analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_fake_all_lockbox_{ckpt_csv}_frames_report.csv"
        rr = pd.read_csv(real_csv)
        ff = pd.read_csv(fake_csv)
        rr["local"] = rr["frame_path"].map(gs_to_local)
        ff["local"] = ff["frame_path"].map(gs_to_local)
        rr_avail = rr[rr["local"].notna()].copy()
        ff_avail = ff[ff["local"].notna()].copy()
        rr_avail["source"] = rr_avail["video_id"].str.split("__").str[0]
        rr_avail["cohort"] = rr_avail.apply(
            lambda r: assign_cohort(r["video_id"], r["source"]), axis=1
        )

        real_v = rr_avail.groupby("video_id").agg(
            prob=("frame_prob", "mean"),
            source=("source", "first"),
            cohort=("cohort", "first"),
        ).reset_index()
        real_v["label"] = 0
        fake_v = ff_avail.groupby("video_id").agg(prob=("frame_prob", "mean")).reset_index()
        fake_v["label"] = 1
        fake_v["cohort"] = "fake"

        all_v = pd.concat(
            [real_v[["video_id", "prob", "label", "cohort"]],
             fake_v[["video_id", "prob", "label", "cohort"]]],
            axis=0,
        )
        auc_full = float(roc_auc_score(all_v["label"], all_v["prob"]))

        cohort_aucs = {}
        for coh in ["chronic_6", "real_dor", "dor_shkedi", "non_chronic"]:
            mask = (all_v["cohort"] == coh) | (all_v["cohort"] == "fake")
            sub = all_v[mask]
            if sub["label"].nunique() < 2:
                cohort_aucs[coh] = float("nan")
                continue
            cohort_aucs[coh] = float(roc_auc_score(sub["label"], sub["prob"]))
        rows.append({
            "ckpt": ckpt_label,
            "split": "lockbox_local_subset",
            "n_videos": int(len(all_v)),
            "n_reals": int((all_v["label"] == 0).sum()),
            "n_fakes": int((all_v["label"] == 1).sum()),
            "trained_head_lockbox_auc": auc_full,
            "trained_head_chronic_6_auc": cohort_aucs["chronic_6"],
            "trained_head_real_dor_auc": cohort_aucs["real_dor"],
            "trained_head_dor_shkedi_auc": cohort_aucs["dor_shkedi"],
            "trained_head_non_chronic_auc": cohort_aucs["non_chronic"],
        })
        print(f"{ckpt_label}: full={auc_full:.4f}, chronic_6={cohort_aucs['chronic_6']:.4f}, "
              f"real_dor={cohort_aucs['real_dor']:.4f}, dor_shkedi={cohort_aucs['dor_shkedi']:.4f}, "
              f"non_chronic={cohort_aucs['non_chronic']:.4f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "trained_head_baseline_aucs.csv", index=False)


if __name__ == "__main__":
    main()
