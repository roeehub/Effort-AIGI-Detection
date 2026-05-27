#!/usr/bin/env python3
"""CURRENT_CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06.

Per paired-sample cohort diagnosis classifying each frame under each of P8A,
E2B, PA_3800 into one of:
    real_OK_fake_OK
    real_OK_fake_missed
    real_FP_fake_OK
    real_FP_fake_missed

Pairing strategy: bipartite cohort by `person_cluster` -- see `_person_cluster()`.
Inside a cluster, every real frame is conceptually "paired" against every fake
frame from the same person. Rather than expand to a Cartesian product (which
explodes the count and is misleading), we operate at *frame level* and report:

  * fake-side cell:   real_OK_fake_OK / real_OK_fake_missed
                      (using the cluster's real-side aggregate as the
                       conditioning context).
  * real-side cell:   real_OK_fake_OK / real_FP_fake_OK
                      (conditioned on the cluster's fake-side aggregate).

Each frame is therefore classified once on its own (label-conditional) AND we
also compute a "joint cell" label per frame as:

   joint_cell(real_frame)  -> real_FP_fake_X  if frame is FP, else real_OK_fake_X
       where X = "missed" if cluster fake-recall <50% else "OK"
   joint_cell(fake_frame)  -> real_X_fake_missed  if frame missed, else real_X_fake_OK
       where X = "FP" if cluster real-FPR >5% else "OK"

The cohort cross-tabs aggregate the joint_cell distributions over (axis_value,
ckpt) so we can read 4-cell fractions per cohort.

Outputs in `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
OUT_DIR = ROOT / "analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A", "E2B", "PA_3800"]
SCORE_COLS = {c: f"score_{c}" for c in CKPTS}
TAU_PRIMARY = 0.5
TAU_SWEEP = (0.3, 0.5, 0.7)


# ----- Pairing key: person cluster -----------------------------------------
def _person_cluster(bi: object) -> str:
    s = str(bi).lower()
    if "dor_shkedi" in s:
        return "dor_shkedi"
    if s.startswith("dor_evening") or s.startswith("dor_morning"):
        return "dor_evening_morning"
    if s.startswith("dor_fake_"):
        return "dor_evening_morning"
    if "team_may5__dor" in s:
        return "dor_evening_morning"
    if "real_dor" in s:
        return "dor_shkedi"
    if "xiang" in s:
        return "xiang"
    if "xinhe" in s or "xinghe" in s:
        return "xinhe"
    if s.startswith("roee_tester") or s.startswith("tester_roee"):
        return "roee_tester"
    if s.startswith("royd") or s == "roy_d" or s == "extra_roy_d":
        return "royd"
    if s.startswith("cam_test__") or s.startswith("test_cam__"):
        return s
    if "pc_generator" in s:
        return "pc_generator"
    if "bla_bla_chow" in s:
        return "bla_bla_chow"
    if "chikara" in s:
        return s
    if "md_noyn" in s:
        return s
    return s


# ----- Method/enhancer/transport derivation --------------------------------
ENHANCER_KEYWORDS = {
    "gfpgan": "gfpgan",
    "codeformer": "codeformer",
    "gpen": "gpen",
}


def _enhancer_family(row) -> str:
    """Best-effort enhancer attribution from path + identity."""
    s = (str(row.get("frame_path", "")) + "|" + str(row.get("base_identity", ""))).lower()
    for k, v in ENHANCER_KEYWORDS.items():
        if k in s:
            return v
    if "_enhanced" in s:
        return "enhanced_unspec"
    if "_regular" in s:
        return "regular"
    return "none"


def _method_family(row) -> str:
    s = (str(row.get("frame_path", "")) + "|" + str(row.get("base_identity", ""))).lower()
    if "deeplive" in s or "deep_live" in s:
        return "deeplive"
    if "inswapper" in s:
        return "inswapper"
    if "ghostface" in s:
        return "ghostface"
    if "instyle_swapper" in s or "instyle" in s:
        return "instyle"
    if "simswap" in s:
        return "simswap"
    if "visomaster" in str(row.get("suite", "")).lower() or "viso" in str(row.get("bucket", "")).lower():
        return "visomaster_unspec"
    if str(row.get("suite", "")).startswith("teams_fake"):
        return "teams_fake_unspec"
    if str(row.get("suite", "")).startswith("live_fakes"):
        return "live_unspec"
    if str(row.get("suite", "")).startswith("dor_fake_local"):
        return "local_unspec"
    return "real_or_unknown"


def _transport(row) -> str:
    suite = str(row.get("suite", ""))
    bucket = str(row.get("bucket", ""))
    if "live" in suite or "live" in bucket:
        return "teams_live"
    if "teams" in suite or "teams" in bucket:
        return "teams_capture"
    if "visomaster" in suite or "visomaster" in bucket:
        return "visomaster"
    if suite in ("dor_morning", "dor_evening", "team_sanity_may5", "xinhe_may6_falseflag", "extra"):
        return "raw_capture"
    if suite == "dor_fake_local":
        return "raw_capture"
    return "unknown"


def _quality_band(row) -> str:
    q = row.get("quality")
    if pd.isna(q) or q == "unknown":
        return "unknown"
    return str(q)


def _face_size_band(row) -> str:
    fs = row.get("face_size")
    if pd.isna(fs) or fs == "unknown":
        return "unknown"
    return str(fs)


def _pair_gap_band(row) -> str:
    """Banded score-gap inside the row -- used as a covariate.

    For a row with score_P8A, score_E2B, score_PA_3800 we compute the cross-ckpt
    spread as a proxy for "ckpts disagree" cohorts.  This is *not* the true
    pair_gap (real vs fake) which we compute separately at cohort level.
    """
    sc = np.array([row.get(c) for c in [SCORE_COLS[k] for k in CKPTS]], dtype=float)
    if np.any(np.isnan(sc)):
        return "unknown"
    spread = float(sc.max() - sc.min())
    if spread < 0.1:
        return "tight_<0.1"
    if spread < 0.3:
        return "mid_0.1-0.3"
    return "wide_>=0.3"


# ----- Frame-level classification helpers ----------------------------------
def classify_frame(score: float, label: int, tau: float) -> str:
    """Return the *frame's own* outcome -- collapses joint cells to per-row."""
    if pd.isna(score):
        return "missing"
    pred_fake = score >= tau
    if label == 1:  # ground truth fake
        return "fake_OK" if pred_fake else "fake_missed"
    else:  # real
        return "real_FP" if pred_fake else "real_OK"


def joint_cell_per_frame(frame_outcome: str, cluster_other_fail_rate: float, fail_threshold: float) -> str:
    """Combine the frame's own outcome with the cluster's other-side aggregate."""
    if frame_outcome == "fake_OK":
        other_bad = cluster_other_fail_rate > fail_threshold
        return "real_FP_fake_OK" if other_bad else "real_OK_fake_OK"
    if frame_outcome == "fake_missed":
        other_bad = cluster_other_fail_rate > fail_threshold
        return "real_FP_fake_missed" if other_bad else "real_OK_fake_missed"
    if frame_outcome == "real_OK":
        other_bad = cluster_other_fail_rate > fail_threshold
        return "real_OK_fake_missed" if other_bad else "real_OK_fake_OK"
    if frame_outcome == "real_FP":
        other_bad = cluster_other_fail_rate > fail_threshold
        return "real_FP_fake_missed" if other_bad else "real_FP_fake_OK"
    return "missing"


# ----- Main routine ---------------------------------------------------------
def main() -> int:
    if not DATA_PATH.exists():
        print(f"manifest not found: {DATA_PATH}", file=sys.stderr)
        return 1
    df = pd.read_csv(DATA_PATH)
    df["person_cluster"] = df["base_identity"].map(_person_cluster)
    df["method_family"] = df.apply(_method_family, axis=1)
    df["enhancer_family"] = df.apply(_enhancer_family, axis=1)
    df["transport"] = df.apply(_transport, axis=1)
    df["quality_band"] = df.apply(_quality_band, axis=1)
    df["face_size_band"] = df.apply(_face_size_band, axis=1)
    df["pair_gap_band"] = df.apply(_pair_gap_band, axis=1)

    # restrict to paired clusters
    pc_counts = df.groupby(["person_cluster", "label"]).size().unstack(fill_value=0)
    paired = set(pc_counts[(pc_counts.get(0, 0) > 0) & (pc_counts.get(1, 0) > 0)].index)
    df_paired = df[df["person_cluster"].isin(paired)].copy()
    print(f"paired clusters: {len(paired)} -- frames: {len(df_paired)} / {len(df)}")
    print(f"clusters: {sorted(paired)}")

    # ---- Frame-level outcome per ckpt at each tau ----
    rows = []
    for tau in TAU_SWEEP:
        for ckpt in CKPTS:
            col = SCORE_COLS[ckpt]
            df_paired[f"outcome_{ckpt}_t{tau}"] = [
                classify_frame(s, l, tau) for s, l in zip(df_paired[col], df_paired["label"])
            ]

    # ---- Cluster-level "other-side fail rate" so we can derive joint_cell ----
    fail_thresh_real = 0.05  # >5% real-FPR == cluster's "real side is bad"
    fail_thresh_fake = 0.50  # >50% missed-fakes == cluster's "fake side is bad"

    cluster_stats: dict = {}
    for tau in TAU_SWEEP:
        for ckpt in CKPTS:
            col = f"outcome_{ckpt}_t{tau}"
            for cluster, sub in df_paired.groupby("person_cluster"):
                real = sub[sub.label == 0]
                fake = sub[sub.label == 1]
                fpr = (real[col] == "real_FP").mean() if len(real) else float("nan")
                miss = (fake[col] == "fake_missed").mean() if len(fake) else float("nan")
                cluster_stats[(tau, ckpt, cluster)] = {"fpr": fpr, "missed_rate": miss}

    # joint cell column per (ckpt, tau)
    for tau in TAU_SWEEP:
        for ckpt in CKPTS:
            colo = f"outcome_{ckpt}_t{tau}"
            colj = f"joint_{ckpt}_t{tau}"
            joint_vals = []
            for outcome, cluster, label in zip(
                df_paired[colo], df_paired["person_cluster"], df_paired["label"]
            ):
                stats = cluster_stats[(tau, ckpt, cluster)]
                if label == 1:
                    other_fail = stats["fpr"] if not np.isnan(stats["fpr"]) else 0.0
                    threshold = fail_thresh_real
                else:
                    other_fail = stats["missed_rate"] if not np.isnan(stats["missed_rate"]) else 0.0
                    threshold = fail_thresh_fake
                joint_vals.append(joint_cell_per_frame(outcome, other_fail, threshold))
            df_paired[colj] = joint_vals

    # ---- Persist enriched manifest ----
    df_paired.to_csv(OUT_DIR / "paired_frames_with_outcomes.csv", index=False)

    # ---- Aggregate cell distributions per axis -----
    AXES = {
        "method": "method_family",
        "enhancer": "enhancer_family",
        "transport": "transport",
        "identity": "base_identity",
        "quality_band": "quality_band",
        "face_size_band": "face_size_band",
        "suite": "suite",
        "pair_gap_band": "pair_gap_band",
        "is_lockbox": "is_lockbox",
        "person_cluster": "person_cluster",
    }
    ALL_CELLS = ["real_OK_fake_OK", "real_OK_fake_missed", "real_FP_fake_OK", "real_FP_fake_missed"]

    def crosstab(axis_col: str, tau: float = TAU_PRIMARY) -> pd.DataFrame:
        out_rows = []
        for ckpt in CKPTS:
            colj = f"joint_{ckpt}_t{tau}"
            for axis_value, sub in df_paired.groupby(axis_col):
                n = len(sub)
                cells = sub[colj].value_counts(normalize=True).to_dict()
                row = {"ckpt": ckpt, "axis": axis_col, "axis_value": axis_value, "n_frames": n}
                for cell in ALL_CELLS:
                    row[f"frac_{cell}"] = cells.get(cell, 0.0)
                # headroom = frames where ckpt's fake-side is failing
                row["headroom_fake_failure"] = (
                    row["frac_real_OK_fake_missed"] + row["frac_real_FP_fake_missed"]
                )
                row["headroom_real_failure"] = (
                    row["frac_real_FP_fake_OK"] + row["frac_real_FP_fake_missed"]
                )
                row["total_failure"] = (
                    row["frac_real_OK_fake_missed"]
                    + row["frac_real_FP_fake_OK"]
                    + row["frac_real_FP_fake_missed"]
                )
                out_rows.append(row)
        return pd.DataFrame(out_rows).sort_values(["axis_value", "ckpt"]).reset_index(drop=True)

    for nice, axis_col in AXES.items():
        ct = crosstab(axis_col)
        ct.to_csv(OUT_DIR / f"cohort_by_{nice}.csv", index=False)
        print(f"wrote cohort_by_{nice}.csv  ({len(ct)} rows)")

    # ---- Aggregated summary per ckpt ----
    summary = {"tau_primary": TAU_PRIMARY, "tau_sweep": list(TAU_SWEEP), "n_paired_frames": int(len(df_paired)), "ckpts": {}}
    for ckpt in CKPTS:
        ck = {"per_tau": {}}
        for tau in TAU_SWEEP:
            colj = f"joint_{ckpt}_t{tau}"
            cells = df_paired[colj].value_counts(normalize=True).to_dict()
            ck["per_tau"][f"tau_{tau}"] = {
                **{c: float(cells.get(c, 0.0)) for c in ALL_CELLS},
                "fake_fpr_aggregate": float((df_paired[df_paired.label == 0][f"outcome_{ckpt}_t{tau}"] == "real_FP").mean()),
                "fake_recall_aggregate": float((df_paired[df_paired.label == 1][f"outcome_{ckpt}_t{tau}"] == "fake_OK").mean()),
            }
        summary["ckpts"][ckpt] = ck
    summary["paired_clusters"] = sorted(paired)

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote summary.json")

    # ---- FT-base recommendation per cohort ----
    rec = {"by_axis": {}, "overall": None}
    for nice, axis_col in AXES.items():
        if axis_col == "is_lockbox":
            continue
        ct = pd.read_csv(OUT_DIR / f"cohort_by_{nice}.csv")
        # for each axis_value, pick ckpt with lowest total_failure
        per_value = {}
        for axis_value, sub in ct.groupby("axis_value"):
            sub = sub.sort_values("total_failure")
            best = sub.iloc[0]
            second = sub.iloc[1] if len(sub) > 1 else None
            per_value[str(axis_value)] = {
                "n_frames": int(best["n_frames"]),
                "best_ckpt": str(best["ckpt"]),
                "best_total_failure": float(best["total_failure"]),
                "second_ckpt": str(second["ckpt"]) if second is not None else None,
                "gap": float(second["total_failure"] - best["total_failure"]) if second is not None else None,
                "frac_real_FP_fake_missed_per_ckpt": {
                    str(r["ckpt"]): float(r["frac_real_FP_fake_missed"]) for _, r in sub.iterrows()
                },
            }
        rec["by_axis"][nice] = per_value

    # overall: vote across axis-values (weighted by n_frames)
    p8a_score = 0.0
    e2b_score = 0.0
    pa_score = 0.0
    p8a_weight = 0.0
    e2b_weight = 0.0
    pa_weight = 0.0
    for axis_dict in rec["by_axis"].values():
        for v in axis_dict.values():
            n = v["n_frames"]
            for ckpt in CKPTS:
                # contribute weight by n; "score" is inverted total-failure
                ct_row = v["frac_real_FP_fake_missed_per_ckpt"]
            # use winner-takes-weight
            winner = v["best_ckpt"]
            if winner == "P8A":
                p8a_score += n
                p8a_weight += n
            elif winner == "E2B":
                e2b_score += n
                e2b_weight += n
            elif winner == "PA_3800":
                pa_score += n
                pa_weight += n
    rec["overall"] = {
        "winner_weighted_by_frames": {"P8A": p8a_score, "E2B": e2b_score, "PA_3800": pa_score},
        "aggregate_fake_fpr": {
            ckpt: summary["ckpts"][ckpt]["per_tau"][f"tau_{TAU_PRIMARY}"]["fake_fpr_aggregate"]
            for ckpt in CKPTS
        },
        "aggregate_fake_recall": {
            ckpt: summary["ckpts"][ckpt]["per_tau"][f"tau_{TAU_PRIMARY}"]["fake_recall_aggregate"]
            for ckpt in CKPTS
        },
    }
    with (OUT_DIR / "ft_base_recommendation.json").open("w") as f:
        json.dump(rec, f, indent=2, default=str)
    print("wrote ft_base_recommendation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
