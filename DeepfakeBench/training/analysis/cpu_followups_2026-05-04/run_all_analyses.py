"""Comprehensive non-ensemble CPU analysis of E1/E2b/E3 packets.

Outputs:
  outputs/01_per_method_recall.csv              — per-suite × per-method recall table at FPR=2/5/10%
  outputs/02_per_ckpt_per_suite_auc_eer.csv     — AUC, EER, FPR=2/5/10% recall per ckpt × suite
  outputs/03_score_distribution_stats.csv       — mean/std/percentiles per ckpt × suite × label
  outputs/04_per_frame_score_correlation.csv    — Pearson r matrix per suite (diagnostic only)
  outputs/05_method_champion.csv                — winning ckpt per (suite, method, FPR floor)
  outputs/06_per_video_disagreements.csv        — videos where ckpts disagree (top 100 most-divergent)

  figures/02_score_histogram_<ckpt>_<suite>.png
  figures/05_roc_curve_<suite>.png
  figures/05_calibration_<suite>.png

Then writes viewer entries to viewer_artifacts/runs_e_packets.yaml for inclusion in
viewer/model_dashboard_runs.yaml.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
RAW_DIR = ANALYSIS_DIR / "raw_reports"
OUT_DIR = ANALYSIS_DIR / "outputs"
FIG_DIR = ANALYSIS_DIR / "figures"
VIEWER_DIR = ANALYSIS_DIR / "viewer_artifacts"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
VIEWER_DIR.mkdir(exist_ok=True)

# Three checkpoints we care about (these are the operational best of each packet)
CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}

REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
]
FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
]
ALL_SUITES = REAL_SUITES + FAKE_SUITES


def load_frames(suite: str, ckpt_token: str) -> list[dict]:
    """Returns list of dict rows from frames_report.csv."""
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        return [
            {
                "method": row["method"],
                "label": int(row["label"]),
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "score": float(row["frame_prob"]),
                "group_key": row.get("group_key", ""),
                "family_key": row.get("family_key", ""),
            }
            for row in rdr
        ]


def calib_tau(real_scores: np.ndarray, target_fpr: float) -> float:
    sorted_s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(sorted_s) * (1 - target_fpr))) - 1)
    return float(sorted_s[idx]) if idx < len(sorted_s) else 1.01


def auc_eer(scores: np.ndarray, labels: np.ndarray):
    """Compute ROC-AUC and EER using the rank-based formula. Avoids sklearn dependency."""
    # AUC via Mann-Whitney
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # rank-based AUC = (sum of ranks of pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_rank_sum = float(ranks[:n_pos].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # EER: threshold where FPR == FNR
    sorted_neg = np.sort(neg)
    sorted_pos = np.sort(pos)
    # Sweep threshold over unique values, track FPR and FNR
    thresholds = np.linspace(0, 1, 1001)
    fprs = np.array([(neg >= t).mean() for t in thresholds])
    fnrs = np.array([(pos < t).mean() for t in thresholds])
    eer_idx = np.argmin(np.abs(fprs - fnrs))
    eer = float((fprs[eer_idx] + fnrs[eer_idx]) / 2)
    return auc, eer


def make_video_id_to_real_label(real_pool):
    """Some real-fake confusion happens via 'real_*' methods inside fake suites; identify via label."""
    return {(r["video_id"], r["method"]): r["label"] for r in real_pool}


# ============================================================
# Analysis 01: per-method recall table at FPR=2/5/10%
# ============================================================
def analysis_01_per_method_recall():
    print("=== 01: per-method recall table ===")
    target_fprs = [0.02, 0.05, 0.10]
    rows = []
    # Pool real frames across the 3 dev real suites for joint calibration per ckpt
    # Note: this matches contract calibration semantics (dev_primary_real_fpr based on teams_real_all_dev)
    # We use ONLY teams_real_all_dev for τ calibration to match contract.
    for ckpt_name, ckpt_token in CKPTS.items():
        # Calibrate τ per ckpt on teams_real_all_dev real frames
        real_frames = load_frames("teams_real_all_dev", ckpt_token)
        real_scores = np.array([f["score"] for f in real_frames if f["label"] == 0])
        if len(real_scores) == 0:
            print(f"  WARN: no real frames for {ckpt_name} on teams_real_all_dev")
            continue

        for target_fpr in target_fprs:
            tau = calib_tau(real_scores, target_fpr)
            achieved_fpr = float((real_scores >= tau).mean())

            # For each suite, group by method and compute recall
            for suite in FAKE_SUITES:
                frames = load_frames(suite, ckpt_token)
                if not frames:
                    continue
                # Group by method (label==1 only — fake methods)
                by_method = defaultdict(list)
                for f in frames:
                    if f["label"] == 1:
                        by_method[f["method"]].append(f["score"])
                for method, method_scores in by_method.items():
                    method_scores = np.array(method_scores)
                    recall = float((method_scores >= tau).mean())
                    rows.append({
                        "ckpt": ckpt_name,
                        "calibration_FPR_target": target_fpr,
                        "calibration_FPR_actual": achieved_fpr,
                        "tau": tau,
                        "suite": suite,
                        "method": method,
                        "n_frames": len(method_scores),
                        "recall": recall,
                    })

    out_path = OUT_DIR / "01_per_method_recall.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out_path} ({len(rows)} rows)")


# ============================================================
# Analysis 02: per-ckpt per-suite AUC/EER + score-distribution stats
# ============================================================
def analysis_02_auc_eer_and_distributions():
    print("=== 02: per-ckpt per-suite AUC/EER + distributions ===")
    rows_metrics = []
    rows_dist = []
    for ckpt_name, ckpt_token in CKPTS.items():
        for suite in ALL_SUITES:
            frames = load_frames(suite, ckpt_token)
            if not frames:
                continue
            scores = np.array([f["score"] for f in frames])
            labels = np.array([f["label"] for f in frames])

            if len(np.unique(labels)) >= 2:
                auc, eer = auc_eer(scores, labels)
            else:
                auc, eer = float("nan"), float("nan")

            rows_metrics.append({
                "ckpt": ckpt_name,
                "suite": suite,
                "n_frames": len(frames),
                "n_real": int((labels == 0).sum()),
                "n_fake": int((labels == 1).sum()),
                "auc": auc,
                "eer": eer,
                "fpr_at_05": float((scores[labels == 0] >= 0.5).mean()) if (labels == 0).sum() > 0 else float("nan"),
                "recall_at_05": float((scores[labels == 1] >= 0.5).mean()) if (labels == 1).sum() > 0 else float("nan"),
            })

            for label_val, label_name in [(0, "real"), (1, "fake")]:
                ls = scores[labels == label_val]
                if len(ls) == 0:
                    continue
                rows_dist.append({
                    "ckpt": ckpt_name,
                    "suite": suite,
                    "label": label_name,
                    "n": len(ls),
                    "mean": float(ls.mean()),
                    "std": float(ls.std()),
                    "min": float(ls.min()),
                    "p10": float(np.percentile(ls, 10)),
                    "p25": float(np.percentile(ls, 25)),
                    "p50": float(np.percentile(ls, 50)),
                    "p75": float(np.percentile(ls, 75)),
                    "p90": float(np.percentile(ls, 90)),
                    "max": float(ls.max()),
                })

    out_path = OUT_DIR / "02_per_ckpt_per_suite_auc_eer.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows_metrics[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows_metrics)
    print(f"  → {out_path} ({len(rows_metrics)} rows)")

    out_path2 = OUT_DIR / "03_score_distribution_stats.csv"
    with out_path2.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows_dist[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows_dist)
    print(f"  → {out_path2} ({len(rows_dist)} rows)")


# ============================================================
# Analysis 04: per-frame score correlation (DIAGNOSTIC ONLY)
# ============================================================
def analysis_04_score_correlation():
    """Pearson r between ckpts on per-frame scores. Tells us whether the 3 ckpts
    are using same vs different signal. NOT used for ensembling (per Roee's instruction)."""
    print("=== 04: per-frame score correlation (diagnostic) ===")
    rows = []
    ckpt_pairs = [("P8A", "E2B_3200"), ("P8A", "E3_6600"), ("E2B_3200", "E3_6600")]
    for suite in ALL_SUITES:
        # Load frames for each ckpt, key by frame_path
        frames_by_ckpt = {}
        for ckpt_name, ckpt_token in CKPTS.items():
            f_list = load_frames(suite, ckpt_token)
            frames_by_ckpt[ckpt_name] = {f["frame_path"]: (f["label"], f["score"]) for f in f_list}
        if not all(frames_by_ckpt.values()):
            continue
        common = sorted(set(frames_by_ckpt["P8A"].keys()) & set(frames_by_ckpt["E2B_3200"].keys()) & set(frames_by_ckpt["E3_6600"].keys()))
        if not common:
            continue
        for c1, c2 in ckpt_pairs:
            scores1 = np.array([frames_by_ckpt[c1][p][1] for p in common])
            scores2 = np.array([frames_by_ckpt[c2][p][1] for p in common])
            labels = np.array([frames_by_ckpt[c1][p][0] for p in common])
            # Pearson r overall
            r_all = float(np.corrcoef(scores1, scores2)[0, 1])
            # Separate for real and fake
            r_real = float(np.corrcoef(scores1[labels == 0], scores2[labels == 0])[0, 1]) if (labels == 0).sum() > 1 else float("nan")
            r_fake = float(np.corrcoef(scores1[labels == 1], scores2[labels == 1])[0, 1]) if (labels == 1).sum() > 1 else float("nan")
            rows.append({
                "suite": suite,
                "ckpt_a": c1,
                "ckpt_b": c2,
                "n_frames": len(common),
                "r_overall": r_all,
                "r_real_only": r_real,
                "r_fake_only": r_fake,
            })

    out_path = OUT_DIR / "04_per_frame_score_correlation.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out_path} ({len(rows)} rows)")


# ============================================================
# Analysis 05: method-champion table
# ============================================================
def analysis_05_method_champion():
    """For each (suite, method, FPR-floor), which ckpt has highest recall?"""
    print("=== 05: method-by-method champion table ===")
    # Re-use 01 output
    perm = OUT_DIR / "01_per_method_recall.csv"
    if not perm.exists():
        print("  SKIP — needs 01 to run first")
        return
    by_key = defaultdict(list)
    with perm.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            key = (row["suite"], row["method"], row["calibration_FPR_target"])
            by_key[key].append((row["ckpt"], float(row["recall"]), int(row["n_frames"])))
    rows = []
    for (suite, method, fpr), entries in by_key.items():
        entries.sort(key=lambda x: -x[1])
        winner = entries[0]
        runner_up = entries[1] if len(entries) > 1 else (None, float("nan"), 0)
        rows.append({
            "suite": suite,
            "method": method,
            "calibration_FPR_target": fpr,
            "n_frames": winner[2],
            "champion": winner[0],
            "champion_recall": winner[1],
            "runner_up": runner_up[0],
            "runner_up_recall": runner_up[1],
            "champion_lift_pct_pts": (winner[1] - runner_up[1]) * 100,
        })
    out_path = OUT_DIR / "05_method_champion.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out_path} ({len(rows)} rows)")


# ============================================================
# Analysis 06: per-video disagreements
# ============================================================
def analysis_06_per_video_disagreements():
    """Find videos where ckpts strongly disagree — useful for failure-case browsing."""
    print("=== 06: per-video disagreements ===")
    rows = []
    for suite in FAKE_SUITES + ["teams_real_all_dev", "teams_real_all_lockbox"]:
        # Aggregate by video_id, mean score per ckpt
        by_video = defaultdict(lambda: {"P8A": [], "E2B_3200": [], "E3_6600": [], "label": None, "method": None})
        for ckpt_name, ckpt_token in CKPTS.items():
            frames = load_frames(suite, ckpt_token)
            for f in frames:
                v = by_video[f["video_id"]]
                v[ckpt_name].append(f["score"])
                v["label"] = f["label"]
                v["method"] = f["method"]
        for vid, v in by_video.items():
            if not (v["P8A"] and v["E2B_3200"] and v["E3_6600"]):
                continue
            p = float(np.mean(v["P8A"]))
            e2 = float(np.mean(v["E2B_3200"]))
            e3 = float(np.mean(v["E3_6600"]))
            scores = [p, e2, e3]
            disagreement = float(max(scores) - min(scores))
            rows.append({
                "suite": suite,
                "video_id": vid,
                "method": v["method"],
                "label": v["label"],
                "P8A_score": p,
                "E2B_3200_score": e2,
                "E3_6600_score": e3,
                "max_minus_min": disagreement,
            })
    rows.sort(key=lambda r: -r["max_minus_min"])
    rows = rows[:200]  # top 200 most-divergent
    out_path = OUT_DIR / "06_per_video_disagreements.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out_path} ({len(rows)} rows, top-200 most-divergent)")


# ============================================================
# Analysis 07: per-identity FPR concentration
# ============================================================
def analysis_07_per_identity_fpr():
    """For each ckpt × real suite, distribution of per-identity FPR rates. Helps see
    whether FPRs concentrate on a few problem identities or are evenly spread."""
    print("=== 07: per-identity FPR concentration ===")
    rows = []
    for ckpt_name, ckpt_token in CKPTS.items():
        for suite in REAL_SUITES:
            frames = load_frames(suite, ckpt_token)
            if not frames:
                continue
            real_frames = [f for f in frames if f["label"] == 0]
            if not real_frames:
                continue
            # Calibrate τ on this ckpt's teams_real_all_dev to FPR=10%
            cal_frames = load_frames("teams_real_all_dev", ckpt_token)
            cal_scores = np.array([f["score"] for f in cal_frames if f["label"] == 0])
            tau = calib_tau(cal_scores, 0.10)

            # Per video_id (identity proxy), compute mean score and "fires" rate
            by_vid = defaultdict(list)
            for f in real_frames:
                by_vid[f["video_id"]].append(f["score"])
            per_vid_fires_rate = []
            for vid, scs in by_vid.items():
                fires = sum(1 for s in scs if s >= tau) / len(scs)
                per_vid_fires_rate.append(fires)
            arr = np.array(per_vid_fires_rate)
            # Concentration metrics
            n_vids = len(arr)
            n_offending = int((arr > 0).sum())
            top10_frac_of_offenders = float(np.sum(np.sort(arr)[-max(1, int(np.ceil(n_offending * 0.1))):])) / max(arr.sum(), 1e-9)
            rows.append({
                "ckpt": ckpt_name,
                "suite": suite,
                "tau_at_FPR10_dev": tau,
                "n_identities": n_vids,
                "n_offending_identities": n_offending,
                "fraction_offending": n_offending / n_vids if n_vids else 0,
                "max_per_identity_fpr": float(arr.max()) if n_vids else 0,
                "p90_per_identity_fpr": float(np.percentile(arr, 90)) if n_vids else 0,
                "p50_per_identity_fpr": float(np.percentile(arr, 50)) if n_vids else 0,
                "top10pct_offenders_share_of_total_fires": top10_frac_of_offenders,
            })
    out_path = OUT_DIR / "07_per_identity_fpr.csv"
    with out_path.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out_path} ({len(rows)} rows)")


# ============================================================
# RUN ALL
# ============================================================
def main():
    analysis_01_per_method_recall()
    analysis_02_auc_eer_and_distributions()
    analysis_04_score_correlation()
    analysis_05_method_champion()
    analysis_06_per_video_disagreements()
    analysis_07_per_identity_fpr()
    print("\nAll analyses written to outputs/")


if __name__ == "__main__":
    main()
