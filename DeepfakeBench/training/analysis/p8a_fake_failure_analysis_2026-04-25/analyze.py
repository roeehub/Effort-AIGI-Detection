#!/usr/bin/env python3
"""
P8A vs RLP6_04 fake-failure analysis on teams_fake_all_dev (2026-04-25).

Quantifies WHICH fakes P8A loses on relative to RLP6_04, and clusters them
by score margin, video-id naming patterns, and frame-level flip behavior.

Inputs (in same dir):
  teams_fake_all_dev_p8a_step5000_videos_report.csv
  teams_fake_all_dev_p8a_step5000_frames_report.csv
  teams_fake_all_dev_rlp6_04_step23500_videos_report.csv
  teams_fake_all_dev_rlp6_04_step23500_frames_report.csv

Outputs:
  summary.json         -- structured findings
  regressed_videos.csv -- one row per video where rlp6 was correct (fake) and p8a was wrong (real)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import pandas as pd

HERE = Path(__file__).resolve().parent
TAU = 0.5

P8A_VIDEOS = HERE / "teams_fake_all_dev_p8a_step5000_videos_report.csv"
P8A_FRAMES = HERE / "teams_fake_all_dev_p8a_step5000_frames_report.csv"
RLP_VIDEOS = HERE / "teams_fake_all_dev_rlp6_04_step23500_videos_report.csv"
RLP_FRAMES = HERE / "teams_fake_all_dev_rlp6_04_step23500_frames_report.csv"

REGRESSING_METHODS = {
    "visomaster_enhanced_macro",
    "deeplive_enhanced",
    "teams_flat_xiang_xiang2_feng",
}


def quantiles(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    def q(p):
        if n == 1:
            return s[0]
        # nearest-rank
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx]
    return {
        "n": n,
        "min": s[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "mean": float(mean(s)),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": s[-1],
    }


def load() -> dict:
    p8a_v = pd.read_csv(P8A_VIDEOS)
    rlp_v = pd.read_csv(RLP_VIDEOS)
    p8a_f = pd.read_csv(P8A_FRAMES)
    rlp_f = pd.read_csv(RLP_FRAMES)
    return {"p8a_v": p8a_v, "rlp_v": rlp_v, "p8a_f": p8a_f, "rlp_f": rlp_f}


def parse_subject_visomaster(video_id: str) -> str:
    """visomaster ids like 'visomaster_enhanced_raw__seq12349__fake' or
    'visomaster_enhanced_teams__seq9800__fake' -- meaningful sub-pool split."""
    m = re.match(r"^(visomaster_enhanced_(?:raw|teams))__seq", video_id)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z0-9_]+?)__seq", video_id)
    if m:
        return m.group(1)
    return video_id


def parse_subject_deeplive(video_id: str) -> str:
    """deeplive ids like 'deeplive_dor__seq10006__fake' -- single subject 'dor'."""
    m = re.match(r"^deeplive_([a-z]+)__seq", video_id)
    if m:
        return m.group(1)
    return video_id


def parse_subject_xiang(video_id: str) -> str:
    """teams_flat_xiang_xiang2_feng -- 'Xiang_Xiang2_Feng__seq46024__fake' single subject."""
    m = re.match(r"^([A-Za-z0-9_]+?)__seq", video_id)
    if m:
        return m.group(1)
    return video_id


SUBJECT_PARSERS = {
    "visomaster_enhanced_macro": parse_subject_visomaster,
    "deeplive_enhanced": parse_subject_deeplive,
    "teams_flat_xiang_xiang2_feng": parse_subject_xiang,
}


def parse_seq_id(video_id: str) -> int | None:
    """Extract the numeric seq id - useful for ordering/temporal proxy."""
    m = re.search(r"seq(\d+)", video_id)
    if m:
        return int(m.group(1))
    return None


def main():
    data = load()
    p8a_v = data["p8a_v"].copy()
    rlp_v = data["rlp_v"].copy()
    p8a_f = data["p8a_f"]
    rlp_f = data["rlp_f"]

    # Sanity: same video_ids on both sides?
    assert p8a_v["video_id"].nunique() == rlp_v["video_id"].nunique(), "video count mismatch"
    n_videos = p8a_v["video_id"].nunique()

    # Merge on (method, video_id)
    merged = p8a_v.merge(
        rlp_v,
        on=["method", "video_id", "label", "group_key", "family_key"],
        suffixes=("_p8a", "_rlp"),
    )
    assert len(merged) == len(p8a_v), f"merge dropped rows: {len(merged)} vs {len(p8a_v)}"

    # Per-row classification at tau=0.5
    merged["p8a_pred_fake"] = (merged["avg_video_prob_p8a"] >= TAU).astype(int)
    merged["rlp_pred_fake"] = (merged["avg_video_prob_rlp"] >= TAU).astype(int)
    merged["score_gap"] = merged["avg_video_prob_rlp"] - merged["avg_video_prob_p8a"]

    # Regression videos: rlp called fake correctly, p8a called real
    regressed = merged[(merged["rlp_pred_fake"] == 1) & (merged["p8a_pred_fake"] == 0)].copy()

    # By method
    summary = {
        "config": {
            "tau": TAU,
            "n_videos_total": int(n_videos),
            "regressing_methods_focus": sorted(REGRESSING_METHODS),
        },
        "aggregate": {},
        "by_method": {},
        "global_score_distributions": {},
        "frame_level": {},
        "video_id_clusters": {},
    }

    # Aggregate counts
    agg_total_regress = int(len(regressed))
    p8a_misses_total = int((merged["p8a_pred_fake"] == 0).sum())
    rlp_misses_total = int((merged["rlp_pred_fake"] == 0).sum())
    summary["aggregate"] = {
        "p8a_total_misses_fake_called_real": p8a_misses_total,
        "rlp_total_misses_fake_called_real": rlp_misses_total,
        "regressed_videos_p8a_lost_rlp_won": agg_total_regress,
        "shared_misses_both_failed": int(((merged["p8a_pred_fake"] == 0) & (merged["rlp_pred_fake"] == 0)).sum()),
        "videos_p8a_won_rlp_lost": int(((merged["p8a_pred_fake"] == 1) & (merged["rlp_pred_fake"] == 0)).sum()),
        "regression_pct_of_total": round(agg_total_regress / n_videos * 100.0, 2),
    }

    # Global score distributions on regressed videos
    summary["global_score_distributions"] = {
        "p8a_score_on_regressed": quantiles(regressed["avg_video_prob_p8a"].tolist()),
        "rlp_score_on_regressed": quantiles(regressed["avg_video_prob_rlp"].tolist()),
        "score_gap_rlp_minus_p8a_on_regressed": quantiles(regressed["score_gap"].tolist()),
    }

    # Score-distribution comparison on the *correctly-classified* portion per method.
    # If P8A's correct-fake scores are tightly clustered near 1.0 like RLP, the issue is
    # method-specific separability; if globally compressed, it's a calibration shift.
    correct_score_dist = {}
    for method, group in merged.groupby("method"):
        both_correct = group[(group["p8a_pred_fake"] == 1) & (group["rlp_pred_fake"] == 1)]
        correct_score_dist[method] = {
            "n_both_correct": int(len(both_correct)),
            "p8a_score_when_correct": quantiles(both_correct["avg_video_prob_p8a"].tolist()),
            "rlp_score_when_correct": quantiles(both_correct["avg_video_prob_rlp"].tolist()),
        }
    summary["score_distribution_when_both_correct"] = correct_score_dist

    # Per-method breakdown
    for method, group in merged.groupby("method"):
        group_regress = regressed[regressed["method"] == method]
        n = int(len(group))
        n_reg = int(len(group_regress))
        block = {
            "n_videos": n,
            "p8a_acc": float((group["p8a_pred_fake"] == 1).mean()),
            "rlp_acc": float((group["rlp_pred_fake"] == 1).mean()),
            "regressed_count": n_reg,
            "regressed_pct": round(n_reg / n * 100.0, 2) if n else 0.0,
            "p8a_score_on_regressed": quantiles(group_regress["avg_video_prob_p8a"].tolist()),
            "rlp_score_on_regressed": quantiles(group_regress["avg_video_prob_rlp"].tolist()),
            "score_gap_on_regressed": quantiles(group_regress["score_gap"].tolist()),
        }
        # Score band buckets on P8A score for regressed
        if n_reg > 0:
            scores = group_regress["avg_video_prob_p8a"].tolist()
            block["p8a_score_buckets_on_regressed"] = {
                "lt_0.10": int(sum(1 for s in scores if s < 0.10)),
                "0.10_to_0.20": int(sum(1 for s in scores if 0.10 <= s < 0.20)),
                "0.20_to_0.30": int(sum(1 for s in scores if 0.20 <= s < 0.30)),
                "0.30_to_0.40": int(sum(1 for s in scores if 0.30 <= s < 0.40)),
                "0.40_to_0.45": int(sum(1 for s in scores if 0.40 <= s < 0.45)),
                "0.45_to_0.50": int(sum(1 for s in scores if 0.45 <= s < 0.50)),
            }
            # tau-recoverable bucket: 0.40-0.50 = could flip with modest threshold change
            tau_recoverable = sum(1 for s in scores if 0.40 <= s < 0.50)
            block["pct_tau_recoverable_score_in_0.40_0.50"] = round(tau_recoverable / n_reg * 100.0, 2)
            confident_real = sum(1 for s in scores if s < 0.20)
            block["pct_confidently_real_score_lt_0.20"] = round(confident_real / n_reg * 100.0, 2)
        summary["by_method"][method] = block

    # Save the regressed table
    regressed_out = HERE / "regressed_videos.csv"
    regressed[
        [
            "method",
            "video_id",
            "avg_video_prob_p8a",
            "avg_video_prob_rlp",
            "score_gap",
            "group_key",
            "family_key",
        ]
    ].to_csv(regressed_out, index=False)

    # Video-id clustering for the 3 regressing methods
    for method in REGRESSING_METHODS:
        method_reg = regressed[regressed["method"] == method]
        if len(method_reg) == 0:
            continue
        parser = SUBJECT_PARSERS[method]
        subjects = [parser(v) for v in method_reg["video_id"]]
        all_subjects = [parser(v) for v in merged[merged["method"] == method]["video_id"]]
        sub_counter = Counter(subjects)
        all_sub_counter = Counter(all_subjects)
        # Compute regression rate per subject
        per_subject_rate = {}
        for sub, n_reg in sub_counter.most_common(20):
            n_total = all_sub_counter[sub]
            per_subject_rate[sub] = {
                "n_total": int(n_total),
                "n_regressed": int(n_reg),
                "regression_rate_pct": round(n_reg / n_total * 100.0, 2) if n_total else 0.0,
            }
        # Token frequency in regressed video_ids
        token_counter = Counter()
        for vid in method_reg["video_id"]:
            for tok in re.split(r"[_\-]", vid):
                if tok and not tok.isdigit() and not re.match(r"^\d+\.\d+$", tok) and tok != "fake":
                    token_counter[tok.lower()] += 1
        # Compare to baseline token freq across method
        all_token_counter = Counter()
        for vid in merged[merged["method"] == method]["video_id"]:
            for tok in re.split(r"[_\-]", vid):
                if tok and not tok.isdigit() and not re.match(r"^\d+\.\d+$", tok) and tok != "fake":
                    all_token_counter[tok.lower()] += 1
        # Enrichment: token frac in regressed / token frac across method
        n_reg_total = len(method_reg)
        n_method_total = (merged["method"] == method).sum()
        enrichment = []
        for tok, c_reg in token_counter.most_common(50):
            c_all = all_token_counter[tok]
            if c_all < 5:
                continue
            frac_reg = c_reg / n_reg_total
            frac_all = c_all / n_method_total
            enrichment.append({
                "token": tok,
                "n_in_regressed": int(c_reg),
                "n_in_method": int(c_all),
                "frac_regressed": round(frac_reg, 3),
                "frac_method": round(frac_all, 3),
                "enrichment_ratio": round(frac_reg / frac_all, 2) if frac_all > 0 else None,
            })
        enrichment.sort(key=lambda x: -(x["enrichment_ratio"] or 0))
        summary["video_id_clusters"][method] = {
            "top_subjects_by_regression_count": per_subject_rate,
            "top_enriched_tokens_in_regressed_video_ids": enrichment[:15],
            "sample_regressed_video_ids": method_reg["video_id"].head(10).tolist(),
        }

    # CAVEAT: capture frames-per-video stats so consumers know frame-level analysis
    # is essentially video-level (avg ~1.26 frames/video).
    frames_per_video = p8a_f.groupby(["method", "video_id"]).size()
    summary["frames_per_video_caveat"] = {
        "note": "Each video has very few frames (~1.26 mean). Frame-level analysis is effectively video-level.",
        "min": int(frames_per_video.min()),
        "max": int(frames_per_video.max()),
        "mean": round(float(frames_per_video.mean()), 3),
        "median": int(frames_per_video.median()),
    }

    # Subject sub-pool split for visomaster (raw vs teams) -- the only method where
    # video_id encodes a meaningful subdivision.
    viso_pool_block = {}
    viso_merged = merged[merged["method"] == "visomaster_enhanced_macro"].copy()
    viso_merged["pool"] = viso_merged["video_id"].map(parse_subject_visomaster)
    viso_regressed = regressed[regressed["method"] == "visomaster_enhanced_macro"].copy()
    viso_regressed["pool"] = viso_regressed["video_id"].map(parse_subject_visomaster)
    for pool, grp in viso_merged.groupby("pool"):
        n = len(grp)
        n_p8a_correct = int((grp["p8a_pred_fake"] == 1).sum())
        n_rlp_correct = int((grp["rlp_pred_fake"] == 1).sum())
        n_reg = int(((grp["rlp_pred_fake"] == 1) & (grp["p8a_pred_fake"] == 0)).sum())
        viso_pool_block[pool] = {
            "n_videos": n,
            "p8a_acc": round(n_p8a_correct / n, 3) if n else 0.0,
            "rlp_acc": round(n_rlp_correct / n, 3) if n else 0.0,
            "regressed_count": n_reg,
            "regressed_pct_of_pool": round(n_reg / n * 100.0, 2) if n else 0.0,
            "p8a_score_quantiles": quantiles(grp["avg_video_prob_p8a"].tolist()),
            "p8a_score_on_regressed": quantiles(
                grp[(grp["rlp_pred_fake"] == 1) & (grp["p8a_pred_fake"] == 0)]["avg_video_prob_p8a"].tolist()
            ),
        }
    summary["visomaster_pool_split_raw_vs_teams"] = viso_pool_block

    # Seq-id range distribution for regressed-vs-not, per regressing method
    seq_block = {}
    for method in REGRESSING_METHODS:
        m_all = merged[merged["method"] == method].copy()
        m_all["seq_id"] = m_all["video_id"].map(parse_seq_id)
        m_reg = m_all[(m_all["rlp_pred_fake"] == 1) & (m_all["p8a_pred_fake"] == 0)]
        m_held = m_all[~((m_all["rlp_pred_fake"] == 1) & (m_all["p8a_pred_fake"] == 0))]
        seq_block[method] = {
            "regressed_seq_id_range": quantiles(m_reg["seq_id"].dropna().tolist()),
            "non_regressed_seq_id_range": quantiles(m_held["seq_id"].dropna().tolist()),
        }
    summary["seq_id_distribution_regressed_vs_not"] = seq_block

    # Frame-level analysis on regressed videos
    # For each regressed video, compare frame_prob in p8a vs rlp.
    # % of frames that flipped (rlp >= 0.5 and p8a < 0.5).
    p8a_f["video_id"] = p8a_f["video_id"].astype(str)
    rlp_f["video_id"] = rlp_f["video_id"].astype(str)

    p8a_f_idx = p8a_f.set_index("video_id")
    rlp_f_idx = rlp_f.set_index("video_id")

    frame_stats_per_method = defaultdict(list)
    overall_flip_rates = []
    for _, row in regressed.iterrows():
        vid = row["video_id"]
        method = row["method"]
        try:
            p_frames = p8a_f_idx.loc[[vid]] if vid in p8a_f_idx.index else None
            r_frames = rlp_f_idx.loc[[vid]] if vid in rlp_f_idx.index else None
        except KeyError:
            continue
        if p_frames is None or r_frames is None:
            continue
        # Frames may not align 1:1 by frame_path - use frame_path join
        merged_frames = p_frames.merge(
            r_frames[["frame_path", "frame_prob"]],
            on="frame_path",
            suffixes=("_p8a", "_rlp"),
            how="inner",
        )
        if len(merged_frames) == 0:
            continue
        n_frames = len(merged_frames)
        n_p8a_real = int((merged_frames["frame_prob_p8a"] < TAU).sum())
        n_rlp_fake = int((merged_frames["frame_prob_rlp"] >= TAU).sum())
        n_flipped = int(((merged_frames["frame_prob_p8a"] < TAU) & (merged_frames["frame_prob_rlp"] >= TAU)).sum())
        flip_rate = n_flipped / n_frames if n_frames else 0.0
        p8a_real_rate = n_p8a_real / n_frames if n_frames else 0.0
        frame_stats_per_method[method].append({
            "video_id": vid,
            "n_frames": int(n_frames),
            "p8a_real_rate": round(p8a_real_rate, 3),
            "flip_rate_rlp_fake_to_p8a_real": round(flip_rate, 3),
            "p8a_frame_prob_mean": round(float(merged_frames["frame_prob_p8a"].mean()), 4),
            "p8a_frame_prob_std": round(float(merged_frames["frame_prob_p8a"].std() or 0), 4),
            "p8a_frame_prob_min": round(float(merged_frames["frame_prob_p8a"].min()), 4),
            "p8a_frame_prob_max": round(float(merged_frames["frame_prob_p8a"].max()), 4),
        })
        overall_flip_rates.append({
            "method": method,
            "video_id": vid,
            "flip_rate": flip_rate,
            "p8a_real_rate": p8a_real_rate,
        })

    # Aggregate frame stats
    frame_summary = {}
    for method, stats in frame_stats_per_method.items():
        if not stats:
            continue
        flip_rates = [s["flip_rate_rlp_fake_to_p8a_real"] for s in stats]
        p8a_real_rates = [s["p8a_real_rate"] for s in stats]
        prob_means = [s["p8a_frame_prob_mean"] for s in stats]
        prob_stds = [s["p8a_frame_prob_std"] for s in stats]
        # Buckets: full-flip (>= 0.95), partial-flip (0.30-0.95), borderline (< 0.30)
        full_flip = sum(1 for r in flip_rates if r >= 0.95)
        partial_flip = sum(1 for r in flip_rates if 0.30 <= r < 0.95)
        small_flip = sum(1 for r in flip_rates if r < 0.30)
        frame_summary[method] = {
            "n_regressed_videos_frame_analyzed": len(stats),
            "flip_rate_distribution": quantiles(flip_rates),
            "p8a_real_rate_distribution": quantiles(p8a_real_rates),
            "p8a_frame_prob_mean_distribution": quantiles(prob_means),
            "p8a_frame_prob_std_distribution_within_video": quantiles(prob_stds),
            "videos_full_flip_pct_frames_flipped_ge_95": int(full_flip),
            "videos_partial_flip_pct_frames_flipped_30_to_95": int(partial_flip),
            "videos_small_flip_pct_frames_flipped_lt_30": int(small_flip),
            "interpretation_full_flip_means": (
                "P8A predicts real on essentially every frame -> deep separability loss, not noise-driven"
            ),
            "interpretation_partial_flip_means": (
                "P8A flickers per-frame -> aggregation pulls below 0.5 but signal still partly there; "
                "score-margin or temporal smoothing might help"
            ),
        }

    summary["frame_level"] = frame_summary

    # Final write
    summary_path = HERE / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {summary_path}")
    print(f"Wrote {regressed_out}")

    # Console snapshot
    print("\n=== AGGREGATE ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print("\n=== PER-METHOD on the 3 regressing methods ===")
    for m in REGRESSING_METHODS:
        if m in summary["by_method"]:
            print(f"\n{m}")
            print(json.dumps(summary["by_method"][m], indent=2))
    print("\n=== FRAME-LEVEL ===")
    for m in REGRESSING_METHODS:
        if m in summary["frame_level"]:
            print(f"\n{m}")
            print(json.dumps(summary["frame_level"][m], indent=2))
    print("\n=== VIDEO-ID CLUSTERS ===")
    for m in REGRESSING_METHODS:
        if m in summary["video_id_clusters"]:
            print(f"\n{m}")
            block = summary["video_id_clusters"][m]
            print("Top subjects (regression count):")
            print(json.dumps(block["top_subjects_by_regression_count"], indent=2))
            print("Top enriched tokens:")
            print(json.dumps(block["top_enriched_tokens_in_regressed_video_ids"][:8], indent=2))
            print("Samples:", block["sample_regressed_video_ids"][:5])


if __name__ == "__main__":
    main()
