#!/usr/bin/env python3
"""
Cross-Checkpoint Failure Analysis for Round 2.5 Models

Compares the top-5 R2.5 checkpoints side-by-side on the SAME evaluation data
to identify:
  1. Consensus failures: videos ALL models get wrong → inherent data issues
  2. Discriminating failures: videos SOME models get wrong → model-specific weaknesses
  3. Quality-correlated errors: do failures cluster by image quality metrics?

Usage:
  # First, run validate_custom_sources.py for each checkpoint to produce per-checkpoint
  # frames_report CSVs. Use the --output_filename_prefix flag to differentiate:
  #
  #   python validate_custom_sources.py \
  #     --checkpoint_gcs_path gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth \
  #     --output_filename_prefix R25_F1_ \
  #     --deeplive_bucket live-deepfake-methods-real-and-fake-frames-cropped \
  #     --external_real_bucket effort-collected-data \
  #     --visomaster_bucket live-deepfake-methods-real-and-fake-frames-cropped-visomaster \
  #     --df40_orientation target_source \
  #     ...
  #
  #   (repeat for R25_F2, R25_F5, R25_F3, R25_F6)
  #
  # Then run this script:
  #   python failure_analysis.py --data_dir ./analysis_results

  # Alternatively, point to a single directory containing all prefixed CSVs.

Requirements:
  pip install pandas numpy scikit-learn
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Top-5 Round 2.5 checkpoints
CHECKPOINTS = {
    "R25_F1": {
        "run_id": "5w453our",
        "gcs_path": "gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
        "rank": 736, "k": 32, "auc": 0.9893, "eer": 0.0356,
        "config_note": "k=32, lr=2e-4, baseline (WINNER)",
    },
    "R25_F2": {
        "run_id": "ug8n870r",
        "gcs_path": "gs://training-job-outputs/phase2r2_experiments/ug8n870r/top_n_effort_20260212_step13500_auc0.9887_eer0.0249.pth",
        "rank": 720, "k": 48, "auc": 0.9887, "eer": 0.0249,
        "config_note": "k=48, lr=2e-4, highest capacity (best EER)",
    },
    "R25_F5": {
        "run_id": "tkh09be0",
        "gcs_path": "gs://training-job-outputs/phase2r2_experiments/tkh09be0/top_n_effort_20260212_step18500_auc0.9885_eer0.0320.pth",
        "rank": 744, "k": 24, "auc": 0.9885, "eer": 0.0320,
        "config_note": "k=24, warmup=3K (3x baseline)",
    },
    "R25_F3": {
        "run_id": "6md2py50",
        "gcs_path": "gs://training-job-outputs/phase2r2_experiments/6md2py50/top_n_effort_20260212_step18500_auc0.9884_eer0.0427.pth",
        "rank": 744, "k": 24, "auc": 0.9884, "eer": 0.0427,
        "config_note": "k=24, lr=1e-4 (half baseline LR)",
    },
    "R25_F6": {
        "run_id": "x8lktnbc",
        "gcs_path": "gs://training-job-outputs/phase2r2_experiments/x8lktnbc/top_n_effort_20260212_step22000_auc0.9882_eer0.0391.pth",
        "rank": 736, "k": 32, "auc": 0.9882, "eer": 0.0391,
        "config_note": "k=32, lr=1e-4 (half baseline LR)",
    },
}

# Data source categories for structured analysis
DATA_CATEGORIES = {
    "deeplive_fakes": {"label": 1, "method_prefix": "deeplive_", "exclude_suffix": None},
    "deeplive_reals": {"label": 0, "method_prefix": "deeplive_", "exclude_suffix": None},
    "df40_target_source_fakes": {"label": 1, "method_prefix": None, "file_suffix": "target_source"},
    "external_reals": {"label": 0, "method_name": "external_youtube_avspeech"},
    "visomaster_fakes": {"label": 1, "method_prefix": "visomaster_", "exclude_method": "visomaster_real"},
    "visomaster_reals": {"label": 0, "method_name": "visomaster_real"},
}

# Threshold defaults (from comprehensive_analysis.py best results)
DEFAULT_PROB_THRESHOLD = 0.5
DEFAULT_VOTE_THRESHOLD = 0.5  # 4/8 frames


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def find_checkpoint_csvs(data_dir: Path, model_name: str) -> Dict[str, Path]:
    """
    Find all frames_report CSVs for a checkpoint.

    Looks for files matching: {model_name}_*frames_report.csv
    Also checks for known report types without prefix (B16/L14 naming convention).
    """
    found = {}
    prefix = f"{model_name}_"

    # Auto-discover: find all files matching {prefix}*frames_report.csv
    for candidate in sorted(data_dir.glob(f"{prefix}*frames_report.csv")):
        # Extract report type from filename: S1_deeplive_frames_report.csv → deeplive_frames_report
        report_type = candidate.name[len(prefix):].replace(".csv", "")
        found[report_type] = candidate

    # Fallback: also check known report types without prefix (for B16/L14 naming)
    if not found:
        known_types = [
            "frames_report",
            "deeplive_frames_report",
            "target_source_frames_report",
            "visomaster_frames_report",
            "extreal_frames_report",
            "qualenhance_frames_report",
        ]
        for report_type in known_types:
            alt = data_dir / f"{report_type}.csv"
            if alt.exists():
                found[report_type] = alt

    return found


def load_frames_for_checkpoint(
    data_dir: Path,
    model_name: str,
) -> pd.DataFrame:
    """
    Load all frame-level data for a single checkpoint into one DataFrame.
    Adds 'source_file' column to track origin.
    """
    csvs = find_checkpoint_csvs(data_dir, model_name)
    if not csvs:
        print(f"  WARNING: No CSVs found for {model_name} in {data_dir}")
        return pd.DataFrame()

    frames = []
    for report_type, path in csvs.items():
        df = pd.read_csv(path)
        df["source_file"] = report_type
        frames.append(df)
        print(f"  Loaded {len(df):,} frames from {path.name}")

    combined = pd.concat(frames, ignore_index=True)

    # De-duplicate: same frame_path across different report files
    combined = combined.drop_duplicates(subset=["frame_path"], keep="first")
    return combined


def load_all_checkpoints(data_dir: Path, model_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Load frame-level data for all checkpoints."""
    all_data = {}
    for name in model_names:
        print(f"\nLoading {name}:")
        df = load_frames_for_checkpoint(data_dir, name)
        if not df.empty:
            all_data[name] = df
            print(f"  Total: {len(df):,} frames, {df['video_id'].nunique():,} videos")
        else:
            print(f"  SKIPPED (no data)")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Video-Level Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_to_video_level(
    frames_df: pd.DataFrame,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    vote_threshold: float = DEFAULT_VOTE_THRESHOLD,
) -> pd.DataFrame:
    """
    Aggregate frame probabilities to video-level predictions.

    Returns DataFrame with columns:
      video_id, method, label, mean_prob, max_prob, min_prob, std_prob,
      n_frames, n_above_thresh, vote_fraction, video_pred, correct
    """
    groups = frames_df.groupby("video_id")

    records = []
    for video_id, group in groups:
        probs = group["frame_prob"].values
        label = group["label"].iloc[0]
        method = group["method"].iloc[0]

        n_frames = len(probs)
        n_above = (probs >= prob_threshold).sum()
        vote_fraction = n_above / n_frames if n_frames > 0 else 0.0
        video_pred = 1 if vote_fraction >= vote_threshold else 0

        records.append({
            "video_id": video_id,
            "method": method,
            "label": label,
            "mean_prob": float(np.mean(probs)),
            "max_prob": float(np.max(probs)),
            "min_prob": float(np.min(probs)),
            "std_prob": float(np.std(probs)),
            "n_frames": n_frames,
            "n_above_thresh": int(n_above),
            "vote_fraction": vote_fraction,
            "video_pred": video_pred,
            "correct": int(video_pred == label),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Checkpoint Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def build_cross_checkpoint_table(
    all_video_preds: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Join video-level predictions from all checkpoints into a single wide table.

    Columns: video_id, method, label, {model}_pred, {model}_prob, {model}_correct, ...
             n_models_correct, n_models_total, consensus_type
    """
    model_names = sorted(all_video_preds.keys())
    if not model_names:
        return pd.DataFrame()

    # Build union of all video_id → method, label mappings
    # Use the first model that has each video to fill method/label
    meta_frames = []
    for name in model_names:
        meta_frames.append(
            all_video_preds[name][["video_id", "method", "label"]].copy()
        )
    meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(
        subset=["video_id"], keep="first"
    )
    base = meta.copy()

    for name in model_names:
        vdf = all_video_preds[name][["video_id", "video_pred", "mean_prob", "correct"]].copy()
        vdf = vdf.rename(columns={
            "video_pred": f"{name}_pred",
            "mean_prob": f"{name}_prob",
            "correct": f"{name}_correct",
        })
        base = base.merge(vdf, on="video_id", how="left")

    # Count how many models got each video correct
    correct_cols = [f"{name}_correct" for name in model_names]
    base["n_models_correct"] = base[correct_cols].sum(axis=1)
    base["n_models_total"] = base[correct_cols].notna().sum(axis=1)

    # Classify consensus
    def classify_consensus(row):
        n_correct = row["n_models_correct"]
        n_total = row["n_models_total"]
        if n_correct == n_total:
            return "all_correct"
        elif n_correct == 0:
            return "consensus_failure"  # ALL models wrong
        else:
            return "discriminating"  # Some right, some wrong
    base["consensus_type"] = base.apply(classify_consensus, axis=1)

    return base


# ═══════════════════════════════════════════════════════════════════════════════
# Failure Analysis Reports
# ═══════════════════════════════════════════════════════════════════════════════

def report_consensus_summary(cross_df: pd.DataFrame, model_names: List[str]):
    """Print high-level consensus summary."""
    total = len(cross_df)
    all_correct = (cross_df["consensus_type"] == "all_correct").sum()
    consensus_fail = (cross_df["consensus_type"] == "consensus_failure").sum()
    discriminating = (cross_df["consensus_type"] == "discriminating").sum()

    print("\n" + "=" * 80)
    print("CONSENSUS SUMMARY")
    print("=" * 80)
    print(f"Total videos evaluated:     {total:,}")
    print(f"All models correct:         {all_correct:,} ({all_correct/total*100:.1f}%)")
    print(f"Consensus failures (ALL wrong): {consensus_fail:,} ({consensus_fail/total*100:.1f}%)")
    print(f"Discriminating (mixed):     {discriminating:,} ({discriminating/total*100:.1f}%)")

    # Per-model accuracy
    print(f"\nPer-model accuracy:")
    for name in model_names:
        col = f"{name}_correct"
        if col in cross_df.columns:
            acc = cross_df[col].mean() * 100
            cfg = CHECKPOINTS.get(name, {}).get("config_note", "")
            print(f"  {name}: {acc:.2f}%  ({cfg})")


def report_consensus_failures(cross_df: pd.DataFrame, model_names: List[str], top_n: int = 30):
    """
    Detailed report on consensus failures — videos ALL models get wrong.
    These are inherently hard or mislabeled.
    """
    failures = cross_df[cross_df["consensus_type"] == "consensus_failure"].copy()
    if failures.empty:
        print("\n✅ No consensus failures found!")
        return failures

    print("\n" + "=" * 80)
    print(f"CONSENSUS FAILURES ({len(failures)} videos — ALL {len(model_names)} models wrong)")
    print("=" * 80)

    # Split by label
    false_positives = failures[failures["label"] == 0]  # Real → predicted fake
    false_negatives = failures[failures["label"] == 1]  # Fake → predicted real

    print(f"\n  False Positives (real → predicted fake): {len(false_positives)}")
    print(f"  False Negatives (fake → predicted real): {len(false_negatives)}")

    # Break down by method
    print(f"\n  By method:")
    for method, group in failures.groupby("method"):
        fp = (group["label"] == 0).sum()
        fn = (group["label"] == 1).sum()
        label = "FP" if fp > fn else "FN"
        print(f"    {method}: {len(group)} videos ({fp} FP, {fn} FN)")

    # Show worst false positives (most confident wrong predictions)
    if not false_positives.empty:
        prob_cols = [f"{name}_prob" for name in model_names if f"{name}_prob" in false_positives.columns]
        false_positives = false_positives.copy()
        false_positives["avg_confidence"] = false_positives[prob_cols].mean(axis=1)
        worst_fp = false_positives.nlargest(min(top_n, len(false_positives)), "avg_confidence")

        print(f"\n  Top-{min(top_n, len(worst_fp))} worst False Positives (real videos ALL models flag as fake):")
        print(f"  {'video_id':<50} {'method':<30} {'avg_confidence':<15}")
        print(f"  {'-'*95}")
        for _, row in worst_fp.iterrows():
            per_model = " | ".join(f"{name}:{row.get(f'{name}_prob', 0):.3f}" for name in model_names)
            print(f"  {row['video_id']:<50} {row['method']:<30} {row['avg_confidence']:.4f}")
            print(f"    Per-model: {per_model}")

    # Show worst false negatives
    if not false_negatives.empty:
        prob_cols = [f"{name}_prob" for name in model_names if f"{name}_prob" in false_negatives.columns]
        false_negatives = false_negatives.copy()
        false_negatives["avg_confidence"] = false_negatives[prob_cols].mean(axis=1)
        worst_fn = false_negatives.nsmallest(min(top_n, len(false_negatives)), "avg_confidence")

        print(f"\n  Top-{min(top_n, len(worst_fn))} worst False Negatives (fake videos ALL models miss):")
        print(f"  {'video_id':<50} {'method':<30} {'avg_confidence':<15}")
        print(f"  {'-'*95}")
        for _, row in worst_fn.iterrows():
            per_model = " | ".join(f"{name}:{row.get(f'{name}_prob', 0):.3f}" for name in model_names)
            print(f"  {row['video_id']:<50} {row['method']:<30} {row['avg_confidence']:.4f}")
            print(f"    Per-model: {per_model}")

    return failures


def report_discriminating_failures(cross_df: pd.DataFrame, model_names: List[str]):
    """
    Report on discriminating failures — videos where SOME models are right and others wrong.
    These reveal which models handle which challenges better.
    """
    disc = cross_df[cross_df["consensus_type"] == "discriminating"].copy()
    if disc.empty:
        print("\n  No discriminating failures found.")
        return disc

    print("\n" + "=" * 80)
    print(f"DISCRIMINATING FAILURES ({len(disc)} videos — models disagree)")
    print("=" * 80)

    # Per-model breakdown: how often does each model get discriminating cases right?
    print(f"\n  Per-model accuracy on discriminating videos:")
    model_disc_acc = {}
    for name in model_names:
        col = f"{name}_correct"
        if col in disc.columns:
            acc = disc[col].mean() * 100
            model_disc_acc[name] = acc
            cfg = CHECKPOINTS.get(name, {}).get("config_note", "")
            print(f"    {name}: {acc:.1f}% correct  ({cfg})")

    # Find videos where exactly one model gets it right (strong discriminators)
    for n_correct in [1, 2, 3, 4]:
        subset = disc[disc["n_models_correct"] == n_correct]
        if subset.empty:
            continue
        n_total = len(model_names)
        print(f"\n  Videos where exactly {n_correct}/{n_total} models correct: {len(subset)}")

        # Which models succeed on these?
        for name in model_names:
            col = f"{name}_correct"
            if col in subset.columns:
                n_right = subset[col].sum()
                pct = n_right / len(subset) * 100 if len(subset) > 0 else 0
                if n_right > 0:
                    print(f"    {name}: correct on {n_right}/{len(subset)} ({pct:.0f}%)")

    # Method-level breakdown of discriminating failures
    print(f"\n  Discriminating failures by method:")
    for method, group in disc.groupby("method"):
        n = len(group)
        per_model = {name: group[f"{name}_correct"].sum() for name in model_names if f"{name}_correct" in group.columns}
        model_str = ", ".join(f"{name}:{correct}/{n}" for name, correct in sorted(per_model.items()))
        print(f"    {method}: {n} videos — {model_str}")

    return disc


def report_per_method_accuracy(cross_df: pd.DataFrame, model_names: List[str]):
    """
    Per-method accuracy table for each model.
    Shows which methods are hardest for which models.
    """
    print("\n" + "=" * 80)
    print("PER-METHOD ACCURACY COMPARISON")
    print("=" * 80)

    methods = sorted(cross_df["method"].dropna().unique())
    header = f"{'Method':<40} {'Label':<6} {'N':<6} " + " ".join(f"{name:<10}" for name in model_names)
    print(f"\n{header}")
    print("-" * len(header))

    for method in methods:
        group = cross_df[cross_df["method"] == method]
        label = int(group["label"].mode().iloc[0]) if not group.empty else -1
        n = len(group)
        accs = []
        for name in model_names:
            col = f"{name}_correct"
            if col in group.columns:
                acc = group[col].mean() * 100
                accs.append(f"{acc:>8.1f}%")
            else:
                accs.append(f"{'N/A':>9}")
        label_str = "real" if label == 0 else "fake"
        print(f"{method:<40} {label_str:<6} {n:<6} " + " ".join(accs))


def report_probability_distribution(cross_df: pd.DataFrame, model_names: List[str]):
    """
    Analyze the distribution of predicted probabilities across models for failure cases.
    Helps identify if failures are borderline or confident.
    """
    failures = cross_df[cross_df["consensus_type"] != "all_correct"]
    if failures.empty:
        return

    print("\n" + "=" * 80)
    print("PROBABILITY DISTRIBUTION ANALYSIS (failure cases only)")
    print("=" * 80)

    for name in model_names:
        prob_col = f"{name}_prob"
        if prob_col not in failures.columns:
            continue

        fp = failures[(failures["label"] == 0)]  # Reals misclassified
        fn = failures[(failures["label"] == 1)]  # Fakes missed

        print(f"\n  {name}:")
        if not fp.empty:
            fp_probs = fp[prob_col].dropna()
            print(f"    False Positives (real→fake): n={len(fp_probs)}, "
                  f"mean_prob={fp_probs.mean():.4f}, median={fp_probs.median():.4f}, "
                  f"std={fp_probs.std():.4f}")
            # Bin into confidence buckets
            bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
            labels = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">0.9"]
            if len(fp_probs) > 0:
                binned = pd.cut(fp_probs, bins=bins, labels=labels)
                dist = binned.value_counts().sort_index()
                print(f"    FP confidence distribution: {dict(dist)}")

        if not fn.empty:
            fn_probs = fn[prob_col].dropna()
            print(f"    False Negatives (fake→real): n={len(fn_probs)}, "
                  f"mean_prob={fn_probs.mean():.4f}, median={fn_probs.median():.4f}, "
                  f"std={fn_probs.std():.4f}")
            bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
            labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", ">0.7"]
            if len(fn_probs) > 0:
                binned = pd.cut(fn_probs, bins=bins, labels=labels)
                dist = binned.value_counts().sort_index()
                print(f"    FN confidence distribution: {dict(dist)}")


def report_model_ranking(cross_df: pd.DataFrame, model_names: List[str]):
    """
    Rank models on different evaluation dimensions to find the 'truly best' model.
    """
    print("\n" + "=" * 80)
    print("MODEL RANKING ACROSS DIMENSIONS")
    print("=" * 80)

    rankings = {}
    for name in model_names:
        correct_col = f"{name}_correct"
        prob_col = f"{name}_prob"
        if correct_col not in cross_df.columns:
            continue

        r = {"model": name}
        r["overall_acc"] = cross_df[correct_col].mean() * 100

        # Per-category accuracy
        reals = cross_df[cross_df["label"] == 0]
        fakes = cross_df[cross_df["label"] == 1]
        if not reals.empty:
            r["real_acc"] = reals[correct_col].mean() * 100
        if not fakes.empty:
            r["fake_acc"] = fakes[correct_col].mean() * 100

        # Accuracy on discriminating cases (model differentiators)
        disc = cross_df[cross_df["consensus_type"] == "discriminating"]
        if not disc.empty:
            r["disc_acc"] = disc[correct_col].mean() * 100

        # DeepLive-specific accuracy
        deeplive = cross_df[cross_df["method"].str.startswith("deeplive_")]
        if not deeplive.empty:
            r["deeplive_acc"] = deeplive[correct_col].mean() * 100

        # External real accuracy (proxy for quality robustness)
        ext_real = cross_df[cross_df["method"] == "external_youtube_avspeech"]
        if not ext_real.empty:
            r["ext_real_acc"] = ext_real[correct_col].mean() * 100

        # Mean confidence gap (how decisively does the model predict?)
        if prob_col in cross_df.columns:
            correct_mask = cross_df[correct_col] == 1
            if correct_mask.any():
                r["mean_correct_confidence"] = cross_df.loc[correct_mask, prob_col].apply(
                    lambda p: abs(p - 0.5)
                ).mean()

        r["config"] = CHECKPOINTS.get(name, {}).get("config_note", "")
        r["auc"] = CHECKPOINTS.get(name, {}).get("auc", 0)
        r["eer"] = CHECKPOINTS.get(name, {}).get("eer", 0)
        rankings[name] = r

    if not rankings:
        return

    rank_df = pd.DataFrame(rankings.values())

    # Print ranking table
    dims = ["overall_acc", "real_acc", "fake_acc", "disc_acc", "deeplive_acc", "ext_real_acc"]
    dim_labels = {
        "overall_acc": "Overall Acc",
        "real_acc": "Real Acc (TNR)",
        "fake_acc": "Fake Acc (TPR)",
        "disc_acc": "Disc. Cases Acc",
        "deeplive_acc": "DeepLive Acc",
        "ext_real_acc": "Ext. Real Acc",
    }

    for dim in dims:
        if dim not in rank_df.columns:
            continue
        sorted_df = rank_df.sort_values(dim, ascending=False)
        print(f"\n  Ranked by {dim_labels.get(dim, dim)}:")
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
            print(f"    {medal} {row['model']}: {row[dim]:.2f}%  (AUC={row['auc']:.4f}, {row['config']})")


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_results(
    cross_df: pd.DataFrame,
    consensus_failures: pd.DataFrame,
    discriminating: pd.DataFrame,
    output_dir: Path,
):
    """Export analysis results to CSVs for further investigation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full cross-checkpoint table
    cross_path = output_dir / "cross_checkpoint_predictions.csv"
    cross_df.to_csv(cross_path, index=False)
    print(f"\n  Exported: {cross_path}")

    # Consensus failures
    if not consensus_failures.empty:
        cf_path = output_dir / "consensus_failures.csv"
        consensus_failures.to_csv(cf_path, index=False)
        print(f"  Exported: {cf_path}")

    # Discriminating failures
    if not discriminating.empty:
        disc_path = output_dir / "discriminating_failures.csv"
        discriminating.to_csv(disc_path, index=False)
        print(f"  Exported: {disc_path}")

    # Per-method summary pivot
    model_names = [col.replace("_correct", "") for col in cross_df.columns if col.endswith("_correct")]
    rows = []
    for method in cross_df["method"].unique():
        group = cross_df[cross_df["method"] == method]
        row = {"method": method, "label": group["label"].mode().iloc[0], "n_videos": len(group)}
        for name in model_names:
            col = f"{name}_correct"
            if col in group.columns:
                row[f"{name}_acc"] = group[col].mean() * 100
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "per_method_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Exported: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Shell Command Generator
# ═══════════════════════════════════════════════════════════════════════════════

def print_evaluation_commands():
    """Print the validate_custom_sources.py commands needed to generate input data."""
    print("\n" + "=" * 80)
    print("STEP 1: Generate per-checkpoint CSVs")
    print("Run these commands (in the Docker container or Vertex AI):")
    print("=" * 80)

    for name, info in CHECKPOINTS.items():
        print(f"""
# --- {name} ({info['config_note']}) ---
python validate_custom_sources.py \\
  --checkpoint_gcs_path {info['gcs_path']} \\
  --output_filename_prefix {name}_ \\
  --df40_orientation target_source \\
  --df40_mode paired \\
  --deeplive_bucket live-deepfake-methods-real-and-fake-frames-cropped \\
  --deeplive_split val \\
  --external_real_bucket effort-collected-data \\
  --visomaster_bucket live-deepfake-methods-real-and-fake-frames-cropped-visomaster \\
  --visomaster_held_out_models "GhostFace-v3,InStyleSwapper256-C" \\
  --frames_per_video 8 \\
  --disable_wandb \\
  --detailed_reports \\
  --output_gcs_folder gs://training-job-outputs/failure_analysis_r25/
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-checkpoint failure analysis for Round 2.5 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print validation commands to generate input CSVs:
  python failure_analysis.py --print_commands

  # Run analysis on existing CSVs (uses B16 naming convention):
  python failure_analysis.py --data_dir ./analysis_results --models B16

  # Run analysis with explicit model prefixes:
  python failure_analysis.py --data_dir ./results --models R25_F1,R25_F2,R25_F5,R25_F3,R25_F6

  # Custom thresholds:
  python failure_analysis.py --data_dir ./results --models R25_F1,R25_F2 --prob_threshold 0.6 --vote_threshold 0.625
        """,
    )
    parser.add_argument("--data_dir", type=str, default="./analysis_results",
                        help="Directory containing frames_report CSVs.")
    parser.add_argument("--models", type=str, default="R25_F1,R25_F2,R25_F5,R25_F3,R25_F6",
                        help="Comma-separated model names (CSV filename prefixes).")
    parser.add_argument("--prob_threshold", type=float, default=DEFAULT_PROB_THRESHOLD,
                        help=f"Frame probability threshold (default: {DEFAULT_PROB_THRESHOLD}).")
    parser.add_argument("--vote_threshold", type=float, default=DEFAULT_VOTE_THRESHOLD,
                        help=f"Vote threshold fraction (default: {DEFAULT_VOTE_THRESHOLD}).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output CSVs (default: data_dir/failure_analysis_output).")
    parser.add_argument("--print_commands", action="store_true",
                        help="Print validate_custom_sources.py commands and exit.")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of worst failures to show in detail.")

    args = parser.parse_args()

    if args.print_commands:
        print_evaluation_commands()
        return

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    model_names = [m.strip() for m in args.models.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "failure_analysis_output"

    print("=" * 80)
    print("CROSS-CHECKPOINT FAILURE ANALYSIS")
    print("=" * 80)
    print(f"Data dir:       {data_dir}")
    print(f"Models:         {model_names}")
    print(f"Prob threshold: {args.prob_threshold}")
    print(f"Vote threshold: {args.vote_threshold} ({args.vote_threshold * 8:.0f}/8 frames)")
    print(f"Output dir:     {output_dir}")

    # 1. Load data
    all_frames = load_all_checkpoints(data_dir, model_names)
    if len(all_frames) < 2:
        print(f"\nERROR: Need at least 2 models for comparison, found {len(all_frames)}.")
        print("Run with --print_commands to see how to generate the input CSVs.")
        sys.exit(1)

    # 2. Aggregate to video level
    print(f"\n{'=' * 80}")
    print("Aggregating to video-level predictions...")
    all_video_preds = {}
    for name, frames_df in all_frames.items():
        video_df = aggregate_to_video_level(frames_df, args.prob_threshold, args.vote_threshold)
        all_video_preds[name] = video_df
        n_correct = video_df["correct"].sum()
        n_total = len(video_df)
        print(f"  {name}: {n_correct}/{n_total} correct ({n_correct/n_total*100:.1f}%)")

    # 3. Build cross-checkpoint table
    cross_df = build_cross_checkpoint_table(all_video_preds)
    active_models = [n for n in model_names if n in all_frames]

    # 4. Reports
    report_consensus_summary(cross_df, active_models)
    consensus_failures = report_consensus_failures(cross_df, active_models, top_n=args.top_n)
    discriminating = report_discriminating_failures(cross_df, active_models)
    report_per_method_accuracy(cross_df, active_models)
    report_probability_distribution(cross_df, active_models)
    report_model_ranking(cross_df, active_models)

    # 5. Export
    export_results(cross_df, consensus_failures, discriminating, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey files in {output_dir}:")
    print("  - cross_checkpoint_predictions.csv  → full per-video, per-model predictions")
    print("  - consensus_failures.csv            → videos ALL models get wrong")
    print("  - discriminating_failures.csv        → videos models disagree on")
    print("  - per_method_summary.csv             → per-method accuracy for each model")


if __name__ == "__main__":
    main()
