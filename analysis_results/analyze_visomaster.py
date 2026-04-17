#!/usr/bin/env python3
"""
Analyze VisoMaster validation results for B16 model.

Computes:
- Per-model accuracy at various thresholds
- Per-tier accuracy at various thresholds
- Optimal threshold grid search
- Comparison with previous validation results
"""

import pandas as pd
import numpy as np
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent

PROB_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VOTE_THRESHOLDS = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]


def load_data():
    df = pd.read_csv(ANALYSIS_DIR / 'visomaster_frames_report.csv')
    print(f"Total frames: {len(df):,}")
    return df


def frame_stats(df):
    """Print frame-level probability distributions."""
    print("\n" + "=" * 80)
    print("FRAME-LEVEL PROBABILITY DISTRIBUTIONS")
    print("=" * 80)

    # Per-tier
    print("\n--- By Tier ---")
    for m in ['visomaster_tier_MINIMAL', 'visomaster_tier_MODERATE', 'visomaster_tier_STRONG']:
        sub = df[df['method'] == m]
        if len(sub) > 0:
            tier = m.replace('visomaster_tier_', '')
            print(f"  {tier:>10}: mean={sub['frame_prob'].mean():.4f}  median={sub['frame_prob'].median():.4f}  "
                  f"std={sub['frame_prob'].std():.4f}  n_frames={len(sub):,}  n_videos={sub['video_id'].nunique()}")

    # Real
    real = df[df['method'] == 'visomaster_real']
    print(f"  {'REAL':>10}: mean={real['frame_prob'].mean():.4f}  median={real['frame_prob'].median():.4f}  "
          f"std={real['frame_prob'].std():.4f}  n_frames={len(real):,}  n_videos={real['video_id'].nunique()}")

    # Per-model
    print("\n--- By Swap Model ---")
    model_methods = sorted([m for m in df['method'].unique()
                            if not m.startswith('visomaster_tier') and m != 'visomaster_real'])
    for m in model_methods:
        sub = df[df['method'] == m]
        model_name = m.replace('visomaster_', '')
        print(f"  {model_name:>22}: mean={sub['frame_prob'].mean():.4f}  median={sub['frame_prob'].median():.4f}  "
              f"std={sub['frame_prob'].std():.4f}  n={len(sub):,}  vids={sub['video_id'].nunique()}")


def compute_video_predictions(frames_df, prob_thresh, vote_thresh):
    """Compute video-level predictions."""
    groups = frames_df.groupby(['method', 'label', 'video_id'])
    results = []
    for (method, label, video_id), group in groups:
        n_frames = len(group)
        n_above = (group['frame_prob'] >= prob_thresh).sum()
        vote_frac = n_above / n_frames
        pred = 1 if vote_frac >= vote_thresh else 0
        results.append({
            'method': method, 'label': label, 'video_id': video_id,
            'n_frames': n_frames, 'avg_prob': group['frame_prob'].mean(),
            'prediction': pred, 'is_correct': int(pred == label),
        })
    return pd.DataFrame(results)


def grid_search(df):
    """Grid search over thresholds."""
    print("\n" + "=" * 80)
    print("GRID SEARCH: OPTIMAL THRESHOLDS")
    print("=" * 80)

    results = []
    for prob in PROB_THRESHOLDS:
        for vote in VOTE_THRESHOLDS:
            vdf = compute_video_predictions(df, prob, vote)

            # Real accuracy
            real_v = vdf[vdf['method'] == 'visomaster_real']
            real_acc = (real_v['prediction'] == 0).mean() if len(real_v) > 0 else 0

            # Per-tier accuracy (fake detection rate)
            tier_accs = {}
            for tier in ['MINIMAL', 'MODERATE', 'STRONG']:
                tier_v = vdf[vdf['method'] == f'visomaster_tier_{tier}']
                tier_accs[tier] = (tier_v['prediction'] == 1).mean() if len(tier_v) > 0 else 0

            # All fake (by model, not tier duplicates)
            model_methods = [m for m in vdf['method'].unique()
                             if not m.startswith('visomaster_tier') and m != 'visomaster_real']
            fake_v = vdf[vdf['method'].isin(model_methods)]
            fake_acc = (fake_v['prediction'] == 1).mean() if len(fake_v) > 0 else 0

            overall_balanced = (real_acc + fake_acc) / 2

            results.append({
                'prob': prob, 'vote': vote, 'vote_frac': f"{int(vote * 8)}/8",
                'real_acc': real_acc, 'fake_acc': fake_acc,
                'balanced_acc': overall_balanced,
                'tier_MINIMAL': tier_accs['MINIMAL'],
                'tier_MODERATE': tier_accs['MODERATE'],
                'tier_STRONG': tier_accs['STRONG'],
            })

    rdf = pd.DataFrame(results)

    # Best balanced
    best = rdf.loc[rdf['balanced_acc'].idxmax()]
    print(f"\n✅ Best BALANCED accuracy: {best['balanced_acc']:.2%}")
    print(f"   prob={best['prob']}, vote={best['vote_frac']}")
    print(f"   Real: {best['real_acc']:.2%}  |  Fake (all models): {best['fake_acc']:.2%}")
    print(f"   Tier MINIMAL: {best['tier_MINIMAL']:.2%}  MODERATE: {best['tier_MODERATE']:.2%}  STRONG: {best['tier_STRONG']:.2%}")

    # Best fake with real >= 90%
    high_real = rdf[rdf['real_acc'] >= 0.90]
    if len(high_real) > 0:
        best_90 = high_real.loc[high_real['fake_acc'].idxmax()]
        print(f"\n✅ Best FAKE acc with Real ≥ 90%: {best_90['fake_acc']:.2%}")
        print(f"   prob={best_90['prob']}, vote={best_90['vote_frac']}")
        print(f"   Real: {best_90['real_acc']:.2%}  |  Fake: {best_90['fake_acc']:.2%}")
        print(f"   Tier MINIMAL: {best_90['tier_MINIMAL']:.2%}  MODERATE: {best_90['tier_MODERATE']:.2%}  STRONG: {best_90['tier_STRONG']:.2%}")

    # Best fake with real >= 95%
    high_real_95 = rdf[rdf['real_acc'] >= 0.95]
    if len(high_real_95) > 0:
        best_95 = high_real_95.loc[high_real_95['fake_acc'].idxmax()]
        print(f"\n✅ Best FAKE acc with Real ≥ 95%: {best_95['fake_acc']:.2%}")
        print(f"   prob={best_95['prob']}, vote={best_95['vote_frac']}")
        print(f"   Real: {best_95['real_acc']:.2%}  |  Fake: {best_95['fake_acc']:.2%}")
        print(f"   Tier MINIMAL: {best_95['tier_MINIMAL']:.2%}  MODERATE: {best_95['tier_MODERATE']:.2%}  STRONG: {best_95['tier_STRONG']:.2%}")

    return rdf, best


def detailed_breakdown(df, prob_thresh, vote_thresh):
    """Detailed per-model and per-tier breakdown at a given threshold."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED BREAKDOWN @ prob={prob_thresh}, vote={int(vote_thresh * 8)}/8")
    print(f"{'=' * 80}")

    vdf = compute_video_predictions(df, prob_thresh, vote_thresh)

    # Per-model
    print(f"\n--- Per Swap Model (fake detection rate) ---")
    print(f"{'Model':<28} {'Videos':>7} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 58)
    model_methods = sorted([m for m in vdf['method'].unique()
                            if not m.startswith('visomaster_tier') and m != 'visomaster_real'])
    total_fake = 0
    total_correct = 0
    for m in model_methods:
        sub = vdf[vdf['method'] == m]
        correct = (sub['prediction'] == 1).sum()
        acc = correct / len(sub) if len(sub) > 0 else 0
        name = m.replace('visomaster_', '')
        print(f"  {name:<26} {len(sub):>7} {correct:>8} {acc:>10.2%}")
        total_fake += len(sub)
        total_correct += correct
    overall_fake_acc = total_correct / total_fake if total_fake > 0 else 0
    print(f"  {'ALL MODELS':<26} {total_fake:>7} {total_correct:>8} {overall_fake_acc:>10.2%}")

    # Real
    real_v = vdf[vdf['method'] == 'visomaster_real']
    real_correct = (real_v['prediction'] == 0).sum()
    real_acc = real_correct / len(real_v) if len(real_v) > 0 else 0
    print(f"\n  {'REAL (visomaster_real)':<26} {len(real_v):>7} {real_correct:>8} {real_acc:>10.2%}")

    # Per-tier
    print(f"\n--- Per Tier (fake detection rate) ---")
    print(f"{'Tier':<28} {'Videos':>7} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 58)
    for tier in ['STRONG', 'MODERATE', 'MINIMAL']:
        sub = vdf[vdf['method'] == f'visomaster_tier_{tier}']
        if len(sub) == 0:
            continue
        correct = (sub['prediction'] == 1).sum()
        acc = correct / len(sub) if len(sub) > 0 else 0
        print(f"  {tier:<26} {len(sub):>7} {correct:>8} {acc:>10.2%}")

    # UNKNOWN tier (models without tier data)
    unknown_models = ['visomaster_SimSwap512', 'visomaster_InStyleSwapper256-C']
    unknown_v = vdf[vdf['method'].isin(unknown_models)]
    if len(unknown_v) > 0:
        correct = (unknown_v['prediction'] == 1).sum()
        acc = correct / len(unknown_v) if len(unknown_v) > 0 else 0
        print(f"  {'UNKNOWN (no tier data)':<26} {len(unknown_v):>7} {correct:>8} {acc:>10.2%}")


def threshold_heatmap(df):
    """Print a compact heatmap of real/fake accuracy at all threshold combos."""
    print(f"\n{'=' * 80}")
    print("THRESHOLD HEATMAP: Real% / Fake% (by-model, de-duplicated)")
    print("=" * 80)
    print(f"\n{'prob':<6}", end="")
    for v in VOTE_THRESHOLDS:
        print(f"  {int(v*8)}/8       ", end="")
    print()
    print("-" * 90)

    for prob in PROB_THRESHOLDS:
        print(f"{prob:<6}", end="")
        for vote in VOTE_THRESHOLDS:
            vdf = compute_video_predictions(df, prob, vote)
            real_v = vdf[vdf['method'] == 'visomaster_real']
            real_acc = (real_v['prediction'] == 0).mean() if len(real_v) > 0 else 0
            model_methods = [m for m in vdf['method'].unique()
                             if not m.startswith('visomaster_tier') and m != 'visomaster_real']
            fake_v = vdf[vdf['method'].isin(model_methods)]
            fake_acc = (fake_v['prediction'] == 1).mean() if len(fake_v) > 0 else 0
            print(f" {real_acc:.0%}/{fake_acc:.0%}  ", end="")
        print()


def compare_with_df40(best_prob, best_vote):
    """Compare VisoMaster results with previous DF40 validation at same thresholds."""
    print(f"\n{'=' * 80}")
    print("COMPARISON: VisoMaster vs DF40 (B16, same thresholds)")
    print("=" * 80)

    # Load DF40 source_target data
    df40_path = ANALYSIS_DIR / 'B16_frames_report.csv'
    if not df40_path.exists():
        print("  (DF40 frames report not found, skipping comparison)")
        return

    df40 = pd.read_csv(df40_path)
    df40_fake = df40[df40['label'] == 1]
    df40_real = df40[df40['label'] == 0]

    # DF40 fake at same thresholds
    df40_fake_v = compute_video_predictions(df40_fake, best_prob, best_vote)
    df40_fake_acc = (df40_fake_v['prediction'] == 1).mean()
    df40_real_v = compute_video_predictions(df40_real, best_prob, best_vote)
    df40_real_acc = (df40_real_v['prediction'] == 0).mean()

    print(f"\n  @ prob={best_prob}, vote={int(best_vote*8)}/8:")
    print(f"  {'Source':<30} {'Real Acc':>10} {'Fake Acc':>10}")
    print(f"  {'-'*52}")
    print(f"  {'VisoMaster (9 models)':<30} {'see above':>10} {'see above':>10}")
    print(f"  {'DF40 source_target (8 methods)':<30} {df40_real_acc:>10.2%} {df40_fake_acc:>10.2%}")


def main():
    df = load_data()

    # Frame-level stats
    frame_stats(df)

    # Grid search
    grid_df, best_row = grid_search(df)

    # Detailed breakdown at best threshold
    best_prob = best_row['prob']
    best_vote = best_row['vote']
    detailed_breakdown(df, best_prob, best_vote)

    # Also show at default 0.5/0.5 for comparison
    detailed_breakdown(df, 0.5, 0.5)

    # Also show at the B16 in-dist optimal (prob=0.9, vote=2/8)
    detailed_breakdown(df, 0.9, 2/8)

    # Threshold heatmap
    threshold_heatmap(df)

    # Compare with DF40
    compare_with_df40(best_prob, best_vote)

    # Save grid search
    grid_df.to_csv(ANALYSIS_DIR / 'B16_visomaster_grid_search.csv', index=False)
    print(f"\nSaved grid search to B16_visomaster_grid_search.csv")


if __name__ == '__main__':
    main()
