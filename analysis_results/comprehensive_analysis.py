#!/usr/bin/env python3
"""
Comprehensive Threshold Analysis for B16 and L14 Models

This script finds the best threshold strategy based on IN-DISTRIBUTION data:
  - target_source fakes (DF40 methods seen during training)
  - deeplive fakes (DeepLive val split)
  - external_youtube_avspeech reals

Then reports OOD performance (source_target fakes) using the best parameters.

Threshold Strategy:
  - prob_threshold: Frame probability threshold (0.1 to 0.9)
  - vote_threshold: Fraction of frames that must pass (1/8 to 7/8)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
PROB_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VOTE_THRESHOLDS = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
FRAMES_PER_VIDEO = 8

def load_frames_data(model: str) -> dict:
    """Load all frame-level data for a model."""
    base_path = Path(__file__).parent
    
    data = {}
    
    # OOD: source_target fakes + external reals (original validation)
    ood_df = pd.read_csv(base_path / f"{model}_frames_report.csv")
    data['ood_fakes'] = ood_df[ood_df['label'] == 1].copy()  # source_target fakes
    data['external_reals'] = ood_df[ood_df['label'] == 0].copy()  # external_youtube_avspeech
    
    # In-distribution: target_source fakes
    indist_df = pd.read_csv(base_path / f"{model}_target_source_frames_report.csv")
    data['indist_fakes'] = indist_df[indist_df['label'] == 1].copy()  # target_source fakes
    
    # DeepLive: both real and fake
    deeplive_df = pd.read_csv(base_path / f"{model}_deeplive_frames_report.csv")
    data['deeplive_fakes'] = deeplive_df[deeplive_df['label'] == 1].copy()
    data['deeplive_reals'] = deeplive_df[deeplive_df['label'] == 0].copy()
    
    return data


def apply_threshold_strategy(df: pd.DataFrame, prob_thresh: float, vote_thresh: float) -> pd.DataFrame:
    """
    Apply threshold strategy to frame-level data.
    
    Returns video-level predictions:
    - video_pred: 1 if fake, 0 if real
    """
    # Group by video
    video_groups = df.groupby('video_id')
    
    results = []
    for video_id, frames in video_groups:
        n_frames = len(frames)
        # Count frames above probability threshold
        n_above = (frames['frame_prob'] >= prob_thresh).sum()
        # Check if fraction exceeds vote threshold
        vote_fraction = n_above / n_frames
        video_pred = 1 if vote_fraction >= vote_thresh else 0
        
        results.append({
            'video_id': video_id,
            'video_pred': video_pred,
            'n_frames': n_frames,
            'n_above_thresh': n_above,
            'vote_fraction': vote_fraction,
            'method': frames['method'].iloc[0],
        })
    
    return pd.DataFrame(results)


def compute_accuracy(video_df: pd.DataFrame, expected_label: int) -> float:
    """Compute accuracy for videos with expected label."""
    if len(video_df) == 0:
        return 0.0
    correct = (video_df['video_pred'] == expected_label).sum()
    return correct / len(video_df) * 100


def grid_search_indist(data: dict, model: str) -> pd.DataFrame:
    """
    Grid search over thresholds using IN-DISTRIBUTION data only.
    
    In-distribution = target_source fakes + deeplive fakes + external reals
    """
    results = []
    
    for prob_thresh in PROB_THRESHOLDS:
        for vote_thresh in VOTE_THRESHOLDS:
            # Apply threshold to each dataset
            indist_fakes_video = apply_threshold_strategy(data['indist_fakes'], prob_thresh, vote_thresh)
            deeplive_fakes_video = apply_threshold_strategy(data['deeplive_fakes'], prob_thresh, vote_thresh)
            deeplive_reals_video = apply_threshold_strategy(data['deeplive_reals'], prob_thresh, vote_thresh)
            external_reals_video = apply_threshold_strategy(data['external_reals'], prob_thresh, vote_thresh)
            
            # Compute accuracies
            indist_fake_acc = compute_accuracy(indist_fakes_video, 1)  # Should predict fake
            deeplive_fake_acc = compute_accuracy(deeplive_fakes_video, 1)  # Should predict fake
            deeplive_real_acc = compute_accuracy(deeplive_reals_video, 0)  # Should predict real
            external_real_acc = compute_accuracy(external_reals_video, 0)  # Should predict real
            
            # Combined metrics
            # All fakes (indist + deeplive)
            all_fakes_correct = (
                (indist_fakes_video['video_pred'] == 1).sum() + 
                (deeplive_fakes_video['video_pred'] == 1).sum()
            )
            all_fakes_total = len(indist_fakes_video) + len(deeplive_fakes_video)
            all_fakes_acc = all_fakes_correct / all_fakes_total * 100 if all_fakes_total > 0 else 0
            
            # All reals (deeplive + external)
            all_reals_correct = (
                (deeplive_reals_video['video_pred'] == 0).sum() + 
                (external_reals_video['video_pred'] == 0).sum()
            )
            all_reals_total = len(deeplive_reals_video) + len(external_reals_video)
            all_reals_acc = all_reals_correct / all_reals_total * 100 if all_reals_total > 0 else 0
            
            # Overall in-distribution accuracy (balanced)
            overall_acc = (all_fakes_acc + all_reals_acc) / 2
            
            # Geometric mean (penalizes imbalance more)
            geo_mean = np.sqrt(all_fakes_acc * all_reals_acc)
            
            results.append({
                'model': model,
                'prob_thresh': prob_thresh,
                'vote_thresh': vote_thresh,
                'vote_thresh_frac': f"{int(vote_thresh*8)}/8",
                # Individual accuracies
                'indist_fake_acc': indist_fake_acc,
                'deeplive_fake_acc': deeplive_fake_acc,
                'deeplive_real_acc': deeplive_real_acc,
                'external_real_acc': external_real_acc,
                # Combined
                'all_fakes_acc': all_fakes_acc,
                'all_reals_acc': all_reals_acc,
                'overall_acc': overall_acc,
                'geo_mean': geo_mean,
                # Counts
                'n_indist_fakes': len(indist_fakes_video),
                'n_deeplive_fakes': len(deeplive_fakes_video),
                'n_deeplive_reals': len(deeplive_reals_video),
                'n_external_reals': len(external_reals_video),
            })
    
    return pd.DataFrame(results)


def compute_ood_metrics(data: dict, prob_thresh: float, vote_thresh: float) -> dict:
    """Compute OOD (source_target) metrics with given thresholds."""
    ood_fakes_video = apply_threshold_strategy(data['ood_fakes'], prob_thresh, vote_thresh)
    ood_fake_acc = compute_accuracy(ood_fakes_video, 1)
    
    # Per-method breakdown
    method_accs = {}
    for method in ood_fakes_video['method'].unique():
        method_df = ood_fakes_video[ood_fakes_video['method'] == method]
        method_accs[method] = compute_accuracy(method_df, 1)
    
    return {
        'ood_fake_acc': ood_fake_acc,
        'n_ood_fakes': len(ood_fakes_video),
        'method_accs': method_accs,
    }


def print_per_method_breakdown(data: dict, prob_thresh: float, vote_thresh: float, label: str):
    """Print per-method accuracy breakdown."""
    print(f"\n  Per-method breakdown ({label}):")
    
    if 'fakes' in label.lower():
        df = data.get('indist_fakes') if 'indist' in label.lower() else data.get('ood_fakes')
        expected = 1
    else:
        df = data.get('external_reals')
        expected = 0
    
    if df is None or len(df) == 0:
        print("    No data")
        return
    
    video_df = apply_threshold_strategy(df, prob_thresh, vote_thresh)
    
    for method in sorted(video_df['method'].unique()):
        method_videos = video_df[video_df['method'] == method]
        acc = compute_accuracy(method_videos, expected)
        print(f"    {method}: {acc:.2f}% ({len(method_videos)} videos)")


def main():
    print("=" * 80)
    print("COMPREHENSIVE THRESHOLD ANALYSIS")
    print("=" * 80)
    print()
    print("IN-DISTRIBUTION DATA (used for threshold selection):")
    print("  - target_source fakes: DF40 methods seen during training")
    print("  - deeplive fakes: DeepLive val split (fake videos)")
    print("  - deeplive reals: DeepLive val split (real videos)")  
    print("  - external_youtube_avspeech reals: ~7.3K YouTube videos")
    print()
    print("OOD DATA (evaluated with best in-dist thresholds):")
    print("  - source_target fakes: DF40 methods NOT seen during training")
    print()
    
    for model in ['B16', 'L14']:
        print("=" * 80)
        print(f"MODEL: {model}")
        print("=" * 80)
        
        # Load data
        print(f"\nLoading {model} data...")
        data = load_frames_data(model)
        
        # Print data statistics
        print(f"\nData Statistics:")
        print(f"  In-dist fakes (target_source): {data['indist_fakes']['video_id'].nunique()} videos, {len(data['indist_fakes'])} frames")
        print(f"  DeepLive fakes: {data['deeplive_fakes']['video_id'].nunique()} videos, {len(data['deeplive_fakes'])} frames")
        print(f"  DeepLive reals: {data['deeplive_reals']['video_id'].nunique()} videos, {len(data['deeplive_reals'])} frames")
        print(f"  External reals: {data['external_reals']['video_id'].nunique()} videos, {len(data['external_reals'])} frames")
        print(f"  OOD fakes (source_target): {data['ood_fakes']['video_id'].nunique()} videos, {len(data['ood_fakes'])} frames")
        
        # Grid search on in-distribution
        print(f"\nGrid searching on IN-DISTRIBUTION data...")
        results_df = grid_search_indist(data, model)
        
        # Save grid search results
        results_df.to_csv(f"{model}_comprehensive_grid_search.csv", index=False)
        
        # Find best thresholds
        # Option 1: Best overall (balanced fake + real accuracy)
        best_overall = results_df.loc[results_df['overall_acc'].idxmax()]
        
        # Option 2: Best geometric mean (more balanced)
        best_geomean = results_df.loc[results_df['geo_mean'].idxmax()]
        
        # Option 3: Best fake detection (minimize false negatives)
        best_fake = results_df.loc[results_df['all_fakes_acc'].idxmax()]
        
        # Option 4: Best real preservation (minimize false positives)
        best_real = results_df.loc[results_df['all_reals_acc'].idxmax()]
        
        # Option 5: Best with real >= 90%
        high_real_df = results_df[results_df['all_reals_acc'] >= 90]
        if len(high_real_df) > 0:
            best_90real = high_real_df.loc[high_real_df['all_fakes_acc'].idxmax()]
        else:
            best_90real = None
        
        # Option 6: Best with real >= 95%
        high_real_df = results_df[results_df['all_reals_acc'] >= 95]
        if len(high_real_df) > 0:
            best_95real = high_real_df.loc[high_real_df['all_fakes_acc'].idxmax()]
        else:
            best_95real = None
        
        print("\n" + "-" * 60)
        print("BEST THRESHOLD STRATEGIES (IN-DISTRIBUTION)")
        print("-" * 60)
        
        strategies = [
            ("Best Overall (balanced)", best_overall),
            ("Best Geometric Mean", best_geomean),
            ("Best Fake Detection", best_fake),
            ("Best Real Preservation", best_real),
        ]
        if best_90real is not None:
            strategies.append(("Best Fake @ Real>=90%", best_90real))
        if best_95real is not None:
            strategies.append(("Best Fake @ Real>=95%", best_95real))
        
        for name, row in strategies:
            print(f"\n{name}:")
            print(f"  Thresholds: prob={row['prob_thresh']}, vote={row['vote_thresh_frac']}")
            print(f"  In-dist Fakes: {row['all_fakes_acc']:.2f}%")
            print(f"    - target_source: {row['indist_fake_acc']:.2f}%")
            print(f"    - deeplive: {row['deeplive_fake_acc']:.2f}%")
            print(f"  Reals: {row['all_reals_acc']:.2f}%")
            print(f"    - deeplive: {row['deeplive_real_acc']:.2f}%")
            print(f"    - external: {row['external_real_acc']:.2f}%")
            print(f"  Overall: {row['overall_acc']:.2f}% | Geo-Mean: {row['geo_mean']:.2f}%")
        
        # OOD evaluation with recommended strategy
        print("\n" + "-" * 60)
        print("OOD EVALUATION (source_target fakes)")
        print("-" * 60)
        
        # Use best overall as the recommended strategy
        rec_prob = best_overall['prob_thresh']
        rec_vote = best_overall['vote_thresh']
        
        print(f"\nUsing BEST OVERALL thresholds: prob={rec_prob}, vote={int(rec_vote*8)}/8")
        
        ood_metrics = compute_ood_metrics(data, rec_prob, rec_vote)
        print(f"\nOOD Fake Detection: {ood_metrics['ood_fake_acc']:.2f}% ({ood_metrics['n_ood_fakes']} videos)")
        print("\nPer-method OOD accuracy:")
        for method, acc in sorted(ood_metrics['method_accs'].items(), key=lambda x: x[1]):
            print(f"  {method}: {acc:.2f}%")
        
        # Also show with geometric mean thresholds
        if best_geomean['prob_thresh'] != rec_prob or best_geomean['vote_thresh'] != rec_vote:
            geo_prob = best_geomean['prob_thresh']
            geo_vote = best_geomean['vote_thresh']
            print(f"\n--- With GEOMETRIC MEAN thresholds: prob={geo_prob}, vote={int(geo_vote*8)}/8 ---")
            ood_geo = compute_ood_metrics(data, geo_prob, geo_vote)
            print(f"OOD Fake Detection: {ood_geo['ood_fake_acc']:.2f}%")
        
        # DeepLive per-method breakdown
        print("\n" + "-" * 60)
        print("DEEPLIVE PER-STRATEGY BREAKDOWN")
        print("-" * 60)
        
        deeplive_fakes_video = apply_threshold_strategy(data['deeplive_fakes'], rec_prob, rec_vote)
        deeplive_reals_video = apply_threshold_strategy(data['deeplive_reals'], rec_prob, rec_vote)
        
        print(f"\nUsing thresholds: prob={rec_prob}, vote={int(rec_vote*8)}/8")
        
        print("\nDeepLive FAKE detection by strategy:")
        for method in sorted(deeplive_fakes_video['method'].unique()):
            method_videos = deeplive_fakes_video[deeplive_fakes_video['method'] == method]
            acc = compute_accuracy(method_videos, 1)
            print(f"  {method}: {acc:.2f}% ({len(method_videos)} videos)")
        
        print("\nDeepLive REAL preservation by strategy:")
        for method in sorted(deeplive_reals_video['method'].unique()):
            method_videos = deeplive_reals_video[deeplive_reals_video['method'] == method]
            acc = compute_accuracy(method_videos, 0)
            print(f"  {method}: {acc:.2f}% ({len(method_videos)} videos)")
        
        # In-dist per-method breakdown
        print("\n" + "-" * 60)
        print("IN-DIST FAKE (target_source) PER-METHOD BREAKDOWN")
        print("-" * 60)
        
        indist_fakes_video = apply_threshold_strategy(data['indist_fakes'], rec_prob, rec_vote)
        print(f"\nUsing thresholds: prob={rec_prob}, vote={int(rec_vote*8)}/8")
        for method in sorted(indist_fakes_video['method'].unique()):
            method_videos = indist_fakes_video[indist_fakes_video['method'] == method]
            acc = compute_accuracy(method_videos, 1)
            print(f"  {method}: {acc:.2f}% ({len(method_videos)} videos)")
        
        print()
    
    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print()
    print("Run complete. Check the *_comprehensive_grid_search.csv files for full results.")
    print()


if __name__ == "__main__":
    main()
