#!/usr/bin/env python3
"""
Analyze validation results with different threshold strategies.

Key questions:
1. What's the best probability threshold?
2. What's the best voting threshold (e.g., 3/8 frames must pass)?
3. How does this affect real data (external_youtube_avspeech)?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
ANALYSIS_DIR = Path(__file__).parent
MODELS = {
    'B16': {
        'frames': ANALYSIS_DIR / 'B16_frames_report.csv',
        'videos': ANALYSIS_DIR / 'B16_videos_report.csv',
    },
    'L14': {
        'frames': ANALYSIS_DIR / 'L14_frames_report.csv',
        'videos': ANALYSIS_DIR / 'L14_videos_report.csv',
    }
}

# Probability thresholds to test
PROB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Voting thresholds (fraction of frames that must pass)
VOTING_THRESHOLDS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]  # 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8


def load_frames_data(filepath: Path) -> pd.DataFrame:
    """Load frames report CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} frames from {filepath.name}")
    return df


def compute_video_level_predictions(frames_df: pd.DataFrame, 
                                     prob_threshold: float = 0.5,
                                     voting_threshold: float = 0.5) -> pd.DataFrame:
    """
    Compute video-level predictions using threshold-based voting.
    
    Args:
        frames_df: DataFrame with frame-level predictions
        prob_threshold: Probability threshold to classify a frame as fake
        voting_threshold: Fraction of frames that must be classified as fake to classify video as fake
    
    Returns:
        DataFrame with video-level predictions
    """
    # Group by video
    video_groups = frames_df.groupby(['method', 'label', 'video_id'])
    
    results = []
    for (method, label, video_id), group in video_groups:
        n_frames = len(group)
        # Count frames above probability threshold
        n_fake_frames = (group['frame_prob'] >= prob_threshold).sum()
        fake_ratio = n_fake_frames / n_frames
        
        # Video is classified as fake if enough frames are fake
        prediction = 1 if fake_ratio >= voting_threshold else 0
        is_correct = int(prediction == label)
        
        results.append({
            'method': method,
            'label': label,
            'video_id': video_id,
            'n_frames': n_frames,
            'n_fake_frames': n_fake_frames,
            'fake_ratio': fake_ratio,
            'avg_prob': group['frame_prob'].mean(),
            'median_prob': group['frame_prob'].median(),
            'max_prob': group['frame_prob'].max(),
            'min_prob': group['frame_prob'].min(),
            'prediction': prediction,
            'is_correct': is_correct,
        })
    
    return pd.DataFrame(results)


def compute_metrics(video_df: pd.DataFrame, method: str = None) -> dict:
    """Compute accuracy, precision, recall, F1 for a subset of data."""
    if method is not None:
        df = video_df[video_df['method'] == method]
    else:
        df = video_df
    
    if len(df) == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'n_videos': 0}
    
    # True labels: label=1 is fake, label=0 is real
    y_true = df['label'].values
    y_pred = df['prediction'].values
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # For real data (label=0), we want to know false positive rate
    real_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate for real data
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_videos': len(df),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'real_accuracy': real_accuracy,  # For real data, this is the key metric
    }


def grid_search_thresholds(frames_df: pd.DataFrame, 
                           prob_thresholds: list = PROB_THRESHOLDS,
                           voting_thresholds: list = VOTING_THRESHOLDS) -> pd.DataFrame:
    """
    Grid search over probability and voting thresholds.
    Returns DataFrame with results for each combination.
    """
    results = []
    
    for prob_thresh in prob_thresholds:
        for vote_thresh in voting_thresholds:
            video_df = compute_video_level_predictions(
                frames_df, 
                prob_threshold=prob_thresh,
                voting_threshold=vote_thresh
            )
            
            # Overall metrics
            overall = compute_metrics(video_df)
            
            # Per-method metrics
            methods = video_df['method'].unique()
            method_metrics = {}
            for method in methods:
                method_metrics[method] = compute_metrics(video_df, method)
            
            # Real data metrics (external_youtube_avspeech)
            real_data = video_df[video_df['method'] == 'external_youtube_avspeech']
            if len(real_data) > 0:
                real_correct = (real_data['prediction'] == 0).sum()  # Real should be predicted as 0
                real_accuracy = real_correct / len(real_data)
                real_fp = (real_data['prediction'] == 1).sum()  # False positives
            else:
                real_accuracy = 0
                real_fp = 0
            
            # Fake data metrics (all methods except real)
            fake_data = video_df[video_df['method'] != 'external_youtube_avspeech']
            if len(fake_data) > 0:
                fake_correct = (fake_data['prediction'] == 1).sum()
                fake_accuracy = fake_correct / len(fake_data)
            else:
                fake_accuracy = 0
            
            results.append({
                'prob_threshold': prob_thresh,
                'voting_threshold': vote_thresh,
                'voting_fraction': f"{int(vote_thresh*8)}/8",
                'overall_accuracy': overall['accuracy'],
                'overall_f1': overall['f1'],
                'real_accuracy': real_accuracy,
                'real_fp_count': real_fp,
                'real_total': len(real_data),
                'fake_accuracy': fake_accuracy,
                'fake_total': len(fake_data),
                **{f'{m}_acc': method_metrics.get(m, {}).get('accuracy', 0) 
                   for m in methods if m != 'external_youtube_avspeech'},
            })
    
    return pd.DataFrame(results)


def print_detailed_analysis(model_name: str, frames_df: pd.DataFrame, best_params: dict):
    """Print detailed analysis for a model with the best parameters."""
    prob_thresh = best_params['prob_threshold']
    vote_thresh = best_params['voting_threshold']
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {model_name}")
    print(f"Best Parameters: prob_threshold={prob_thresh}, voting_threshold={vote_thresh} ({int(vote_thresh*8)}/8 frames)")
    print(f"{'='*80}")
    
    video_df = compute_video_level_predictions(frames_df, prob_thresh, vote_thresh)
    
    # Per-method breakdown
    print(f"\n{'Method':<35} {'Videos':>8} {'Correct':>8} {'Accuracy':>10} {'Label'}")
    print("-" * 75)
    
    methods = sorted(video_df['method'].unique())
    for method in methods:
        method_df = video_df[video_df['method'] == method]
        n_videos = len(method_df)
        label = method_df['label'].iloc[0]
        
        if label == 0:  # Real data - should predict 0
            correct = (method_df['prediction'] == 0).sum()
            label_str = "REAL"
        else:  # Fake data - should predict 1
            correct = (method_df['prediction'] == 1).sum()
            label_str = "FAKE"
        
        accuracy = correct / n_videos if n_videos > 0 else 0
        print(f"{method:<35} {n_videos:>8} {correct:>8} {accuracy:>10.2%} {label_str}")
    
    # Summary
    overall = compute_metrics(video_df)
    print(f"\n{'OVERALL':<35} {overall['n_videos']:>8} {int(overall['accuracy']*overall['n_videos']):>8} {overall['accuracy']:>10.2%}")
    print(f"Precision: {overall['precision']:.2%}, Recall: {overall['recall']:.2%}, F1: {overall['f1']:.2%}")


def analyze_real_data_in_detail(model_name: str, frames_df: pd.DataFrame):
    """Detailed analysis specifically for real data."""
    print(f"\n{'='*80}")
    print(f"REAL DATA ANALYSIS: {model_name}")
    print(f"{'='*80}")
    
    real_frames = frames_df[frames_df['method'] == 'external_youtube_avspeech']
    
    print(f"\nTotal real videos: {real_frames['video_id'].nunique():,}")
    print(f"Total real frames: {len(real_frames):,}")
    print(f"Frames per video: {len(real_frames) / real_frames['video_id'].nunique():.1f} avg")
    
    # Frame-level probability distribution
    print(f"\nFrame probability distribution (real data):")
    print(f"  Mean:   {real_frames['frame_prob'].mean():.4f}")
    print(f"  Median: {real_frames['frame_prob'].median():.4f}")
    print(f"  Std:    {real_frames['frame_prob'].std():.4f}")
    print(f"  Min:    {real_frames['frame_prob'].min():.4f}")
    print(f"  Max:    {real_frames['frame_prob'].max():.4f}")
    
    # What % of frames are classified as fake at different thresholds
    print(f"\n% of real frames classified as FAKE at different thresholds:")
    for thresh in PROB_THRESHOLDS:
        pct = (real_frames['frame_prob'] >= thresh).mean() * 100
        print(f"  prob >= {thresh}: {pct:.2f}%")
    
    # Video-level with different voting strategies
    print(f"\nReal video accuracy with different voting strategies:")
    print(f"{'Prob Thresh':<12} {'1/8':<8} {'2/8':<8} {'3/8':<8} {'4/8':<8} {'5/8':<8} {'6/8':<8} {'7/8':<8}")
    print("-" * 72)
    
    for prob_thresh in PROB_THRESHOLDS:
        row = f"{prob_thresh:<12}"
        for vote_thresh in VOTING_THRESHOLDS:
            video_df = compute_video_level_predictions(
                real_frames, prob_thresh, vote_thresh
            )
            # For real data, prediction should be 0
            accuracy = (video_df['prediction'] == 0).mean()
            row += f" {accuracy:.1%}  "
        print(row)


def main():
    print("=" * 80)
    print("VALIDATION RESULTS THRESHOLD ANALYSIS")
    print("=" * 80)
    
    all_results = {}
    
    for model_name, paths in MODELS.items():
        print(f"\n\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}")
        
        # Load data
        frames_df = load_frames_data(paths['frames'])
        
        # Grid search
        print("\n--- Grid Search Results ---")
        grid_results = grid_search_thresholds(frames_df)
        all_results[model_name] = grid_results
        
        # Find best parameters for different objectives
        # 1. Best overall accuracy
        best_overall = grid_results.loc[grid_results['overall_accuracy'].idxmax()]
        print(f"\nBest OVERALL ACCURACY: {best_overall['overall_accuracy']:.2%}")
        print(f"  prob_threshold={best_overall['prob_threshold']}, voting={best_overall['voting_fraction']}")
        print(f"  Real accuracy: {best_overall['real_accuracy']:.2%} (FP: {int(best_overall['real_fp_count'])})")
        print(f"  Fake accuracy: {best_overall['fake_accuracy']:.2%}")
        
        # 2. Best F1
        best_f1 = grid_results.loc[grid_results['overall_f1'].idxmax()]
        print(f"\nBest F1 SCORE: {best_f1['overall_f1']:.4f}")
        print(f"  prob_threshold={best_f1['prob_threshold']}, voting={best_f1['voting_fraction']}")
        print(f"  Real accuracy: {best_f1['real_accuracy']:.2%} (FP: {int(best_f1['real_fp_count'])})")
        print(f"  Fake accuracy: {best_f1['fake_accuracy']:.2%}")
        
        # 3. Best real accuracy (with constraint on fake accuracy > 80%)
        constrained = grid_results[grid_results['fake_accuracy'] >= 0.80]
        if len(constrained) > 0:
            best_real = constrained.loc[constrained['real_accuracy'].idxmax()]
            print(f"\nBest REAL ACCURACY (with fake acc >= 80%): {best_real['real_accuracy']:.2%}")
            print(f"  prob_threshold={best_real['prob_threshold']}, voting={best_real['voting_fraction']}")
            print(f"  Real accuracy: {best_real['real_accuracy']:.2%} (FP: {int(best_real['real_fp_count'])})")
            print(f"  Fake accuracy: {best_real['fake_accuracy']:.2%}")
        
        # Detailed analysis with best overall params
        print_detailed_analysis(model_name, frames_df, {
            'prob_threshold': best_overall['prob_threshold'],
            'voting_threshold': best_overall['voting_threshold'],
        })
        
        # Real data deep dive
        analyze_real_data_in_detail(model_name, frames_df)
    
    # Save grid search results
    for model_name, results in all_results.items():
        output_path = ANALYSIS_DIR / f'{model_name}_grid_search_results.csv'
        results.to_csv(output_path, index=False)
        print(f"\nSaved grid search results to {output_path}")
    
    # Compare models
    print("\n\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    for model_name, results in all_results.items():
        best = results.loc[results['overall_accuracy'].idxmax()]
        print(f"\n{model_name}:")
        print(f"  Best params: prob={best['prob_threshold']}, vote={best['voting_fraction']}")
        print(f"  Overall: {best['overall_accuracy']:.2%}, Real: {best['real_accuracy']:.2%}, Fake: {best['fake_accuracy']:.2%}")


if __name__ == '__main__':
    main()
