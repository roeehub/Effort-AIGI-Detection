#!/usr/bin/env python3
"""
Analyze in-distribution validation results (real + target_source fake data).

This script analyzes:
1. Real data from external YouTube videos
2. Target_source fake data (seen methods during training)

With splits:
- All target_source data
- Train identities only (90%)
- Val identities only (10%)

Key outputs:
- Optimal thresholds for different operating points
- Voting strategies (k/8 frames)
- Per-method breakdown
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
ANALYSIS_DIR = Path(__file__).parent
SPLIT_DIR = Path(__file__).parent.parent / 'DeepfakeBench/training/split_exports/target_source'

MODELS = {
    'B16': {
        'real_frames': ANALYSIS_DIR / 'B16_frames_report.csv',  # Contains external_youtube_avspeech
        'fake_frames': ANALYSIS_DIR / 'B16_target_source_frames_report.csv',
    },
    'L14': {
        'real_frames': ANALYSIS_DIR / 'L14_frames_report.csv',
        'fake_frames': ANALYSIS_DIR / 'L14_target_source_frames_report.csv',
    }
}

# Thresholds to test
PROB_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VOTING_THRESHOLDS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]  # 1/8 to 7/8


def load_split_identities():
    """Load train/val identity split from exported files."""
    train_ids = set()
    val_ids = set()
    
    train_path = SPLIT_DIR / 'train_identities.json'
    val_path = SPLIT_DIR / 'val_identities.json'
    
    if train_path.exists():
        with open(train_path) as f:
            train_ids = set(json.load(f))
        print(f"Loaded {len(train_ids)} train identities")
    else:
        print(f"WARNING: Train identities not found at {train_path}")
    
    if val_path.exists():
        with open(val_path) as f:
            val_ids = set(json.load(f))
        print(f"Loaded {len(val_ids)} val identities")
    else:
        print(f"WARNING: Val identities not found at {val_path}")
    
    return train_ids, val_ids


def load_and_combine_data(real_path: Path, fake_path: Path) -> pd.DataFrame:
    """Load real and fake data, filtering real to only external_youtube_avspeech."""
    # Load real data (filter to external real only)
    real_df = pd.read_csv(real_path)
    real_df = real_df[real_df['method'] == 'external_youtube_avspeech']
    print(f"Loaded {len(real_df):,} real frames from external_youtube_avspeech")
    
    # Load fake data (all target_source methods)
    fake_df = pd.read_csv(fake_path)
    print(f"Loaded {len(fake_df):,} fake frames from target_source methods")
    print(f"  Methods: {fake_df['method'].unique().tolist()}")
    
    # Combine
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    return combined


def extract_identity_from_video_id(video_id: str) -> str:
    """Extract identity from video_id like '001_870' -> '001' (target identity)."""
    if '_' in str(video_id):
        return str(video_id).split('_')[0]
    return str(video_id)


def compute_video_predictions(frames_df: pd.DataFrame, 
                               prob_threshold: float = 0.5,
                               voting_threshold: float = 0.5) -> pd.DataFrame:
    """Compute video-level predictions using threshold-based voting."""
    video_groups = frames_df.groupby(['method', 'label', 'video_id'])
    
    results = []
    for (method, label, video_id), group in video_groups:
        n_frames = len(group)
        n_fake_frames = (group['frame_prob'] >= prob_threshold).sum()
        fake_ratio = n_fake_frames / n_frames
        
        prediction = 1 if fake_ratio >= voting_threshold else 0
        is_correct = int(prediction == label)
        
        # Extract identity for split filtering
        identity = extract_identity_from_video_id(video_id)
        
        results.append({
            'method': method,
            'label': label,
            'video_id': video_id,
            'identity': identity,
            'n_frames': n_frames,
            'n_fake_frames': n_fake_frames,
            'fake_ratio': fake_ratio,
            'avg_prob': group['frame_prob'].mean(),
            'prediction': prediction,
            'is_correct': is_correct,
        })
    
    return pd.DataFrame(results)


def compute_metrics(video_df: pd.DataFrame) -> dict:
    """Compute accuracy, precision, recall, F1."""
    if len(video_df) == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'n_videos': 0}
    
    y_true = video_df['label'].values
    y_pred = video_df['prediction'].values
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / len(video_df) if len(video_df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Real accuracy (true negative rate)
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Fake accuracy (true positive rate / recall)
    fake_acc = recall
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_videos': len(video_df),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
    }


def grid_search(frames_df: pd.DataFrame, 
                train_ids: set = None,
                val_ids: set = None,
                split_name: str = "all") -> pd.DataFrame:
    """Grid search over thresholds, optionally filtering by identity split."""
    results = []
    
    for prob_thresh in PROB_THRESHOLDS:
        for vote_thresh in VOTING_THRESHOLDS:
            video_df = compute_video_predictions(frames_df, prob_thresh, vote_thresh)
            
            # Filter by split if specified
            if split_name == "train" and train_ids:
                # For fake data, filter by identity; for real data, include all
                fake_mask = (video_df['label'] == 1) & (video_df['identity'].isin(train_ids))
                real_mask = video_df['label'] == 0
                video_df = video_df[fake_mask | real_mask]
            elif split_name == "val" and val_ids:
                fake_mask = (video_df['label'] == 1) & (video_df['identity'].isin(val_ids))
                real_mask = video_df['label'] == 0
                video_df = video_df[fake_mask | real_mask]
            
            if len(video_df) == 0:
                continue
            
            metrics = compute_metrics(video_df)
            
            # Count by type
            real_videos = video_df[video_df['label'] == 0]
            fake_videos = video_df[video_df['label'] == 1]
            
            results.append({
                'prob_threshold': prob_thresh,
                'voting_threshold': vote_thresh,
                'voting_fraction': f"{int(vote_thresh*8)}/8",
                'split': split_name,
                'overall_accuracy': metrics['accuracy'],
                'overall_f1': metrics['f1'],
                'real_accuracy': metrics['real_accuracy'],
                'fake_accuracy': metrics['fake_accuracy'],
                'real_videos': len(real_videos),
                'fake_videos': len(fake_videos),
                'real_fp': metrics['fp'],
                'fake_fn': metrics['fn'],
            })
    
    return pd.DataFrame(results)


def find_optimal_thresholds(grid_df: pd.DataFrame, split_name: str) -> dict:
    """Find optimal thresholds for different objectives."""
    df = grid_df[grid_df['split'] == split_name]
    
    if len(df) == 0:
        return {}
    
    results = {}
    
    # 1. Best overall accuracy
    best_acc = df.loc[df['overall_accuracy'].idxmax()]
    results['best_accuracy'] = {
        'prob': best_acc['prob_threshold'],
        'vote': best_acc['voting_fraction'],
        'accuracy': best_acc['overall_accuracy'],
        'real_acc': best_acc['real_accuracy'],
        'fake_acc': best_acc['fake_accuracy'],
    }
    
    # 2. Best F1
    best_f1 = df.loc[df['overall_f1'].idxmax()]
    results['best_f1'] = {
        'prob': best_f1['prob_threshold'],
        'vote': best_f1['voting_fraction'],
        'f1': best_f1['overall_f1'],
        'real_acc': best_f1['real_accuracy'],
        'fake_acc': best_f1['fake_accuracy'],
    }
    
    # 3. Best real accuracy with fake_acc >= 90%
    constrained = df[df['fake_accuracy'] >= 0.90]
    if len(constrained) > 0:
        best_real_90 = constrained.loc[constrained['real_accuracy'].idxmax()]
        results['best_real_with_90_fake'] = {
            'prob': best_real_90['prob_threshold'],
            'vote': best_real_90['voting_fraction'],
            'real_acc': best_real_90['real_accuracy'],
            'fake_acc': best_real_90['fake_accuracy'],
        }
    
    # 4. Best real accuracy with fake_acc >= 95%
    constrained = df[df['fake_accuracy'] >= 0.95]
    if len(constrained) > 0:
        best_real_95 = constrained.loc[constrained['real_accuracy'].idxmax()]
        results['best_real_with_95_fake'] = {
            'prob': best_real_95['prob_threshold'],
            'vote': best_real_95['voting_fraction'],
            'real_acc': best_real_95['real_accuracy'],
            'fake_acc': best_real_95['fake_accuracy'],
        }
    
    # 5. Equal error rate approximation (real_acc ≈ fake_acc)
    df['error_diff'] = abs(df['real_accuracy'] - df['fake_accuracy'])
    best_eer = df.loc[df['error_diff'].idxmin()]
    results['balanced_eer'] = {
        'prob': best_eer['prob_threshold'],
        'vote': best_eer['voting_fraction'],
        'real_acc': best_eer['real_accuracy'],
        'fake_acc': best_eer['fake_accuracy'],
        'diff': best_eer['error_diff'],
    }
    
    return results


def analyze_per_method(frames_df: pd.DataFrame, prob_thresh: float, vote_thresh: float, 
                       train_ids: set = None, val_ids: set = None, split_name: str = "all"):
    """Detailed per-method breakdown."""
    video_df = compute_video_predictions(frames_df, prob_thresh, vote_thresh)
    
    # Filter by split
    if split_name == "train" and train_ids:
        fake_mask = (video_df['label'] == 1) & (video_df['identity'].isin(train_ids))
        real_mask = video_df['label'] == 0
        video_df = video_df[fake_mask | real_mask]
    elif split_name == "val" and val_ids:
        fake_mask = (video_df['label'] == 1) & (video_df['identity'].isin(val_ids))
        real_mask = video_df['label'] == 0
        video_df = video_df[fake_mask | real_mask]
    
    print(f"\n{'Method':<30} {'Videos':>8} {'Correct':>8} {'Accuracy':>10} {'Type'}")
    print("-" * 70)
    
    methods = sorted(video_df['method'].unique())
    for method in methods:
        method_df = video_df[video_df['method'] == method]
        n_videos = len(method_df)
        label = method_df['label'].iloc[0]
        
        if label == 0:
            correct = (method_df['prediction'] == 0).sum()
            type_str = "REAL"
        else:
            correct = (method_df['prediction'] == 1).sum()
            type_str = "FAKE"
        
        accuracy = correct / n_videos if n_videos > 0 else 0
        print(f"{method:<30} {n_videos:>8} {correct:>8} {accuracy:>10.2%} {type_str}")


def main():
    print("=" * 80)
    print("IN-DISTRIBUTION VALIDATION ANALYSIS")
    print("Real (external_youtube_avspeech) + Fake (target_source methods)")
    print("=" * 80)
    
    # Load identity splits
    train_ids, val_ids = load_split_identities()
    
    for model_name, paths in MODELS.items():
        print(f"\n\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}")
        
        # Load data
        frames_df = load_and_combine_data(paths['real_frames'], paths['fake_frames'])
        
        # Analyze three splits: all, train, val
        all_results = []
        
        for split_name, ids in [("all", None), ("train", train_ids), ("val", val_ids)]:
            print(f"\n{'='*60}")
            print(f"SPLIT: {split_name.upper()}")
            print(f"{'='*60}")
            
            # Grid search
            grid_df = grid_search(frames_df, train_ids, val_ids, split_name)
            all_results.append(grid_df)
            
            # Find optimal thresholds
            optimal = find_optimal_thresholds(grid_df, split_name)
            
            print(f"\n--- Optimal Thresholds ({split_name}) ---")
            
            if 'best_accuracy' in optimal:
                o = optimal['best_accuracy']
                print(f"\n✓ Best ACCURACY: {o['accuracy']:.2%}")
                print(f"    prob={o['prob']}, vote={o['vote']}")
                print(f"    Real: {o['real_acc']:.2%}, Fake: {o['fake_acc']:.2%}")
            
            if 'best_f1' in optimal:
                o = optimal['best_f1']
                print(f"\n✓ Best F1: {o['f1']:.4f}")
                print(f"    prob={o['prob']}, vote={o['vote']}")
                print(f"    Real: {o['real_acc']:.2%}, Fake: {o['fake_acc']:.2%}")
            
            if 'best_real_with_95_fake' in optimal:
                o = optimal['best_real_with_95_fake']
                print(f"\n✓ Best REAL (≥95% fake): {o['real_acc']:.2%}")
                print(f"    prob={o['prob']}, vote={o['vote']}")
                print(f"    Real: {o['real_acc']:.2%}, Fake: {o['fake_acc']:.2%}")
            
            if 'best_real_with_90_fake' in optimal:
                o = optimal['best_real_with_90_fake']
                print(f"\n✓ Best REAL (≥90% fake): {o['real_acc']:.2%}")
                print(f"    prob={o['prob']}, vote={o['vote']}")
                print(f"    Real: {o['real_acc']:.2%}, Fake: {o['fake_acc']:.2%}")
            
            if 'balanced_eer' in optimal:
                o = optimal['balanced_eer']
                print(f"\n✓ Balanced (EER-like): Real={o['real_acc']:.2%}, Fake={o['fake_acc']:.2%}")
                print(f"    prob={o['prob']}, vote={o['vote']}, diff={o['diff']:.4f}")
            
            # Per-method breakdown with best accuracy settings
            if 'best_accuracy' in optimal:
                print(f"\n--- Per-Method Breakdown ({split_name}, best accuracy) ---")
                best = optimal['best_accuracy']
                analyze_per_method(frames_df, best['prob'], 
                                   VOTING_THRESHOLDS[int(best['vote'].split('/')[0])-1],
                                   train_ids, val_ids, split_name)
        
        # Save combined grid search results
        combined_grid = pd.concat(all_results, ignore_index=True)
        output_path = ANALYSIS_DIR / f'{model_name}_indist_grid_search.csv'
        combined_grid.to_csv(output_path, index=False)
        print(f"\n\nSaved grid search results to {output_path}")
        
        # Print voting strategy table
        print(f"\n\n--- Voting Strategy Table ({model_name}, ALL data) ---")
        print(f"Real accuracy at different thresholds:\n")
        print(f"{'Prob':<8}", end="")
        for v in VOTING_THRESHOLDS:
            print(f"{int(v*8)}/8     ", end="")
        print()
        print("-" * 70)
        
        for prob in PROB_THRESHOLDS:
            print(f"{prob:<8}", end="")
            for vote in VOTING_THRESHOLDS:
                row = combined_grid[(combined_grid['split'] == 'all') & 
                                    (combined_grid['prob_threshold'] == prob) &
                                    (combined_grid['voting_threshold'] == vote)]
                if len(row) > 0:
                    real_acc = row['real_accuracy'].values[0]
                    fake_acc = row['fake_accuracy'].values[0]
                    print(f"{real_acc:.1%}/{fake_acc:.1%} ", end="")
                else:
                    print("---   ", end="")
            print()


if __name__ == '__main__':
    main()
