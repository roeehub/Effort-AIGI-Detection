#!/usr/bin/env python3
"""
Simulation script to validate real-time detection strategy on existing data.

This script:
1. Takes your existing per-frame predictions (from training CSVs)
2. Simulates real-time streaming by processing frames sequentially
3. Implements both Option A (frame selection) and Option B (per-frame fusion)
4. Measures temporal stability, latency, and accuracy
5. Compares against your validated batch results

Usage:
  python simulate_realtime_detection.py \
    --frame-csvs /path/to/model1.csv /path/to/model2.csv ... \
    --out-dir ./realtime_validation \
    --window-size 27 \
    --strategy option_a
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(__name__)


def topk_mean(x: np.ndarray, k: int) -> float:
    """Average of top k values."""
    if x.size == 0:
        return 0.0
    k = max(1, min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]
    return float(np.mean(x[idx]))


def softmax_pool(x: np.ndarray, beta: float) -> float:
    """Softmax pooling with temperature beta."""
    if x.size == 0:
        return 0.0
    m = np.max(beta * x)
    return float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)


def noisy_or_fusion(probs: np.ndarray) -> float:
    """Noisy-OR fusion: P(fake) = 1 - ∏(1 - p_i)"""
    return float(1.0 - np.prod(1.0 - probs))


class RealtimeSimulator:
    """
    Simulates real-time detection by processing frames sequentially.
    """
    
    def __init__(self,
                 window_size: int,
                 aggregators: Dict[str, callable],
                 calibrators: Dict[str, IsotonicRegression],
                 thresholds: Dict[str, float],
                 strategy: str = 'option_a'):
        """
        Args:
            window_size: Number of frames in sliding window
            aggregators: {'model1': lambda x: topk_mean(x, 4), ...}
            calibrators: {'model1': IsotonicRegression(), ...}
            thresholds: {'T_low': 0.996700, 'T_high': 0.998248}
            strategy: 'option_a' or 'option_b'
        """
        self.window_size = window_size
        self.aggregators = aggregators
        self.calibrators = calibrators
        self.T_low = thresholds['T_low']
        self.T_high = thresholds['T_high']
        self.strategy = strategy
        
        # Sliding window buffer
        self.buffer = deque(maxlen=window_size)
        
        # History tracking
        self.history = {
            'scores': [],
            'decisions': [],
            'confidences': [],
            'buffer_fills': [],
        }
    
    def process_frame(self, frame_preds: Dict[str, float]) -> Dict:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame_preds: {'model1': 0.89, 'model2': 0.87, ...}
            
        Returns:
            Dict with decision, score, confidence, etc.
        """
        # Add to buffer
        self.buffer.append(frame_preds)
        buffer_fill = len(self.buffer) / self.window_size
        
        if self.strategy == 'option_a':
            # Option A: Aggregate then calibrate (matches training)
            aggregated = {}
            for model_id, aggregator in self.aggregators.items():
                scores = np.array([f[model_id] for f in self.buffer])
                aggregated[model_id] = aggregator(scores)
            
            # Calibrate aggregated scores
            calibrated = {}
            for model_id, score in aggregated.items():
                calibrated[model_id] = self.calibrators[model_id].transform([score])[0]
            
        elif self.strategy == 'option_b':
            # Option B: Calibrate per frame, then fuse, then average
            fused_scores = []
            for frame in self.buffer:
                # Calibrate each frame's predictions
                cal_frame = {
                    model_id: self.calibrators[model_id].transform([score])[0]
                    for model_id, score in frame.items()
                }
                # Fuse per frame
                fused = noisy_or_fusion(np.array(list(cal_frame.values())))
                fused_scores.append(fused)
            
            # Average fused scores (this is the final score)
            final_score = np.mean(fused_scores)
            
            # For option B, we return early
            decision = self._classify(final_score)
            confidence = self._compute_confidence(final_score, decision, buffer_fill)
            
            result = {
                'decision': decision,
                'score': final_score,
                'confidence': confidence,
                'buffer_fill': buffer_fill,
                'window_std': np.std(fused_scores),
            }
            self._update_history(result)
            return result
        
        # Option A continues: Fuse calibrated scores
        fusion_score = noisy_or_fusion(np.array(list(calibrated.values())))
        
        # Classify
        decision = self._classify(fusion_score)
        confidence = self._compute_confidence(fusion_score, decision, buffer_fill)
        
        result = {
            'decision': decision,
            'score': fusion_score,
            'confidence': confidence,
            'buffer_fill': buffer_fill,
            'aggregated': aggregated,
            'calibrated': calibrated,
        }
        
        self._update_history(result)
        return result
    
    def _classify(self, score: float) -> str:
        """Three-way classification."""
        if score < self.T_low:
            return "REAL"
        elif score > self.T_high:
            return "FAKE"
        else:
            return "UNCERTAIN"
    
    def _compute_confidence(self, score: float, decision: str, buffer_fill: float) -> float:
        """Compute confidence score [0, 1]."""
        if decision == "REAL":
            conf = 1.0 - score
        elif decision == "FAKE":
            conf = score
        else:  # UNCERTAIN
            conf = 0.5
        
        # Reduce confidence if buffer not full
        return conf * buffer_fill
    
    def _update_history(self, result: Dict):
        """Track decision history for stability analysis."""
        self.history['scores'].append(result['score'])
        self.history['decisions'].append(result['decision'])
        self.history['confidences'].append(result['confidence'])
        self.history['buffer_fills'].append(result['buffer_fill'])
    
    def reset(self):
        """Clear buffer and history (between videos)."""
        self.buffer.clear()
        self.history = {
            'scores': [],
            'decisions': [],
            'confidences': [],
            'buffer_fills': [],
        }
    
    def get_final_decision(self, method: str = 'last') -> str:
        """
        Get final video-level decision after processing all frames.
        
        Args:
            method: 'last' (use last frame), 'majority' (vote), 'conservative' (if any FAKE)
        """
        if not self.history['decisions']:
            return "UNCERTAIN"
        
        if method == 'last':
            return self.history['decisions'][-1]
        elif method == 'majority':
            from collections import Counter
            counts = Counter(self.history['decisions'])
            return counts.most_common(1)[0][0]
        elif method == 'conservative':
            # If any frame was FAKE, video is FAKE
            if "FAKE" in self.history['decisions']:
                return "FAKE"
            elif "UNCERTAIN" in self.history['decisions']:
                return "UNCERTAIN"
            else:
                return "REAL"
        else:
            raise ValueError(f"Unknown method: {method}")


def load_frame_data(csv_path: str, model_name: str) -> pd.DataFrame:
    """Load per-frame predictions for one model."""
    df = pd.read_csv(
        csv_path,
        usecols=["method", "label", "frame_path", "frame_prob"],
        dtype={
            "method": "string",
            "label": "int8",
            "frame_path": "string",
            "frame_prob": "float32"
        }
    )
    
    # Extract clip_key (video ID) and frame index
    df["clip_key"] = df["frame_path"].apply(lambda x: "/".join(x.split("/")[:-1]))
    df["frame_idx"] = df["frame_path"].apply(
        lambda x: int(''.join(filter(str.isdigit, x.split("/")[-1])) or 0)
    )
    df["model"] = model_name
    
    return df


def fit_calibrators_oof(dfs: List[pd.DataFrame], 
                        aggregators: Dict[str, callable],
                        n_splits: int = 5) -> Dict[str, IsotonicRegression]:
    """
    Fit isotonic calibrators using out-of-fold strategy.
    
    This simulates what was done in training.
    """
    from sklearn.model_selection import StratifiedKFold
    
    logger = setup_logging()
    logger.info("Fitting isotonic calibrators (OOF)...")
    
    # First, aggregate per video per model
    video_data = defaultdict(lambda: {'label': None, 'method': None, 'scores': {}})
    
    for df in dfs:
        model_name = df['model'].iloc[0]
        for clip_key, group in df.groupby('clip_key'):
            video_data[clip_key]['label'] = int(group['label'].iloc[0])
            video_data[clip_key]['method'] = str(group['method'].iloc[0])
            
            # Aggregate this model's scores for this video
            scores = group['frame_prob'].to_numpy()
            agg_score = aggregators[model_name](scores)
            video_data[clip_key]['scores'][model_name] = agg_score
    
    # Convert to arrays
    clip_keys = list(video_data.keys())
    y = np.array([video_data[k]['label'] for k in clip_keys])
    
    # Get raw scores per model
    raw_scores = {}
    for model_name in aggregators.keys():
        raw_scores[model_name] = np.array([
            video_data[k]['scores'][model_name] for k in clip_keys
        ])
    
    # Fit calibrators OOF
    calibrators = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for model_name, scores in raw_scores.items():
        oof_calibrated = np.zeros_like(scores)
        
        for train_idx, val_idx in skf.split(scores, y):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(scores[train_idx], y[train_idx])
            oof_calibrated[val_idx] = ir.transform(scores[val_idx])
        
        # Fit final calibrator on all data
        final_ir = IsotonicRegression(out_of_bounds="clip")
        final_ir.fit(scores, y)
        calibrators[model_name] = final_ir
        
        logger.info(f"  {model_name}: Calibrator fitted")
    
    return calibrators


def simulate_video(frames_df: pd.DataFrame,
                   simulator: RealtimeSimulator,
                   ground_truth: int) -> Dict:
    """
    Simulate real-time processing of a single video.
    
    Args:
        frames_df: DataFrame with all frames for this video, all models
        simulator: RealtimeSimulator instance
        ground_truth: True label (0=real, 1=fake)
        
    Returns:
        Dict with results and metrics
    """
    simulator.reset()
    
    # Pivot to get per-frame predictions from all models
    frames_pivot = frames_df.pivot_table(
        index='frame_idx',
        columns='model',
        values='frame_prob'
    ).sort_index()
    
    # Process each frame
    frame_results = []
    for idx, row in frames_pivot.iterrows():
        preds = row.to_dict()
        result = simulator.process_frame(preds)
        frame_results.append(result)
    
    # Get final video-level decision
    final_decision = simulator.get_final_decision(method='last')
    
    # Analyze temporal stability
    decisions = simulator.history['decisions']
    scores = simulator.history['scores']
    
    # Count decision flips
    flips = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i-1])
    
    # Measure convergence time (frames until decision stabilizes)
    convergence_frame = 0
    if decisions:
        for i in range(len(decisions)-1, 0, -1):
            if decisions[i] != final_decision:
                convergence_frame = i + 1
                break
    
    # Calculate metrics
    correct = (final_decision == "FAKE" and ground_truth == 1) or \
              (final_decision == "REAL" and ground_truth == 0)
    
    return {
        'ground_truth': ground_truth,
        'final_decision': final_decision,
        'correct': correct,
        'n_frames': len(decisions),
        'n_flips': flips,
        'convergence_frame': convergence_frame,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_confidence': np.mean(simulator.history['confidences']),
        'frame_results': frame_results,
    }


def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-csvs", nargs='+', required=True,
                       help="Paths to per-frame CSV files (one per model)")
    parser.add_argument("--model-names", nargs='+', required=True,
                       help="Model names (e.g., 9rfa62j1, 1mjgo9w1, ...)")
    parser.add_argument("--aggregators", nargs='+', default=None,
                       help="Aggregator types per model (topk4, softmax_b5, ...)")
    parser.add_argument("--out-dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--window-size", type=int, default=27,
                       help="Sliding window size (number of frames)")
    parser.add_argument("--strategy", choices=['option_a', 'option_b'], default='option_a',
                       help="Detection strategy")
    parser.add_argument("--t-low", type=float, default=0.996700,
                       help="Low threshold (REAL if below)")
    parser.add_argument("--t-high", type=float, default=0.998248,
                       help="High threshold (FAKE if above)")
    parser.add_argument("--sample-videos", type=int, default=None,
                       help="Limit to N videos per method (for fast testing)")
    
    args = parser.parse_args()
    
    # Validation
    if len(args.frame_csvs) != len(args.model_names):
        parser.error("--frame-csvs and --model-names must have same length")
    
    # Default aggregators
    if args.aggregators is None:
        args.aggregators = ['topk4'] * len(args.model_names)
        # Set model 2 to softmax_b5 if we have 4 models (matches your setup)
        if len(args.model_names) == 4:
            args.aggregators[1] = 'softmax_b5'
    
    if len(args.aggregators) != len(args.model_names):
        parser.error("--aggregators must have same length as --model-names")
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading frame-level data...")
    dfs = []
    for csv_path, model_name in zip(args.frame_csvs, args.model_names):
        df = load_frame_data(csv_path, model_name)
        dfs.append(df)
        logger.info(f"  {model_name}: {len(df):,} frames")
    
    # Merge data
    logger.info("Merging data from all models...")
    merged = pd.concat(dfs, ignore_index=True)
    
    # Find common clip keys (videos present in all models)
    common_clips = set(dfs[0]['clip_key'].unique())
    for df in dfs[1:]:
        common_clips &= set(df['clip_key'].unique())
    logger.info(f"Common videos across all models: {len(common_clips):,}")
    
    merged = merged[merged['clip_key'].isin(common_clips)]
    
    # Sample if requested
    if args.sample_videos:
        logger.info(f"Sampling {args.sample_videos} videos per method...")
        sampled_clips = []
        for method in merged['method'].unique():
            method_clips = merged[merged['method'] == method]['clip_key'].unique()
            sample_size = min(args.sample_videos, len(method_clips))
            sampled_clips.extend(np.random.choice(method_clips, sample_size, replace=False))
        merged = merged[merged['clip_key'].isin(sampled_clips)]
        logger.info(f"After sampling: {len(merged['clip_key'].unique())} videos")
    
    # Setup aggregators
    agg_funcs = {
        'topk4': lambda x: topk_mean(x, 4),
        'softmax_b5': lambda x: softmax_pool(x, 5.0),
        'mean': lambda x: float(np.mean(x)),
        'max': lambda x: float(np.max(x)),
    }
    
    aggregators = {
        model_name: agg_funcs[agg_type]
        for model_name, agg_type in zip(args.model_names, args.aggregators)
    }
    
    # Fit calibrators
    calibrators = fit_calibrators_oof(
        [merged[merged['model'] == m] for m in args.model_names],
        aggregators
    )
    
    # Save calibrators
    with open(Path(args.out_dir) / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    logger.info(f"Saved calibrators to {args.out_dir}/calibrators.pkl")
    
    # Initialize simulator
    thresholds = {'T_low': args.t_low, 'T_high': args.t_high}
    simulator = RealtimeSimulator(
        window_size=args.window_size,
        aggregators=aggregators,
        calibrators=calibrators,
        thresholds=thresholds,
        strategy=args.strategy
    )
    
    # Simulate real-time detection on all videos
    logger.info("Simulating real-time detection...")
    results = []
    
    for clip_key in tqdm(merged['clip_key'].unique(), desc="Processing videos"):
        video_frames = merged[merged['clip_key'] == clip_key].copy()
        ground_truth = int(video_frames['label'].iloc[0])
        method = str(video_frames['method'].iloc[0])
        
        result = simulate_video(video_frames, simulator, ground_truth)
        result['clip_key'] = clip_key
        result['method'] = method
        results.append(result)
    
    # Analyze results
    logger.info("\n" + "="*60)
    logger.info("SIMULATION RESULTS")
    logger.info("="*60)
    
    # Overall metrics
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'frame_results'}
        for r in results
    ])
    
    accuracy = df_results['correct'].mean()
    logger.info(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Confusion matrix
    y_true = df_results['ground_truth'].to_numpy()
    y_pred = df_results['final_decision'].map({'REAL': 0, 'FAKE': 1, 'UNCERTAIN': 2}).to_numpy()
    
    # Three-way confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    logger.info("\nConfusion Matrix (Ground Truth vs Predicted):")
    logger.info(f"               Predicted REAL  Predicted FAKE  Predicted UNCERTAIN")
    
    # Count uncertain separately
    uncertain_real = ((y_true == 0) & (df_results['final_decision'] == 'UNCERTAIN')).sum()
    uncertain_fake = ((y_true == 1) & (df_results['final_decision'] == 'UNCERTAIN')).sum()
    
    correct_real = cm[0, 0]
    wrong_real = cm[0, 1]
    correct_fake = cm[1, 1]
    wrong_fake = cm[1, 0]
    
    logger.info(f"Actual REAL:   {correct_real:6d}         {wrong_real:6d}         {uncertain_real:6d}")
    logger.info(f"Actual FAKE:   {wrong_fake:6d}         {correct_fake:6d}         {uncertain_fake:6d}")
    
    # Performance metrics
    total_real = (y_true == 0).sum()
    total_fake = (y_true == 1).sum()
    
    tnr = correct_real / total_real if total_real > 0 else 0
    tpr = correct_fake / total_fake if total_fake > 0 else 0
    fpr = wrong_real / total_real if total_real > 0 else 0
    fnr = wrong_fake / total_fake if total_fake > 0 else 0
    
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  TPR (Recall on fakes): {tpr:.2%}")
    logger.info(f"  TNR (Recall on reals): {tnr:.2%}")
    logger.info(f"  FPR (false alarms):    {fpr:.2%}")
    logger.info(f"  FNR (missed fakes):    {fnr:.2%}")
    logger.info(f"  Uncertain rate:        {(df_results['final_decision'] == 'UNCERTAIN').mean():.2%}")
    
    # Temporal stability
    logger.info(f"\nTemporal Stability:")
    logger.info(f"  Mean flips per video:      {df_results['n_flips'].mean():.1f}")
    logger.info(f"  Mean convergence frame:    {df_results['convergence_frame'].mean():.1f}")
    logger.info(f"  Mean frames per video:     {df_results['n_frames'].mean():.1f}")
    logger.info(f"  Mean score std per video:  {df_results['std_score'].mean():.4f}")
    
    # Per-method breakdown
    logger.info(f"\nPer-Method Performance:")
    for method in sorted(df_results['method'].unique()):
        method_df = df_results[df_results['method'] == method]
        if len(method_df) < 5:
            continue
        acc = method_df['correct'].mean()
        label = int(method_df['ground_truth'].iloc[0])
        logger.info(f"  {method:20s}: {acc:.1%} ({len(method_df):4d} videos, label={label})")
    
    # Save results
    logger.info(f"\nSaving results to {args.out_dir}...")
    df_results.to_csv(Path(args.out_dir) / "video_results.csv", index=False)
    
    # Save configuration
    config = {
        'strategy': args.strategy,
        'window_size': args.window_size,
        'thresholds': thresholds,
        'model_names': args.model_names,
        'aggregators': args.aggregators,
        'metrics': {
            'accuracy': float(accuracy),
            'tpr': float(tpr),
            'tnr': float(tnr),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'uncertain_rate': float((df_results['final_decision'] == 'UNCERTAIN').mean()),
            'mean_flips': float(df_results['n_flips'].mean()),
            'mean_convergence': float(df_results['convergence_frame'].mean()),
        }
    }
    
    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Done!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
