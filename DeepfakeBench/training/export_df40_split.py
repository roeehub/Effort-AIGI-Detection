#!/usr/bin/env python3
"""
Export DF40 data split information for reproducibility.

This script recreates the exact train/val/test split used during training
and exports it to JSON files for reference.

Usage:
    python export_df40_split.py --seed 737 --output_dir ./split_exports
    
The output includes:
- train_identities.json: List of identities used for training (90%)
- val_identities.json: List of identities used for in-dist validation (10%) 
- split_details.json: Full mapping of identity -> split and sample counts
"""

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class DF40Sample:
    """Lightweight sample class - no heavy imports needed."""
    method: str
    orientation: str
    target_identity: str
    real_video: str
    fake_video: str


def load_samples_from_json(pair_json_path: str, methods: Optional[List[str]] = None) -> List[DF40Sample]:
    """
    Load samples directly from the pair JSON file.
    This bypasses all the heavy dataset imports.
    """
    with open(pair_json_path, 'r') as f:
        pair_data = json.load(f)
    
    # Get method orientation mapping
    method_orientation = pair_data.get('method_orientation', {})
    
    samples = []
    for pair in pair_data.get('pairs', []):
        method = pair.get('method', '')
        
        if methods and method not in methods:
            continue
        
        # Get orientation from the method_orientation mapping
        orientation = method_orientation.get(method, 'unknown')
        
        # Extract identity info
        target_identity = pair.get('target_identity', '')
        source_identity = pair.get('source_identity', '')
        
        # Get paths
        fake_path = pair.get('fake', {}).get('path', '')
        real_path = pair.get('real', {}).get('path', '')
        
        samples.append(DF40Sample(
            method=method,
            orientation=orientation,
            target_identity=target_identity,
            real_video=real_path,
            fake_video=fake_path,
        ))
    
    return samples


def split_samples_and_export(
    samples,
    train_split: float,
    val_split: float,
    seed: int,
    output_dir: str
):
    """
    Recreate the exact split used during training and export it.
    
    This mirrors the logic in data/sources/df40_paired.py::_split_samples
    """
    rng = random.Random(seed)
    
    # Group samples by target_identity (the real person)
    by_identity = defaultdict(list)
    for sample in samples:
        by_identity[sample.target_identity].append(sample)
    
    # Get list of unique identities and shuffle (same as training)
    identities = list(by_identity.keys())
    rng.shuffle(identities)
    
    # Split identities (not pairs!)
    n_identities = len(identities)
    n_train_ids = int(n_identities * train_split)
    n_val_ids = int(n_identities * val_split)
    
    train_identities = identities[:n_train_ids]
    val_identities = identities[n_train_ids:n_train_ids + n_val_ids]
    test_identities = identities[n_train_ids + n_val_ids:]
    
    # Count samples per split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in samples:
        identity = sample.target_identity
        if identity in train_identities:
            train_samples.append(sample)
        elif identity in val_identities:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export identity lists
    with open(os.path.join(output_dir, 'train_identities.json'), 'w') as f:
        json.dump(train_identities, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_identities.json'), 'w') as f:
        json.dump(val_identities, f, indent=2)
    
    with open(os.path.join(output_dir, 'test_identities.json'), 'w') as f:
        json.dump(test_identities, f, indent=2)
    
    # Create detailed split info
    split_details = {
        'metadata': {
            'seed': seed,
            'train_split': train_split,
            'val_split': val_split,
            'test_split': 1.0 - train_split - val_split,
            'exported_at': datetime.now().isoformat(),
        },
        'summary': {
            'total_identities': n_identities,
            'train_identities': len(train_identities),
            'val_identities': len(val_identities),
            'test_identities': len(test_identities),
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
        },
        'identity_to_split': {
            identity: 'train' if identity in train_identities 
                      else ('val' if identity in val_identities else 'test')
            for identity in by_identity.keys()
        },
        'samples_by_identity': {
            identity: [
                {
                    'method': s.method,
                    'real_video': s.real_video,
                    'fake_video': s.fake_video,
                }
                for s in samples_list
            ]
            for identity, samples_list in by_identity.items()
        }
    }
    
    with open(os.path.join(output_dir, 'split_details.json'), 'w') as f:
        json.dump(split_details, f, indent=2)
    
    # Export video-level lists for easy reference
    train_videos = [(s.method, s.target_identity, s.fake_video) for s in train_samples]
    val_videos = [(s.method, s.target_identity, s.fake_video) for s in val_samples]
    test_videos = [(s.method, s.target_identity, s.fake_video) for s in test_samples]
    
    with open(os.path.join(output_dir, 'train_videos.json'), 'w') as f:
        json.dump(train_videos, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_videos.json'), 'w') as f:
        json.dump(val_videos, f, indent=2)
    
    with open(os.path.join(output_dir, 'test_videos.json'), 'w') as f:
        json.dump(test_videos, f, indent=2)
    
    return {
        'train_identities': train_identities,
        'val_identities': val_identities,
        'test_identities': test_identities,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
    }


def main():
    parser = argparse.ArgumentParser(description='Export DF40 data split information')
    parser.add_argument('--seed', type=int, default=737,
                        help='Random seed used during training (default: 737)')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Training set proportion (default: 0.9)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set proportion (default: 0.1)')
    parser.add_argument('--pair_json', type=str, 
                        default='dataset/df40_pairs/df40-pair-matching.json',
                        help='Path to DF40 pair matching JSON')
    parser.add_argument('--gcs_bucket', type=str, default='df40-frames-recropped-rfa85',
                        help='GCS bucket name')
    parser.add_argument('--output_dir', type=str, default='./split_exports',
                        help='Output directory for split files')
    parser.add_argument('--orientation', type=str, default='target_source',
                        choices=['target_source', 'source_target', 'all'],
                        help='DF40 orientation filter')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to include (default: all)')
    
    args = parser.parse_args()
    
    # Handle relative paths
    pair_json_path = args.pair_json
    if not os.path.isabs(pair_json_path):
        training_dir = os.path.dirname(os.path.abspath(__file__))
        pair_json_path = os.path.join(training_dir, pair_json_path)
    
    if not os.path.exists(pair_json_path):
        print(f"ERROR: Pair JSON not found: {pair_json_path}")
        return 1
    
    # Parse methods
    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    
    print(f"=" * 60)
    print(f"DF40 Split Export")
    print(f"=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Train/Val/Test: {args.train_split}/{args.val_split}/{1-args.train_split-args.val_split}")
    print(f"Pair JSON: {pair_json_path}")
    print(f"Orientation: {args.orientation}")
    print(f"Methods: {methods or 'ALL'}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Load samples directly from JSON (no heavy imports needed)
    print("Loading samples from pair JSON...")
    samples = load_samples_from_json(pair_json_path, methods)
    
    # Filter by orientation if needed
    if args.orientation != 'all':
        samples = [s for s in samples if s.orientation == args.orientation]
    
    print(f"Found {len(samples)} samples (orientation={args.orientation})")
    
    # Export split
    result = split_samples_and_export(
        samples=samples,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    print()
    print(f"=" * 60)
    print(f"Split Summary")
    print(f"=" * 60)
    print(f"Train identities: {len(result['train_identities'])} ({result['train_samples']} samples)")
    print(f"Val identities: {len(result['val_identities'])} ({result['val_samples']} samples)")
    print(f"Test identities: {len(result['test_identities'])} ({result['test_samples']} samples)")
    print()
    print(f"Files exported to: {args.output_dir}/")
    print(f"  - train_identities.json")
    print(f"  - val_identities.json")
    print(f"  - test_identities.json")
    print(f"  - train_videos.json")
    print(f"  - val_videos.json")
    print(f"  - test_videos.json")
    print(f"  - split_details.json")
    
    return 0


if __name__ == '__main__':
    exit(main())
