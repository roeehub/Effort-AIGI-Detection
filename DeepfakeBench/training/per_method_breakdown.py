#!/usr/bin/env python3
"""
Per-method accuracy breakdown for the recommended strategies.
Evaluates window-level performance per method across all 3 buckets.
"""
import csv
import numpy as np
from collections import defaultdict
import sys
import os

# --- Config ---
INFERENCE_DIR = "inference_results"
CSVS = {
    'teams_flat': os.path.join(INFERENCE_DIR, 'r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv'),
    'live_deepfake': os.path.join(INFERENCE_DIR, 'r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv'),
    'poc_phase1': os.path.join(INFERENCE_DIR, 'r9a_run1__poc-phase-1-test.csv'),
}
SEED = 42

# --- Strategies to evaluate ---
STRATEGIES = [
    # (label, W, T, K, margin)
    ("Baseline (current)",     16, 0.80, 8,  0),
    ("Best W=16 binary",       16, 0.42, 9,  0),
    ("Best W=16 uncertain",    16, 0.43, 9,  1),
    ("Best W=24 binary",       24, 0.47, 12, 0),
    ("Best W=32 binary",       32, 0.56, 16, 0),
    ("Best W=32 uncertain",    32, 0.48, 16, 2),
]


def load_frames(path, bucket_name):
    """Load frames from CSV."""
    frames = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue
            frames.append({
                'bucket': bucket_name,
                'method': r.get('method', 'unknown'),
                'label': int(r.get('label', 0)),
                'prob_fake': float(r.get('prob_fake', 0.5)),
                'video_id': r.get('video_id', ''),
                'frame_name': r.get('frame_name', ''),
                'blob_path': r.get('blob_path', ''),
            })
    return frames


def build_windows_per_method(frames, W, bucket_name):
    """Build windows grouped by method, returns list of (method, label, probs)."""
    np.random.seed(SEED)
    windows = []

    if bucket_name == 'poc_phase1':
        # Natural video grouping
        by_vid = defaultdict(list)
        for f in frames:
            by_vid[f['video_id']].append(f)
        for vid, vframes in by_vid.items():
            vframes.sort(key=lambda x: x['frame_name'])
            method = vframes[0]['method']
            label = vframes[0]['label']
            probs = [f['prob_fake'] for f in vframes]
            for i in range(0, len(probs) - W + 1, W):
                chunk = probs[i:i+W]
                if len(chunk) == W:
                    windows.append((method, label, np.array(chunk)))

    elif bucket_name == 'live_deepfake':
        # Group by video_id then split by label, pool same-(method,label) across videos
        by_vid = defaultdict(list)
        for f in frames:
            by_vid[f['video_id']].append(f)

        pools = defaultdict(list)  # (method, label) -> [prob_fake, ...]
        for vid, vframes in by_vid.items():
            for f in vframes:
                pools[(f['method'], f['label'])].append(f['prob_fake'])

        for (method, label), probs in pools.items():
            np.random.shuffle(probs)
            for i in range(0, len(probs) - W + 1, W):
                chunk = probs[i:i+W]
                if len(chunk) == W:
                    windows.append((method, label, np.array(chunk)))

    elif bucket_name == 'teams_flat':
        # Random partition by (method, label)
        pools = defaultdict(list)
        for f in frames:
            pools[(f['method'], f['label'])].append(f['prob_fake'])

        for (method, label), probs in pools.items():
            arr = np.array(probs)
            np.random.shuffle(arr)
            for i in range(0, len(arr) - W + 1, W):
                chunk = arr[i:i+W]
                if len(chunk) == W:
                    windows.append((method, label, chunk))

    return windows


def evaluate_strategy(windows, T, K, margin=0):
    """Evaluate a strategy on windows. Returns per-method stats."""
    K_high = K + margin
    K_low = K - margin

    results_by_method = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'uncertain': 0, 'n_fake': 0, 'n_real': 0})

    for method, label, probs in windows:
        above = np.sum(probs >= T)

        if margin > 0:
            if above >= K_high:
                pred = 1
            elif above <= K_low:
                pred = 0
            else:
                pred = -1  # uncertain
        else:
            pred = 1 if above >= K else 0

        stats = results_by_method[method]
        if label == 1:
            stats['n_fake'] += 1
        else:
            stats['n_real'] += 1

        if pred == -1:
            stats['uncertain'] += 1
        elif pred == 1 and label == 1:
            stats['tp'] += 1
        elif pred == 1 and label == 0:
            stats['fp'] += 1
        elif pred == 0 and label == 0:
            stats['tn'] += 1
        elif pred == 0 and label == 1:
            stats['fn'] += 1

    return dict(results_by_method)


def main():
    # Load all frames
    all_frames = {}
    for bucket, path in CSVS.items():
        all_frames[bucket] = load_frames(path, bucket)
        print(f"Loaded {len(all_frames[bucket])} frames from {bucket}")

    print()
    print("=" * 120)
    print("PER-METHOD ACCURACY BREAKDOWN")
    print("=" * 120)

    for strat_label, W, T, K, margin in STRATEGIES:
        print()
        print(f"{'─'*120}")
        print(f"Strategy: {strat_label}  (W={W}, T={T}, K={K}", end="")
        if margin > 0:
            print(f", margin={margin}, K_hi={K+margin}, K_lo={K-margin})", end="")
        else:
            print(")", end="")
        print()
        print(f"{'─'*120}")

        # Build windows per bucket
        all_method_stats = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'uncertain': 0, 'n_fake': 0, 'n_real': 0})

        for bucket in ['teams_flat', 'live_deepfake', 'poc_phase1']:
            windows = build_windows_per_method(all_frames[bucket], W, bucket)
            method_stats = evaluate_strategy(windows, T, K, margin)

            # Print per-bucket header
            total_win = len(windows)
            print(f"\n  [{bucket}] ({total_win} windows)")
            print(f"  {'Method':<30s} {'n':>5s} {'nF':>4s} {'nR':>4s} {'TPR':>8s} {'TNR':>8s} {'Acc':>8s} {'FPR':>8s} {'unc%':>6s} {'ClnAcc':>8s}")
            print(f"  {'─'*100}")

            # Sort methods: fake methods first (by TPR), then real
            methods_sorted = sorted(method_stats.keys())

            bucket_total = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'uncertain': 0, 'n_fake': 0, 'n_real': 0}

            for method in methods_sorted:
                s = method_stats[method]
                n = s['n_fake'] + s['n_real']
                unc_rate = s['uncertain'] / n if n > 0 else 0
                n_decided = n - s['uncertain']

                # TPR = tp / n_fake, TNR = tn / n_real
                tpr = s['tp'] / s['n_fake'] if s['n_fake'] > 0 else float('nan')
                tnr = s['tn'] / s['n_real'] if s['n_real'] > 0 else float('nan')
                fpr = s['fp'] / s['n_real'] if s['n_real'] > 0 else float('nan')
                acc = (s['tp'] + s['tn']) / n if n > 0 else float('nan')
                clean_acc = (s['tp'] + s['tn']) / n_decided if n_decided > 0 else float('nan')

                tpr_s = f"{tpr*100:7.1f}%" if tpr == tpr else "    n/a "
                tnr_s = f"{tnr*100:7.1f}%" if tnr == tnr else "    n/a "
                fpr_s = f"{fpr*100:7.1f}%" if fpr == fpr else "    n/a "
                acc_s = f"{acc*100:7.1f}%" if acc == acc else "    n/a "
                cln_s = f"{clean_acc*100:7.1f}%" if clean_acc == clean_acc and unc_rate > 0 else "       -"

                print(f"  {method:<30s} {n:>5d} {s['n_fake']:>4d} {s['n_real']:>4d} {tpr_s:>8s} {tnr_s:>8s} {acc_s:>8s} {fpr_s:>8s} {unc_rate*100:>5.1f}% {cln_s:>8s}")

                # Aggregate
                for k in bucket_total:
                    bucket_total[k] += s[k]

                # Also aggregate into global
                for k in all_method_stats[method]:
                    all_method_stats[method][k] += s[k]

            # Bucket totals
            bt = bucket_total
            n = bt['n_fake'] + bt['n_real']
            n_dec = n - bt['uncertain']
            tpr = bt['tp'] / bt['n_fake'] if bt['n_fake'] > 0 else float('nan')
            tnr = bt['tn'] / bt['n_real'] if bt['n_real'] > 0 else float('nan')
            fpr = bt['fp'] / bt['n_real'] if bt['n_real'] > 0 else float('nan')
            bal = (tpr + tnr) / 2 if tpr == tpr and tnr == tnr else float('nan')
            acc = (bt['tp'] + bt['tn']) / n if n > 0 else float('nan')
            cln = (bt['tp'] + bt['tn']) / n_dec if n_dec > 0 else float('nan')
            unc = bt['uncertain'] / n if n > 0 else 0
            print(f"  {'─'*100}")
            bal_s = f"{bal*100:7.1f}%" if bal == bal else "    n/a "
            acc_s = f"{acc*100:7.1f}%" if acc == acc else "    n/a "
            cln_s = f"{cln*100:7.1f}%" if cln == cln and unc > 0 else f"{acc*100:7.1f}%"
            print(f"  {'TOTAL':<30s} {n:>5d} {bt['n_fake']:>4d} {bt['n_real']:>4d} {tpr*100:>7.1f}% {tnr*100:>7.1f}% {acc_s:>8s} {fpr*100:>7.1f}% {unc*100:>5.1f}% {cln_s:>8s}  bal_acc={bal_s}")

        # --- Summary across all buckets ---
        print(f"\n  [ALL THREE COMBINED]")
        print(f"  {'Method':<30s} {'n':>5s} {'nF':>4s} {'nR':>4s} {'TPR':>8s} {'TNR':>8s} {'Acc':>8s} {'unc%':>6s}")
        print(f"  {'─'*80}")
        grand = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'uncertain': 0, 'n_fake': 0, 'n_real': 0}
        for method in sorted(all_method_stats.keys()):
            s = all_method_stats[method]
            n = s['n_fake'] + s['n_real']
            tpr = s['tp'] / s['n_fake'] if s['n_fake'] > 0 else float('nan')
            tnr = s['tn'] / s['n_real'] if s['n_real'] > 0 else float('nan')
            acc = (s['tp'] + s['tn']) / n if n > 0 else float('nan')
            unc = s['uncertain'] / n if n > 0 else 0
            tpr_s = f"{tpr*100:7.1f}%" if tpr == tpr else "    n/a "
            tnr_s = f"{tnr*100:7.1f}%" if tnr == tnr else "    n/a "
            acc_s = f"{acc*100:7.1f}%" if acc == acc else "    n/a "
            print(f"  {method:<30s} {n:>5d} {s['n_fake']:>4d} {s['n_real']:>4d} {tpr_s:>8s} {tnr_s:>8s} {acc_s:>8s} {unc*100:>5.1f}%")
            for k in grand:
                grand[k] += s[k]

        n = grand['n_fake'] + grand['n_real']
        tpr = grand['tp'] / grand['n_fake'] if grand['n_fake'] > 0 else 0
        tnr = grand['tn'] / grand['n_real'] if grand['n_real'] > 0 else 0
        bal = (tpr + tnr) / 2
        acc = (grand['tp'] + grand['tn']) / n if n > 0 else 0
        unc = grand['uncertain'] / n if n > 0 else 0
        print(f"  {'─'*80}")
        print(f"  {'GRAND TOTAL':<30s} {n:>5d} {grand['n_fake']:>4d} {grand['n_real']:>4d} {tpr*100:>7.1f}% {tnr*100:>7.1f}% {acc*100:>7.1f}% {unc*100:>5.1f}%  bal_acc={bal*100:.1f}%")


if __name__ == "__main__":
    main()
