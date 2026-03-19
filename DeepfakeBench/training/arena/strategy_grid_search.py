#!/usr/bin/env python3
"""
Strategy Grid Search for Deepfake Detection Window-Based Voting.

Finds optimal (W, T, K/W) and uncertain-zone strategy across 3 test buckets.

Stages:
  1. Coarse grid:  W × T × K/W  (225 combos)
  2. Fine grid:    zoom into top regions
  3. Uncertain:    add uncertain label (≤5%) to boost clean accuracy

Usage:
  python3 strategy_grid_search.py [--stage 1|2|3|all] [--output_dir ./strategy_results]
"""

import csv
import json
import os
import sys
import time
import argparse
import logging
from collections import defaultdict
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = "inference_results"
CSV_FILES = {
    "teams_flat":     f"{RESULTS_DIR}/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
    "live_deepfake":  f"{RESULTS_DIR}/r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    "poc_phase1":     f"{RESULTS_DIR}/r9a_run1__poc-phase-1-test.csv",
}

SEED = 42

# Stage 1 coarse grid
WINDOW_SIZES    = [16, 24, 32]
THRESHOLDS_COARSE = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
K_RATIOS_COARSE = [0.25, 0.375, 0.50, 0.625, 0.75]

UNCERTAIN_BUDGET = 0.05  # max 5% uncertain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [strategy-grid] %(levelname)s: %(message)s")
log = logging.getLogger("strategy-grid")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------
def build_windows(rows, bucket_name, W, seed=SEED):
    """
    Build non-overlapping windows of W frames, each with a single ground-truth label.

    Returns list of dicts: {probs: np.array, label: int, label_str: str, method: str, bucket: str}
    """
    windows = []

    if bucket_name == "poc_phase1":
        # Natural video grouping, single label per video
        by_video = defaultdict(list)
        for r in rows:
            by_video[r['video_id']].append(r)

        for vid in sorted(by_video.keys()):
            frames = sorted(by_video[vid], key=lambda r: r['frame_name'])
            label = int(frames[0]['label'])
            label_str = frames[0]['label_str']
            method = frames[0]['method']
            n_windows = len(frames) // W
            for i in range(n_windows):
                chunk = frames[i * W : (i + 1) * W]
                probs = np.array([float(r['prob_fake']) for r in chunk])
                windows.append({
                    'probs': probs, 'label': label, 'label_str': label_str,
                    'method': method, 'bucket': bucket_name,
                })

    elif bucket_name == "live_deepfake":
        # Group by video_id, split by label, then pool same-label frames across samples
        by_video = defaultdict(list)
        for r in rows:
            by_video[r['video_id']].append(r)

        fake_frames, real_frames = [], []
        for vid in sorted(by_video.keys()):
            for r in sorted(by_video[vid], key=lambda r: r['frame_name']):
                if r['label_str'] == 'fake':
                    fake_frames.append(r)
                else:
                    real_frames.append(r)

        for pool, label, label_str in [(fake_frames, 1, 'fake'), (real_frames, 0, 'real')]:
            n_windows = len(pool) // W
            for i in range(n_windows):
                chunk = pool[i * W : (i + 1) * W]
                probs = np.array([float(r['prob_fake']) for r in chunk])
                # Dominant method in this chunk
                methods = defaultdict(int)
                for r in chunk:
                    methods[r['method']] += 1
                method = max(methods, key=methods.get)
                windows.append({
                    'probs': probs, 'label': label, 'label_str': label_str,
                    'method': method, 'bucket': bucket_name,
                })

    elif bucket_name == "teams_flat":
        # No natural video grouping → random partition by label
        rng = np.random.RandomState(seed)
        fake_frames = [r for r in rows if r['label_str'] == 'fake']
        real_frames = [r for r in rows if r['label_str'] == 'real']

        for pool, label, label_str in [(fake_frames, 1, 'fake'), (real_frames, 0, 'real')]:
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            n_windows = len(indices) // W
            for i in range(n_windows):
                chunk_idx = indices[i * W : (i + 1) * W]
                probs = np.array([float(pool[j]['prob_fake']) for j in chunk_idx])
                method = pool[chunk_idx[0]]['method']
                windows.append({
                    'probs': probs, 'label': label, 'label_str': label_str,
                    'method': method, 'bucket': bucket_name,
                })

    return windows


# ---------------------------------------------------------------------------
# Metrics computation (vectorised)
# ---------------------------------------------------------------------------
def compute_metrics_vec(labels, preds, uncertain_mask=None):
    """
    labels:          np.array of int (1=fake, 0=real)
    preds:           np.array of int (1=fake, 0=real) — for decided windows
    uncertain_mask:  np.array of bool — True if window is uncertain (optional)

    Returns dict of metrics.
    """
    n = len(labels)
    if uncertain_mask is None:
        uncertain_mask = np.zeros(n, dtype=bool)

    decided = ~uncertain_mask
    n_decided = decided.sum()
    n_uncertain = uncertain_mask.sum()

    total_fake = (labels == 1).sum()
    total_real = (labels == 0).sum()

    # Among decided windows
    d_labels = labels[decided]
    d_preds = preds[decided]

    tp = ((d_preds == 1) & (d_labels == 1)).sum()
    tn = ((d_preds == 0) & (d_labels == 0)).sum()
    fp = ((d_preds == 1) & (d_labels == 0)).sum()
    fn = ((d_preds == 0) & (d_labels == 1)).sum()

    # Uncertain breakdown
    u_fake = (labels[uncertain_mask] == 1).sum() if n_uncertain > 0 else 0
    u_real = (labels[uncertain_mask] == 0).sum() if n_uncertain > 0 else 0

    # Rates — use total population (not just decided) for TPR/TNR/FPR
    # TPR = fraction of all fakes that are correctly called fake
    # TNR = fraction of all reals that are correctly called real
    tpr = float(tp) / total_fake if total_fake > 0 else float('nan')
    tnr = float(tn) / total_real if total_real > 0 else float('nan')
    fpr = float(fp) / total_real if total_real > 0 else float('nan')
    fnr = float(fn) / total_fake if total_fake > 0 else float('nan')

    balanced_acc = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else float('nan')
    accuracy = float(tp + tn) / n if n > 0 else float('nan')
    clean_accuracy = float(tp + tn) / n_decided if n_decided > 0 else float('nan')
    uncertain_rate = float(n_uncertain) / n if n > 0 else 0.0

    return {
        'balanced_acc': balanced_acc,
        'accuracy': accuracy,
        'clean_accuracy': clean_accuracy,
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'uncertain_rate': uncertain_rate,
        'n_total': int(n),
        'n_fake': int(total_fake),
        'n_real': int(total_real),
        'n_uncertain': int(n_uncertain),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'u_fake': int(u_fake), 'u_real': int(u_real),
    }


# ---------------------------------------------------------------------------
# Grid search engine
# ---------------------------------------------------------------------------
def run_grid(windows_by_bucket, W, thresholds, k_ratios, uncertain_margins=None):
    """
    Run grid over (T, K_ratio) combos for a given W.
    Returns list of result rows (dicts).
    """
    if uncertain_margins is None:
        uncertain_margins = [0]  # no uncertain

    # Build combined arrays for each evaluation scope
    scopes = {}  # scope_name -> (probs_matrix, labels, bucket_ids, method_ids)

    # Flatten all windows with tracking
    all_windows = []
    bucket_names_list = []
    method_names_list = []
    label_list = []

    for bname in ["teams_flat", "live_deepfake", "poc_phase1"]:
        for w in windows_by_bucket.get(bname, []):
            all_windows.append(w['probs'])
            bucket_names_list.append(bname)
            method_names_list.append(w['method'])
            label_list.append(w['label'])

    if not all_windows:
        return []

    probs_matrix = np.stack(all_windows)        # (N, W)
    labels = np.array(label_list)                # (N,)
    bucket_ids = np.array(bucket_names_list)     # (N,) strings
    method_ids = np.array(method_names_list)     # (N,) strings

    # Pre-compute masks for scopes
    mask_teams_flat = bucket_ids == "teams_flat"
    mask_live_df = bucket_ids == "live_deepfake"
    mask_poc = bucket_ids == "poc_phase1"
    mask_teams_only = mask_teams_flat | mask_live_df
    mask_all = np.ones(len(labels), dtype=bool)

    scope_masks = {
        'all_three': mask_all,
        'teams_only': mask_teams_only,
        'teams_flat': mask_teams_flat,
        'live_deepfake': mask_live_df,
        'poc_phase1': mask_poc,
    }

    results = []
    total_combos = len(thresholds) * len(k_ratios) * len(uncertain_margins)
    combo_count = 0

    for T in thresholds:
        votes = (probs_matrix >= T).sum(axis=1)  # (N,)

        for K_ratio in k_ratios:
            K = int(np.ceil(K_ratio * W))

            for margin in uncertain_margins:
                combo_count += 1
                K_high = K + margin
                K_low = K - 1 - margin

                # Predictions
                pred_fake = votes >= K_high
                pred_real = votes <= K_low
                uncertain = ~pred_fake & ~pred_real

                preds = np.where(pred_fake, 1, 0)  # default real for uncertain too

                row = {
                    'W': W, 'T': T, 'K_ratio': K_ratio, 'K': K,
                    'margin': margin, 'K_high': K_high, 'K_low': K_low,
                }

                # Compute metrics for each scope
                for scope_name, mask in scope_masks.items():
                    if mask.sum() == 0:
                        # No windows in this scope → NaN metrics
                        for m in ['balanced_acc', 'accuracy', 'clean_accuracy',
                                  'tpr', 'tnr', 'fpr', 'fnr', 'uncertain_rate',
                                  'n_total', 'n_fake', 'n_real', 'n_uncertain',
                                  'tp', 'tn', 'fp', 'fn']:
                            row[f'{scope_name}_{m}'] = float('nan') if m != 'n_total' else 0
                        continue

                    m = compute_metrics_vec(
                        labels[mask],
                        preds[mask],
                        uncertain[mask]
                    )
                    for metric_name, val in m.items():
                        row[f'{scope_name}_{metric_name}'] = val

                results.append(row)

    return results


# ---------------------------------------------------------------------------
# Per-method breakdown (poc-phase-1 only)
# ---------------------------------------------------------------------------
def per_method_breakdown(windows_by_bucket, W, T, K_ratio, margin=0):
    """Detailed per-method metrics for poc-phase-1 at a specific strategy."""
    poc_windows = windows_by_bucket.get("poc_phase1", [])
    if not poc_windows:
        return []

    by_method = defaultdict(list)
    for w in poc_windows:
        by_method[w['method']].append(w)

    rows = []
    for method in sorted(by_method.keys()):
        windows = by_method[method]
        probs_matrix = np.stack([w['probs'] for w in windows])
        labels = np.array([w['label'] for w in windows])

        votes = (probs_matrix >= T).sum(axis=1)
        K = int(np.ceil(K_ratio * W))
        K_high = K + margin
        K_low = K - 1 - margin
        pred_fake = votes >= K_high
        pred_real = votes <= K_low
        uncertain = ~pred_fake & ~pred_real
        preds = np.where(pred_fake, 1, 0)

        m = compute_metrics_vec(labels, preds, uncertain)
        m['method'] = method
        m['W'] = W
        m['T'] = T
        m['K_ratio'] = K_ratio
        m['K'] = K
        m['margin'] = margin
        m['n_windows'] = len(windows)
        rows.append(m)

    return rows


# ---------------------------------------------------------------------------
# Stage 1: Coarse grid
# ---------------------------------------------------------------------------
def stage1(raw_data, output_dir):
    log.info("=" * 60)
    log.info("STAGE 1: Coarse Grid Search")
    log.info("=" * 60)

    all_results = []

    for W in WINDOW_SIZES:
        log.info(f"Building windows W={W}...")
        windows_by_bucket = {}
        for bname, rows in raw_data.items():
            windows = build_windows(rows, bname, W)
            windows_by_bucket[bname] = windows
            n_fake = sum(1 for w in windows if w['label'] == 1)
            n_real = sum(1 for w in windows if w['label'] == 0)
            log.info(f"  {bname}: {len(windows)} windows (fake={n_fake}, real={n_real})")

        total = sum(len(v) for v in windows_by_bucket.values())
        log.info(f"  Total windows for W={W}: {total}")

        log.info(f"Running grid: {len(THRESHOLDS_COARSE)} thresholds × {len(K_RATIOS_COARSE)} k_ratios = {len(THRESHOLDS_COARSE)*len(K_RATIOS_COARSE)} combos")
        t0 = time.time()
        results = run_grid(windows_by_bucket, W, THRESHOLDS_COARSE, K_RATIOS_COARSE)
        dt = time.time() - t0
        log.info(f"  Completed in {dt:.1f}s ({len(results)} rows)")
        all_results.extend(results)

    # Write CSV
    out_path = os.path.join(output_dir, "stage1_coarse_grid.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Stage 1 results: {out_path} ({len(all_results)} rows)")

    # Find best per calibration set
    log.info("\n--- STAGE 1 BEST STRATEGIES ---")
    for cal_set in ['teams_only', 'all_three']:
        key = f'{cal_set}_balanced_acc'
        valid = [r for r in all_results if not np.isnan(r.get(key, float('nan')))]
        if not valid:
            continue
        best = max(valid, key=lambda r: r[key])
        log.info(f"\nBest by {cal_set} balanced_acc:")
        log.info(f"  W={best['W']}, T={best['T']}, K_ratio={best['K_ratio']} (K={best['K']})")
        for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
            ba = best.get(f'{scope}_balanced_acc', float('nan'))
            tpr = best.get(f'{scope}_tpr', float('nan'))
            tnr = best.get(f'{scope}_tnr', float('nan'))
            acc = best.get(f'{scope}_accuracy', float('nan'))
            n = best.get(f'{scope}_n_total', 0)
            log.info(f"  {scope:>15s}: bal_acc={ba:.4f}  tpr={tpr:.4f}  tnr={tnr:.4f}  acc={acc:.4f}  n={n}")

    return all_results


# ---------------------------------------------------------------------------
# Stage 2: Fine grid around best regions
# ---------------------------------------------------------------------------
def stage2(raw_data, stage1_results, output_dir, top_n=5):
    log.info("\n" + "=" * 60)
    log.info("STAGE 2: Fine Grid Search (zoom into top regions)")
    log.info("=" * 60)

    all_results = []

    for cal_set in ['teams_only', 'all_three']:
        key = f'{cal_set}_balanced_acc'
        valid = [r for r in stage1_results if not np.isnan(r.get(key, float('nan')))]
        if not valid:
            continue

        # Get top N combos PER WINDOW SIZE to ensure all W values are explored
        top_combos = []
        seen = set()
        for W in WINDOW_SIZES:
            w_valid = [r for r in valid if r['W'] == W]
            sorted_w = sorted(w_valid, key=lambda r: r[key], reverse=True)
            count = 0
            for r in sorted_w:
                combo = (r['W'], r['T'], r['K_ratio'])
                if combo not in seen:
                    seen.add(combo)
                    top_combos.append(combo)
                    count += 1
                if count >= top_n:
                    break

        log.info(f"\nCalibration: {cal_set} — top {len(top_combos)} combos:")
        for W, T, K_ratio in top_combos:
            log.info(f"  W={W}, T={T:.2f}, K_ratio={K_ratio:.3f}")

        for W, T_center, K_center in top_combos:
            # Fine threshold grid: ±0.06 around T, step 0.01
            T_fine = sorted(set(np.clip(
                np.arange(T_center - 0.06, T_center + 0.061, 0.01), 0.01, 0.99
            ).round(3)))
            # Fine K_ratio grid: ±0.15 around K_ratio, step 0.0625
            K_fine = sorted(set(np.clip(
                np.arange(K_center - 0.15, K_center + 0.151, 0.0625), 0.0625, 0.9375
            ).round(4)))

            log.info(f"  Fine grid around W={W}, T={T_center:.2f}, K={K_center:.3f}: "
                     f"{len(T_fine)} T × {len(K_fine)} K = {len(T_fine)*len(K_fine)} combos")

            windows_by_bucket = {}
            for bname, rows in raw_data.items():
                windows_by_bucket[bname] = build_windows(rows, bname, W)

            results = run_grid(windows_by_bucket, W, T_fine, K_fine)
            for r in results:
                r['calibration_source'] = cal_set
            all_results.extend(results)

    # Write CSV
    out_path = os.path.join(output_dir, "stage2_fine_grid.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Stage 2 results: {out_path} ({len(all_results)} rows)")

    # Find best per calibration set
    log.info("\n--- STAGE 2 BEST STRATEGIES ---")
    for cal_set in ['teams_only', 'all_three']:
        key = f'{cal_set}_balanced_acc'
        subset = [r for r in all_results
                  if r.get('calibration_source') == cal_set
                  and not np.isnan(r.get(key, float('nan')))]
        if not subset:
            continue
        best = max(subset, key=lambda r: r[key])
        log.info(f"\nBest (fine) by {cal_set} balanced_acc:")
        log.info(f"  W={best['W']}, T={best['T']}, K_ratio={best['K_ratio']} (K={best['K']})")
        for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
            ba = best.get(f'{scope}_balanced_acc', float('nan'))
            tpr = best.get(f'{scope}_tpr', float('nan'))
            tnr = best.get(f'{scope}_tnr', float('nan'))
            acc = best.get(f'{scope}_accuracy', float('nan'))
            n = best.get(f'{scope}_n_total', 0)
            log.info(f"  {scope:>15s}: bal_acc={ba:.4f}  tpr={tpr:.4f}  tnr={tnr:.4f}  acc={acc:.4f}  n={n}")

    return all_results


# ---------------------------------------------------------------------------
# Stage 3: Uncertain zone
# ---------------------------------------------------------------------------
def stage3(raw_data, stage2_results, output_dir, top_n=3):
    log.info("\n" + "=" * 60)
    log.info("STAGE 3: Uncertain Zone Optimization (≤5% budget)")
    log.info("=" * 60)

    all_results = []

    for cal_set in ['teams_only', 'all_three']:
        key = f'{cal_set}_balanced_acc'
        # Use stage 2 results if available, else stage 1
        subset = [r for r in stage2_results if not np.isnan(r.get(key, float('nan')))]
        if not subset:
            continue

        # Get top N combos PER WINDOW SIZE
        top_combos = []
        seen = set()
        for W in WINDOW_SIZES:
            w_subset = [r for r in subset if r['W'] == W]
            sorted_w = sorted(w_subset, key=lambda r: r[key], reverse=True)
            count = 0
            for r in sorted_w:
                combo = (r['W'], r['T'], r['K_ratio'])
                if combo not in seen:
                    seen.add(combo)
                    top_combos.append(combo)
                    count += 1
                if count >= top_n:
                    break

        log.info(f"\nCalibration: {cal_set} — top {len(top_combos)} combos for uncertain tuning:")

        for W, T, K_ratio in top_combos:
            log.info(f"\n  Strategy: W={W}, T={T:.3f}, K_ratio={K_ratio:.4f}")

            windows_by_bucket = {}
            for bname, rows in raw_data.items():
                windows_by_bucket[bname] = build_windows(rows, bname, W)

            # Sweep margin from 0 upward
            max_margin = W // 4  # reasonable upper bound
            margins = list(range(0, max_margin + 1))
            results = run_grid(windows_by_bucket, W, [T], [K_ratio], uncertain_margins=margins)

            for r in results:
                r['calibration_source'] = cal_set

            # Filter by uncertain budget on the calibration scope
            for r in results:
                ur = r.get(f'{cal_set}_uncertain_rate', 0)
                r['within_budget'] = 1 if ur <= UNCERTAIN_BUDGET else 0

            # Log results
            for r in results:
                m = r['margin']
                ur_t = r.get(f'{cal_set}_uncertain_rate', 0)
                ca_t = r.get(f'{cal_set}_clean_accuracy', float('nan'))
                ba_t = r.get(f'{cal_set}_balanced_acc', float('nan'))
                budget_ok = "✓" if r['within_budget'] else "✗"
                log.info(f"    margin={m}: uncertain={ur_t:.3f} {budget_ok}  "
                         f"clean_acc={ca_t:.4f}  bal_acc={ba_t:.4f}")

            all_results.extend(results)

    # Write CSV
    out_path = os.path.join(output_dir, "stage3_uncertain.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Stage 3 results: {out_path} ({len(all_results)} rows)")

    # Find best within budget
    log.info("\n--- STAGE 3 BEST UNCERTAIN STRATEGIES ---")
    for cal_set in ['teams_only', 'all_three']:
        within_budget = [r for r in all_results
                         if r.get('calibration_source') == cal_set
                         and r.get('within_budget', 0) == 1]
        if not within_budget:
            continue
        key_clean = f'{cal_set}_clean_accuracy'
        best = max(within_budget, key=lambda r: r.get(key_clean, 0))
        log.info(f"\nBest (uncertain ≤5%) by {cal_set} clean_accuracy:")
        log.info(f"  W={best['W']}, T={best['T']}, K_ratio={best['K_ratio']} "
                 f"(K={best['K']}), margin={best['margin']} "
                 f"(K_high={best['K_high']}, K_low={best['K_low']})")
        for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
            ca = best.get(f'{scope}_clean_accuracy', float('nan'))
            ba = best.get(f'{scope}_balanced_acc', float('nan'))
            ur = best.get(f'{scope}_uncertain_rate', 0)
            tpr = best.get(f'{scope}_tpr', float('nan'))
            tnr = best.get(f'{scope}_tnr', float('nan'))
            n = best.get(f'{scope}_n_total', 0)
            log.info(f"  {scope:>15s}: clean_acc={ca:.4f}  bal_acc={ba:.4f}  "
                     f"tpr={tpr:.4f}  tnr={tnr:.4f}  uncertain={ur:.3f}  n={n}")

    return all_results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def write_summary(raw_data, stage1_results, stage2_results, stage3_results, output_dir):
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)

    summary_rows = []

    # For each calibration set, pick:
    #   a) best binary strategy (stage 2 or stage 1)
    #   b) best uncertain strategy (stage 3)
    for cal_set in ['teams_only', 'all_three']:
        # Best binary
        key = f'{cal_set}_balanced_acc'
        all_binary = stage2_results if stage2_results else stage1_results
        valid = [r for r in all_binary if not np.isnan(r.get(key, float('nan')))]
        if valid:
            best_binary = max(valid, key=lambda r: r[key])
            best_binary['strategy_type'] = 'binary'
            best_binary['selected_by'] = cal_set
            summary_rows.append(best_binary)

        # Best uncertain
        within_budget = [r for r in stage3_results
                         if r.get('calibration_source') == cal_set
                         and r.get('within_budget', 0) == 1]
        if within_budget:
            key_clean = f'{cal_set}_clean_accuracy'
            best_unc = max(within_budget, key=lambda r: r.get(key_clean, 0))
            best_unc['strategy_type'] = 'uncertain'
            best_unc['selected_by'] = cal_set
            summary_rows.append(best_unc)

    # Write summary CSV
    out_path = os.path.join(output_dir, "best_strategies_summary.csv")
    if summary_rows:
        # Merge all fieldnames across rows (stage2 and stage3 rows may differ)
        all_fields = []
        seen_fields = set()
        for row in summary_rows:
            for k in row.keys():
                if k not in seen_fields:
                    all_fields.append(k)
                    seen_fields.add(k)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary_rows)
        log.info(f"Summary: {out_path}")

    # Per-method breakdown for best strategies
    for row in summary_rows:
        W, T, K_ratio, margin = row['W'], row['T'], row['K_ratio'], row.get('margin', 0)
        stype = row.get('strategy_type', 'binary')
        cal = row.get('selected_by', '?')

        log.info(f"\n--- Per-method (poc_phase1) | {stype} | calibrated on {cal} ---")
        log.info(f"    W={W}, T={T}, K_ratio={K_ratio}, margin={margin}")

        windows_by_bucket = {}
        for bname, rows in raw_data.items():
            windows_by_bucket[bname] = build_windows(rows, bname, W)

        method_rows = per_method_breakdown(windows_by_bucket, W, T, K_ratio, margin)
        for mr in method_rows:
            log.info(f"    {mr['method']:>18s}: n={mr['n_windows']:>4d}  "
                     f"bal_acc={mr['balanced_acc']:.4f}  tpr={mr['tpr']:.4f}  "
                     f"tnr={mr['tnr']:.4f}  uncertain={mr['uncertain_rate']:.3f}")

    # Also write per-method CSV
    all_method_rows = []
    for row in summary_rows:
        W, T, K_ratio, margin = row['W'], row['T'], row['K_ratio'], row.get('margin', 0)
        windows_by_bucket = {}
        for bname, rows_data in raw_data.items():
            windows_by_bucket[bname] = build_windows(rows_data, bname, W)
        method_rows = per_method_breakdown(windows_by_bucket, W, T, K_ratio, margin)
        for mr in method_rows:
            mr['strategy_type'] = row.get('strategy_type', 'binary')
            mr['selected_by'] = row.get('selected_by', '?')
            all_method_rows.append(mr)

    if all_method_rows:
        out_path = os.path.join(output_dir, "best_strategies_per_method.csv")
        fieldnames = list(all_method_rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_method_rows)
        log.info(f"Per-method detail: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Strategy Grid Search")
    parser.add_argument("--stage", default="all", help="Stage to run: 1, 2, 3, or all")
    parser.add_argument("--output_dir", default="./strategy_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading inference CSVs...")
    raw_data = {}
    for bname, path in CSV_FILES.items():
        raw_data[bname] = load_csv(path)
        log.info(f"  {bname}: {len(raw_data[bname])} frames")

    stage1_results = []
    stage2_results = []
    stage3_results = []

    run_stage1 = args.stage in ('1', 'all')
    run_stage2 = args.stage in ('2', 'all')
    run_stage3 = args.stage in ('3', 'all')

    t_total = time.time()

    if run_stage1:
        stage1_results = stage1(raw_data, args.output_dir)

    if run_stage2:
        if not stage1_results:
            # Try loading from file
            s1_path = os.path.join(args.output_dir, "stage1_coarse_grid.csv")
            if os.path.exists(s1_path):
                log.info("Loading stage 1 results from file...")
                with open(s1_path) as f:
                    reader = csv.DictReader(f)
                    stage1_results = []
                    for r in reader:
                        for k in r:
                            try:
                                r[k] = float(r[k])
                                if r[k] == int(r[k]):
                                    r[k] = int(r[k])
                            except (ValueError, OverflowError):
                                pass
                        stage1_results.append(r)
        stage2_results = stage2(raw_data, stage1_results, args.output_dir)

    if run_stage3:
        # Use stage2 results if available, else stage1
        base_results = stage2_results if stage2_results else stage1_results
        if not base_results:
            # Try loading stage 2
            s2_path = os.path.join(args.output_dir, "stage2_fine_grid.csv")
            if os.path.exists(s2_path):
                log.info("Loading stage 2 results from file...")
                with open(s2_path) as f:
                    reader = csv.DictReader(f)
                    base_results = []
                    for r in reader:
                        for k in r:
                            try:
                                r[k] = float(r[k])
                                if r[k] == int(r[k]):
                                    r[k] = int(r[k])
                            except (ValueError, OverflowError):
                                pass
                        base_results.append(r)
        stage3_results = stage3(raw_data, base_results, args.output_dir)

    # Final summary
    write_summary(raw_data, stage1_results, stage2_results, stage3_results, args.output_dir)

    dt_total = time.time() - t_total
    log.info(f"\nTotal runtime: {dt_total:.1f}s")
    log.info("Done.")


if __name__ == "__main__":
    main()
