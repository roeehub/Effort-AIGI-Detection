import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


def youden_threshold(y_true, y_score):
    """Calculates the Youden's J statistic and the optimal threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    # Add a small epsilon to TPR to handle the case where TPR and FPR are both 0
    j_scores = tpr - fpr + 1e-9
    j_opt_idx = np.argmax(j_scores)
    return thr[j_opt_idx]


# --- Main Analysis ---

print("Loading fusion_oof_scores.csv...")
df = pd.read_csv("out_full/fusion_oof_scores.csv")

# --- Establish the Baseline (without faceforensics++) ---
print("\n--- Establishing New Baseline (without faceforensics++) ---")
baseline_df = df[df['method'] != 'faceforensics++'].copy()
reals_baseline = baseline_df[baseline_df['label'] == 0]
fakes_baseline = baseline_df[baseline_df['label'] == 1]

# Baseline thresholds and recalls
threshold_1pct_baseline = reals_baseline['noisy_or'].quantile(0.99)
recall_1pct_baseline = (fakes_baseline['noisy_or'] >= threshold_1pct_baseline).mean()

threshold_youden_baseline = youden_threshold(baseline_df['label'], baseline_df['noisy_or'])
recall_youden_baseline = (fakes_baseline['noisy_or'] >= threshold_youden_baseline).mean()

print(f"Baseline Recall (<1% FPR):   {recall_1pct_baseline:.2%}")
print(f"Baseline Recall (Youden):    {recall_youden_baseline:.2%}")

# --- Create the "Improved" Dataset by removing top 3 bottlenecks ---
print("\n--- Simulating Performance after Removing Top 3 Bottlenecks ---")
bottlenecks_to_remove = ['veo3_creations', 'sadtalker', 'deepfakedetection']
improved_df = baseline_df[~baseline_df['method'].isin(bottlenecks_to_remove)].copy()

# The real videos are the same as the baseline
reals_improved = improved_df[improved_df['label'] == 0]
# The fake videos are now a subset
fakes_improved = improved_df[improved_df['label'] == 1]

print(f"Improved set has {len(reals_improved)} real videos and {len(fakes_improved)} fake videos.")

# --- Scenario 1: Strict <1% FPR ---
# The threshold is determined by the 99th percentile of REAL videos.
# Since the real videos haven't changed, the threshold remains the same.
new_threshold_1pct = reals_improved['noisy_or'].quantile(0.99)
new_recall_1pct = (fakes_improved['noisy_or'] >= new_threshold_1pct).mean()

print(f"\n--- Analysis for <1% FPR Scenario ---")
print(f"Old Threshold: {threshold_1pct_baseline:.6f}")
print(f"New Threshold: {new_threshold_1pct:.6f} (No change as expected)")
print(f"New Recall at <1% FPR: {new_recall_1pct:.2%}")

# --- Scenario 2: General Performance (Youden's J) ---
# Here, the threshold MIGHT change because the distribution of fakes has changed.
new_threshold_youden = youden_threshold(improved_df['label'], improved_df['noisy_or'])
new_recall_youden = (fakes_improved['noisy_or'] >= new_threshold_youden).mean()

# Let's also see what the recall would have been with the OLD threshold
recall_with_old_youden_thr = (fakes_improved['noisy_or'] >= threshold_youden_baseline).mean()

print(f"\n--- Analysis for Youden Scenario ---")
print(f"Old Threshold: {threshold_youden_baseline:.6f}")
print(f"New Threshold: {new_threshold_youden:.6f}")
print(f"Recall with OLD threshold: {recall_with_old_youden_thr:.2%}")
print(f"New Recall with NEW (re-optimized) threshold: {new_recall_youden:.2%}")
