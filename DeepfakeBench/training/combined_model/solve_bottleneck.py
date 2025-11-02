import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


def youden_threshold(y_true, y_score):
    """Calculates the Youden's J statistic and the optimal threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    j_opt_idx = np.argmax(j_scores)
    return thr[j_opt_idx]


# --- Main Analysis ---

# 1. Load the full dataset of OOF scores
print("Loading fusion_oof_scores.csv...")
df = pd.read_csv("out_full/fusion_oof_scores.csv")
print(f"Loaded {len(df)} total video scores.")

# 2. Establish the new baseline by removing all faceforensics++ data
print("\n--- Establishing New Baseline (without faceforensics++) ---")
baseline_df = df[df['method'] != 'faceforensics++'].copy()
reals_baseline = baseline_df[baseline_df['label'] == 0]
fakes_baseline = baseline_df[baseline_df['label'] == 1]

print(f"Baseline now has {len(reals_baseline)} real videos and {len(fakes_baseline)} fake videos.")

# 3. Calculate the new operating point thresholds for this baseline
# Threshold for <1% FPR
threshold_1pct = reals_baseline['noisy_or'].quantile(0.99)
# Threshold for general performance (Youden's J)
threshold_youden = youden_threshold(baseline_df['label'], baseline_df['noisy_or'])

baseline_recall_1pct = (fakes_baseline['noisy_or'] >= threshold_1pct).mean()
baseline_recall_youden = (fakes_baseline['noisy_or'] >= threshold_youden).mean()

print(f"\nNew Baseline Threshold (<1% FPR): {threshold_1pct:.6f}")
print(f"New Baseline Recall (<1% FPR):   {baseline_recall_1pct:.2%}")
print(f"\nNew Baseline Threshold (Youden):  {threshold_youden:.6f}")
print(f"New Baseline Recall (Youden):    {baseline_recall_youden:.2%}")

# 4. Simulate removing each fake method one-by-one
print("\n--- Simulating Removal of Each Fake Method ---")
impact_results = []
unique_fake_methods = fakes_baseline['method'].unique()

for method_to_remove in unique_fake_methods:
    # Create a temporary subset of fakes excluding the current method
    fakes_subset = fakes_baseline[fakes_baseline['method'] != method_to_remove]

    # Calculate recall on this subset using the FIXED baseline thresholds
    recall_after_removal_1pct = (fakes_subset['noisy_or'] >= threshold_1pct).mean()
    recall_after_removal_youden = (fakes_subset['noisy_or'] >= threshold_youden).mean()

    # The individual TPR of the method we are considering removing
    method_tpr_1pct = (
            fakes_baseline[fakes_baseline['method'] == method_to_remove]['noisy_or'] >= threshold_1pct).mean()
    method_tpr_youden = (
            fakes_baseline[fakes_baseline['method'] == method_to_remove]['noisy_or'] >= threshold_youden).mean()

    impact_results.append({
        'method_removed': method_to_remove,
        'recall_gain_1pct': recall_after_removal_1pct - baseline_recall_1pct,
        'recall_gain_youden': recall_after_removal_youden - baseline_recall_youden,
        'individual_tpr_1pct': method_tpr_1pct,
        'individual_tpr_youden': method_tpr_youden,
        'count': len(fakes_baseline[fakes_baseline['method'] == method_to_remove])
    })

# 5. Analyze and display the results
results_df = pd.DataFrame(impact_results)

# Sort by impact at the <1% FPR threshold
top_bottlenecks_1pct = results_df.sort_values(by='recall_gain_1pct', ascending=False)
print("\n\n--- TOP 5 BOTTLENECKS for Recall at <1% FPR ---")
print("(Removing these methods provides the biggest boost to the strict <1% FPR metric)")
print(top_bottlenecks_1pct[['method_removed', 'recall_gain_1pct', 'individual_tpr_1pct', 'count']].head(5).to_string(
    index=False))

# Sort by impact at the Youden threshold
top_bottlenecks_youden = results_df.sort_values(by='recall_gain_youden', ascending=False)
print("\n\n--- TOP 5 BOTTLENECKS for General Recall (Youden) ---")
print("(Removing these methods provides the biggest boost to the balanced metric)")
print(top_bottlenecks_youden[['method_removed', 'recall_gain_youden', 'individual_tpr_youden', 'count']].head(
    5).to_string(index=False))
