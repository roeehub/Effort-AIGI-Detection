import pandas as pd
import numpy as np

# --- Main Analysis ---

print("Loading fusion_oof_scores.csv...")
df = pd.read_csv("out_full/fusion_oof_scores.csv")

# --- Define the thresholds from the "Supported Methods" strategy ---
T_low = 0.964434
T_high = 0.983175

print(f"\nUsing thresholds optimized for SUPPORTED methods:")
print(f"  - T_low:  {T_low}")
print(f"  - T_high: {T_high}")

# --- Isolate the "Unsupported" fake videos ---
unsupported_methods = ['veo3_creations', 'sadtalker', 'deepfakedetection']
unsupported_fakes_df = df[df['method'].isin(unsupported_methods)].copy()

print(f"\nTesting these {len(unsupported_fakes_df)} videos from unsupported methods...")

# --- Apply the thresholds and calculate the outcome ---
# Note: Since these are all fakes (label=1), we are calculating TPR.

# How many are correctly classified as FAKE?
count_fake = (unsupported_fakes_df['noisy_or'] > T_high).sum()

# How many fall into the UNCERTAIN band?
count_uncertain = (
        (unsupported_fakes_df['noisy_or'] >= T_low) & (unsupported_fakes_df['noisy_or'] <= T_high)
).sum()

# How many are incorrectly classified as REAL?
count_real = (unsupported_fakes_df['noisy_or'] < T_low).sum()

total_count = len(unsupported_fakes_df)

# --- Report the Results ---
print("\n--- Performance on UNSUPPORTED Fake Methods ---")
print("If an unsupported fake appears, this is how it will be classified:")
print(f"\n  - Classified as FAKE:      {count_fake / total_count:.2%}  (This is your incidental TPR)")
print(f"  - Classified as UNCERTAIN: {count_uncertain / total_count:.2%}")
print(f"  - Classified as REAL (Miss): {count_real / total_count:.2%}")

# Optional: Per-method breakdown
print("\n--- Breakdown by Method ---")
for method in unsupported_methods:
    method_df = unsupported_fakes_df[unsupported_fakes_df['method'] == method]
    total_method_count = len(method_df)

    tpr = (method_df['noisy_or'] > T_high).mean()
    pct_uncertain = ((method_df['noisy_or'] >= T_low) & (method_df['noisy_or'] <= T_high)).mean()
    pct_miss = (method_df['noisy_or'] < T_low).mean()

    print(f"\nMethod: {method} ({total_method_count} videos)")
    print(f"  - Caught as FAKE:      {tpr:.1%}")
    print(f"  - Flagged as UNCERTAIN: {pct_uncertain:.1%}")
    print(f"  - Missed as REAL:      {pct_miss:.1%}")
