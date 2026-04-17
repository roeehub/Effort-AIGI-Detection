import pandas as pd

# Load B16 grid search results
df = pd.read_csv('B16_grid_search_results.csv')

# Find thresholds where real accuracy >= 95%
df_95 = df[df['real_accuracy'] >= 0.95].sort_values('fake_accuracy', ascending=False)

print('B16: Thresholds achieving >= 95% Real Accuracy')
print('='*70)
print(f"{'Prob':>6} {'Vote':>6} {'Real Acc':>10} {'Fake Acc':>10} {'Real FP':>8}")
print('-'*70)
for _, row in df_95.iterrows():
    print(f"{row['prob_threshold']:>6} {row['voting_fraction']:>6} {row['real_accuracy']:>10.2%} {row['fake_accuracy']:>10.2%} {int(row['real_fp_count']):>8}")

# Also show near-95% options
print()
print('Near 95% options (94-95%):')
print('-'*70)
df_near = df[(df['real_accuracy'] >= 0.94) & (df['real_accuracy'] < 0.95)].sort_values('fake_accuracy', ascending=False)
for _, row in df_near.iterrows():
    print(f"{row['prob_threshold']:>6} {row['voting_fraction']:>6} {row['real_accuracy']:>10.2%} {row['fake_accuracy']:>10.2%} {int(row['real_fp_count']):>8}")

# Best trade-off
print()
print('='*70)
print('RECOMMENDATION for ~95% Real Accuracy:')
if len(df_95) > 0:
    best = df_95.iloc[0]
    print(f"  Settings: prob_threshold={best['prob_threshold']}, voting={best['voting_fraction']}")
    print(f"  Real Accuracy: {best['real_accuracy']:.2%} ({int(best['real_fp_count'])} false positives)")
    print(f"  Fake Accuracy: {best['fake_accuracy']:.2%}")
    print(f"  Drop from best fake acc (91.41%): {91.41 - best['fake_accuracy']*100:.1f} percentage points")
