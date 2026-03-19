#!/usr/bin/env python3
"""Quick analysis of stage 1 grid search results — look at W=16 specifically and overall patterns."""
import csv
import numpy as np
from collections import defaultdict

with open("strategy_results/stage1_coarse_grid.csv") as f:
    rows = []
    for r in csv.DictReader(f):
        for k in r:
            try:
                r[k] = float(r[k])
                if r[k] == int(r[k]) and abs(r[k]) < 1e9:
                    r[k] = int(r[k])
            except (ValueError, OverflowError):
                pass
        rows.append(r)

print("=" * 80)
print("TOP 10 by teams_only balanced_acc (all W sizes)")
print("=" * 80)
sorted_teams = sorted(rows, key=lambda r: r.get('teams_only_balanced_acc', 0), reverse=True)
print(f"{'W':>3} {'T':>5} {'K_ratio':>7} {'K':>3} | {'teams_ba':>8} {'teams_tpr':>9} {'teams_tnr':>9} | {'all3_ba':>8} {'poc_ba':>7} {'live_ba':>7} {'flat_ba':>7}")
for r in sorted_teams[:10]:
    print(f"{r['W']:>3} {r['T']:>5.2f} {r['K_ratio']:>7.3f} {r['K']:>3} | "
          f"{r.get('teams_only_balanced_acc',0):>8.4f} {r.get('teams_only_tpr',0):>9.4f} {r.get('teams_only_tnr',0):>9.4f} | "
          f"{r.get('all_three_balanced_acc',0):>8.4f} {r.get('poc_phase1_balanced_acc',0):>7.4f} "
          f"{r.get('live_deepfake_balanced_acc',0):>7.4f} {r.get('teams_flat_balanced_acc',0):>7.4f}")

print("\n" + "=" * 80)
print("TOP 10 by teams_only balanced_acc — W=16 ONLY")
print("=" * 80)
w16 = [r for r in rows if r['W'] == 16]
sorted_w16 = sorted(w16, key=lambda r: r.get('teams_only_balanced_acc', 0), reverse=True)
print(f"{'W':>3} {'T':>5} {'K_ratio':>7} {'K':>3} | {'teams_ba':>8} {'teams_tpr':>9} {'teams_tnr':>9} | {'all3_ba':>8} {'poc_ba':>7} {'live_ba':>7}")
for r in sorted_w16[:10]:
    print(f"{r['W']:>3} {r['T']:>5.2f} {r['K_ratio']:>7.3f} {r['K']:>3} | "
          f"{r.get('teams_only_balanced_acc',0):>8.4f} {r.get('teams_only_tpr',0):>9.4f} {r.get('teams_only_tnr',0):>9.4f} | "
          f"{r.get('all_three_balanced_acc',0):>8.4f} {r.get('poc_phase1_balanced_acc',0):>7.4f} "
          f"{r.get('live_deepfake_balanced_acc',0):>7.4f}")

print("\n" + "=" * 80)
print("TOP 10 by all_three balanced_acc (all W sizes)")
print("=" * 80)
sorted_all = sorted(rows, key=lambda r: r.get('all_three_balanced_acc', 0), reverse=True)
for r in sorted_all[:10]:
    print(f"{r['W']:>3} {r['T']:>5.2f} {r['K_ratio']:>7.3f} {r['K']:>3} | "
          f"{r.get('teams_only_balanced_acc',0):>8.4f} {r.get('teams_only_tpr',0):>9.4f} {r.get('teams_only_tnr',0):>9.4f} | "
          f"{r.get('all_three_balanced_acc',0):>8.4f} {r.get('poc_phase1_balanced_acc',0):>7.4f} "
          f"{r.get('live_deepfake_balanced_acc',0):>7.4f}")

# Check the user's previous strategy: W=16, T=0.82, K=8
print("\n" + "=" * 80)
print("USER's PREVIOUS STRATEGY: W=16, T=0.82, K/W=0.5")
print("=" * 80)
user_strat = [r for r in w16 if r['T'] == 0.82 and r['K_ratio'] == 0.5]
if user_strat:
    r = user_strat[0]
    print(f"teams_only: bal_acc={r.get('teams_only_balanced_acc',0):.4f} tpr={r.get('teams_only_tpr',0):.4f} tnr={r.get('teams_only_tnr',0):.4f}")
    print(f"all_three:  bal_acc={r.get('all_three_balanced_acc',0):.4f} tpr={r.get('all_three_tpr',0):.4f} tnr={r.get('all_three_tnr',0):.4f}")
    print(f"poc_phase1: bal_acc={r.get('poc_phase1_balanced_acc',0):.4f} tpr={r.get('poc_phase1_tpr',0):.4f} tnr={r.get('poc_phase1_tnr',0):.4f}")
    print(f"live_df:    bal_acc={r.get('live_deepfake_balanced_acc',0):.4f} tpr={r.get('live_deepfake_tpr',0):.4f} tnr={r.get('live_deepfake_tnr',0):.4f}")
    print(f"teams_flat: bal_acc={r.get('teams_flat_balanced_acc',0):.4f} tpr={r.get('teams_flat_tpr',0):.4f} tnr={r.get('teams_flat_tnr',0):.4f}")

# Data balance analysis per W
print("\n" + "=" * 80)
print("DATA BALANCE per W (fake/real windows)")
print("=" * 80)
for W in [16, 24, 32]:
    subset = [r for r in rows if r['W'] == W]
    if subset:
        r = subset[0]
        for scope in ['teams_flat', 'live_deepfake', 'poc_phase1', 'teams_only', 'all_three']:
            nf = r.get(f'{scope}_n_fake', 0)
            nr = r.get(f'{scope}_n_real', 0)
            nt = r.get(f'{scope}_n_total', 0)
            pct_fake = nf / nt * 100 if nt > 0 else 0
            print(f"  W={W} {scope:>15s}: total={nt:>5} fake={nf:>5} real={nr:>5} ({pct_fake:.1f}% fake)")
        print()
