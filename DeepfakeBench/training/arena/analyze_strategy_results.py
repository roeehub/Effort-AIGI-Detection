#!/usr/bin/env python3
"""
Comprehensive analysis of strategy grid search results.
Produces a clean, readable report with best strategies per window size,
data balance context, and per-method breakdowns.
"""
import csv
import os
from collections import defaultdict

RESULTS_DIR = "./strategy_results"

def load_csv(path):
    """Load CSV and convert numeric fields."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    fval = float(v)
                    if fval == fval and fval == int(fval):  # not NaN and integer-valued
                        row[k] = int(fval)
                    else:
                        row[k] = fval
                except (ValueError, TypeError, OverflowError):
                    row[k] = v
            rows.append(row)
    return rows


def fmt(val, pct=True):
    """Format a numeric value."""
    if val is None or val == '':
        return "  n/a "
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "  n/a "
    if v != v:  # NaN
        return "  n/a "
    if pct:
        return f"{v*100:6.2f}%"
    return f"{v:6.3f}"


def fmt_int(val):
    return f"{int(val):>5d}"


def print_strategy_header():
    print(f"{'W':>3s} {'T':>6s} {'K':>3s} {'K/W':>5s} {'margin':>6s}")


def print_strategy(r, show_margin=False):
    margin_str = ""
    if show_margin and 'margin' in r and r.get('margin', 0) > 0:
        margin_str = f"  margin={int(r['margin'])} (K_hi={int(r.get('K_high', 0))}, K_lo={int(r.get('K_low', 0))})"
    print(f"  W={int(r['W']):>2d}, T={r['T']:.3f}, K={int(r['K']):>2d} (K/W={r['K_ratio']:.3f}){margin_str}")


def print_scope_metrics(r, scope, show_clean=False):
    """Print metrics for a given calibration scope."""
    prefix = f"{scope}_"
    bal = r.get(f"{prefix}balanced_acc")
    acc = r.get(f"{prefix}accuracy")
    tpr = r.get(f"{prefix}tpr")
    tnr = r.get(f"{prefix}tnr")
    fpr = r.get(f"{prefix}fpr")
    unc = r.get(f"{prefix}uncertain_rate", 0)
    n = r.get(f"{prefix}n_total", 0)
    n_fake = r.get(f"{prefix}n_fake", 0)
    n_real = r.get(f"{prefix}n_real", 0)
    n_unc = r.get(f"{prefix}n_uncertain", 0)

    clean_str = ""
    if show_clean and unc > 0:
        clean = r.get(f"{prefix}clean_accuracy")
        clean_str = f"  clean_acc={fmt(clean)}"

    bal_str = fmt(bal)
    print(f"    {scope:>20s}: bal_acc={bal_str}  acc={fmt(acc)}  TPR={fmt(tpr)}  TNR={fmt(tnr)}  FPR={fmt(fpr)}  uncertain={fmt(unc)}{clean_str}  (n={int(n)}, {int(n_fake)}F/{int(n_real)}R)")


def find_best(rows, scope, metric_suffix="balanced_acc", W_filter=None, max_uncertain=None):
    """Find best row by metric in a scope, optionally filtered."""
    key = f"{scope}_{metric_suffix}"
    candidates = rows
    if W_filter is not None:
        candidates = [r for r in candidates if int(r['W']) == W_filter]
    if max_uncertain is not None:
        unc_key = f"{scope}_uncertain_rate"
        candidates = [r for r in candidates if r.get(unc_key, 0) <= max_uncertain]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get(key, -999))


def main():
    # Load all stages
    stage1 = load_csv(os.path.join(RESULTS_DIR, "stage1_coarse_grid.csv"))
    stage2 = load_csv(os.path.join(RESULTS_DIR, "stage2_fine_grid.csv"))
    stage3 = load_csv(os.path.join(RESULTS_DIR, "stage3_uncertain.csv"))
    per_method = load_csv(os.path.join(RESULTS_DIR, "best_strategies_per_method.csv"))

    # Combine stage1 + stage2 for "binary" results (stage3 adds uncertain)
    all_binary = stage1 + stage2
    all_rows = all_binary + stage3

    print("=" * 100)
    print("COMPREHENSIVE STRATEGY GRID SEARCH RESULTS")
    print("=" * 100)
    print(f"\nEvaluated: {len(stage1)} coarse + {len(stage2)} fine + {len(stage3)} uncertain = {len(all_rows)} total combos")
    print()

    # ═══════════════════════════════════════════════════════════
    # 1. DATA BALANCE — context for interpreting results
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("1. DATA BALANCE (windows per scope per W)")
    print("─" * 100)
    # Use stage1 to get counts per W (coarse grid has all W)
    for W in [16, 24, 32]:
        sample = next((r for r in stage1 if int(r['W']) == W), None)
        if not sample:
            continue
        print(f"\n  W = {W}:")
        for scope in ['teams_flat', 'live_deepfake', 'poc_phase1', 'teams_only', 'all_three']:
            n = int(sample.get(f"{scope}_n_total", 0))
            nf = int(sample.get(f"{scope}_n_fake", 0))
            nr = int(sample.get(f"{scope}_n_real", 0))
            if n > 0:
                pct_fake = nf / n * 100
                print(f"    {scope:>20s}: {n:>5d} windows  ({nf:>4d} fake / {nr:>4d} real = {pct_fake:.1f}% fake)")

    print()

    # ═══════════════════════════════════════════════════════════
    # 2. BEST BINARY STRATEGIES PER WINDOW SIZE
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("2. BEST BINARY STRATEGIES (no uncertain zone) — per window size")
    print("─" * 100)

    for cal_scope in ['teams_only', 'all_three']:
        print(f"\n  Calibrated on: {cal_scope.upper()}")
        print(f"  {'─'*90}")
        for W in [16, 24, 32]:
            best = find_best(all_binary, cal_scope, "balanced_acc", W_filter=W)
            if not best:
                print(f"  W={W}: no results")
                continue
            print()
            print_strategy(best)
            for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
                print_scope_metrics(best, scope)

    print()

    # ═══════════════════════════════════════════════════════════
    # 3. BEST UNCERTAIN STRATEGIES (≤5% uncertain budget)
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("3. BEST STRATEGIES WITH UNCERTAIN ZONE (≤5% budget) — per window size")
    print("─" * 100)

    for cal_scope in ['teams_only', 'all_three']:
        print(f"\n  Calibrated on: {cal_scope.upper()}")
        print(f"  {'─'*90}")
        for W in [16, 24, 32]:
            best = find_best(stage3, cal_scope, "clean_accuracy", W_filter=W, max_uncertain=0.05)
            if not best:
                # Fallback: any uncertain result for this W
                best = find_best(stage3, cal_scope, "clean_accuracy", W_filter=W, max_uncertain=1.0)
                if best:
                    unc = best.get(f"{cal_scope}_uncertain_rate", 0)
                    print(f"\n  W={W}: NONE within 5% budget (best has {unc*100:.1f}% uncertain)")
                    print_strategy(best, show_margin=True)
                    for scope in ['teams_only', 'all_three']:
                        print_scope_metrics(best, scope, show_clean=True)
                else:
                    print(f"\n  W={W}: no uncertain results")
                continue

            unc = best.get(f"{cal_scope}_uncertain_rate", 0)
            print(f"\n  W={W}: uncertain={unc*100:.1f}%")
            print_strategy(best, show_margin=True)
            for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
                print_scope_metrics(best, scope, show_clean=True)

    print()

    # ═══════════════════════════════════════════════════════════
    # 4. BASELINE COMPARISON — user's previous strategy
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("4. BASELINE COMPARISON — Previous Strategy (W=16, T=0.82, K=8)")
    print("─" * 100)

    # Find the closest match in stage1 (T=0.82 might not be exact, find ~0.80)
    baseline = None
    for r in all_binary:
        if int(r['W']) == 16 and abs(r['T'] - 0.82) < 0.03 and int(r['K']) == 8:
            baseline = r
            break

    if not baseline:
        # Interpolate from coarse grid
        for r in all_binary:
            if int(r['W']) == 16 and abs(r['T'] - 0.80) < 0.06 and int(r['K']) == 8:
                if baseline is None or abs(r['T'] - 0.82) < abs(baseline['T'] - 0.82):
                    baseline = r

    if baseline:
        print(f"\n  Found closest match: W={int(baseline['W'])}, T={baseline['T']:.3f}, K={int(baseline['K'])}")
        for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
            print_scope_metrics(baseline, scope)

        # Compare with best W=16
        print("\n  vs. Best W=16 (binary, calibrated on teams_only):")
        best_w16 = find_best(all_binary, 'teams_only', "balanced_acc", W_filter=16)
        if best_w16:
            print_strategy(best_w16)
            for scope in ['teams_only', 'all_three', 'teams_flat', 'live_deepfake', 'poc_phase1']:
                print_scope_metrics(best_w16, scope)

            # Delta
            delta_teams = best_w16.get('teams_only_balanced_acc', 0) - baseline.get('teams_only_balanced_acc', 0)
            delta_all = best_w16.get('all_three_balanced_acc', 0) - baseline.get('all_three_balanced_acc', 0)
            print(f"\n  Δ teams_only bal_acc: {delta_teams*100:+.2f}pp")
            print(f"  Δ all_three  bal_acc: {delta_all*100:+.2f}pp")
    else:
        print("  (baseline not found in grid — T=0.82 may be outside search range)")

    print()

    # ═══════════════════════════════════════════════════════════
    # 5. CROSS-WINDOW COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("5. CROSS-WINDOW COMPARISON TABLE (best per W, calibrated on teams_only)")
    print("─" * 100)
    print()
    print(f"{'Type':<12s} {'W':>3s} {'T':>6s} {'K':>3s} {'K/W':>5s} {'mgn':>4s} "
          f"{'teams_bal':>10s} {'teams_cln':>10s} {'all3_bal':>10s} {'all3_cln':>10s} "
          f"{'poc_bal':>10s} {'live_bal':>10s} {'unc%':>6s}")
    print("─" * 110)

    strats = []
    for W in [16, 24, 32]:
        # Binary best
        best_bin = find_best(all_binary, 'teams_only', "balanced_acc", W_filter=W)
        if best_bin:
            strats.append(('binary', best_bin))

        # Uncertain best
        best_unc = find_best(stage3, 'teams_only', "clean_accuracy", W_filter=W, max_uncertain=0.05)
        if best_unc:
            strats.append(('uncertain', best_unc))

    for stype, r in strats:
        W = int(r['W'])
        T = r['T']
        K = int(r['K'])
        kr = r['K_ratio']
        mgn = int(r.get('margin', 0))
        t_bal = r.get('teams_only_balanced_acc', 0)
        t_cln = r.get('teams_only_clean_accuracy', r.get('teams_only_accuracy', 0))
        a_bal = r.get('all_three_balanced_acc', 0)
        a_cln = r.get('all_three_clean_accuracy', r.get('all_three_accuracy', 0))
        p_bal = r.get('poc_phase1_balanced_acc', 0)
        l_bal = r.get('live_deepfake_balanced_acc', 0)
        unc = r.get('teams_only_uncertain_rate', 0)

        print(f"{stype:<12s} {W:>3d} {T:>6.3f} {K:>3d} {kr:>5.3f} {mgn:>4d} "
              f"{t_bal*100:>9.2f}% {t_cln*100:>9.2f}% {a_bal*100:>9.2f}% {a_cln*100:>9.2f}% "
              f"{p_bal*100:>9.2f}% {l_bal*100:>9.2f}% {unc*100:>5.1f}%")

    print()

    # ═══════════════════════════════════════════════════════════
    # 6. PER-METHOD BREAKDOWN (poc_phase1)
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("6. PER-METHOD BREAKDOWN (poc_phase1 — W=32 binary, calibrated on teams_only)")
    print("─" * 100)

    # Group per-method rows
    by_strat = defaultdict(list)
    for r in per_method:
        key = f"{r.get('selected_by','?')}_{r.get('strategy_type','?')}"
        by_strat[key].append(r)

    target = by_strat.get('teams_only_binary', [])
    if target:
        print(f"\n  {'Method':<18s} {'n':>5s} {'TPR':>8s} {'TNR':>8s} {'unc%':>8s}")
        print(f"  {'─'*50}")
        hard_methods = []

        for r in sorted(target, key=lambda x: x.get('method', '')):
            method = r.get('method', '?')
            n = int(r.get('n_total', r.get('n', 0)))
            tpr = r.get('tpr')
            tnr = r.get('tnr')
            unc = r.get('uncertain_rate', r.get('uncertain', 0)) or 0

            def _is_valid(v):
                return v is not None and isinstance(v, (int, float)) and v == v

            tpr_str = fmt(tpr) if _is_valid(tpr) else "   n/a "
            tnr_str = fmt(tnr) if _is_valid(tnr) else "   n/a "

            det_rate = tpr if _is_valid(tpr) else tnr if _is_valid(tnr) else 0
            if _is_valid(tpr) and tpr < 0.7 and method != 'real':
                hard_methods.append((method, n, tpr, unc if isinstance(unc, (int, float)) else 0))

            unc_val = float(unc) if isinstance(unc, (int, float)) and unc == unc else 0.0
            print(f"  {method:<18s} {n:>5d} {tpr_str:>8s} {tnr_str:>8s} {unc_val*100:>7.1f}%")

        if hard_methods:
            print(f"\n  ⚠ Hard methods (TPR < 70%):")
            for m, n, tpr, unc in hard_methods:
                print(f"    {m}: TPR={tpr*100:.1f}% on {n} windows")

    print()

    # ═══════════════════════════════════════════════════════════
    # 7. KEY INSIGHTS & RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("7. KEY INSIGHTS & RECOMMENDATIONS")
    print("─" * 100)

    # Best overall
    best_w16_teams = find_best(all_binary, 'teams_only', "balanced_acc", W_filter=16)
    best_w24_teams = find_best(all_binary, 'teams_only', "balanced_acc", W_filter=24)
    best_w32_teams = find_best(all_binary, 'teams_only', "balanced_acc", W_filter=32)

    print("""
  A. THRESHOLD IS TOO HIGH AT 0.82
     The current deployment uses T=0.82, which maximizes TNR (specificity) at the
     cost of TPR (sensitivity). Lowering to T≈0.42-0.56 range significantly improves
     balanced accuracy by catching more fakes without losing much on reals.
""")

    if best_w16_teams:
        print(f"  B. BEST W=16 STRATEGY: T={best_w16_teams['T']:.3f}, K={int(best_w16_teams['K'])}")
        print(f"     teams_only bal_acc = {best_w16_teams.get('teams_only_balanced_acc',0)*100:.2f}%")
        print(f"     all_three  bal_acc = {best_w16_teams.get('all_three_balanced_acc',0)*100:.2f}%")
        print()

    if best_w24_teams:
        print(f"  C. BEST W=24 STRATEGY: T={best_w24_teams['T']:.3f}, K={int(best_w24_teams['K'])}")
        print(f"     teams_only bal_acc = {best_w24_teams.get('teams_only_balanced_acc',0)*100:.2f}%")
        print(f"     all_three  bal_acc = {best_w24_teams.get('all_three_balanced_acc',0)*100:.2f}%")
        print()

    if best_w32_teams:
        print(f"  D. BEST W=32 STRATEGY: T={best_w32_teams['T']:.3f}, K={int(best_w32_teams['K'])}")
        print(f"     teams_only bal_acc = {best_w32_teams.get('teams_only_balanced_acc',0)*100:.2f}%")
        print(f"     all_three  bal_acc = {best_w32_teams.get('all_three_balanced_acc',0)*100:.2f}%")
        print()

    print("""  E. W=32 POC DATA CAVEAT
     poc_phase1 at W=32 has only 83 real windows (94.3% fake). The TNR estimate
     for poc_phase1 at W=32 has high variance. Teams-only data is better balanced
     and more reliable for calibration.

  F. ADDING POC DATA SHIFTS OPTIMAL STRATEGY
     When calibrating on all_three (which includes biased poc_phase1), the optimal
     thresholds shift lower (T≈0.30-0.47) and K_ratio shifts higher (0.54-0.63).
     This is partly driven by the poc imbalance. Calibrating on teams_only yields
     T≈0.42-0.56, K_ratio≈0.47-0.54, which is more trustworthy.

  G. UNCERTAIN ZONE VALUE
     Adding a 2-4 frame margin at W=32 pushes clean accuracy to ~98.9% on teams_only
     at only 3.6% uncertain rate. This is the best trade-off if uncertain labels
     trigger human review.
""")

    # ═══════════════════════════════════════════════════════════
    # 8. RECOMMENDED STRATEGIES
    # ═══════════════════════════════════════════════════════════
    print("─" * 100)
    print("8. RECOMMENDED STRATEGIES (ranked)")
    print("─" * 100)

    best_unc_w32 = find_best(stage3, 'teams_only', "clean_accuracy", W_filter=32, max_uncertain=0.05)
    best_unc_w16 = find_best(stage3, 'teams_only', "clean_accuracy", W_filter=16, max_uncertain=0.05)

    recs = []
    if best_unc_w32:
        recs.append(("★ #1 — W=32 with uncertain zone (production-best if latency OK)", best_unc_w32, True))
    if best_w32_teams:
        recs.append(("  #2 — W=32 binary (simple, high accuracy)", best_w32_teams, False))
    if best_w16_teams:
        recs.append(("  #3 — W=16 binary (fast decisions, current window size)", best_w16_teams, False))
    if best_unc_w16:
        recs.append(("  #4 — W=16 with uncertain zone", best_unc_w16, True))
    if best_w24_teams:
        recs.append(("  #5 — W=24 binary (middle ground)", best_w24_teams, False))

    for label, r, show_unc in recs:
        print(f"\n  {label}")
        print_strategy(r, show_margin=show_unc)
        t_bal = r.get('teams_only_balanced_acc', 0)
        a_bal = r.get('all_three_balanced_acc', 0)
        t_unc = r.get('teams_only_uncertain_rate', 0)
        t_cln = r.get('teams_only_clean_accuracy', r.get('teams_only_accuracy', 0))
        print(f"    teams_only: bal_acc={t_bal*100:.2f}%  clean_acc={t_cln*100:.2f}%  uncertain={t_unc*100:.1f}%")
        print(f"    all_three:  bal_acc={a_bal*100:.2f}%")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
