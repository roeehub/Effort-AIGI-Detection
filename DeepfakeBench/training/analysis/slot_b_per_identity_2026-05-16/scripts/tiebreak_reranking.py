"""Re-rank the 2026-05-16 scorecard with alternative tiebreaks.

v3-fix policy gates:
  1. dev_fake_macro_recall >= 0.30
  2. dev_worst_real_stress_fpr <= 0.10
  3. dev_primary_real_fpr <= 0.07
  4. Tiebreak: ascending lockbox_real_fpr  <-- the load-bearing question

Compare against:
  Tiebreak alt-A: descending dev_fake_macro_recall  (recall-favoring)
  Tiebreak alt-B: descending teams_fake_all_dev + visomaster_enhanced_macro_dev
                  + deeplive_enhanced_dev (3-fake-suite sum)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
INP = HERE / 'inputs'
OUT = HERE / 'outputs'

GATE1_FLOOR = 0.30
GATE2_STRESS = 0.10
GATE3_REAL = 0.07


def check_gates(row):
    g1 = row['dev_fake_macro_recall'] >= GATE1_FLOOR - 1e-9
    g2 = row['dev_worst_real_stress_fpr'] <= GATE2_STRESS + 1e-9
    g3 = row['dev_primary_real_fpr'] <= GATE3_REAL + 1e-9
    return g1 and g2 and g3, g1, g2, g3


def rank_with_tiebreak(df: pd.DataFrame, tiebreak_col: str, ascending: bool) -> pd.DataFrame:
    df = df.copy()
    df['three_suite_sum'] = (df['teams_fake_all_dev__fake_recall']
                             + df['visomaster_enhanced_macro_dev__fake_recall']
                             + df['deeplive_enhanced_dev__fake_recall'])
    df[['pass_all', 'g1', 'g2', 'g3']] = df.apply(
        lambda r: pd.Series(check_gates(r)), axis=1)
    pass_df = df[df['pass_all']].copy()
    fail_df = df[~df['pass_all']].copy()
    pass_df = pass_df.sort_values(tiebreak_col, ascending=ascending).reset_index(drop=True)
    pass_df['rank'] = pass_df.index + 1
    fail_df = fail_df.sort_values('dev_fake_macro_recall', ascending=False).reset_index(drop=True)
    fail_df['rank'] = len(pass_df) + fail_df.index + 1
    return pd.concat([pass_df, fail_df], ignore_index=True)


def main():
    df = pd.read_csv(INP / 'checkpoint_summary.csv')
    print('=== Original v3-fix tiebreak (ascending lockbox_real_fpr) ===')
    r0 = rank_with_tiebreak(df, 'lockbox_real_fpr', ascending=True)
    cols = ['rank', 'checkpoint_key', 'selected_threshold', 'dev_fake_macro_recall',
            'dev_worst_real_stress_fpr', 'dev_primary_real_fpr',
            'lockbox_real_fpr', 'three_suite_sum', 'pass_all']
    print(r0[cols].to_string(index=False))

    print('\n=== Alt-A tiebreak: descending dev_fake_macro_recall ===')
    rA = rank_with_tiebreak(df, 'dev_fake_macro_recall', ascending=False)
    print(rA[cols].to_string(index=False))

    print('\n=== Alt-B tiebreak: descending 3-fake-suite sum ===')
    rB = rank_with_tiebreak(df, 'three_suite_sum', ascending=False)
    print(rB[cols].to_string(index=False))

    r0.to_csv(OUT / 'rerank_orig_lockbox_fpr.csv', index=False)
    rA.to_csv(OUT / 'rerank_alt_dev_recall.csv', index=False)
    rB.to_csv(OUT / 'rerank_alt_three_suite_sum.csv', index=False)


if __name__ == '__main__':
    main()
