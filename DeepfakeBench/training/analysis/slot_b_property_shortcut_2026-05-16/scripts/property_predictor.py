"""Test whether the property axis (not identity) predicts Slot β over-firing.

Two key tests:
  A) Within bla_bla_chow (the only OTHER over-firing identity in lockbox), do
     the over-firing 8 frames share the same property profile as the
     over-firing dor_shkedi frames?
  B) If we train a property-based linear classifier on Slot β over-fire status
     on dor_shkedi only, does it predict over-firing on bla_bla_chow without
     looking at identity?
  C) Compute per-frame Slot β score correlation with each property axis,
     stratified by identity, to confirm the shortcut is universal.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'


PROPS = ['sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
         'lab_a_std', 'lab_b_std', 'skin_frac_hsv', 'edge_density',
         'min_dim', 'file_size_bytes']


def main():
    df = pd.read_csv(OUT / 'frames_with_props_and_scores.csv')

    # Test A: bla_bla_chow over-fire property profile
    print('=== Test A: bla_bla_chow over (n=8) vs non-over (n=60) within identity ===')
    bbc = df[df['identity'] == 'bla_bla_chow']
    over_bbc = bbc[bbc['SLOT_B_overfire'] == 1]
    notover_bbc = bbc[bbc['SLOT_B_overfire'] == 0]
    rows = []
    for p in PROPS:
        try:
            u = stats.mannwhitneyu(over_bbc[p], notover_bbc[p], alternative='two-sided')
            pv = u.pvalue
        except Exception:
            pv = np.nan
        pooled_sd = np.sqrt((over_bbc[p].var() + notover_bbc[p].var()) / 2)
        d = (over_bbc[p].mean() - notover_bbc[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows.append({'property': p,
                     'over_mean': over_bbc[p].mean(),
                     'notover_mean': notover_bbc[p].mean(),
                     'over_median': over_bbc[p].median(),
                     'notover_median': notover_bbc[p].median(),
                     'cohens_d': d,
                     'mw_p': pv})
    a = pd.DataFrame(rows).sort_values('cohens_d', key=abs, ascending=False)
    print(a.to_string(index=False))
    a.to_csv(OUT / 'contrast_bla_bla_chow_overfire_vs_not.csv', index=False)

    # Test B: train property classifier on dor_shkedi, test on bla_bla_chow
    print('\n=== Test B: linear property classifier trained on dor_shkedi, tested on bla_bla_chow ===')
    dor = df[df['identity'] == 'dor_shkedi']
    X_train = dor[PROPS].values
    y_train = dor['SLOT_B_overfire'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_train_s, y_train)

    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train_s)[:, 1])
    print(f'Train AUC on dor_shkedi: {train_auc:.4f}')

    # Test on bla_bla_chow
    bbc = df[df['identity'] == 'bla_bla_chow']
    X_test = scaler.transform(bbc[PROPS].values)
    y_test = bbc['SLOT_B_overfire'].values
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) if y_test.sum() > 0 else np.nan
    print(f'Test AUC on bla_bla_chow: {test_auc:.4f}  (n={len(bbc)}, positives={int(y_test.sum())})')

    # Also: do non-chronic identities (real_dor, Chikara_Takahashi, PC_Generator) score
    # below the dor-trained classifier's decision boundary?
    for other_id in ['real_dor', 'Chikara_Takahashi', 'PC_Generator']:
        sub = df[df['identity'] == other_id]
        X_o = scaler.transform(sub[PROPS].values)
        probs = clf.predict_proba(X_o)[:, 1]
        print(f'  {other_id}: n={len(sub)}, classifier mean prob = {probs.mean():.4f}, '
              f'max = {probs.max():.4f}, frac > 0.5 = {(probs > 0.5).mean():.4f}')

    # Show classifier coefficients (which properties matter)
    print('\nClassifier coefficients (scaled props):')
    for name, c in sorted(zip(PROPS, clf.coef_[0]), key=lambda kv: abs(kv[1]), reverse=True):
        print(f'  {name}: {c:+.4f}')

    # Test C: per-identity Spearman correlation of Slot β score with each property
    print('\n=== Test C: per-identity Spearman ρ(Slot β score, property) ===')
    rows = []
    for ident in ['dor_shkedi', 'bla_bla_chow', 'real_dor', 'Chikara_Takahashi', 'PC_Generator']:
        sub = df[df['identity'] == ident]
        if len(sub) < 10:
            continue
        for p in PROPS:
            r, pval = stats.spearmanr(sub['SLOT_B'], sub[p])
            rows.append({'identity': ident, 'property': p, 'rho': r, 'p': pval, 'n': len(sub)})
    cdf = pd.DataFrame(rows)
    pivot = cdf.pivot(index='property', columns='identity', values='rho')
    print(pivot.to_string())
    cdf.to_csv(OUT / 'per_identity_spearman_slot_b_vs_property.csv', index=False)


if __name__ == '__main__':
    main()
