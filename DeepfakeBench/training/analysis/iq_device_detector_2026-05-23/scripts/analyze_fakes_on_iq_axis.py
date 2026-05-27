"""Project fake frames onto the IQ-margin axis and quantify the cost of an
IQ-margin-based abstain rule.

Key questions:
  1. Are fakes Mac-like (margin > 0) or Windows-like (margin < 0)?
  2. Per-decile P8A recall on fakes — does Mac-like fakes get detected less?
  3. If we abstain at margin > T, what fraction of fakes are dropped vs reals?
  4. Net effect: does abstain rule improve real-FPR without crushing fake-recall?

The IQ-margin axis is RE-FIT here on real Roee-Mac vs Roee-Windows (identical
to build_iq_detector.py); fakes are projected onto the same axis. We do not
include fakes in training — they must be held out for the test to be valid.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

ROOT = Path('.')
SRC_REAL = ROOT / "analysis/iq_device_detector_2026-05-23/outputs/per_frame_iq_v2.parquet"
SRC_FAKE = ROOT / "analysis/iq_device_detector_2026-05-23/outputs/per_frame_iq_fakes_v2.parquet"
OUT = ROOT / "analysis/iq_device_detector_2026-05-23/outputs"

# ---- Load ----
real = pd.read_parquet(SRC_REAL)
real["role"] = "real"
fake = pd.read_parquet(SRC_FAKE)
print(f"Real frames: {len(real)}  |  Fake frames: {len(fake)}")
print(f"Real per-human: {real.groupby('human').size().to_dict()}")
print(f"Fake per-role: {fake.groupby('role').size().to_dict()}")
print(f"Fake per-human: {fake.groupby('human').size().to_dict()}")

META = {'frame_path','human','base_identity','deploy_relevant','prob_P8A','role'}
SKIP = META | {'jpeg_qf'}
iq_cols = [c for c in real.columns if c not in SKIP and real[c].dtype != object]
print(f"\nIQ features ({len(iq_cols)}): {iq_cols}")

# Drop NaN
real_clean = real.dropna(subset=iq_cols).copy()
fake_clean = fake.dropna(subset=iq_cols).copy()
print(f"\nAfter NaN drop: real={len(real_clean)}, fake={len(fake_clean)}")

# ---- Train axis on REAL Roee-Mac vs Roee-Windows ONLY (held-out: fakes) ----
mac_mask = (real_clean['human']=='Roee_Mac')
win_mask = (real_clean['human']=='Roee_Windows')
X_mac = real_clean[mac_mask][iq_cols].to_numpy(dtype=float)
X_win = real_clean[win_mask][iq_cols].to_numpy(dtype=float)
print(f"\nTrain set: Roee-Windows={len(X_win)}, Roee-Mac={len(X_mac)}")

X = np.concatenate([X_win, X_mac], axis=0)
y = np.concatenate([np.zeros(len(X_win), dtype=int), np.ones(len(X_mac), dtype=int)])

pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver='lbfgs'))])
cv_acc = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=1)
print(f"CV accuracy: {cv_acc.mean():.4f}")
pipe.fit(X, y)

# Project both reals and fakes
real_clean['iq_margin'] = pipe.decision_function(real_clean[iq_cols].to_numpy(dtype=float))
fake_clean['iq_margin'] = pipe.decision_function(fake_clean[iq_cols].to_numpy(dtype=float))

# ---- Q1: per-fake-cohort margin distribution ----
print('\n' + '='*120)
print('Q1: Per-fake-cohort IQ-margin distribution (positive = Mac-like, dangerous if we abstain)')
print('='*120)
fake_summary = fake_clean.groupby(['role','base_identity']).agg(
    n=('iq_margin','size'),
    margin_mean=('iq_margin','mean'),
    margin_p50=('iq_margin','median'),
    margin_p95=('iq_margin', lambda s: s.quantile(0.95)),
    frac_mac_like=('iq_margin', lambda s: (s>0).mean()),
    frac_above_p3=('iq_margin', lambda s: (s>3.0).mean()),
    p8a_recall_010=('prob_P8A', lambda s: (s>=0.10).mean()),
    p8a_recall_059=('prob_P8A', lambda s: (s>=0.59).mean()),
    deploy_relevant=('deploy_relevant','first'),
).reset_index().sort_values('margin_mean', ascending=False)
pd.set_option('display.float_format','{:.3f}'.format)
pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
print(fake_summary.to_string(index=False))
fake_summary.to_csv(OUT / 'per_fake_cohort_iq_margin.csv', index=False)

# Roll-up by role
print('\n' + '-'*120)
print('Roll-up by fake-role:')
role_roll = fake_clean.groupby('role').agg(
    n=('iq_margin','size'),
    margin_mean=('iq_margin','mean'),
    frac_mac_like=('iq_margin', lambda s: (s>0).mean()),
    frac_above_p3=('iq_margin', lambda s: (s>3.0).mean()),
    p8a_recall_010=('prob_P8A', lambda s: (s>=0.10).mean()),
    p8a_recall_059=('prob_P8A', lambda s: (s>=0.59).mean()),
).reset_index()
print(role_roll.to_string(index=False))

# ---- Q2: per-decile fake recall (using REAL-frame decile boundaries) ----
print('\n' + '='*120)
print('Q2: Per-margin-decile — FAKE recall vs REAL FPR')
print('='*120)
# Decile boundaries from the COMBINED distribution (so deciles are population-anchored)
combined = pd.concat([
    real_clean[['iq_margin','prob_P8A']].assign(role='real'),
    fake_clean[['iq_margin','prob_P8A']].assign(role='fake'),
], ignore_index=True)
qs = combined['iq_margin'].quantile(np.linspace(0.0, 1.0, 11)).to_numpy()
qs[0] -= 1e-9; qs[-1] += 1e-9

def per_dec(df, qs, tau_list=(0.10, 0.59)):
    out = []
    for i in range(len(qs)-1):
        mask = (df['iq_margin']>qs[i]) & (df['iq_margin']<=qs[i+1])
        sub = df[mask]
        row = {'decile': i, 'm_low': qs[i], 'm_high': qs[i+1], 'n': len(sub)}
        for t in tau_list:
            row[f'rate_tau{t}'] = (sub['prob_P8A']>=t).mean() if len(sub) else np.nan
        out.append(row)
    return pd.DataFrame(out)

real_dec = per_dec(real_clean, qs)
fake_dec = per_dec(fake_clean, qs)
merged = real_dec.merge(fake_dec, on=['decile','m_low','m_high'], suffixes=('_real','_fake'))
merged.columns = ['decile','m_low','m_high','n_real','real_fpr_010','real_fpr_059','n_fake','fake_recall_010','fake_recall_059']
print(merged.to_string(index=False))
merged.to_csv(OUT / 'per_decile_fake_vs_real.csv', index=False)

# ---- Q3: Abstain-at-threshold cost curve ----
print('\n' + '='*120)
print('Q3: Abstain-at-threshold cost curve (abstain when iq_margin > T)')
print('  - frac_real_abstain: fraction of real frames dropped (HIGHER = MORE FPR REDUCTION)')
print('  - frac_fake_abstain: fraction of fake frames dropped (HIGHER = MORE RECALL LOSS)')
print('  - For best outcome: real_abstain ≫ fake_abstain at chosen threshold')
print('='*120)
thresholds = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
rows = []
for T in thresholds:
    real_abs = (real_clean['iq_margin']>T).mean()
    fake_abs = (fake_clean['iq_margin']>T).mean()
    # FPR/recall on the *retained* subset
    real_ret = real_clean[real_clean['iq_margin']<=T]
    fake_ret = fake_clean[fake_clean['iq_margin']<=T]
    rows.append({
        'T': T,
        'frac_real_abstain': real_abs,
        'frac_fake_abstain': fake_abs,
        'ratio_real_per_fake': real_abs / fake_abs if fake_abs > 0 else np.inf,
        'n_real_retained': len(real_ret),
        'n_fake_retained': len(fake_ret),
        'real_fpr_010_retained': (real_ret['prob_P8A']>=0.10).mean() if len(real_ret) else np.nan,
        'real_fpr_059_retained': (real_ret['prob_P8A']>=0.59).mean() if len(real_ret) else np.nan,
        'fake_recall_010_retained': (fake_ret['prob_P8A']>=0.10).mean() if len(fake_ret) else np.nan,
        'fake_recall_059_retained': (fake_ret['prob_P8A']>=0.59).mean() if len(fake_ret) else np.nan,
    })
abs_curve = pd.DataFrame(rows)
print(abs_curve.to_string(index=False))
abs_curve.to_csv(OUT / 'abstain_threshold_cost_curve.csv', index=False)

# Baseline (no abstain)
print('\nBaseline (no abstain):')
print(f"  real_fpr@τ=0.10: {(real_clean['prob_P8A']>=0.10).mean():.4f}")
print(f"  real_fpr@τ=0.59: {(real_clean['prob_P8A']>=0.59).mean():.4f}")
print(f"  fake_recall@τ=0.10: {(fake_clean['prob_P8A']>=0.10).mean():.4f}")
print(f"  fake_recall@τ=0.59: {(fake_clean['prob_P8A']>=0.59).mean():.4f}")

# Same for deploy-relevant subset
real_dep = real_clean[real_clean['deploy_relevant']==True]
fake_dep = fake_clean[fake_clean['deploy_relevant']==True]
print(f"\nBaseline (no abstain), deploy_relevant=True only:')")
print(f"  real_fpr@τ=0.10: {(real_dep['prob_P8A']>=0.10).mean():.4f}  (n={len(real_dep)})")
print(f"  real_fpr@τ=0.59: {(real_dep['prob_P8A']>=0.59).mean():.4f}")
print(f"  fake_recall@τ=0.10: {(fake_dep['prob_P8A']>=0.10).mean():.4f}  (n={len(fake_dep)})")
print(f"  fake_recall@τ=0.59: {(fake_dep['prob_P8A']>=0.59).mean():.4f}")

print('\n' + '='*120)
print('Q3b: Same curve but on deploy_relevant=True subset only')
print('='*120)
rows = []
for T in thresholds:
    real_abs = (real_dep['iq_margin']>T).mean() if len(real_dep) else np.nan
    fake_abs = (fake_dep['iq_margin']>T).mean() if len(fake_dep) else np.nan
    real_ret = real_dep[real_dep['iq_margin']<=T]
    fake_ret = fake_dep[fake_dep['iq_margin']<=T]
    rows.append({
        'T': T,
        'frac_real_abstain': real_abs,
        'frac_fake_abstain': fake_abs,
        'ratio_real_per_fake': real_abs / fake_abs if (fake_abs and fake_abs > 0) else np.inf,
        'n_real_retained': len(real_ret),
        'n_fake_retained': len(fake_ret),
        'real_fpr_010_retained': (real_ret['prob_P8A']>=0.10).mean() if len(real_ret) else np.nan,
        'real_fpr_059_retained': (real_ret['prob_P8A']>=0.59).mean() if len(real_ret) else np.nan,
        'fake_recall_010_retained': (fake_ret['prob_P8A']>=0.10).mean() if len(fake_ret) else np.nan,
        'fake_recall_059_retained': (fake_ret['prob_P8A']>=0.59).mean() if len(fake_ret) else np.nan,
    })
abs_curve_dep = pd.DataFrame(rows)
print(abs_curve_dep.to_string(index=False))
abs_curve_dep.to_csv(OUT / 'abstain_threshold_cost_curve_deploy.csv', index=False)

# ---- Q4: Per-frame correlation iq_margin → P8A score (within fakes) ----
print('\n' + '='*120)
print('Q4: Per-frame Spearman correlation (iq_margin → P8A score) within fakes')
print('='*120)
rho_f, p_f = spearmanr(fake_clean['iq_margin'], fake_clean['prob_P8A'])
print(f'All fakes: rho={rho_f:.4f}, p={p_f:.2e}, n={len(fake_clean)}')
fake_dep = fake_clean[fake_clean['deploy_relevant']==True]
if len(fake_dep) > 10:
    rho_fd, p_fd = spearmanr(fake_dep['iq_margin'], fake_dep['prob_P8A'])
    print(f'Deploy fakes: rho={rho_fd:.4f}, p={p_fd:.2e}, n={len(fake_dep)}')

# ---- Q5: Summary verdict ----
print('\n' + '='*120)
print('VERDICT SUMMARY')
print('='*120)
real_mac_like = (real_clean['iq_margin']>0).mean()
fake_mac_like = (fake_clean['iq_margin']>0).mean()
real_p3 = (real_clean['iq_margin']>3.0).mean()
fake_p3 = (fake_clean['iq_margin']>3.0).mean()
print(f"Fraction Mac-like (margin>0): real={real_mac_like:.1%}, fake={fake_mac_like:.1%}")
print(f"Fraction extreme Mac-like (margin>+3.0): real={real_p3:.1%}, fake={fake_p3:.1%}")
print(f"Selectivity ratio (real_p3/fake_p3): {real_p3/fake_p3 if fake_p3 > 0 else np.inf:.2f}x")
print('  Interpretation: if reals have a much higher Mac-like fraction than fakes,')
print('  an IQ-margin abstain rule selectively reduces FPR with minimal recall cost.')
print('  If they\'re comparable, the abstain rule is non-selective and probably bad.')
