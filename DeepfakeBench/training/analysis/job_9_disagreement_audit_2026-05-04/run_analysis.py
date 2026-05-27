"""Job 9 - per-frame disagreement audit across P8A / E2B_3200 / E3_6600.

Pure CPU. Cached scores only. n_jobs=1.

Symmetric treatment: NO ckpt is "the leader" in this analysis.
Disagreement bands defined symmetrically. All "uniquely catches" / "uniquely
FPs" reported in BOTH directions.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training')
RAW = ROOT / 'analysis' / 'cpu_followups_2026-05-04' / 'raw_reports'
OUT = ROOT / 'analysis' / 'job_9_disagreement_audit_2026-05-04'
CROP_ATTR = ROOT / 'analysis' / 'score_distribution_2026-05-02' / 'outputs' / 'crop_attributes.csv'
FULL_TAGS = ROOT / 'analysis' / 'lockbox_tagging' / 'full_tags_2026-04-27.parquet'

OUT.mkdir(parents=True, exist_ok=True)

CKPT_FILES = {
    # ckpt_label: { suite_label: filename }
    'P8A': {
        'visomaster_enhanced_macro_dev':
            'visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv',
        'teams_real_all_dev':
            'teams_real_all_dev_p8a_reference_step5000_frames_report.csv',
        'teams_real_all_lockbox':
            'teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv',
    },
    'E2B': {
        'visomaster_enhanced_macro_dev':
            'visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv',
        'teams_real_all_dev':
            'teams_real_all_dev_e2b_top_n_step3200_frames_report.csv',
        'teams_real_all_lockbox':
            'teams_real_all_lockbox_e2b_top_n_step3200_frames_report.csv',
    },
    'E3': {
        'visomaster_enhanced_macro_dev':
            'visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv',
        'teams_real_all_dev':
            'teams_real_all_dev_e3_top_n_step6600_frames_report.csv',
        'teams_real_all_lockbox':
            'teams_real_all_lockbox_e3_top_n_step6600_frames_report.csv',
    },
}

CHRONIC_OFFENDERS = {
    'bla_bla_chow',
    'bla_bla_chow__s2',
    'PC_Generator__s22',
    'PC_Generator__s45',
    'roy_d',     # case-insensitive match below also tries 'Roy_D'
    'Q__s6',
}

# canonicalise: identity comparisons are done case-insensitively
CHRONIC_OFFENDERS_LC = {x.lower() for x in CHRONIC_OFFENDERS}

PAIRS = [('P8A', 'E2B'), ('P8A', 'E3'), ('E2B', 'E3')]

THR_HIGH = 0.8
THR_LOW = 0.2

SUITES = ['visomaster_enhanced_macro_dev', 'teams_real_all_dev', 'teams_real_all_lockbox']


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
HEX_RE = re.compile(r'^[a-f0-9]+$')


def extract_identity(path: str) -> str:
    """Identity for reals is the speaker token, e.g. `Test_Cam__s41`, `dor_shkedi`,
    `bla_bla_chow__s1`. Strips trailing hex hash, `_frame*` suffix, trailing
    `_<float>` angle indicator, and trailing underscores."""
    if not isinstance(path, str):
        return 'unknown'
    base = path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    parts = base.rsplit('__', 1)
    if len(parts) == 2 and 6 <= len(parts[1]) <= 12 and HEX_RE.match(parts[1]):
        base = parts[0]
    if '_frame' in base:
        prefix = base.split('_frame')[0]
        prefix = re.sub(r'_\d+(\.\d+)?$', '', prefix)
        prefix = prefix.rstrip('_')
        return prefix
    return base


def viso_subtype(path: str) -> str:
    base = path.rsplit('/', 1)[-1] if isinstance(path, str) else ''
    if 'visomaster_enhanced_raw' in base:
        return 'raw'
    if 'visomaster_enhanced_teams' in base:
        return 'teams'
    return 'unknown'


def viso_seq(path: str) -> str:
    """For viso fakes, parse seqNNN — used as a coarse identity / clip id proxy."""
    base = path.rsplit('/', 1)[-1] if isinstance(path, str) else ''
    m = re.search(r'seq(\d+)', base)
    return f'seq{m.group(1)}' if m else 'unknown'


def viso_filename(path: str) -> str:
    return path.rsplit('/', 1)[-1] if isinstance(path, str) else ''


def is_chronic(identity: str) -> bool:
    if not isinstance(identity, str):
        return False
    return identity.lower() in CHRONIC_OFFENDERS_LC


# -----------------------------------------------------------------------------
# Step 1: load + join
# -----------------------------------------------------------------------------
def load_suite(suite: str) -> pd.DataFrame:
    parts = []
    for ckpt, files in CKPT_FILES.items():
        path = RAW / files[suite]
        df = pd.read_csv(path)
        df = df[['frame_path', 'label', 'frame_prob']].rename(
            columns={'frame_prob': f'{ckpt}_score'}
        )
        parts.append((ckpt, df))
    base = parts[0][1]
    for ckpt, df in parts[1:]:
        base = base.merge(df, on=['frame_path', 'label'], how='inner')
    base['suite'] = suite
    return base


def load_all() -> pd.DataFrame:
    frames = [load_suite(s) for s in SUITES]
    df = pd.concat(frames, ignore_index=True)
    return df[['frame_path', 'label', 'suite', 'P8A_score', 'E2B_score', 'E3_score']]


# -----------------------------------------------------------------------------
# Step 2: bands per pair
# -----------------------------------------------------------------------------
def band_for_pair(scores_a: pd.Series, scores_b: pd.Series) -> pd.Series:
    out = pd.Series('MIXED_MIDDLE', index=scores_a.index, dtype=object)
    a_hi = scores_a > THR_HIGH
    a_lo = scores_a < THR_LOW
    b_hi = scores_b > THR_HIGH
    b_lo = scores_b < THR_LOW
    out[a_hi & b_hi] = 'AGREE_HIGH'
    out[a_lo & b_lo] = 'AGREE_LOW'
    out[a_hi & b_lo] = 'DISAGREE_A_HIGH_B_LOW'
    out[a_lo & b_hi] = 'DISAGREE_A_LOW_B_HIGH'
    return out


def make_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, b in PAIRS:
        df[f'band_{a}_{b}'] = band_for_pair(df[f'{a}_score'], df[f'{b}_score'])
    return df


# -----------------------------------------------------------------------------
# Step 3: counts table
# -----------------------------------------------------------------------------
def build_band_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        col = f'band_{a}_{b}'
        for suite in SUITES:
            for label in [0, 1]:
                sub = df[(df['suite'] == suite) & (df['label'] == label)]
                if sub.empty:
                    continue
                vc = sub[col].value_counts().to_dict()
                for band in [
                    'AGREE_HIGH', 'AGREE_LOW',
                    'DISAGREE_A_HIGH_B_LOW', 'DISAGREE_A_LOW_B_HIGH',
                    'MIXED_MIDDLE',
                ]:
                    rows.append({
                        'pair': f'{a}_vs_{b}',
                        'ckpt_A': a,
                        'ckpt_B': b,
                        'suite': suite,
                        'label': label,
                        'band': band,
                        'count': int(vc.get(band, 0)),
                        'total_in_cell': int(len(sub)),
                    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 4: viso-fake characterisation per disagreement band
# -----------------------------------------------------------------------------
def build_iq_per_pair(df_with_bands: pd.DataFrame, crop_attr: pd.DataFrame) -> pd.DataFrame:
    """For each pair x band on visomaster_enhanced_macro_dev label=1, compute
    image-quality stats (mean/std) across IQ features and subtype composition."""
    iq_features = [
        'luma_p10', 'luma_p90', 'laplacian_var',
        'sobel_edge_mean', 'saturation_mean', 'skin_frac',
    ]
    viso_fakes = df_with_bands[
        (df_with_bands['suite'] == 'visomaster_enhanced_macro_dev') &
        (df_with_bands['label'] == 1)
    ].copy()
    viso_fakes['filename'] = viso_fakes['frame_path'].apply(viso_filename)
    merged = viso_fakes.merge(
        crop_attr[['filename', 'subtype'] + iq_features],
        on='filename', how='left',
    )
    miss = merged['laplacian_var'].isna().sum()
    if miss:
        print(f'  [iq] WARNING: {miss} viso fakes did not match crop_attributes', file=sys.stderr)

    rows = []
    for a, b in PAIRS:
        col = f'band_{a}_{b}'
        for band in [
            'DISAGREE_A_HIGH_B_LOW', 'DISAGREE_A_LOW_B_HIGH',
            'AGREE_HIGH', 'AGREE_LOW', 'MIXED_MIDDLE',
        ]:
            sub = merged[merged[col] == band]
            row: dict[str, object] = {
                'pair': f'{a}_vs_{b}',
                'ckpt_A': a,
                'ckpt_B': b,
                'band': band,
                'n': int(len(sub)),
                'n_raw': int((sub['subtype'] == 'raw').sum()),
                'n_teams': int((sub['subtype'] == 'teams').sum()),
                'n_seq_unique': int(sub['frame_path'].apply(viso_seq).nunique()),
            }
            for f in iq_features:
                row[f'{f}_mean'] = float(sub[f].mean()) if len(sub) else np.nan
                row[f'{f}_std'] = float(sub[f].std()) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 5: identity composition per pair x band x suite
# -----------------------------------------------------------------------------
def build_identity_table(df_with_bands: pd.DataFrame, full_tags: pd.DataFrame) -> pd.DataFrame:
    df = df_with_bands.copy()
    df['identity'] = df['frame_path'].apply(extract_identity)
    df['is_chronic'] = df['identity'].apply(is_chronic)

    # Capture-mode lookup from full_tags (best-effort; overlap is partial for lockbox)
    cm_map = dict(zip(full_tags['gcs_uri'], full_tags['clip_capture_mode']))
    w_map = dict(zip(full_tags['gcs_uri'], full_tags['width']))
    h_map = dict(zip(full_tags['gcs_uri'], full_tags['height']))
    df['clip_capture_mode'] = df['frame_path'].map(cm_map)
    df['_w'] = df['frame_path'].map(w_map)
    df['_h'] = df['frame_path'].map(h_map)
    df['low_res_lt200'] = ((df['_w'] < 200) | (df['_h'] < 200))

    rows = []
    for a, b in PAIRS:
        col = f'band_{a}_{b}'
        for suite in SUITES:
            for band in ['DISAGREE_A_HIGH_B_LOW', 'DISAGREE_A_LOW_B_HIGH']:
                sub = df[(df['suite'] == suite) & (df[col] == band)]
                # only meaningful for the labels where the disagreement is interpretable
                # (we'll keep both labels)
                for label in [0, 1]:
                    sub_l = sub[sub['label'] == label]
                    if sub_l.empty:
                        continue
                    n = len(sub_l)
                    n_chronic = int(sub_l['is_chronic'].sum())
                    n_lowres = int(sub_l['low_res_lt200'].fillna(False).sum())
                    cm_counts = sub_l['clip_capture_mode'].value_counts().to_dict()
                    cm_top = ','.join(f'{k}:{v}' for k, v in
                                      sorted(cm_counts.items(),
                                             key=lambda kv: -kv[1])[:5]
                                      if pd.notna(k))
                    id_counts = sub_l['identity'].value_counts().to_dict()
                    id_top = ','.join(f'{k}:{v}' for k, v in
                                      sorted(id_counts.items(),
                                             key=lambda kv: -kv[1])[:8])
                    rows.append({
                        'pair': f'{a}_vs_{b}',
                        'ckpt_A': a,
                        'ckpt_B': b,
                        'suite': suite,
                        'band': band,
                        'label': label,
                        'n': int(n),
                        'n_unique_identities': int(sub_l['identity'].nunique()),
                        'n_chronic_offender_frames': n_chronic,
                        'pct_chronic_offender': round(100.0 * n_chronic / n, 2) if n else 0.0,
                        'n_low_res_lt200': n_lowres,
                        'top_identities': id_top,
                        'capture_mode_top': cm_top,
                    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 6: unique-catch table on viso fakes
# -----------------------------------------------------------------------------
def build_unique_catches(df_with_bands: pd.DataFrame) -> pd.DataFrame:
    """For viso fakes (label=1), per pair, count how many viso fakes each ckpt
    UNIQUELY catches (one ckpt > THR_HIGH and the other < THR_LOW). Stratified
    by subtype."""
    viso = df_with_bands[
        (df_with_bands['suite'] == 'visomaster_enhanced_macro_dev') &
        (df_with_bands['label'] == 1)
    ].copy()
    viso['subtype'] = viso['frame_path'].apply(viso_subtype)

    rows = []
    for a, b in PAIRS:
        col = f'band_{a}_{b}'
        # ckpt A uniquely catches: A > 0.8, B < 0.2 -- DISAGREE_A_HIGH_B_LOW
        # ckpt B uniquely catches: A < 0.2, B > 0.8 -- DISAGREE_A_LOW_B_HIGH
        for catcher, band in [(a, 'DISAGREE_A_HIGH_B_LOW'),
                              (b, 'DISAGREE_A_LOW_B_HIGH')]:
            sub = viso[viso[col] == band]
            for subtype in ['raw', 'teams', 'unknown', 'ALL']:
                if subtype == 'ALL':
                    n = len(sub)
                else:
                    n = int((sub['subtype'] == subtype).sum())
                rows.append({
                    'pair': f'{a}_vs_{b}',
                    'ckpt_A': a,
                    'ckpt_B': b,
                    'unique_catcher': catcher,
                    'band': band,
                    'subtype': subtype,
                    'n_uniquely_caught': n,
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 7: unique-FP table on real frames
# -----------------------------------------------------------------------------
def build_unique_fps(df_with_bands: pd.DataFrame) -> pd.DataFrame:
    """For real frames (label=0), per pair, count how many reals each ckpt
    UNIQUELY false-positives on. Concentrated on chronic offenders?"""
    df = df_with_bands.copy()
    df['identity'] = df['frame_path'].apply(extract_identity)
    df['is_chronic'] = df['identity'].apply(is_chronic)

    rows = []
    for a, b in PAIRS:
        col = f'band_{a}_{b}'
        for fp_owner, band in [(a, 'DISAGREE_A_HIGH_B_LOW'),
                               (b, 'DISAGREE_A_LOW_B_HIGH')]:
            for suite in ['teams_real_all_dev', 'teams_real_all_lockbox']:
                sub = df[(df['suite'] == suite) &
                         (df['label'] == 0) &
                         (df[col] == band)]
                n = len(sub)
                n_chr = int(sub['is_chronic'].sum())
                top_ids = ','.join(f'{k}:{v}' for k, v in
                                   sorted(sub['identity']
                                          .value_counts()
                                          .to_dict().items(),
                                          key=lambda kv: -kv[1])[:8])
                rows.append({
                    'pair': f'{a}_vs_{b}',
                    'ckpt_A': a,
                    'ckpt_B': b,
                    'unique_fp_owner': fp_owner,
                    'band': band,
                    'suite': suite,
                    'n_unique_fps': n,
                    'n_on_chronic_offenders': n_chr,
                    'pct_on_chronic_offenders':
                        round(100.0 * n_chr / n, 2) if n else 0.0,
                    'top_identities': top_ids,
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print('[Job 9] loading suites...')
    df = load_all()
    print(f'  joined rows: {len(df):,} (expected ~6532)')
    for s in SUITES:
        sub = df[df['suite'] == s]
        for lab in sorted(sub['label'].unique()):
            print(f'  {s} label={lab}: {len(sub[sub["label"] == lab]):,}')

    print('[Job 9] computing bands...')
    df_b = make_bands(df)

    # Step 2 output
    counts = build_band_counts(df_b)
    counts.to_csv(OUT / 'disagreement_bands_counts.csv', index=False)
    print(f'  -> {OUT/"disagreement_bands_counts.csv"} ({len(counts)} rows)')

    # Step 3 output (IQ on viso fakes)
    print('[Job 9] loading crop_attributes...')
    crop_attr = pd.read_csv(CROP_ATTR)
    iq = build_iq_per_pair(df_b, crop_attr)
    iq.to_csv(OUT / 'disagreement_iq_per_pair.csv', index=False)
    print(f'  -> {OUT/"disagreement_iq_per_pair.csv"} ({len(iq)} rows)')

    # Step 4 output (identities)
    print('[Job 9] loading full_tags parquet...')
    full_tags = pd.read_parquet(FULL_TAGS)
    ids = build_identity_table(df_b, full_tags)
    ids.to_csv(OUT / 'disagreement_identities_per_pair.csv', index=False)
    print(f'  -> {OUT/"disagreement_identities_per_pair.csv"} ({len(ids)} rows)')

    # Step 5 output (unique catches)
    uc = build_unique_catches(df_b)
    uc.to_csv(OUT / 'unique_catches_viso_per_ckpt.csv', index=False)
    print(f'  -> {OUT/"unique_catches_viso_per_ckpt.csv"} ({len(uc)} rows)')

    # Step 6 output (unique FPs)
    ufp = build_unique_fps(df_b)
    ufp.to_csv(OUT / 'unique_fps_real_per_ckpt.csv', index=False)
    print(f'  -> {OUT/"unique_fps_real_per_ckpt.csv"} ({len(ufp)} rows)')

    # Save the full joined+banded frame for downstream re-analysis
    df_b.to_csv(OUT / '_joined_with_bands.csv', index=False)
    print(f'  -> {OUT/"_joined_with_bands.csv"} ({len(df_b)} rows)')

    print('[Job 9] done.')


if __name__ == '__main__':
    main()
