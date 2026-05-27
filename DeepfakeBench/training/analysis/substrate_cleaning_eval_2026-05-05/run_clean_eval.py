"""
Reusable substrate-cleaning evaluation for new checkpoints.

Defines the F4 cleaned substrate (drop chronic-6 identities + low-res frames
with min(W,H)<200 + frames with no face). Reproduces the Job 14
(analysis/job_14_substrate_clean_simulation_2026-05-04) numbers when fed the
same per-frame score reports.

Inputs (per checkpoint):
  --ckpt-name <label>            Logical name for the ckpt (used in output filenames).
  --real-csv <path>              Per-frame CSV for the REAL dev pool used to calibrate tau.
                                 Default real pool: teams_real_all_dev (4564 frames).
                                 CSV must have columns: frame_path, frame_prob, video_id.
  --fake-csv suite=path [..]     One or more fake-suite CSVs to evaluate at the
                                 cleaned-substrate calibrated tau.
                                 Each CSV must have columns: frame_path, frame_prob.
  --out-dir <path>               Output directory.
  --target-fpr <float>           FPR target for tau calibration (default 0.10).

Outputs (under --out-dir):
  cleaned_substrate_manifest.json   Filter rules + chronic-6 list + per-frame keep mask
  <ckpt>_real_cleaned.csv           Cleaned real CSV with KEEP/DROP and reason columns
  <ckpt>_per_filter_fpr.csv         FPR @ F0-calibrated tau across F0..F4 filters
  <ckpt>_per_suite_recall_lift.csv  recall_F0 vs recall_F4 per fake suite
  <ckpt>_summary.json               Summary stats (n frames before/after, recall@FPR={5,10}%)

Reference run (no inputs needed):
  --reference-run                Reproduces Job 14 P8A and E2B_3200 numbers from existing
                                 raw_reports CSVs and writes:
                                   reference_run_p8a.json
                                   reference_run_e2b.json
                                 Verifies P8A viso recall ~27 -> ~67 % at F0->F4 tau.

CPU only. Requires pandas, numpy, pyarrow.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — frozen contract for the F4 cleaned substrate.
# ---------------------------------------------------------------------------

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
PARQUET_TAGS = os.path.join(ROOT, 'analysis/lockbox_tagging/full_tags_2026-04-27.parquet')
REFERENCE_RAW_DIR = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
REFERENCE_OUT_DIR = os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05')

CHRONIC_6 = [
    'bla_bla_chow',
    'bla_bla_chow__s2',
    'pc_generator__s22',
    'pc_generator__s45',
    'roy_d',
    'q__s6',
]

MIN_WH_PX = 200             # F2: drop frames where min(width, height) < this
DROP_NO_FACE = True         # F3: drop is_no_face == True
TARGET_FPR_DEFAULT = 0.10

# Regex strips frame/segment suffixes from `video_id` to recover the canonical
# subject-session token. We DO NOT strip `__s\d+` because the chronic-6 list
# carries that token (e.g. `pc_generator__s22`).
_IDENT_STRIP = re.compile(
    r'(__seq\d+|__seg_[\d.]+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)

# Reference targets from Job 14 (analysis/job_14_substrate_clean_simulation_2026-05-04).
# Used by --reference-run to assert reproducibility.
_REF_TARGETS = {
    'P8A': {
        'real_csv': 'teams_real_all_dev_p8a_reference_step5000_frames_report.csv',
        'fake_csvs': {
            'visomaster_enhanced_macro_dev':
                'visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv',
            'deeplive_enhanced_dev':
                'deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv',
            'teams_fake_all_dev':
                'teams_fake_all_dev_p8a_reference_step5000_frames_report.csv',
        },
        'expected': {
            'visomaster_enhanced_macro_dev': {'recall_F0': 0.2691, 'recall_F4': 0.6709},
            'deeplive_enhanced_dev':         {'recall_F0': 0.4239, 'recall_F4': 0.9248},
            'teams_fake_all_dev':            {'recall_F0': 0.6989, 'recall_F4': 0.9223},
            'fpr_F0': 0.1001, 'fpr_F4': 0.0086,
            'n_real_F0': 4564, 'n_real_F4': 2091,
        },
    },
    'E2B_3200': {
        'real_csv': 'teams_real_all_dev_e2b_top_n_step3200_frames_report.csv',
        'fake_csvs': {
            'visomaster_enhanced_macro_dev':
                'visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv',
            'deeplive_enhanced_dev':
                'deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv',
            'teams_fake_all_dev':
                'teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv',
        },
        'expected': {
            'visomaster_enhanced_macro_dev': {'recall_F0': 0.0836, 'recall_F4': 0.3091},
            'deeplive_enhanced_dev':         {'recall_F0': 0.9394, 'recall_F4': 1.0000},
            'teams_fake_all_dev':            {'recall_F0': 0.7940, 'recall_F4': 0.8713},
            'fpr_F0': 0.1001, 'fpr_F4': 0.0316,
            'n_real_F0': 4564, 'n_real_F4': 2091,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def extract_identity_from_video_id(vid: str) -> str:
    """Recover canonical subject-session token from a video_id string.

    Iteratively strips known frame/segment/crop suffixes; preserves __s\\d+.
    Returns 'UNK' if input is NaN/empty.
    """
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP.sub('', s)
    return s.strip('_').lower() or 'UNK'


def load_parquet_meta(parquet_path: str = PARQUET_TAGS) -> pd.DataFrame:
    """Load the subset of parquet tags needed for filtering."""
    pq = pd.read_parquet(parquet_path)
    meta = pq[['gcs_uri', 'identity_key', 'width', 'height', 'is_no_face']].copy()
    meta['min_wh'] = meta[['width', 'height']].min(axis=1)
    return meta


def annotate_real(real_df: pd.DataFrame, parquet_meta: pd.DataFrame) -> pd.DataFrame:
    """Add identity + parquet metadata columns to a real-pool frames CSV."""
    df = real_df.copy()
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(parquet_meta, how='left',
                      left_on='frame_path', right_on='gcs_uri')
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    return merged


def build_filter_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return KEEP masks for filters F0..F4.

    F0: keep all
    F1: drop chronic-6 identities (regex-derived OR parquet identity_key)
    F2: drop frames where min(W,H) < MIN_WH_PX (parquet-covered only;
        uncovered frames KEPT — caveat documented in Job 14)
    F3: drop frames where is_no_face == True (parquet-covered only;
        uncovered frames KEPT)
    F4: F1 AND F2 AND F3
    """
    chronic_set = set(CHRONIC_6)
    is_chronic = df['identity'].isin(chronic_set) | df['identity_pq'].isin(chronic_set)
    keep_F1 = ~is_chronic

    min_wh = pd.to_numeric(df['min_wh'], errors='coerce')
    keep_F2 = ~(min_wh < MIN_WH_PX).fillna(False)

    no_face = df['is_no_face'].fillna(False).astype(bool)
    keep_F3 = ~no_face if DROP_NO_FACE else pd.Series(True, index=df.index)

    keep_F0 = pd.Series(True, index=df.index)
    keep_F4 = keep_F1 & keep_F2 & keep_F3
    return {'F0': keep_F0, 'F1': keep_F1, 'F2': keep_F2, 'F3': keep_F3, 'F4': keep_F4}


def fpr_threshold(scores: pd.Series, target_fpr: float) -> float:
    """tau such that fraction(scores > tau) == target_fpr."""
    s = pd.to_numeric(scores, errors='coerce').dropna().to_numpy()
    if len(s) == 0:
        return float('nan')
    return float(np.quantile(s, 1.0 - target_fpr))


def fpr_at_tau(scores: pd.Series, tau: float) -> float:
    s = pd.to_numeric(scores, errors='coerce').dropna().to_numpy()
    if len(s) == 0:
        return float('nan')
    return float((s > tau).sum() / len(s))


def recall_at_tau(scores: pd.Series, tau: float) -> float:
    return fpr_at_tau(scores, tau)  # recall on fakes is just fraction(score>tau)


# ---------------------------------------------------------------------------
# Manifest writer (called once or on demand; manifest is content-addressed).
# ---------------------------------------------------------------------------

def write_manifest(out_dir: str, parquet_meta: pd.DataFrame) -> str:
    """Write the cleaned-substrate manifest with the frozen filter rules.

    The manifest is the single source of truth for which frames belong to F4.
    It includes the chronic-6 list, the rules, and a list of dropped GCS URIs
    keyed by reason so future audits can reproduce the F4 set without any code.
    """
    # Build per-frame keep mask over parquet-covered frames.
    df = parquet_meta.copy()
    df['identity_pq'] = df['identity_key'].astype(str).str.lower()
    chronic_set = set(CHRONIC_6)
    is_chronic = df['identity_pq'].isin(chronic_set)
    is_lowres = (pd.to_numeric(df['min_wh'], errors='coerce') < MIN_WH_PX).fillna(False)
    is_noface = df['is_no_face'].fillna(False).astype(bool)
    drop_chronic = df.loc[is_chronic, 'gcs_uri'].dropna().tolist()
    drop_lowres = df.loc[is_lowres & ~is_chronic, 'gcs_uri'].dropna().tolist()
    drop_noface = df.loc[is_noface & ~is_chronic & ~is_lowres, 'gcs_uri'].dropna().tolist()

    manifest = {
        'schema_version': 1,
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'source_parquet': PARQUET_TAGS,
        'parquet_n_rows': int(len(df)),
        'filter_rules': {
            'F1_drop_chronic_identities': CHRONIC_6,
            'F2_drop_low_resolution': {
                'predicate': 'min(width, height) < threshold_px',
                'threshold_px': MIN_WH_PX,
                'note': 'Applied only where parquet covers the frame; uncovered frames KEPT.',
            },
            'F3_drop_no_face': {
                'predicate': 'is_no_face == True',
                'note': 'Applied only where parquet covers the frame; uncovered frames KEPT.',
            },
            'F4': 'F1 AND F2 AND F3',
        },
        'dropped_uris_by_reason': {
            'chronic_6_identity': sorted(drop_chronic),
            'low_resolution_lt200': sorted(drop_lowres),
            'is_no_face': sorted(drop_noface),
        },
        'counts': {
            'dropped_chronic_6': len(drop_chronic),
            'dropped_lowres_only_excl_chronic': len(drop_lowres),
            'dropped_noface_only_excl_chronic_and_lowres': len(drop_noface),
        },
    }
    out_path = os.path.join(out_dir, 'cleaned_substrate_manifest.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Core evaluation: one ckpt, one real CSV, N fake CSVs.
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_name: str,
    real_csv: str,
    fake_csvs: Dict[str, str],
    out_dir: str,
    target_fpr: float = TARGET_FPR_DEFAULT,
    parquet_meta: pd.DataFrame | None = None,
) -> Dict:
    """Evaluate one checkpoint on the cleaned substrate.

    Writes:
      <ckpt>_real_cleaned.csv
      <ckpt>_per_filter_fpr.csv
      <ckpt>_per_suite_recall_lift.csv
      <ckpt>_summary.json

    Returns the summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    if parquet_meta is None:
        parquet_meta = load_parquet_meta()

    real_df = pd.read_csv(real_csv)
    real_anno = annotate_real(real_df, parquet_meta)
    masks = build_filter_masks(real_anno)

    # Annotated cleaned CSV: include keep_F4 + per-axis reason columns.
    chronic_set = set(CHRONIC_6)
    real_anno['drop_chronic_6'] = (real_anno['identity'].isin(chronic_set)
                                   | real_anno['identity_pq'].isin(chronic_set))
    real_anno['drop_lowres_lt200'] = (
        pd.to_numeric(real_anno['min_wh'], errors='coerce') < MIN_WH_PX
    ).fillna(False)
    real_anno['drop_no_face'] = real_anno['is_no_face'].fillna(False).astype(bool)
    real_anno['keep_F4'] = masks['F4']
    real_cleaned_path = os.path.join(out_dir, f'{ckpt_name}_real_cleaned.csv')
    cols_to_save = ['frame_path', 'video_id', 'identity', 'identity_pq', 'frame_prob',
                    'width', 'height', 'min_wh', 'is_no_face',
                    'drop_chronic_6', 'drop_lowres_lt200', 'drop_no_face', 'keep_F4']
    real_anno[cols_to_save].to_csv(real_cleaned_path, index=False)

    # Per-filter FPR table at F0-calibrated tau.
    tau_F0 = fpr_threshold(real_anno['frame_prob'], target_fpr)
    fpr_rows = []
    for fname, mask in masks.items():
        sub = real_anno.loc[mask, 'frame_prob']
        fpr_rows.append({
            'ckpt': ckpt_name,
            'filter': fname,
            'n_frames': int(mask.sum()),
            'tau_used_F0_calibrated': round(tau_F0, 4),
            'fpr_actual_pct': round(100 * fpr_at_tau(sub, tau_F0), 2),
        })
    fpr_df = pd.DataFrame(fpr_rows)
    fpr_path = os.path.join(out_dir, f'{ckpt_name}_per_filter_fpr.csv')
    fpr_df.to_csv(fpr_path, index=False)

    # Recalibrate tau on F4 reals; also compute tau@5%.
    f4_real_scores = real_anno.loc[masks['F4'], 'frame_prob']
    tau_F4_at_10 = fpr_threshold(f4_real_scores, 0.10)
    tau_F4_at_5 = fpr_threshold(f4_real_scores, 0.05)

    # Recall on each fake suite at both tau_F0 and tau_F4.
    recall_rows = []
    for suite_short, fake_csv in fake_csvs.items():
        if not os.path.exists(fake_csv):
            print(f'  WARN: missing fake CSV: {fake_csv}', file=sys.stderr)
            continue
        df_fake = pd.read_csv(fake_csv)
        rec_F0 = recall_at_tau(df_fake['frame_prob'], tau_F0)
        rec_F4_at_10 = recall_at_tau(df_fake['frame_prob'], tau_F4_at_10)
        rec_F4_at_5 = recall_at_tau(df_fake['frame_prob'], tau_F4_at_5)
        recall_rows.append({
            'ckpt': ckpt_name,
            'fake_suite': suite_short,
            'n_fake_frames': int(len(df_fake)),
            'tau_F0_calibrated_at_target_fpr': round(tau_F0, 4),
            'tau_F4_calibrated_at_FPR=10%': round(tau_F4_at_10, 4),
            'tau_F4_calibrated_at_FPR=5%': round(tau_F4_at_5, 4),
            'recall_at_tau_F0_pct': round(100 * rec_F0, 2),
            'recall_at_tau_F4_FPR10_pct': round(100 * rec_F4_at_10, 2),
            'recall_at_tau_F4_FPR5_pct': round(100 * rec_F4_at_5, 2),
            'lift_abs_pp_F0_to_F4_FPR10': round(100 * (rec_F4_at_10 - rec_F0), 2),
        })
    recall_df = pd.DataFrame(recall_rows)
    recall_path = os.path.join(out_dir, f'{ckpt_name}_per_suite_recall_lift.csv')
    recall_df.to_csv(recall_path, index=False)

    # Summary.
    n_real_F0 = int(masks['F0'].sum())
    n_real_F4 = int(masks['F4'].sum())
    summary = {
        'ckpt_name': ckpt_name,
        'real_csv': real_csv,
        'fake_csvs': fake_csvs,
        'target_fpr': target_fpr,
        'n_real_frames_F0': n_real_F0,
        'n_real_frames_F4': n_real_F4,
        'pct_real_kept_F4': round(100 * n_real_F4 / max(1, n_real_F0), 2),
        'tau_F0_calibrated': round(tau_F0, 4),
        'tau_F4_at_FPR_5pct': round(tau_F4_at_5, 4),
        'tau_F4_at_FPR_10pct': round(tau_F4_at_10, 4),
        'fpr_F0_pct': round(100 * fpr_at_tau(real_anno['frame_prob'], tau_F0), 2),
        'fpr_F4_pct': round(100 * fpr_at_tau(f4_real_scores, tau_F0), 2),
        'fpr_at_tau_0_5_F0_pct': round(100 * fpr_at_tau(real_anno['frame_prob'], 0.5), 2),
        'fpr_at_tau_0_5_F4_pct': round(100 * fpr_at_tau(f4_real_scores, 0.5), 2),
        'per_suite': recall_rows,
        'outputs': {
            'real_cleaned_csv': real_cleaned_path,
            'per_filter_fpr_csv': fpr_path,
            'per_suite_recall_lift_csv': recall_path,
        },
    }
    summary_path = os.path.join(out_dir, f'{ckpt_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Reference run — reproduces Job 14 numbers as a regression check.
# ---------------------------------------------------------------------------

def _reference_one(label: str, parquet_meta: pd.DataFrame, out_dir: str) -> Tuple[Dict, List[str]]:
    """Run one reference checkpoint and return (summary, list of mismatch strings)."""
    spec = _REF_TARGETS[label]
    real_csv = os.path.join(REFERENCE_RAW_DIR, spec['real_csv'])
    fake_csvs = {k: os.path.join(REFERENCE_RAW_DIR, v) for k, v in spec['fake_csvs'].items()}
    summary = evaluate_checkpoint(
        ckpt_name=f'reference_{label.lower()}',
        real_csv=real_csv,
        fake_csvs=fake_csvs,
        out_dir=out_dir,
        target_fpr=TARGET_FPR_DEFAULT,
        parquet_meta=parquet_meta,
    )

    # Compare against Job 14 expected values within reasonable absolute tolerance.
    expected = spec['expected']
    mismatches = []
    tol_recall = 0.005  # 0.5pp absolute tolerance on recall fractions
    tol_fpr = 0.005
    tol_count = 5

    if abs(summary['n_real_frames_F0'] - expected['n_real_F0']) > tol_count:
        mismatches.append(
            f'n_real_F0 expected {expected["n_real_F0"]} got {summary["n_real_frames_F0"]}'
        )
    if abs(summary['n_real_frames_F4'] - expected['n_real_F4']) > tol_count:
        mismatches.append(
            f'n_real_F4 expected {expected["n_real_F4"]} got {summary["n_real_frames_F4"]}'
        )
    if abs(summary['fpr_F0_pct'] / 100 - expected['fpr_F0']) > tol_fpr:
        mismatches.append(
            f'fpr_F0 expected {expected["fpr_F0"]:.4f} got {summary["fpr_F0_pct"]/100:.4f}'
        )
    if abs(summary['fpr_F4_pct'] / 100 - expected['fpr_F4']) > tol_fpr:
        mismatches.append(
            f'fpr_F4 expected {expected["fpr_F4"]:.4f} got {summary["fpr_F4_pct"]/100:.4f}'
        )

    suite_results = {row['fake_suite']: row for row in summary['per_suite']}
    for suite_short, exp in expected.items():
        if not isinstance(exp, dict):
            continue
        if 'recall_F0' not in exp:
            continue
        if suite_short not in suite_results:
            mismatches.append(f'suite {suite_short} missing from results')
            continue
        got_rec_F0 = suite_results[suite_short]['recall_at_tau_F0_pct'] / 100
        got_rec_F4 = suite_results[suite_short]['recall_at_tau_F4_FPR10_pct'] / 100
        if abs(got_rec_F0 - exp['recall_F0']) > tol_recall:
            mismatches.append(
                f'{suite_short} recall_F0 expected {exp["recall_F0"]:.4f} got {got_rec_F0:.4f}'
            )
        if abs(got_rec_F4 - exp['recall_F4']) > tol_recall:
            mismatches.append(
                f'{suite_short} recall_F4 expected {exp["recall_F4"]:.4f} got {got_rec_F4:.4f}'
            )

    summary['reference_check'] = {
        'expected': expected,
        'mismatches': mismatches,
        'reproduces_job_14': len(mismatches) == 0,
    }
    return summary, mismatches


def reference_run(out_dir: str = REFERENCE_OUT_DIR) -> Dict[str, Dict]:
    """Reproduce Job 14 P8A and E2B_3200 numbers; write reference_run_*.json files."""
    print('=== Reference run vs Job 14 ===')
    parquet_meta = load_parquet_meta()
    write_manifest(out_dir, parquet_meta)
    print(f'wrote manifest -> {out_dir}/cleaned_substrate_manifest.json')

    # Map internal labels to the deliverable filenames required by the spec.
    label_to_outname = {'P8A': 'p8a', 'E2B_3200': 'e2b'}

    results: Dict[str, Dict] = {}
    for label in _REF_TARGETS:
        summary, mismatches = _reference_one(label, parquet_meta, out_dir)
        ref_path = os.path.join(out_dir, f'reference_run_{label_to_outname.get(label, label.lower())}.json')
        with open(ref_path, 'w') as f:
            json.dump(summary, f, indent=2)
        results[label] = summary
        verdict = 'PASS' if not mismatches else f'FAIL ({len(mismatches)} mismatches)'
        print(f'\n[{label}] {verdict}')
        for line in mismatches:
            print(f'  - {line}')
        # Also print headline numbers.
        print(f'  n_real F0={summary["n_real_frames_F0"]} F4={summary["n_real_frames_F4"]}')
        print(f'  tau_F0={summary["tau_F0_calibrated"]:.4f}  tau_F4@10%={summary["tau_F4_at_FPR_10pct"]:.4f}')
        for row in summary['per_suite']:
            print(
                f'  {row["fake_suite"]}: recall_F0={row["recall_at_tau_F0_pct"]:.2f}%  '
                f'recall_F4@FPR10={row["recall_at_tau_F4_FPR10_pct"]:.2f}%  '
                f'lift={row["lift_abs_pp_F0_to_F4_FPR10"]:+.2f}pp'
            )
        print(f'  -> {ref_path}')

    overall_pass = all(r['reference_check']['reproduces_job_14'] for r in results.values())
    print('\n=== Overall: ' + ('PASS' if overall_pass else 'FAIL') + ' ===')
    return results


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_fake_csv_arg(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        if '=' not in it:
            raise SystemExit(f'--fake-csv expects suite=path, got: {it}')
        k, v = it.split('=', 1)
        out[k] = v
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference-run', action='store_true',
                   help='Reproduce Job 14 P8A and E2B_3200 numbers and exit.')
    p.add_argument('--ckpt-name', type=str, default=None)
    p.add_argument('--real-csv', type=str, default=None,
                   help='Per-frame CSV for the REAL dev pool (teams_real_all_dev).')
    p.add_argument('--fake-csv', action='append', default=[],
                   help='suite_short=/path/to/fake.csv (repeatable).')
    p.add_argument('--out-dir', type=str, default=REFERENCE_OUT_DIR)
    p.add_argument('--target-fpr', type=float, default=TARGET_FPR_DEFAULT)
    p.add_argument('--write-manifest-only', action='store_true',
                   help='Only (re-)write cleaned_substrate_manifest.json and exit.')
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.write_manifest_only:
        path = write_manifest(args.out_dir, load_parquet_meta())
        print(f'wrote {path}')
        return 0

    if args.reference_run:
        reference_run(args.out_dir)
        return 0

    # Single-checkpoint evaluation.
    if not args.ckpt_name or not args.real_csv:
        p.error('Must pass --reference-run, OR (--ckpt-name AND --real-csv) [+ --fake-csv ...].')

    fake_csvs = _parse_fake_csv_arg(args.fake_csv)
    parquet_meta = load_parquet_meta()
    write_manifest(args.out_dir, parquet_meta)
    summary = evaluate_checkpoint(
        ckpt_name=args.ckpt_name,
        real_csv=args.real_csv,
        fake_csvs=fake_csvs,
        out_dir=args.out_dir,
        target_fpr=args.target_fpr,
        parquet_meta=parquet_meta,
    )
    print(f'\n[{args.ckpt_name}] cleaned eval done.')
    print(f'  n_real F0={summary["n_real_frames_F0"]} -> F4={summary["n_real_frames_F4"]} '
          f'({summary["pct_real_kept_F4"]:.1f}%)')
    print(f'  tau_F0={summary["tau_F0_calibrated"]:.4f}  '
          f'tau_F4@5%={summary["tau_F4_at_FPR_5pct"]:.4f}  '
          f'tau_F4@10%={summary["tau_F4_at_FPR_10pct"]:.4f}')
    for row in summary['per_suite']:
        print(f'  {row["fake_suite"]}: recall_F0={row["recall_at_tau_F0_pct"]:.2f}%  '
              f'recall_F4@FPR10={row["recall_at_tau_F4_FPR10_pct"]:.2f}%  '
              f'recall_F4@FPR5={row["recall_at_tau_F4_FPR5_pct"]:.2f}%  '
              f'lift={row["lift_abs_pp_F0_to_F4_FPR10"]:+.2f}pp')
    return 0


if __name__ == '__main__':
    sys.exit(main())
