"""Per-identity decomposition of lockbox real over-fires for the 3 main ckpts.

For each ckpt (P8A, T5C, Slot β step3500): count frames > τ_cal per identity,
compute identity-level FPR, and check whether the 0.0882 Slot β FPR concentrates
on chronic-6 identities or distributes across new ones.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
INP = HERE / 'inputs'
OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = {
    'P8A': ('teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv', 0.916),
    'T5C': ('teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv', 0.831),
    'SLOT_B': ('teams_real_all_lockbox_slot_b_6axis_grl_step3500_frames_report.csv', 0.816),
}

CHRONIC6 = {
    'dor_shkedi', 'bla_bla_chow', 'Roy_D', 'xiang', 'dor',
    'PC_Generator',
}

VIDEO_RE = re.compile(r'^(?P<identity>.+?)__s(?P<session>\d+)__seg_(?P<seg>[\d.]+)__real$')


def extract_identity(video_id: str) -> str:
    m = VIDEO_RE.match(video_id)
    if not m:
        return video_id.split('__')[0]
    return m.group('identity')


def per_identity_summary(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    df = df.copy()
    df['identity'] = df['video_id'].apply(extract_identity)
    df['above_tau'] = (df['frame_prob'] >= tau).astype(int)
    agg = (df.groupby('identity')
             .agg(n_frames=('frame_prob', 'size'),
                  n_overfires=('above_tau', 'sum'),
                  mean_score=('frame_prob', 'mean'),
                  p95_score=('frame_prob', lambda s: s.quantile(0.95)))
             .reset_index())
    agg['frac_overfires'] = agg['n_overfires'] / agg['n_frames']
    agg = agg.sort_values('n_overfires', ascending=False)
    return agg


def main():
    summaries = {}
    headlines = []
    for name, (fname, tau) in CKPTS.items():
        df = pd.read_csv(INP / fname)
        per_id = per_identity_summary(df, tau)
        per_id['ckpt'] = name
        per_id['tau'] = tau
        per_id['is_chronic6'] = per_id['identity'].isin(CHRONIC6)
        summaries[name] = per_id

        total_frames = int(per_id['n_frames'].sum())
        total_over = int(per_id['n_overfires'].sum())
        chronic_over = int(per_id.loc[per_id['is_chronic6'], 'n_overfires'].sum())
        non_chronic_over = total_over - chronic_over
        identities_with_overfires = int((per_id['n_overfires'] > 0).sum())
        chronic_with_overfires = int(((per_id['n_overfires'] > 0) & per_id['is_chronic6']).sum())

        headlines.append({
            'ckpt': name,
            'tau': tau,
            'total_frames': total_frames,
            'total_overfires': total_over,
            'overall_fpr': total_over / total_frames,
            'chronic6_overfires': chronic_over,
            'chronic6_frac_of_total': (chronic_over / total_over) if total_over else 0.0,
            'non_chronic_overfires': non_chronic_over,
            'identities_with_overfires': identities_with_overfires,
            'chronic_with_overfires': chronic_with_overfires,
            'identities_total': int(per_id['identity'].nunique()),
        })

        per_id.to_csv(OUT / f'per_identity_{name.lower()}.csv', index=False)

    summary_df = pd.DataFrame(headlines)
    summary_df.to_csv(OUT / 'summary_chronic6_share.csv', index=False)
    print(summary_df.to_string(index=False))
    print()
    print('=== Top 10 over-firing identities per ckpt ===')
    for name, per_id in summaries.items():
        print(f'\n[{name}] (tau={CKPTS[name][1]})')
        top = per_id.head(10)[['identity', 'n_frames', 'n_overfires', 'frac_overfires', 'p95_score', 'is_chronic6']]
        print(top.to_string(index=False))


if __name__ == '__main__':
    main()
