"""Apply the Option-3 per-identity rule to Slot β lockbox reals to see whether
it would correctly classify each identity as real.

Rule (memory project_blend_unsharp_lever_2026-05-14):
   frac_above_0.6 >= 0.4  AND  count_above_0.9 >= 1   =>  identity is FAKE
Otherwise: identity is REAL.

For lockbox-real frames, we want every identity classified REAL. Any identity
that the rule flags as FAKE would be a chronic-FP NOT rescuable by this rule.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
INP = HERE / 'inputs'
OUT = HERE / 'outputs'

CKPTS = {
    'P8A': ('teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv', 0.916),
    'T5C': ('teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv', 0.831),
    'SLOT_B': ('teams_real_all_lockbox_slot_b_6axis_grl_step3500_frames_report.csv', 0.816),
}

VIDEO_RE = re.compile(r'^(?P<identity>.+?)__s(?P<session>\d+)__seg_(?P<seg>[\d.]+)__real$')


def extract_identity(video_id: str) -> str:
    m = VIDEO_RE.match(video_id)
    if not m:
        return video_id.split('__')[0]
    return m.group('identity')


def apply_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['identity'] = df['video_id'].apply(extract_identity)
    g = df.groupby('identity')['frame_prob']
    out = pd.DataFrame({
        'n_frames': g.size(),
        'frac_above_0_6': g.apply(lambda s: (s > 0.6).mean()),
        'count_above_0_9': g.apply(lambda s: (s > 0.9).sum()),
        'p95_score': g.apply(lambda s: s.quantile(0.95)),
        'max_score': g.max(),
    }).reset_index()
    out['rule_says_fake'] = (out['frac_above_0_6'] >= 0.4) & (out['count_above_0_9'] >= 1)
    return out


def main():
    rows = []
    for name, (fname, _tau) in CKPTS.items():
        df = pd.read_csv(INP / fname)
        ruled = apply_rule(df)
        ruled['ckpt'] = name
        rows.append(ruled)
        print(f'\n=== {name} ===')
        print(ruled[['identity', 'n_frames', 'frac_above_0_6', 'count_above_0_9',
                     'p95_score', 'max_score', 'rule_says_fake']].to_string(index=False))
        n_flagged = int(ruled['rule_says_fake'].sum())
        print(f'Identities flagged FAKE by rule (errors): {n_flagged}/{len(ruled)}')
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(OUT / 'per_identity_rule_verdicts.csv', index=False)


if __name__ == '__main__':
    main()
