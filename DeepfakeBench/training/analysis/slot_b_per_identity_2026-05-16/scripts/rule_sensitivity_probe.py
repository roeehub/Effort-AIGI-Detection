"""Rule-sensitivity probe — what's the smallest rule perturbation that rescues
Slot β dor_shkedi AND preserves the rule's correctness on the 5-identity
lockbox real pool for T5C / P8A?

The default Option-3 rule is:
   frac_above_0.6 >= 0.4  AND  count_above_0.9 >= 1   ==>  FAKE

Slot β dor_shkedi has frac_0.6 = 0.505 and count_0.9 = 1, max_score = 0.909.
T5C dor_shkedi has frac_0.6 = 0.279 and count_0.9 = 4, max_score = 0.922.

Probe two adjustments:
  A) Bump count threshold from >0.9 to >0.92  (dor_shkedi max=0.909 stays under)
  B) Bump frac threshold from 0.4 to 0.55

For each adjustment, compute: how many of the 5 lockbox identities flip
classification on each ckpt? Goal: minimal lockbox-FPR after applied rule.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
INP = HERE / 'inputs'
OUT = HERE / 'outputs'

CKPTS = {
    'P8A': 'teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv',
    'T5C': 'teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv',
    'SLOT_B': 'teams_real_all_lockbox_slot_b_6axis_grl_step3500_frames_report.csv',
}
TAUS = {'P8A': 0.916, 'T5C': 0.831, 'SLOT_B': 0.816}

VIDEO_RE = re.compile(r'^(?P<identity>.+?)__s(?P<session>\d+)__seg_(?P<seg>[\d.]+)__real$')


def extract_identity(video_id: str) -> str:
    m = VIDEO_RE.match(video_id)
    if not m:
        return video_id.split('__')[0]
    return m.group('identity')


def apply_rule(df, frac_thresh=0.6, frac_min=0.4, count_thresh=0.9, count_min=1):
    df = df.copy()
    df['identity'] = df['video_id'].apply(extract_identity)
    g = df.groupby('identity')['frame_prob']
    out = pd.DataFrame({
        'n_frames': g.size(),
        f'frac_above_{frac_thresh}': g.apply(lambda s: (s > frac_thresh).mean()),
        f'count_above_{count_thresh}': g.apply(lambda s: (s > count_thresh).sum()),
        'max_score': g.max(),
    }).reset_index()
    out['rule_says_fake'] = (
        (out[f'frac_above_{frac_thresh}'] >= frac_min)
        & (out[f'count_above_{count_thresh}'] >= count_min)
    )
    return out


def compute_post_rule_lockbox_fpr(df, identities_kept_as_real, tau):
    """Deployment-style rule: rule acts as a per-frame suppression layer.

    A frame is predicted FAKE iff (frame_prob >= tau) AND (rule says identity is FAKE).
    Otherwise the frame is predicted REAL.

    Two interpretations exist; this is the practical one (rule overrides per-frame
    only on identities the rule says FAKE; on rule-says-REAL identities, all frames
    are forced to REAL). The alternative (rule = pure identity-level verdict)
    gives much worse numbers when the rule misfires on a real identity (100%
    FPR on that identity). Both are reported in the output for completeness.
    """
    df = df.copy()
    df['identity'] = df['video_id'].apply(extract_identity)
    df['above_tau'] = (df['frame_prob'] >= tau).astype(int)
    rule_fake = ~df['identity'].isin(identities_kept_as_real)

    # Per-frame suppression: frame is FP iff above_tau AND rule says identity FAKE
    suppression_fp = int(((df['above_tau'] == 1) & rule_fake).sum())
    # Identity-level verdict: all frames of rule-says-FAKE identities are FP
    identity_verdict_fp = int(rule_fake.sum())

    total = len(df)
    return {
        'rule_real_identities': sorted(identities_kept_as_real),
        'rule_fake_identities': sorted(set(df['identity']) - set(identities_kept_as_real)),
        'total_frames': total,
        'suppression_fp_frames': suppression_fp,
        'suppression_fpr': suppression_fp / total,
        'identity_verdict_fp_frames': identity_verdict_fp,
        'identity_verdict_fpr': identity_verdict_fp / total,
    }


def main():
    variants = [
        ('default (frac>=0.4 + count_0.9>=1)', dict(frac_thresh=0.6, frac_min=0.4,
                                                      count_thresh=0.9, count_min=1)),
        ('A: count_0.92>=1', dict(frac_thresh=0.6, frac_min=0.4,
                                   count_thresh=0.92, count_min=1)),
        ('B: frac_0.6>=0.55', dict(frac_thresh=0.6, frac_min=0.55,
                                    count_thresh=0.9, count_min=1)),
        ('C: count_0.95>=1', dict(frac_thresh=0.6, frac_min=0.4,
                                   count_thresh=0.95, count_min=1)),
    ]
    rows = []
    for label, kwargs in variants:
        print(f'\n=== Variant: {label} ===')
        for name, fname in CKPTS.items():
            df = pd.read_csv(INP / fname)
            ruled = apply_rule(df, **kwargs)
            real_ids = set(ruled.loc[~ruled['rule_says_fake'], 'identity'])
            stats = compute_post_rule_lockbox_fpr(df, real_ids, TAUS[name])
            stats['ckpt'] = name
            stats['variant'] = label
            rows.append(stats)
            print(f'  {name}: suppression_FPR = {stats["suppression_fpr"]*100:.2f}%  |  '
                  f'identity_verdict_FPR = {stats["identity_verdict_fpr"]*100:.2f}%  '
                  f'(rule says REAL on {len(real_ids)}/5 identities)')
    pd.DataFrame(rows).to_csv(OUT / 'rule_sensitivity_variants.csv', index=False)


if __name__ == '__main__':
    main()
