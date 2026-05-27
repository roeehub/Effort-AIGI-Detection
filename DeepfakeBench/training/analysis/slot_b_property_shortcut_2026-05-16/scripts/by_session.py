"""Per-session decomposition of dor_shkedi over-fires.

dor_shkedi videos are tagged `dor_shkedi__s{NN}__seg_{X.Y}__real`. The `s{NN}`
is the recording session. If over-firing concentrates on a few sessions,
that's strong evidence of capture-session shortcut.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'

VIDEO_RE = re.compile(r'^(?P<identity>.+?)__s(?P<session>\d+)__seg_(?P<seg>[\d.]+)__real$')


def session_of(vid: str):
    m = VIDEO_RE.match(vid)
    if m:
        return int(m.group('session'))
    # real_dor doesn't follow this pattern
    return None


def main():
    df = pd.read_csv(OUT / 'frames_with_props_and_scores.csv')
    df['session'] = df['video_id'].apply(session_of)

    print('=== Per-session breakdown for dor_shkedi ===')
    dor = df[df['identity'] == 'dor_shkedi']
    sess = dor.groupby('session').agg(
        n_frames=('SLOT_B_overfire', 'size'),
        n_overfires=('SLOT_B_overfire', 'sum'),
        slot_b_mean_score=('SLOT_B', 'mean'),
        slot_b_p95=('SLOT_B', lambda s: s.quantile(0.95)),
        sharpness_med=('sharpness_laplacian', 'median'),
        lab_b_dev_med=('lab_b_dev', 'median'),
        lab_a_std_med=('lab_a_std', 'median'),
        skin_frac_med=('skin_frac_hsv', 'median'),
        luma_mean_med=('luma_mean', 'median'),
        min_dim_med=('min_dim', 'median'),
    ).reset_index()
    sess['rate'] = sess['n_overfires'] / sess['n_frames']
    sess = sess.sort_values('rate', ascending=False)
    print(sess.to_string(index=False))
    sess.to_csv(OUT / 'per_session_dor_shkedi.csv', index=False)

    print('\n=== Per-session breakdown for bla_bla_chow ===')
    bbc = df[df['identity'] == 'bla_bla_chow']
    sess_bbc = bbc.groupby('session').agg(
        n_frames=('SLOT_B_overfire', 'size'),
        n_overfires=('SLOT_B_overfire', 'sum'),
        slot_b_mean_score=('SLOT_B', 'mean'),
        slot_b_p95=('SLOT_B', lambda s: s.quantile(0.95)),
        sharpness_med=('sharpness_laplacian', 'median'),
        lab_b_dev_med=('lab_b_dev', 'median'),
        skin_frac_med=('skin_frac_hsv', 'median'),
    ).reset_index()
    sess_bbc['rate'] = sess_bbc['n_overfires'] / sess_bbc['n_frames']
    sess_bbc = sess_bbc.sort_values('rate', ascending=False)
    print(sess_bbc.to_string(index=False))
    sess_bbc.to_csv(OUT / 'per_session_bla_bla_chow.csv', index=False)

    # Also: for dor_shkedi, what fraction of over-fires concentrate in top-k sessions?
    print('\n=== dor_shkedi: cumulative concentration of over-fires by session (sorted by rate) ===')
    sess_sorted = sess.sort_values('n_overfires', ascending=False)
    sess_sorted['cum_overfires'] = sess_sorted['n_overfires'].cumsum()
    sess_sorted['cum_frac_of_total'] = sess_sorted['cum_overfires'] / sess_sorted['n_overfires'].sum()
    print(sess_sorted[['session', 'n_frames', 'n_overfires', 'cum_overfires',
                        'cum_frac_of_total', 'rate', 'sharpness_med', 'lab_b_dev_med',
                        'skin_frac_med']].to_string(index=False))


if __name__ == '__main__':
    main()
