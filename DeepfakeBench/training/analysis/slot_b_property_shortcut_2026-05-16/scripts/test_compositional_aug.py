"""Follow-up 1: test whether a compositional augmentation can push training
real frames into the 6+ band region.

The aug chain (applied jointly with probability 1.0 for the test, p=0.1-0.2
in practice):
  1. Gaussian blur (sigma 3-6)        -> sharpness < 142, edge_density < 0.25
  2. Luma multiplication (× 0.6-0.85) -> luma_mean < 130
  3. LAB a-channel shift (+ or - 20)  -> lab_a_dev > 16
  4. Resize up                        -> min_dim > 266
  5. Re-JPEG (q 70-90)                -> file_size > 110210 maybe, edge_density
"""
from __future__ import annotations
import os, sys, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'scripts'))
from compute_properties import compute_props

CACHE_IN = HERE / 'training_cache'
CACHE_OUT = HERE / 'aug_cache'
OUT = HERE / 'outputs'
CACHE_OUT.mkdir(exist_ok=True)


def push_into_band_region(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a compositional aug chain to walk into the 6+ band region.

    Tuned to produce Roy_D-like values: sharpness ~67, luma ~108, lab_a_dev
    ~18, min_dim ~278, edge_density ~0.20, file_size ~110kB.
    """
    img = img_bgr.astype(np.float32)
    # 1. Mild Gaussian blur — sharpness 50-100, edge_density 0.15-0.25
    sigma = rng.uniform(0.8, 1.6)
    ksize = max(3, int(2 * round(2 * sigma) + 1))
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    # 2. Luma multiplication — luma_mean ~108
    factor = rng.uniform(0.75, 0.90)
    img *= factor
    img = np.clip(img, 0, 255)
    # 3. LAB a-channel shift — lab_a_dev around 18
    img_u8 = img.astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    shift = rng.choice([+12, -12, +14, -14, +10, -10])
    lab_a = lab[..., 1].astype(np.int16) + shift
    lab[..., 1] = np.clip(lab_a, 0, 255).astype(np.uint8)
    img_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 4. Resize up — min_dim 270-310
    target = rng.randint(275, 310)
    img_u8 = cv2.resize(img_u8, (target, target), interpolation=cv2.INTER_LINEAR)
    return img_u8


def main():
    src_files = sorted(CACHE_IN.glob('*.jpg'))[:100]
    print(f'Sampling {len(src_files)} training reals for aug test')
    rng = random.Random(0)

    rows_aug = []
    rows_orig = []
    for fp in src_files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None: continue

        # Compute on original
        props_o = compute_props(img, fp.stat().st_size)
        props_o['fname'] = fp.name
        props_o['kind'] = 'orig'
        rows_orig.append(props_o)

        # Apply aug + re-save as JPEG to mimic training transport
        aug = push_into_band_region(img, rng)
        out_path = CACHE_OUT / f'AUG_{fp.stem}.jpg'
        # Use PNG to mimic Roy_D's file format (production format per user)
        out_path = CACHE_OUT / f'AUG_{fp.stem}.png'
        cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        # Re-read so file_size_bytes reflects post-aug compression
        aug_reread = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        props_a = compute_props(aug_reread, out_path.stat().st_size)
        props_a['fname'] = out_path.name
        props_a['kind'] = 'aug'
        rows_aug.append(props_a)

    df = pd.concat([pd.DataFrame(rows_orig), pd.DataFrame(rows_aug)], ignore_index=True)

    BANDS = {
        'sharpness_laplacian': ('< 142', lambda s: s < 142),
        'luma_mean': ('< 130', lambda s: s < 130),
        'lab_a_dev': ('> 16', lambda s: s > 16),
        'lab_a_std': ('in [10.3, 11.9]', lambda s: (s >= 10.3) & (s <= 11.9)),
        'lab_b_dev': ('in [14.7, 17.7]', lambda s: (s >= 14.7) & (s <= 17.7)),
        'skin_frac_hsv': ('> 0.88', lambda s: s > 0.88),
        'edge_density': ('< 0.25', lambda s: s < 0.25),
        'min_dim': ('> 266', lambda s: s > 266),
        'file_size_bytes': ('> 110210', lambda s: s > 110210),
    }

    def count_bands(row):
        n = 0
        for col, (_, fn) in BANDS.items():
            if fn(pd.Series([row[col]])).iloc[0]:
                n += 1
        return n
    df['n_bands'] = df.apply(count_bands, axis=1)
    df.to_csv(OUT / 'aug_test_properties.csv', index=False)

    print('\n=== Per-band hit rate: orig vs aug ===')
    print(f'{"property":<22} {"band":<22} {"%orig":>8} {"%aug":>8}')
    for col, (label, fn) in BANDS.items():
        po = fn(df[df['kind'] == 'orig'][col]).mean() * 100
        pa = fn(df[df['kind'] == 'aug'][col]).mean() * 100
        print(f'{col:<22} {label:<22} {po:>7.2f}% {pa:>7.2f}%')

    print('\n=== n_bands distribution: orig vs aug ===')
    for kind in ['orig', 'aug']:
        sub = df[df['kind'] == kind]
        print(f'\n[{kind}] n={len(sub)}')
        counts = sub['n_bands'].value_counts().sort_index()
        for n, c in counts.items():
            print(f'  n_bands={n}: {c} ({100*c/len(sub):.1f}%)')
        print(f'  mean n_bands: {sub["n_bands"].mean():.2f}')

    print('\n=== Property values: aug samples vs Roy_D target ===')
    aug_sub = df[df['kind'] == 'aug']
    PROPS = ['sharpness_laplacian', 'luma_mean', 'lab_a_dev', 'skin_frac_hsv',
             'edge_density', 'min_dim', 'file_size_bytes']
    print(f'{"property":<22} {"aug_med":>10} {"roy_d_med":>12}')
    roy_d = {
        'sharpness_laplacian': 67.4,
        'luma_mean': 108.2,
        'lab_a_dev': 18.0,
        'skin_frac_hsv': 0.902,
        'edge_density': 0.202,
        'min_dim': 277.5,
        'file_size_bytes': 111664,
    }
    for p in PROPS:
        print(f'{p:<22} {aug_sub[p].median():>10.2f} {roy_d[p]:>12.2f}')


if __name__ == '__main__':
    main()
