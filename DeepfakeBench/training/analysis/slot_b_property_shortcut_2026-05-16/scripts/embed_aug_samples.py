"""Embed the augmented samples + a sample of training reals + dev frames.
Do they cluster near Roy_D in embedding space?"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def embed_dir(model, preprocess, src_dir: Path, label: str, max_n: int = None):
    files = sorted(src_dir.glob('*'))
    if max_n:
        files = files[:max_n]
    rows = []
    embs = []
    with torch.no_grad():
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB')
                t = preprocess(img).unsqueeze(0).to(DEVICE)
                feat = model.encode_image(t).cpu().numpy()[0]
                embs.append(feat)
                rows.append({'fname': fp.name, 'cohort': label})
            except Exception as e:
                continue
    return rows, np.array(embs) if embs else np.empty((0, 512))


def main():
    print('Loading OpenCLIP ViT-B-16...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
    model.eval().to(DEVICE)

    # Load the prior dev embeddings + identities
    dev = pd.read_csv(OUT / 'encoder_embeddings_dev_png.csv')
    dev_emb = dev[[c for c in dev.columns if c.startswith('e')]].values
    dev_rows = dev[['crop_basename', 'identity']].to_dict('records')
    for r in dev_rows:
        r['cohort'] = r['identity']
        r['fname'] = r.pop('crop_basename')
    print(f'Dev: {len(dev_emb)} frames')

    # Embed augmented samples
    aug_rows, aug_emb = embed_dir(model, preprocess, HERE / 'aug_cache', 'AUG_synthetic')
    print(f'Aug: {len(aug_emb)} frames')

    # Embed a sample of training reals (max 100)
    train_rows, train_emb = embed_dir(model, preprocess, HERE / 'training_cache', 'training_real', max_n=100)
    print(f'Training reals: {len(train_emb)} frames')

    # Stack
    all_rows = dev_rows + aug_rows + train_rows
    all_emb = np.vstack([dev_emb, aug_emb, train_emb])

    print(f'\nTotal embeddings: {all_emb.shape}')

    # Fit PCA on dev only (to keep axes consistent with Test 3 in probe)
    pca = PCA(n_components=2)
    pca.fit(dev_emb)
    proj_all = pca.transform(all_emb)

    df = pd.DataFrame(all_rows)
    df['PC1'] = proj_all[:, 0]
    df['PC2'] = proj_all[:, 1]

    print('\n=== PCA-2D centroids by cohort ===')
    for c in df['cohort'].unique():
        sub = df[df['cohort'] == c]
        print(f'  {c} (n={len(sub)}): PC1={sub["PC1"].mean():.3f} ±{sub["PC1"].std():.3f}, '
              f'PC2={sub["PC2"].mean():.3f} ±{sub["PC2"].std():.3f}')

    # Distance from each aug sample to nearest Roy_D embedding
    print('\n=== Augmented samples: distance to Roy_D distribution ===')
    roy_mask = (df['cohort'] == 'Roy_D').values
    roy_emb_norm = dev_emb[roy_mask[:len(dev_emb)]] / np.linalg.norm(dev_emb[roy_mask[:len(dev_emb)]], axis=1, keepdims=True)
    roy_centroid = dev_emb[roy_mask[:len(dev_emb)]].mean(axis=0)
    ilan_orel_mask = df['cohort'].isin(['ilan', 'orel']).values
    clean_centroid = dev_emb[ilan_orel_mask[:len(dev_emb)]].mean(axis=0)

    # For each aug sample, compute cosine sim to Roy_D centroid vs Clean centroid
    aug_emb_arr = aug_emb if len(aug_emb) else np.empty((0, 512))
    train_emb_arr = train_emb
    if len(aug_emb_arr) > 0:
        roy_c_norm = roy_centroid / np.linalg.norm(roy_centroid)
        clean_c_norm = clean_centroid / np.linalg.norm(clean_centroid)

        aug_norm = aug_emb_arr / np.linalg.norm(aug_emb_arr, axis=1, keepdims=True)
        sim_to_roy = aug_norm @ roy_c_norm
        sim_to_clean = aug_norm @ clean_c_norm
        roy_minus_clean = sim_to_roy - sim_to_clean
        print(f'  Aug n={len(aug_emb_arr)}')
        print(f'    cos_sim to Roy_D centroid:    mean={sim_to_roy.mean():.4f}, median={np.median(sim_to_roy):.4f}')
        print(f'    cos_sim to clean centroid:    mean={sim_to_clean.mean():.4f}, median={np.median(sim_to_clean):.4f}')
        print(f'    closer to Roy_D than clean: {(roy_minus_clean > 0).mean()*100:.1f}% of aug samples')

        train_norm = train_emb_arr / np.linalg.norm(train_emb_arr, axis=1, keepdims=True)
        sim_train_roy = train_norm @ roy_c_norm
        sim_train_clean = train_norm @ clean_c_norm
        print(f'\n  Original training reals n={len(train_emb_arr)} (no aug):')
        print(f'    cos_sim to Roy_D centroid:    mean={sim_train_roy.mean():.4f}, median={np.median(sim_train_roy):.4f}')
        print(f'    cos_sim to clean centroid:    mean={sim_train_clean.mean():.4f}, median={np.median(sim_train_clean):.4f}')
        print(f'    closer to Roy_D than clean: {((sim_train_roy - sim_train_clean) > 0).mean()*100:.1f}% of training reals')

    df.to_csv(OUT / 'pca_with_aug_and_training.csv', index=False)
    print('\nSaved pca_with_aug_and_training.csv')


if __name__ == '__main__':
    main()
