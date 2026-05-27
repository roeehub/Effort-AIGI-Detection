"""Embed the dor anchor pools and check where they fall in the embedding
space relative to Roy_D (dev PNG) and training reals.

This validates the anchor_aware setup: if the false-flag pool frames
inhabit a similar encoder region as Roy_D, then anchoring on them at
training time should generalize to Roy_D. If they're in a different
region, Slot A's anchor_aware lever may not generalize.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def embed_dir(model, preprocess, src_dir: Path, cohort_label: str):
    files = sorted(src_dir.glob('*'))
    embs = []
    names = []
    with torch.no_grad():
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB')
                t = preprocess(img).unsqueeze(0).to(DEVICE)
                feat = model.encode_image(t).cpu().numpy()[0]
                embs.append(feat)
                names.append(fp.name)
            except Exception:
                continue
    rows = [{'fname': n, 'cohort': cohort_label} for n in names]
    return rows, np.array(embs) if embs else np.empty((0, 512))


def main():
    print('Loading OpenCLIP ViT-B-16...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
    model.eval().to(DEVICE)

    # Load prior dev embeddings
    dev = pd.read_csv(OUT / 'encoder_embeddings_dev_png.csv')
    dev_emb = dev[[c for c in dev.columns if c.startswith('e')]].values
    dev_rows = dev[['crop_basename', 'identity']].rename(columns={'crop_basename': 'fname', 'identity': 'cohort'}).to_dict('records')
    print(f'Dev: {len(dev_emb)}')

    # Embed false-flag anchor pool
    aa_rows, aa_emb = embed_dir(model, preprocess, HERE / 'anchor_pool_cache', 'anchor_pool_falseflag')
    print(f'Anchor false-flag pool: {len(aa_emb)}')

    # Embed correct pool (model-corrects-on)
    cp_rows, cp_emb = embed_dir(model, preprocess, HERE / 'correct_pool_cache', 'correct_pool_whiteish')
    print(f'Correct pool: {len(cp_emb)}')

    # Stack
    all_rows = dev_rows + aa_rows + cp_rows
    all_emb = np.vstack([dev_emb, aa_emb, cp_emb])
    print(f'Total: {all_emb.shape}')

    # Fit PCA on dev only
    pca = PCA(n_components=2)
    pca.fit(dev_emb)
    proj = pca.transform(all_emb)

    df = pd.DataFrame(all_rows)
    df['PC1'] = proj[:, 0]
    df['PC2'] = proj[:, 1]

    print('\n=== Per-cohort PCA centroids (PCA fit on dev PNG only) ===')
    for c in df['cohort'].unique():
        sub = df[df['cohort'] == c]
        print(f'  {c} (n={len(sub)}): PC1={sub["PC1"].mean():.3f} ±{sub["PC1"].std():.3f}, '
              f'PC2={sub["PC2"].mean():.3f} ±{sub["PC2"].std():.3f}')

    # Cosine sim to Roy_D centroid
    roy_mask = (df['cohort'] == 'Roy_D').values
    roy_centroid = all_emb[roy_mask].mean(axis=0)
    roy_c_norm = roy_centroid / np.linalg.norm(roy_centroid)

    clean_mask = df['cohort'].isin(['ilan', 'orel']).values
    clean_centroid = all_emb[clean_mask].mean(axis=0)
    clean_c_norm = clean_centroid / np.linalg.norm(clean_centroid)

    print('\n=== Cosine similarity to Roy_D and clean centroids ===')
    for cohort in ['Roy_D', 'ilan', 'orel', 'anchor_pool_falseflag', 'correct_pool_whiteish']:
        mask = (df['cohort'] == cohort).values
        if mask.sum() == 0: continue
        sub_emb = all_emb[mask]
        sub_norm = sub_emb / np.linalg.norm(sub_emb, axis=1, keepdims=True)
        sim_roy = (sub_norm @ roy_c_norm).mean()
        sim_clean = (sub_norm @ clean_c_norm).mean()
        print(f'  {cohort} (n={mask.sum()}): cos_sim to Roy_D={sim_roy:.4f}, cos_sim to clean={sim_clean:.4f}, '
              f'Roy_D-bias={sim_roy - sim_clean:+.4f}')

    # The critical test: do anchor-pool frames cluster near Roy_D?
    # If yes, training on them anchors the Roy_D-style region.
    # If they cluster near training reals, anchor_aware adds redundant signal.
    print('\n=== KEY DIAGNOSTIC: where do anchor-pool frames live? ===')
    aa_centroid_pc1 = df[df['cohort'] == 'anchor_pool_falseflag']['PC1'].mean()
    cp_centroid_pc1 = df[df['cohort'] == 'correct_pool_whiteish']['PC1'].mean()
    roy_centroid_pc1 = df[df['cohort'] == 'Roy_D']['PC1'].mean()
    clean_centroid_pc1 = df[df['cohort'].isin(['ilan', 'orel'])]['PC1'].mean()

    print(f'  PC1 axis: Roy_D={roy_centroid_pc1:.2f}, clean(ilan/orel)={clean_centroid_pc1:.2f}')
    print(f'  Anchor false-flag pool PC1: {aa_centroid_pc1:.2f}')
    print(f'  Correct pool PC1: {cp_centroid_pc1:.2f}')

    # If anchor pool is between Roy_D and clean, the anchoring signal partly
    # overlaps with Roy_D region.
    # If anchor pool is at clean side, anchor_aware adds redundant signal.

    df.to_csv(OUT / 'embeddings_with_anchor_pools.csv', index=False)


if __name__ == '__main__':
    main()
