"""Follow-up 2: Encoder embedding probe.

Compute encoder embeddings for the 194 dev PNG frames (Roy_D + ilan + orel)
and test whether Roy_D separates from ilan+orel at the embedding level.

Strategy:
  - Use vanilla OpenCLIP ViT-B-16 (datacomp_xl_s13b_b90k) as encoder. This
    matches the BASE that T5C / Slot β were fine-tuned from.
  - Test 1: cosine similarity within Roy_D vs within ilan+orel vs across.
  - Test 2: linear probe (logistic regression) on embeddings → Roy_D vs not.
    If AUC is high, the encoder DOES distinguish Roy_D from clean controls.
  - Test 3: PCA visualization (2D) of all 194 frames.

If the encoder DOES separate them, the augmentation lever should work
(anchoring augmented reals in that region of feature space). If the encoder
DOESN'T separate them, the problem is downstream (head/decision boundary)
and augmentation won't help.
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
CACHE = HERE / 'dev_cache'
OUT = HERE / 'outputs'

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def main():
    print('Loading OpenCLIP ViT-B-16 (datacomp_xl_s13b_b90k)...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
    model.eval()
    model = model.to(DEVICE)
    print('Model loaded.')

    # Load dev frame data — frame to identity mapping
    dev_props = pd.read_csv(OUT / 'dev_png_frames_with_props_and_scores.csv')

    rows = []
    embeddings = []
    with torch.no_grad():
        for i, row in dev_props.iterrows():
            fp = CACHE / row['crop_basename']
            if not fp.exists():
                continue
            img = Image.open(fp).convert('RGB')
            t = preprocess(img).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(t).cpu().numpy()[0]
            embeddings.append(feat)
            rows.append({
                'crop_basename': row['crop_basename'],
                'identity': row['identity'],
                'SLOT_B': row['SLOT_B'],
                'T5C': row['T5C'],
                'P8A': row['P8A'],
                'SLOT_B_overfire': row['SLOT_B_overfire'],
            })
            if (i+1) % 30 == 0:
                print(f'  [{i+1}/{len(dev_props)}]')

    emb = np.array(embeddings)
    print(f'\nEmbeddings shape: {emb.shape}')
    df = pd.DataFrame(rows)
    df_emb = pd.DataFrame(emb, columns=[f'e{i}' for i in range(emb.shape[1])])
    df_full = pd.concat([df.reset_index(drop=True), df_emb], axis=1)
    df_full.to_csv(OUT / 'encoder_embeddings_dev_png.csv', index=False)
    print(f'Saved encoder_embeddings_dev_png.csv ({df_full.shape})')

    # Normalize embeddings (cosine similarity prep)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # Test 1: cosine similarity
    print('\n=== Test 1: cosine similarity within/across cohorts ===')
    roy_mask = (df['identity'] == 'Roy_D').values
    clean_mask = df['identity'].isin(['ilan', 'orel']).values
    roy_emb = emb_norm[roy_mask]
    clean_emb = emb_norm[clean_mask]

    def avg_within(X):
        if len(X) < 2: return np.nan
        sims = X @ X.T
        n = len(X)
        return (sims.sum() - n) / (n * (n - 1))  # exclude self-similarity

    def avg_across(A, B):
        return (A @ B.T).mean()

    print(f'  within Roy_D (n={roy_mask.sum()}): {avg_within(roy_emb):.4f}')
    print(f'  within ilan+orel (n={clean_mask.sum()}): {avg_within(clean_emb):.4f}')
    print(f'  across Roy_D <-> ilan+orel: {avg_across(roy_emb, clean_emb):.4f}')

    # Test 2: linear probe
    print('\n=== Test 2: linear probe (Roy_D vs ilan+orel) ===')
    y = roy_mask.astype(int)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(emb)
    # 5-fold CV
    from sklearn.model_selection import cross_val_predict
    probs = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                              X_s, y, cv=5, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, probs)
    print(f'  Cross-validated AUC: {auc:.4f}')

    # Also: predict over-fire status (not just identity)
    y_over = df['SLOT_B_overfire'].values.astype(int)
    if y_over.sum() > 0 and y_over.sum() < len(y_over):
        probs_over = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                                       X_s, y_over, cv=5, method='predict_proba')[:, 1]
        auc_over = roc_auc_score(y_over, probs_over)
        print(f'  Cross-validated AUC for Slot β over-fire (any identity): {auc_over:.4f}')

    # Test 3: PCA 2D
    print('\n=== Test 3: PCA 2D projection ===')
    pca = PCA(n_components=2)
    proj = pca.fit_transform(emb)
    print(f'  Explained variance ratio: {pca.explained_variance_ratio_}')

    print('  Per-identity PCA centroid:')
    for ident in ['Roy_D', 'ilan', 'orel']:
        mask = (df['identity'] == ident).values
        if mask.sum() == 0: continue
        centroid = proj[mask].mean(axis=0)
        print(f'    {ident}: PC1={centroid[0]:.3f}, PC2={centroid[1]:.3f}, n={mask.sum()}')

    # Save PCA results
    pca_df = pd.DataFrame({
        'crop_basename': df['crop_basename'],
        'identity': df['identity'],
        'PC1': proj[:, 0],
        'PC2': proj[:, 1],
        'SLOT_B': df['SLOT_B'],
        'SLOT_B_overfire': df['SLOT_B_overfire'],
    })
    pca_df.to_csv(OUT / 'pca_dev_png_2d.csv', index=False)


if __name__ == '__main__':
    main()
