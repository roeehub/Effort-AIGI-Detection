"""SVD-aware encoder probe: load the trained T5C / Slot β backbone and probe
whether the encoder's PC1 axis shifts vs vanilla openclip.

Strategy:
  1. Build vanilla openclip ViT-B-16 (datacomp_xl)
  2. Apply SVD wrappers (apply_svd_residual_to_openclip_attn) with rank=736
  3. Load state_dict from the trained ckpt
  4. Wrap in OpenCLIPVisionModelWrapper
  5. Run encode_image on Roy_D + ilan + orel dev PNG frames
  6. Compare PCA / AUC / cosine sim vs the vanilla openclip baseline
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add training root to path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
import open_clip

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / 'dev_cache'
OUT = HERE / 'outputs'
CKPT_CACHE = HERE / 'ckpt_cache'

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

T5C_CKPT = CKPT_CACHE / 'periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth'
SLOTB_CKPT = CKPT_CACHE / 'periodic_effort_20260516_step3500_auc0.9910_eer0.0175.pth'


def build_visual_encoder_with_svd(rank=736, apply_to_mlp=True):
    """Build openclip ViT-B-16 + apply SVD wrappers using the same code path as EffortDetector."""
    from detectors.effort_detector import apply_svd_residual_to_openclip_attn

    print('Loading vanilla openclip...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k'
    )
    visual = model.visual
    print(f'Applying SVD with rank={rank}, apply_to_mlp={apply_to_mlp}...')
    visual = apply_svd_residual_to_openclip_attn(visual, r=rank, block_indices=None, apply_to_mlp=apply_to_mlp)
    return visual, preprocess


def load_backbone_weights(visual, ckpt_path):
    """Load state_dict's backbone.visual.* keys into the SVD-wrapped visual encoder."""
    print(f'Loading ckpt: {ckpt_path.name}')
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    # Strip 'backbone.visual.' prefix
    visual_sd = {}
    for k, v in sd.items():
        if k.startswith('backbone.visual.'):
            visual_sd[k[len('backbone.visual.'):]] = v
    print(f'  Visual sub-state-dict: {len(visual_sd)} keys')
    missing, unexpected = visual.load_state_dict(visual_sd, strict=False)
    print(f'  Missing keys: {len(missing)} (first 3: {missing[:3]})')
    print(f'  Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})')
    return visual


def encode_dir(visual, preprocess, src_dir, identity_map):
    visual.eval()
    visual.to(DEVICE)
    rows = []
    embs = []
    files = sorted(Path(src_dir).glob('*'))
    with torch.no_grad():
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB')
                t = preprocess(img).unsqueeze(0).to(DEVICE)
                feat = visual(t).cpu().numpy()[0]
                embs.append(feat)
                rows.append({'fname': fp.name, 'identity': identity_map.get(fp.name, 'unknown')})
            except Exception as e:
                print(f'  fail on {fp.name}: {e}')
    return rows, np.array(embs) if embs else np.empty((0, 512))


def main():
    # Identity map for dev PNG frames
    dev = pd.read_csv(OUT / 'dev_png_frames_with_props_and_scores.csv')
    id_map = dict(zip(dev['crop_basename'], dev['identity']))

    # Probe T5C (baseline; this is the FT base for both A and B)
    visual_t5c, preprocess = build_visual_encoder_with_svd()
    visual_t5c = load_backbone_weights(visual_t5c, T5C_CKPT)
    rows_t5c, emb_t5c = encode_dir(visual_t5c, preprocess, CACHE, id_map)
    print(f'T5C embeddings: {emb_t5c.shape}')

    # Probe Slot β (already-trained, lockbox-bad 0.0882 FPR)
    del visual_t5c
    torch.mps.empty_cache() if DEVICE == 'mps' else None

    visual_sb, _ = build_visual_encoder_with_svd()
    visual_sb = load_backbone_weights(visual_sb, SLOTB_CKPT)
    rows_sb, emb_sb = encode_dir(visual_sb, preprocess, CACHE, id_map)
    print(f'Slot β embeddings: {emb_sb.shape}')

    # Load vanilla openclip embeddings for reference (we already saved these)
    vanilla = pd.read_csv(OUT / 'encoder_embeddings_dev_png.csv')
    emb_vanilla = vanilla[[c for c in vanilla.columns if c.startswith('e')]].values

    # Compare PC1 separations
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    def probe_separation(emb, label):
        df = pd.DataFrame(rows_t5c if label != 'vanilla' else vanilla[['crop_basename', 'identity']].rename(columns={'crop_basename': 'fname'}).to_dict('records'))
        # For T5C/Slot β we use rows_t5c; for vanilla we use the prior csv
        if label != 'vanilla':
            df = pd.DataFrame(rows_t5c if label == 'T5C' else rows_sb)
        y = (df['identity'] == 'Roy_D').values.astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(emb)
        probs = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                                  X, y, cv=5, method='predict_proba')[:, 1]
        auc = roc_auc_score(y, probs)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        roy_pc1 = proj[y == 1, 0].mean()
        clean_mask = df['identity'].isin(['ilan', 'orel']).values
        clean_pc1 = proj[clean_mask, 0].mean()

        emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        roy_c = emb[y == 1].mean(0); roy_c /= np.linalg.norm(roy_c)
        clean_c = emb[clean_mask].mean(0); clean_c /= np.linalg.norm(clean_c)
        within_roy = float(np.mean(emb_n[y == 1] @ roy_c))
        within_clean = float(np.mean(emb_n[clean_mask] @ clean_c))
        across = float(np.mean(emb_n[y == 1] @ clean_c))

        print(f'\n=== {label} encoder ===')
        print(f'  Roy_D-vs-clean linear probe AUC: {auc:.4f}')
        print(f'  PC1 centroid: Roy_D={roy_pc1:+.2f}, clean={clean_pc1:+.2f}, separation={clean_pc1 - roy_pc1:+.2f}')
        print(f'  Cosine: within Roy_D={within_roy:.4f}, within clean={within_clean:.4f}, across={across:.4f}')

    probe_separation(emb_vanilla, 'vanilla')
    probe_separation(emb_t5c, 'T5C')
    probe_separation(emb_sb, 'Slot β')

    # Save embeddings
    pd.DataFrame(rows_t5c).join(pd.DataFrame(emb_t5c, columns=[f'e{i}' for i in range(emb_t5c.shape[1])])).to_csv(OUT / 'encoder_embeddings_t5c_trained.csv', index=False)
    pd.DataFrame(rows_sb).join(pd.DataFrame(emb_sb, columns=[f'e{i}' for i in range(emb_sb.shape[1])])).to_csv(OUT / 'encoder_embeddings_slot_b_trained.csv', index=False)


if __name__ == '__main__':
    main()
