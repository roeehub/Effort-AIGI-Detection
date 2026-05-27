# Track H.2 — ArcFace identity-purity audit

Generated 2026-04-28 by `analysis/arcface_training_audit_2026-04-29/run_audit.py`.
Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (dev+lockbox; proxy for training-set identity purity).
All compute: n_jobs=1 (per `feedback_sklearn_njobs.md`).

## Methodology

- Bucket by (identity_key, label). Mixing real+fake under one identity_key is
  guaranteed to depress purity because deepfake morphs ArcFace embedding by
  design — that's not corruption, it's the swap working. Within-label purity
  isolates the actual H.2 hypothesis (mixed-identity content under one key).
- 512-dim L2-normalized ArcFace embeddings.
- Per bucket: centroid = mean(embeddings); purity = mean cosine-sim to centroid.
- k=2 split-test: KMeans(k=2) silhouette score (≥ 0.2 suggests two sub-identities).
- Flagged corrupted if mean purity < 0.85 OR silhouette ≥ 0.2.
- **Decision rule operates on REAL-bucket purity only.**

## Headline

- Buckets audited: **41** (real=25, fake=16)
- **Avg purity (REAL buckets): 0.772** (threshold 0.85)
- Avg purity (fake buckets): 0.814 *(lower-by-design, see Methodology)*
- REAL buckets flagged purity < 0.85: **18** / 25
- Fake buckets flagged purity < 0.85: **9** / 16
- Buckets flagged by k=2 split (≥ 0.2): **0**

**Verdict: REAL identity corruption confirmed** (REAL avg purity < threshold).

## Implication for P14 / P15

- **If corruption confirmed (REAL avg < 0.85)**: drop `contrastive_regularization` in any retry (per Plan v6 §3.11).
- **If pure**: contrastive_reg can stay. Audit refutes the H9 corruption hypothesis from Plan v3 §1.6.
- **Fake-bucket purity is informational** — it indicates how distinct the deepfake morphs are from the source identity. Lower fake-bucket purity = more aggressive face-swap = more identity-loss in the training signal.

## Buckets flagged (worst purity first)

| identity_key | label | n_frames | centroid_purity | k2_silhouette |
|---|---|---|---|---|
| PC_Generator__s34 | real | 92 | 0.551 | nan |
| Test_Cam__s41 | real | 815 | 0.556 | nan |
| orel | real | 35 | 0.577 | nan |
| deeplive_dor | fake | 545 | 0.588 | nan |
| Test_Cam__s76 | real | 214 | 0.636 | nan |
| Cam_Test__s35 | fake | 365 | 0.648 | nan |
| bla_bla_chow | real | 311 | 0.656 | nan |
| PC_Generator__s13 | real | 212 | 0.696 | nan |
| PC_Generator__s45 | real | 91 | 0.704 | nan |
| Test_Cam__s76 | fake | 138 | 0.713 | nan |
| ilan | real | 29 | 0.714 | nan |
| Cam_Test__s38 | fake | 85 | 0.723 | nan |
| Xiang_Xiang2_Feng | real | 301 | 0.732 | nan |
| PC_Generator__s14 | real | 103 | 0.735 | nan |
| Cam_Test__s32 | real | 81 | 0.764 | nan |
| Xiang_Xiang2_Feng__s23 | real | 102 | 0.776 | nan |
| Cam_Test__s46 | fake | 124 | 0.778 | nan |
| Q__s6 | real | 54 | 0.781 | nan |
| Xiang_Xiang2_Feng | fake | 135 | 0.794 | nan |
| Cam_Test__s38 | real | 85 | 0.796 | nan |
| Cam_Test__s33 | fake | 334 | 0.805 | nan |
| Test_Cam__s53 | fake | 165 | 0.812 | nan |
| bla_bla_chow__s2 | real | 180 | 0.813 | nan |
| Cam_Test__s32 | fake | 235 | 0.829 | nan |
| Md_noyn_Sharker__s15 | real | 682 | 0.837 | nan |
| PC_Generator__s22 | real | 227 | 0.842 | nan |
| PC_Generator__s15 | real | 29 | 0.845 | nan |

## All buckets (sorted by purity)

| identity_key | label | n_frames | centroid_purity | k2_silhouette |
|---|---|---|---|---|
| PC_Generator__s34 | real | 92 | 0.551 | nan |
| Test_Cam__s41 | real | 815 | 0.556 | nan |
| orel | real | 35 | 0.577 | nan |
| deeplive_dor | fake | 545 | 0.588 | nan |
| Test_Cam__s76 | real | 214 | 0.636 | nan |
| Cam_Test__s35 | fake | 365 | 0.648 | nan |
| bla_bla_chow | real | 311 | 0.656 | nan |
| PC_Generator__s13 | real | 212 | 0.696 | nan |
| PC_Generator__s45 | real | 91 | 0.704 | nan |
| Test_Cam__s76 | fake | 138 | 0.713 | nan |
| ilan | real | 29 | 0.714 | nan |
| Cam_Test__s38 | fake | 85 | 0.723 | nan |
| Xiang_Xiang2_Feng | real | 301 | 0.732 | nan |
| PC_Generator__s14 | real | 103 | 0.735 | nan |
| Cam_Test__s32 | real | 81 | 0.764 | nan |
| Xiang_Xiang2_Feng__s23 | real | 102 | 0.776 | nan |
| Cam_Test__s46 | fake | 124 | 0.778 | nan |
| Q__s6 | real | 54 | 0.781 | nan |
| Xiang_Xiang2_Feng | fake | 135 | 0.794 | nan |
| Cam_Test__s38 | real | 85 | 0.796 | nan |
| Cam_Test__s33 | fake | 334 | 0.805 | nan |
| Test_Cam__s53 | fake | 165 | 0.812 | nan |
| bla_bla_chow__s2 | real | 180 | 0.813 | nan |
| Cam_Test__s32 | fake | 235 | 0.829 | nan |
| Md_noyn_Sharker__s15 | real | 682 | 0.837 | nan |
| PC_Generator__s22 | real | 227 | 0.842 | nan |
| PC_Generator__s15 | real | 29 | 0.845 | nan |
| Noyn_sharker__s23 | fake | 324 | 0.856 | nan |
| Test_Cam__s73 | real | 251 | 0.856 | nan |
| dor_shkedi__s16 | fake | 78 | 0.864 | nan |
| PC_Generator__s8 | real | 101 | 0.869 | nan |
| dor_shkedi__s16 | real | 31 | 0.887 | nan |
| PC_Generator__s3 | fake | 118 | 0.892 | nan |
| bla_bla_chow__s1 | real | 68 | 0.898 | nan |
| PC_Generator__s15 | fake | 91 | 0.904 | nan |
| dor_shkedi | real | 275 | 0.912 | nan |
| PC_Generator__s4 | real | 9 | 0.926 | nan |
| Test_Cam__s73 | fake | 101 | 0.931 | nan |
| PC_Generator__s4 | fake | 30 | 0.935 | nan |
| Chikara_Takahashi__s22 | real | 42 | 0.936 | nan |
| PC_Generator__s9 | fake | 46 | 0.955 | nan |

