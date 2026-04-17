"""Check what was actually sampled in the DF40 meta-analysis."""
import pandas as pd
import numpy as np

df = pd.read_csv("../meta_analysis_properties.csv")
df40f = df[df["source"] == "df40_fake"]
df40r = df[df["source"] == "df40_real"]

print(f"DF40 fake: {len(df40f)} images")
print(f"DF40 real: {len(df40r)} images\n")

# The filenames lost their method subdirectory during meta-analysis
# (see meta_analysis_enhancer.py line 451: fb.name.split('/')[-1])
# So we can't tell which method each image came from
print("DF40 fake filenames (first 10):")
for f in sorted(df40f["filename"].tolist())[:10]:
    print(f"  {f}")

print("\nDF40 real filenames (first 10):")
for f in sorted(df40r["filename"].tolist())[:10]:
    print(f"  {f}")

print("\n" + "="*70)
print("DF40 FAKE statistics:")
print(f"  Sharpness (Lap var): mean={df40f['sharpness_laplacian_var'].mean():.1f}, "
      f"median={df40f['sharpness_laplacian_var'].median():.1f}, "
      f"std={df40f['sharpness_laplacian_var'].std():.1f}")
print(f"  Model prob: mean={df40f['model_fake_prob'].mean():.3f}, "
      f"median={df40f['model_fake_prob'].median():.3f}")
print(f"  Accuracy @0.5: {(df40f['model_fake_prob'] > 0.5).mean()*100:.1f}%")

print("\nDF40 REAL statistics:")
print(f"  Sharpness (Lap var): mean={df40r['sharpness_laplacian_var'].mean():.1f}, "
      f"median={df40r['sharpness_laplacian_var'].median():.1f}, "
      f"std={df40r['sharpness_laplacian_var'].std():.1f}")
print(f"  Model prob: mean={df40r['model_fake_prob'].mean():.3f}, "
      f"median={df40r['model_fake_prob'].median():.3f}")
print(f"  Accuracy @0.5 (real=correct if prob<0.5): {(df40r['model_fake_prob'] < 0.5).mean()*100:.1f}%")

# CRITICAL: The meta-analysis sampled randomly from ALL fake/ methods
# including 12 methods NOT in training (StyleGAN, DiT, ddim, etc.)
# Let's check the holdout AUC to compare
print("\n" + "="*70)
print("KEY QUESTION: The meta-analysis sampled from fake/ which has 29 methods.")
print("Training only used 17 methods (from the pair JSON).")
print("The 12 excluded methods include GAN-generated faces (StyleGAN2/3/XL, VQGAN)")
print("and diffusion models (DiT, SiT, RDDM, ddim) — these produce fundamentally")
print("different images than face-swaps/reenactments.")
print()
print("If the 150 sampled fakes include GAN/diffusion outputs that the model")
print("never saw during training, the 53.3% accuracy is misleading — it's mixing")
print("in-distribution and completely OOD fakes in one number.")
print()
print("The holdout AUC of 0.9893 is computed only on the 17 training methods")
print("(identity-split), suggesting the model IS good at those methods.")

# Check sharpness distribution — GAN outputs might have very different sharpness
print("\n" + "="*70)
print("SHARPNESS DISTRIBUTION of DF40 fake sample:")
bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 200), (200, 1000)]
for lo, hi in bins:
    mask = (df40f["sharpness_laplacian_var"] >= lo) & (df40f["sharpness_laplacian_var"] < hi)
    n = mask.sum()
    if n > 0:
        acc = (df40f.loc[mask, "model_fake_prob"] > 0.5).mean() * 100
        print(f"  Sharpness [{lo:4d}-{hi:4d}): {n:3d} images, accuracy={acc:.0f}%")
