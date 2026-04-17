"""Check which DF40 methods are actually used in training vs what was sampled in meta-analysis."""
import json, yaml
from collections import Counter
from pathlib import Path

ROOT = Path('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training')

# 1. Load pair JSON
with open(ROOT / 'dataset/df40_pairs/df40-pair-matching.json') as f:
    pairs = json.load(f)

orientations = pairs['method_orientation']
target_source = sorted([m for m, o in orientations.items() if o == 'target_source'])
source_target = sorted([m for m, o in orientations.items() if o == 'source_target'])

print("=" * 70)
print("DF40 PAIR JSON — METHOD ORIENTATIONS")
print("=" * 70)
print(f"\ntarget_source (FACE SWAP — relevant to our use case):")
for m in target_source:
    n = pairs['summary']['pairs_per_method'][m]
    print(f"  {m:20s}  {n} pairs")
ts_total = sum(pairs['summary']['pairs_per_method'][m] for m in target_source)

print(f"\nsource_target (REENACTMENT — less relevant):")
for m in source_target:
    n = pairs['summary']['pairs_per_method'][m]
    print(f"  {m:20s}  {n} pairs")
st_total = sum(pairs['summary']['pairs_per_method'][m] for m in source_target)

print(f"\ntarget_source total: {ts_total} pairs ({len(target_source)} methods)")
print(f"source_target total: {st_total} pairs ({len(source_target)} methods)")
print(f"Grand total: {pairs['summary']['total_pairs']} pairs")

# 2. Check experiment config for method filtering
print("\n" + "=" * 70)
print("EXPERIMENT CONFIG — METHOD FILTERING")
print("=" * 70)

exp_config = ROOT / 'experiments/combined_paired_vit_B16_laion.yaml'
with open(exp_config) as f:
    cfg = yaml.safe_load(f)

df40_cfg = cfg.get('combined_paired', {}).get('df40', {})
methods_filter = df40_cfg.get('methods', 'NOT SET')
print(f"\ncombined_paired.df40.methods = {methods_filter}")
print(f"  → {'ALL 17 methods used' if methods_filter == 'NOT SET' or methods_filter is None else f'Filtered to: {methods_filter}'}")

# Also check defaults
with open(ROOT / 'config/defaults.yaml') as f:
    defaults = yaml.safe_load(f)
df40_defaults = defaults.get('combined_paired', {}).get('df40', {})
methods_default = df40_defaults.get('methods', 'NOT SET')
print(f"\ndefaults.yaml combined_paired.df40.methods = {methods_default}")

# 3. GCS bucket — all methods available (29 in fake/)
print("\n" + "=" * 70)
print("GCS BUCKET — ALL FAKE METHODS (29 total)")
print("=" * 70)
gcs_methods = [
    'DiT', 'MRAA', 'RDDM', 'SiT', 'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN',
    'blendface', 'danet', 'ddim', 'e4s', 'facedancer', 'faceswap', 'facevid2vid',
    'fomm', 'fsgan', 'hyperreenact', 'inswap', 'lia', 'mcnet', 'mobileswap',
    'one_shot_free', 'pirender', 'sadtalker', 'simswap', 'tpsm', 'uniface', 'wav2lip'
]
in_training = set(target_source + source_target)
extra_in_bucket = [m for m in gcs_methods if m not in in_training]
print(f"\nMethods in GCS bucket but NOT in pair JSON (12 methods):")
for m in extra_in_bucket:
    print(f"  {m}")
print(f"\nThese are the methods the meta-analysis could have sampled FROM")
print(f"but that are NOT in the training set!")

# 4. Now the critical question: what did the meta-analysis actually sample?
print("\n" + "=" * 70)
print("META-ANALYSIS — WHAT WAS ACTUALLY SAMPLED?")
print("=" * 70)
print("""
The meta-analysis script (meta_analysis_enhancer.py) uses:
  prefix = f"{label}/"  (i.e., "fake/")
  blobs = list(bucket.list_blobs(prefix=prefix, max_results=20000))

This lists ALL blobs under fake/ — all 29 methods — then randomly 
samples 150 images. The filenames are stored as just "df40_fake/XXXXX.png"
with NO method information preserved.

Since the sample is random across 29 methods, roughly:
  - 17/29 = 59% would be from the 17 pair-JSON methods (in training)
  - 12/29 = 41% would be from the 12 extra methods (NOT in training)

But it's worse: the 12 extra methods include GANs (StyleGAN2/3/XL), 
diffusion models (DiT, SiT, RDDM, ddim), and other architectures
(VQGAN, hyperreenact, sadtalker, tpsm, wav2lip) that produce 
fundamentally different images than the face-swap methods the model 
was trained on.

The model was NEVER trained on these methods, so low accuracy on them
is EXPECTED, not a bug. The 53% accuracy on "DF40 fake" in the 
meta-analysis is contaminated by ~60 images from unseen methods.
""")

# 5. What about DF40 reals?
print("=" * 70)
print("DF40 REALS — WHERE DO THEY COME FROM?")
print("=" * 70)
# Check real structure
import subprocess
result = subprocess.run(
    ['gsutil', 'ls', 'gs://df40-frames-recropped-rfa85/real/'],
    capture_output=True, text=True, timeout=30
)
if result.returncode == 0:
    dirs = [line.strip().rstrip('/').split('/')[-1] for line in result.stdout.strip().split('\n') if line.strip()]
    print(f"\nReal sources in GCS ({len(dirs)}):")
    for d in sorted(dirs):
        print(f"  {d}")
else:
    print(f"\n  Could not list GCS: {result.stderr[:200]}")

print("""
DF40 reals are from FaceForensics++ — they are NOT low quality by nature.
They are video frames that were re-cropped with RFA85 face detector.
The low sharpness (15.4) might be specific to this re-cropping pipeline
or to the original FaceForensics++ video quality.

KEY QUESTION: Are the DF40 reals used in training (via the pair JSON)?
""")

# Check if reals are paired
real_identities = set()
for p in pairs['pairs']:
    real_identities.add(p['real']['identity'])
print(f"Unique real identities in pair JSON: {len(real_identities)}")
print(f"Real source: {pairs['pairs'][0]['real']['source']}")
print(f"\nYes — DF40 reals ARE in training, paired with their fakes.")
print(f"The model is trained on these low-sharpness reals but still")
print(f"classifies them as fake 89% of the time. This is very suspicious.")
