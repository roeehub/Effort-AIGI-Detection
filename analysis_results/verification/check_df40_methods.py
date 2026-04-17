"""Check which DF40 methods were sampled in the meta-analysis."""
import pandas as pd
import subprocess
import re

# 1. Load the meta-analysis data
df = pd.read_csv('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/analysis_results/meta_analysis_properties.csv')

df40_fake = df[df['source'] == 'df40_fake']
df40_real = df[df['source'] == 'df40_real']

print(f"DF40 fake samples: {len(df40_fake)}")
print(f"DF40 real samples: {len(df40_real)}")
print()

# 2. Show the filenames to understand the structure
print("=== DF40 FAKE sample filenames (first 10) ===")
for f in sorted(df40_fake['filename'].tolist())[:10]:
    print(f"  {f}")
print()

print("=== DF40 REAL sample filenames (first 10) ===")
for f in sorted(df40_real['filename'].tolist())[:10]:
    print(f"  {f}")
print()

# 3. Check if filenames contain method info or just numeric IDs
print("=== Filename patterns ===")
fake_fnames = df40_fake['filename'].tolist()
# Check if any path components reveal the method
has_slash = [f for f in fake_fnames if '/' in f]
print(f"Filenames with path separators: {len(has_slash)}")
if has_slash:
    print(f"  Examples: {has_slash[:5]}")
no_slash = [f for f in fake_fnames if '/' not in f]
print(f"Filenames without path separators: {len(no_slash)}")
if no_slash:
    print(f"  Examples: {no_slash[:5]}")
print()

# 4. Check DF40 stats from the meta-analysis
print("=== DF40 FAKE properties ===")
for col in ['sharpness_laplacian_var', 'edge_density', 'noise_estimate', 'model_fake_prob']:
    if col in df40_fake.columns:
        print(f"  {col}: mean={df40_fake[col].mean():.4f}, median={df40_fake[col].median():.4f}, std={df40_fake[col].std():.4f}")

print()
print("=== DF40 REAL properties ===")
for col in ['sharpness_laplacian_var', 'edge_density', 'noise_estimate', 'model_fake_prob']:
    if col in df40_real.columns:
        print(f"  {col}: mean={df40_real[col].mean():.4f}, median={df40_real[col].median():.4f}, std={df40_real[col].std():.4f}")

print()

# 5. DF40 fake accuracy distribution
probs = df40_fake['model_fake_prob']
print("=== DF40 FAKE accuracy breakdown ===")
print(f"  Accuracy @0.5 threshold: {(probs > 0.5).mean()*100:.1f}%")
print(f"  % with prob > 0.9 (confident fake): {(probs > 0.9).mean()*100:.1f}%")
print(f"  % with prob < 0.1 (confident real): {(probs < 0.1).mean()*100:.1f}%")
print(f"  % with prob 0.3-0.7 (uncertain): {((probs > 0.3) & (probs < 0.7)).mean()*100:.1f}%")
print()

# 6. DF40 real FPR distribution
probs_real = df40_real['model_fake_prob']
print("=== DF40 REAL FPR breakdown ===")
print(f"  Correct (prob < 0.5): {(probs_real < 0.5).mean()*100:.1f}%")
print(f"  FP (prob > 0.5): {(probs_real > 0.5).mean()*100:.1f}%")
print(f"  Confident FP (prob > 0.9): {(probs_real > 0.9).mean()*100:.1f}%")
print()

# 7. Look at the meta-analysis script to see how DF40 was sampled
print("=== Checking meta-analysis sampling code ===")
import ast
meta_script = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/meta_analysis_enhancer.py'
try:
    with open(meta_script, 'r') as f:
        content = f.read()
    # Find the DF40 sampling section
    for i, line in enumerate(content.split('\n')):
        if 'df40' in line.lower() or 'DF40' in line:
            start = max(0, i-2)
            end = min(len(content.split('\n')), i+3)
            for j in range(start, end):
                print(f"  L{j+1}: {content.split(chr(10))[j]}")
            print()
except Exception as e:
    print(f"  Could not read: {e}")

# 8. Check the training config to see which DF40 methods are used
print("=== Checking training config for DF40 methods ===")
import yaml
import glob

config_paths = [
    '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/config/defaults.yaml',
]
# Also check experiment configs
config_paths += glob.glob('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/*.yaml')

for cp in config_paths:
    try:
        with open(cp, 'r') as f:
            cfg = yaml.safe_load(f)
        # Look for df40 related config
        def find_df40_keys(d, prefix=''):
            if isinstance(d, dict):
                for k, v in d.items():
                    if 'df40' in str(k).lower() or 'df40' in str(v).lower() if isinstance(v, str) else False:
                        print(f"  {cp}: {prefix}{k} = {v}")
                    if isinstance(v, (dict, list)):
                        find_df40_keys(v, prefix=f"{prefix}{k}.")
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    if isinstance(v, str) and 'df40' in v.lower():
                        print(f"  {cp}: {prefix}[{i}] = {v}")
                    elif isinstance(v, (dict, list)):
                        find_df40_keys(v, prefix=f"{prefix}[{i}].")
        find_df40_keys(cfg)
    except Exception as e:
        pass

# 9. Check the DF40 paired data source to see method filtering
print()
print("=== Checking DF40 paired data source code ===")
df40_source_files = glob.glob('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/*df40*')
df40_source_files += glob.glob('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/*paired*')
for f in df40_source_files:
    print(f"  Found: {f}")
    with open(f, 'r') as fh:
        content = fh.read()
    # Find method filtering logic
    for i, line in enumerate(content.split('\n')):
        if any(kw in line.lower() for kw in ['method', 'source_target', 'target_source', 'filter', 'exclude', 'include']):
            print(f"    L{i+1}: {line.rstrip()}")

print()
print("=== Checking GCS bucket structure for DF40 ===")
print("  (listing DF40 fake subdirectories from GCS...)")
try:
    result = subprocess.run(
        ['gsutil', 'ls', 'gs://deepfake-detection-training-data/DF40/fake/'],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        dirs = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        print(f"  Found {len(dirs)} method directories:")
        for d in sorted(dirs):
            method = d.rstrip('/').split('/')[-1]
            print(f"    {method}")
    else:
        print(f"  Error: {result.stderr[:200]}")
except Exception as e:
    print(f"  Error: {e}")
