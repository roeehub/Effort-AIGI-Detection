#!/usr/bin/env python3
"""Download remaining audit data (fakes + teams) that the bash script missed."""
import os, subprocess, random, shutil, sys, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "audit_data"
GSUTIL = os.path.expanduser("~/google-cloud-sdk/bin/gsutil")
DF40 = "gs://df40-frames-recropped-rfa85"
TEAMS = "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
METHODS = ["simswap", "facedancer", "blendface", "e4s", "inswap", "mobileswap", "uniface"]

os.makedirs(f"{OUT}/df40_crops", exist_ok=True)
os.makedirs(f"{OUT}/teams_real", exist_ok=True)

def gsutil_ls(path):
    """List GCS path, return list of URLs."""
    try:
        r = subprocess.run(
            [GSUTIL, "ls", path],
            capture_output=True, text=True, timeout=30
        )
        return [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception as e:
        print(f"  WARNING: gsutil ls failed for {path}: {e}")
        return []

def download_files(gcs_urls, dest_dir, tag, max_files=999):
    """Download GCS files to dest_dir with tag prefix to avoid collisions."""
    tmpdir = f"{dest_dir}/.tmp_{tag}_{os.getpid()}"
    os.makedirs(tmpdir, exist_ok=True)
    
    # Filter to image files only
    urls = [u for u in gcs_urls if u.endswith('.png') or u.endswith('.jpg')][:max_files]
    if not urls:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 0
    
    # Download all at once
    proc = subprocess.run(
        [GSUTIL, "-o", "GSUtil:parallel_thread_count=4",
         "-o", "GSUtil:parallel_process_count=2",
         "-m", "cp"] + urls + [tmpdir + "/"],
        capture_output=True, text=True, timeout=120
    )
    
    # Rename with tag prefix
    count = 0
    for f in sorted(os.listdir(tmpdir)):
        fp = os.path.join(tmpdir, f)
        if os.path.isfile(fp):
            dest = os.path.join(dest_dir, f"{tag}__{f}")
            os.rename(fp, dest)
            count += 1
    shutil.rmtree(tmpdir, ignore_errors=True)
    return count


# ── Section 2: DF40 FAKES ──────────────────────────────────────────
print("[2/3] DF40 fake frames (7 methods × ~15)...")
rng = random.Random(42)
fake_total = 0

for method in METHODS:
    print(f"      {method}...", flush=True)
    id_dirs = gsutil_ls(f"{DF40}/fake/{method}/")
    if not id_dirs:
        print(f"        (no dirs found)")
        continue
    
    # Sample 3 identity-pair folders
    sampled = rng.sample(id_dirs, min(3, len(id_dirs)))
    
    for id_dir in sampled:
        id_name = id_dir.rstrip('/').split('/')[-1]
        tag = f"fake_{method}_{id_name}"
        
        # List files in this identity folder
        files = gsutil_ls(id_dir)
        n = download_files(files, f"{OUT}/df40_crops", tag, max_files=5)
        fake_total += n
        print(f"        {id_name}: {n} frames")
    
    time.sleep(0.5)

print(f"      Fake total: {fake_total}")


# ── Section 3: TEAMS REAL ──────────────────────────────────────────
print("\n[3/3] Teams-v2 real frames (30 diverse samples)...")
all_samples = gsutil_ls(f"{TEAMS}/samples/")
print(f"      Found {len(all_samples)} samples")

if not all_samples:
    print("      WARNING: No samples found")
    teams_total = 0
else:
    rng2 = random.Random(42)
    sampled = rng2.sample(all_samples, min(30, len(all_samples)))
    
    teams_total = 0
    for i, sample_dir in enumerate(sampled, 1):
        sample_name = sample_dir.rstrip('/').split('/')[-1]
        real_dir = sample_dir.rstrip('/') + "/frames/real/"
        files = gsutil_ls(real_dir)
        n = download_files(files, f"{OUT}/teams_real", sample_name, max_files=999)
        teams_total += n
        print(f"      [{i}/30] {sample_name}: {n} real frames")
        time.sleep(0.3)

print(f"      Teams real total: {teams_total}")

# ── Summary ────────────────────────────────────────────────────────
df40_count = len([f for f in os.listdir(f"{OUT}/df40_crops") if not f.startswith('.')])
teams_count = len([f for f in os.listdir(f"{OUT}/teams_real") if not f.startswith('.')])
print(f"\n=== Download Complete ===")
print(f"  DF40 crops:    {df40_count} files → {OUT}/df40_crops/")
print(f"  Teams real:    {teams_count} files → {OUT}/teams_real/")
print(f"\nNext: python tools/lighting_showcase.py audit \\")
print(f"  --images-dir {OUT}/df40_crops \\")
print(f"  --real-captures-dir {OUT}/teams_real \\")
print(f"  --shadow-p 0.10 --gamma-up-p 0.12 \\")
print(f"  --output {OUT}/proposed_lighting_audit.png")
