import json
import random
from pathlib import Path
import fsspec # You might need to install with: pip install fsspec gcsfs

# --- Configuration ---
# IMPORTANT: Set your bucket name here.
# This is the name of the bucket itself, not the full gs:// path.
BUCKET_NAME = "df40-frames" 
# Example from your previous output

# --- Sampling Parameters ---
SAMPLE_PERCENTAGE = 0.05  # 5% of the total files
SEED = 42                 # Use a seed for reproducible random sampling
OUTPUT_FILENAME = "partial_manifest.json"

# --- Main Logic ---
def create_sampled_manifest():
    """
    Connects to GCS, lists all image files, randomly samples them,
    and saves the list to a local JSON file.
    """
    print("Connecting to Google Cloud Storage...")
    # The gcsfs library handles authentication automatically if you are logged in
    # via the gcloud CLI or running on a GCP machine with appropriate permissions.
    try:
        fs = fsspec.filesystem("gcs")
    except Exception as e:
        print(f"Error connecting to GCS. Ensure you have authenticated.")
        print(f"Original error: {e}")
        return

    print(f"Listing all frame objects in '{BUCKET_NAME}'...")
    print("This is the slow step and may take a long time depending on bucket size.")
    
    # 1. Get all paths first (the slow part)
    # The pattern includes the bucket name. The '**' makes it recursive.
    glob_pattern = f"{BUCKET_NAME}/**"
    try:
        all_object_paths = fs.glob(glob_pattern)
    except Exception as e:
        print(f"Error during fs.glob(). Does the bucket '{BUCKET_NAME}' exist and do you have permissions?")
        print(f"Original error: {e}")
        return
        
    # Filter for images and prepend the 'gs://' protocol
    all_image_paths = [
        f"gs://{p}" for p in all_object_paths
        if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    
    total_found = len(all_image_paths)
    if total_found == 0:
        print("Warning: No image files were found. The manifest will be empty.")
        return
        
    print(f"Found {total_found:,} total image files.")

    # 2. Calculate how many samples to take
    num_samples = int(total_found * SAMPLE_PERCENTAGE)
    print(f"Preparing to sample {num_samples:,} files ({SAMPLE_PERCENTAGE:.1%})...")

    # 3. Perform the random sampling
    random.seed(SEED)
    sampled_paths = random.sample(all_image_paths, num_samples)

    # 4. Save the sampled list to a JSON file
    output_path = Path(OUTPUT_FILENAME)
    print(f"Saving {len(sampled_paths):,} sampled paths to '{output_path}'...")
    with open(output_path, 'w') as f:
        json.dump(sampled_paths, f, indent=2)

    print("\nDone. Your partial manifest file has been created successfully.")
    print(f"You can now use '{output_path}' in your main application for faster loading.")

if __name__ == "__main__":
    create_sampled_manifest()