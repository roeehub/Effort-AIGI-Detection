import os
import json
import torch
import shutil
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

# Configuration
REPO_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
OUTPUT_DIR = "./anonymous_backbone_temp"
FINAL_PACKAGE_NAME = "custom_vision_backbone"

# Mapping specific filenames to generic ones to hide origin and fix loading
RENAME_MAP = {
    "open_clip_config.json": "config.json",
    "open_clip_model.safetensors": "model.safetensors",
    "open_clip_pytorch_model.bin": "pytorch_model.bin"
}

def clean_json_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keys to remove
    keys_to_remove = ["_name_or_path", "model_name", "hub_model_id", "name", "checkpoint_name"]
    
    modified = False
    for key in keys_to_remove:
        if key in data:
            print(f"  [CLEANING] Removing '{key}'")
            del data[key]
            modified = True
            
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def clean_safetensors_metadata(file_path):
    print(f"  [PROCESSING] scrubbing safetensors metadata...")
    tensors = load_file(file_path)
    # Save back with empty metadata (or generic format tag only)
    save_file(tensors, file_path, metadata={"format": "pt"})
    print("  [SUCCESS] Safetensors header scrubbed.")

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"=== Starting Download of {REPO_ID} ===")
    try:
        snapshot_download(
            repo_id=REPO_ID, 
            local_dir=OUTPUT_DIR,
            ignore_patterns=["*.md", ".gitattributes", "LICENSE", "msgpack"],
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print(f"\n=== Renaming Files for Anonymity & Compatibility ===")
    for original, new in RENAME_MAP.items():
        old_path = os.path.join(OUTPUT_DIR, original)
        new_path = os.path.join(OUTPUT_DIR, new)
        if os.path.exists(old_path):
            print(f"  [RENAME] {original} -> {new}")
            os.rename(old_path, new_path)

    print(f"\n=== Cleaning Metadata ===")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file == "config.json":
                clean_json_metadata(file_path)
            elif file == "model.safetensors":
                clean_safetensors_metadata(file_path)
            # We can delete the .bin file if we have safetensors to save space/confusion
            # or keep it if the team needs legacy support. Let's keep it but clean it if needed.
    
    print(f"\n=== Packaging ===")
    if os.path.exists(f"{FINAL_PACKAGE_NAME}.zip"):
        os.remove(f"{FINAL_PACKAGE_NAME}.zip")
        
    shutil.make_archive(FINAL_PACKAGE_NAME, 'zip', OUTPUT_DIR)
    print(f"SUCCESS: Model packaged as '{FINAL_PACKAGE_NAME}.zip'")
    
    # Clean up temp folder so we don't accidentally use unzipped files
    shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    main()