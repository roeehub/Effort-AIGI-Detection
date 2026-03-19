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

def clean_json_metadata(file_path):
    """
    Removes explicit identity fields from JSON configs.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # List of keys that often reveal the model name
    keys_to_remove = [
        "_name_or_path", 
        "model_name", 
        "hub_model_id", 
        "name", 
        "checkpoint_name"
    ]
    
    modified = False
    for key in keys_to_remove:
        if key in data:
            print(f"  [CLEANING] Removing '{key}': {data[key]}")
            del data[key]
            modified = True
            
    # Genericize the architecture name if it's too specific (optional, usually safe to keep generic 'CLIPModel')
    # if "architectures" in data:
    #     print(f"  [INFO] Architecture listed as: {data['architectures']}")

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("  [SUCCESS] JSON metadata sanitized.")
    else:
        print("  [OK] No sensitive metadata found.")

def clean_safetensors_metadata(file_path):
    """
    Loads safetensors weights and saves them back without the original metadata header.
    """
    print(f"  [PROCESSING] scrubbing safetensors metadata...")
    # Load tensors (this drops the metadata by default)
    tensors = load_file(file_path)
    # Save back to disk. By default, save_file writes empty metadata unless specified.
    save_file(tensors, file_path, metadata={"format": "pt"})
    print("  [SUCCESS] Safetensors header scrubbed.")

def clean_pytorch_bin(file_path):
    """
    Loads .bin pickle files and re-saves them to ensure no hidden dict attributes exist.
    """
    print(f"  [PROCESSING] Refreshing PyTorch binary file...")
    # Load onto CPU
    state_dict = torch.load(file_path, map_location='cpu')
    
    # Check if it's a wrapper dict (e.g. has 'state_dict', 'epoch', 'args')
    # If it is, we usually just want the weights for a backbone.
    # However, to be safe and keep it working, we just re-save the object.
    # Torch save does not typically persist external metadata unless it was in the dict keys.
    torch.save(state_dict, file_path)
    print("  [SUCCESS] Binary file refreshed.")

def main():
    print(f"=== Starting Download of {REPO_ID} ===")
    # Download only the necessary files (exclude system files like .gitattributes or README)
    try:
        model_path = snapshot_download(
            repo_id=REPO_ID, 
            local_dir=OUTPUT_DIR,
            ignore_patterns=["*.md", ".gitattributes", "LICENSE", "msgpack"],
            local_dir_use_symlinks=False  # Important: get actual files, not symlinks
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print(f"\n=== Inspecting and Cleaning Files in {OUTPUT_DIR} ===")
    
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"\nChecking: {file}")
            
            if file.endswith(".json"):
                clean_json_metadata(file_path)
            
            elif file.endswith(".safetensors"):
                clean_safetensors_metadata(file_path)
                
            elif file.endswith(".bin"):
                clean_pytorch_bin(file_path)
    
    print("\n=== Verification ===")
    # Check if it really is just the image encoder
    # We load the config to check the structure
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = json.load(f)
            # CLIP usually has text_config and vision_config
            if 'text_config' in conf:
                print("NOTE: The model contains both Vision and Text encoders (standard CLIP).")
                print("If you strictly need ONLY the image encoder, the receiving team can just ignore the text part,")
                print("or you can modify the script to delete keys starting with 'text_model' in the weight files.")
            else:
                print("Verified: Seems to be vision-only configuration.")

    print(f"\n=== Packaging ===")
    # Create zip
    shutil.make_archive(FINAL_PACKAGE_NAME, 'zip', OUTPUT_DIR)
    print(f"SUCCESS: Model packaged as '{FINAL_PACKAGE_NAME}.zip'")
    print(f"You can now send '{FINAL_PACKAGE_NAME}.zip' to the team.")

    # Optional: Cleanup temp dir
    # shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    main()