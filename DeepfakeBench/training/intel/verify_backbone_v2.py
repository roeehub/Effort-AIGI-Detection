import os
import zipfile
import json
import torch
import shutil
from safetensors.torch import load_file
from transformers import CLIPModel

# Configuration
ZIP_FILE = "custom_vision_backbone.zip"
EXTRACT_DIR = "./verify_temp"

def main():
    # 1. Extraction
    print(f"--- 1. Extracting {ZIP_FILE} ---")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # 2. Filename Audit
    print(f"\n--- 2. Checking Filenames ---")
    files = os.listdir(EXTRACT_DIR)
    print(f"Files found: {files}")
    if "open_clip_model.safetensors" in files:
        print("[FAIL] 'open_clip' filename detected! Anonymization failed.")
        return
    if "model.safetensors" in files:
        print("[PASS] generic 'model.safetensors' found.")
    else:
        print("[WARNING] No model.safetensors found.")

    # 3. Model Loading Test
    print(f"\n--- 3. Loading Weights ---")
    model_path = os.path.join(EXTRACT_DIR, "model.safetensors")
    
    try:
        # Try Method A: Transformers (Strict)
        print("Attempting load via Transformers AutoModel...")
        model = CLIPModel.from_pretrained(EXTRACT_DIR)
        print("[SUCCESS] Loaded via Transformers!")
        print(f"Params: {sum(p.numel() for p in model.parameters())}")
        
    except Exception as e_trans:
        print(f"Transformers load check failed (Expected if config structure differs): {e_trans}")
        print("Falling back to raw SafeTensors load check...")
        
        try:
            # Try Method B: Raw SafeTensors (Robust)
            # This proves the weights are readable and intact, even if config is custom
            st_dict = load_file(model_path)
            print("[SUCCESS] Raw SafeTensors file loaded successfully.")
            
            # Check for a known key to ensure it's a CLIP model
            keys = list(st_dict.keys())
            if any("visual" in k for k in keys):
                print(f"[VERIFIED] Found 'visual' keys in weights. This is a CLIP-like model.")
                print(f"Total keys found: {len(keys)}")
            else:
                print("[WARNING] Weights loaded, but 'visual' keys missing. Check architecture.")
                
        except Exception as e_raw:
            print(f"[FATAL] Could not load weights at all: {e_raw}")

    # Cleanup
    shutil.rmtree(EXTRACT_DIR)
    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    main()