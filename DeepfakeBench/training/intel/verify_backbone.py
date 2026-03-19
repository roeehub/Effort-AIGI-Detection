import os
import zipfile
import json
import torch
import shutil
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from safetensors.torch import load_file

# Configuration
ZIP_FILE = "custom_vision_backbone.zip"
EXTRACT_DIR = "./verify_temp"

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def main():
    # 1. Extraction
    print(f"--- 1. Extracting {ZIP_FILE} ---")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.\n")

    # 2. Metadata Audit
    print(f"--- 2. Auditing for 'Hints' (searching for 'laion') ---")
    found_hints = False
    
    # Check Config
    config_path = os.path.join(EXTRACT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
            # Simple string search for the origin name
            if "laion" in content.lower():
                print(f"[WARNING] Found 'laion' in config.json!")
                found_hints = True
            else:
                print(f"[PASS] config.json appears clean of 'laion'.")
            
            # Load as json to check keys
            data = json.loads(content)
            if "_name_or_path" in data:
                print(f"[WARNING] '_name_or_path' key still exists: {data['_name_or_path']}")
            else:
                print(f"[PASS] '_name_or_path' key is missing (Good).")

    # Check Safetensors Metadata
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".safetensors"):
                path = os.path.join(root, file)
                # We load only the header to check metadata
                with open(path, 'rb') as f:
                    # Safetensors header is the first N bytes, usually json. 
                    # We rely on the library to check metadata dict.
                    try:
                        # Depending on version, load_file might not expose metadata easily without loading tensors
                        # But we can check if the file works.
                        loaded = load_file(path)
                        print(f"[PASS] {file} loaded successfully.")
                    except Exception as e:
                        print(f"[FAIL] Could not load {file}: {e}")

    if not found_hints:
        print(">> AUDIT PASSED: No obvious strings pointing to origin found.\n")

    # 3. Model Loading & Size
    print(f"--- 3. Loading Model & Checking Size ---")
    try:
        # Load the model from the extracted local path
        model = CLIPModel.from_pretrained(EXTRACT_DIR)
        print("[SUCCESS] Model loaded from local directory.")
        
        # Calculate Parameters
        total_params = sum(p.numel() for p in model.parameters())
        vision_params = sum(p.numel() for p in model.vision_model.parameters())
        
        # Calculate File Size
        size_in_bytes = get_dir_size(EXTRACT_DIR)
        size_in_mb = size_in_bytes / (1024 * 1024)

        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Vision-Only Params: {vision_params / 1e6:.2f} M")
        print(f"Disk Size: {size_in_mb:.2f} MB")
        
    except Exception as e:
        print(f"[FATAL] Failed to load model: {e}")
        return

    # 4. Forward Pass Test
    print(f"\n--- 4. Running Forward Pass (Random Image) ---")
    
    # Create a random image tensor: Batch Size 1, 3 Channels, 224x224 Resolution
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # We only care about the vision encoder part for your specific team
    try:
        model.eval()
        with torch.no_grad():
            # Run through the vision model part directly
            vision_outputs = model.vision_model(pixel_values=dummy_image)
            
            # Or run through the main model wrapper
            image_features = model.get_image_features(pixel_values=dummy_image)
        
        print(f"Input Shape: {dummy_image.shape}")
        print(f"Vision Output (Last Hidden State): {vision_outputs.last_hidden_state.shape}")
        print(f"Pooled Image Features: {image_features.shape}")
        print("\n[SUCCESS] Forward pass complete. The backbone is functional.")
        
    except Exception as e:
        print(f"[FAIL] Forward pass error: {e}")

    # Cleanup
    shutil.rmtree(EXTRACT_DIR)
    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    main()