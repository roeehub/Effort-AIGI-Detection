import torch
import open_clip
from transformers import CLIPModel
import numpy as np
import os
import zipfile
import shutil

# --- Configuration ---
# The OpenCLIP short-tag for "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
OPENCLIP_TAG = "laion2b_s34b_b88k" 
YOUR_PACKAGE = "custom_vision_backbone.zip"
TEMP_DIR = "./comparison_temp"

def main():
    print(f"=== 1. Loading ORIGINAL Model (ViT-B-16 / {OPENCLIP_TAG}) via OpenCLIP ===")
    try:
        # We load the specific pre-trained checkpoint that matches your repo
        orig_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=OPENCLIP_TAG)
        orig_model.eval()
        print("   [SUCCESS] Original OpenCLIP model loaded.")
    except Exception as e:
        print(f"   [FAIL] Could not load original via open_clip: {e}")
        return

    print(f"\n=== 2. Loading YOUR PACKAGE via Transformers ===")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    with zipfile.ZipFile(YOUR_PACKAGE, 'r') as z:
        z.extractall(TEMP_DIR)
    
    try:
        # We load purely from the config/weights in the zip
        # This tests if your package works standalone
        my_model = CLIPModel.from_pretrained(TEMP_DIR)
        my_model.eval()
        print("   [SUCCESS] Your anonymous package loaded.")
    except Exception as e:
        print(f"   [FAIL] Could not load your package: {e}")
        return

    print(f"\n=== 3. Running Comparison Test ===")
    # Create a random image tensor (1, 3, 224, 224)
    torch.manual_seed(42)
    dummy_image = torch.randn(1, 3, 224, 224)

    # --- Run Original (OpenCLIP) ---
    with torch.no_grad():
        # OpenCLIP encode_image returns normalized features
        orig_features = orig_model.encode_image(dummy_image)

    # --- Run Yours (Hugging Face) ---
    with torch.no_grad():
        # HF get_image_features returns unnormalized features usually, 
        # but the raw projection weights should match.
        my_features = my_model.get_image_features(dummy_image)

    # --- Compare ---
    v1 = orig_features.detach().cpu().numpy()
    v2 = my_features.detach().cpu().numpy()

    # Calculate Difference
    mse = np.mean((v1 - v2) ** 2)
    max_diff = np.max(np.abs(v1 - v2))
    
    print(f"\n--- Results ---")
    print(f"Original Vector Shape: {v1.shape}")
    print(f"Your Vector Shape:     {v2.shape}")
    print(f"-"*30)
    print(f"Mean Squared Error:    {mse:.8f}")
    print(f"Max Absolute Diff:     {max_diff:.8f}")
    
    # We use a slightly looser threshold (1e-4) because HF and OpenCLIP 
    # sometimes have tiny implementation details (like epsilon in LayerNorm) 
    # that cause drift in the 6th decimal place.
    if mse < 1e-4:
        print("\n✅ VERIFIED: Models are functionally identical.")
        print("The backbone in your zip file matches the original LAION model.")
    else:
        print("\n❌ WARNING: Outputs differ significantly. Check conversion logic.")

    # Cleanup
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()