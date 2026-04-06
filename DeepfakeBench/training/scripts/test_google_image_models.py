# test_google_image_models.py
#
# Usage:
#   export GOOGLE_API_KEY="your_key_here"
#   python test_google_image_models.py --nano
#
# Requirements:
#   pip install --upgrade google-genai pillow matplotlib

import base64
import os
import argparse
from pathlib import Path

# Vertex AI imports (for Imagen)
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
import vertexai

# Unified GenAI imports (for Gemini/Nano)
from google import genai
from google.genai import types
from PIL import Image as PILImage
import matplotlib.pyplot as plt

# =========================
# ARGUMENT PARSING
# =========================
parser = argparse.ArgumentParser(description="Test Google Image Models")
parser.add_argument("--nano", action="store_true", help="Run ONLY the Gemini 3 Pro (Nano Banana) section.")
args = parser.parse_args()

# =========================
# CONFIG
# =========================
PROJECT_ID = "train-cvit2"
LOCATION = "us-central1"
INPUT_IMAGE = "/Users/roeedar/Downloads/real_ffhq_00015.png"
PROMPT = "Give this guy realistic sunglasses and a baseball cap"

OUT_DIR = Path("outputs_google_image_models")
OUT_DIR.mkdir(exist_ok=True)
results = {}

# =========================
# INIT CLIENTS
# =========================
print(f"Initializing Vertex AI (Project: {PROJECT_ID})...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize GenAI Client
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print("Detected GOOGLE_API_KEY. Using AI Studio Client for Gemini.")
    genai_client = genai.Client(api_key=api_key)
else:
    print("No GOOGLE_API_KEY found. Using Vertex AI Client for Gemini.")
    genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =========================
# HELPERS
# =========================
def save_and_return_pil(path: Path):
    if not path.exists(): return None
    try:
        img = PILImage.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        return None

# =========================
# 1. IMAGEN v2
# =========================
if not args.nano:
    print("\n--- Running Imagen v2 (imagegeneration@002) ---")
    try:
        imagen_v2 = ImageGenerationModel.from_pretrained("imagegeneration@002")
        base_image = VertexImage.load_from_file(INPUT_IMAGE)
        res_imagen_v2 = imagen_v2.edit_image(base_image=base_image, prompt=PROMPT, mask_mode="foreground")
        p = OUT_DIR / "imagen_v2.png"
        res_imagen_v2.images[0].save(p)
        print(f"Saved: {p}")
        results["Imagen v2"] = save_and_return_pil(p)
    except Exception as e:
        print(f"Imagen v2 Failed: {e}")
        results["Imagen v2"] = None

# =========================
# 2. IMAGEN v3-class
# =========================
if not args.nano:
    print("\n--- Running Imagen v3-class (imagegeneration@006) ---")
    try:
        imagen_v3 = ImageGenerationModel.from_pretrained("imagegeneration@006")
        base_image = VertexImage.load_from_file(INPUT_IMAGE)
        res_imagen_v3 = imagen_v3.edit_image(base_image=base_image, prompt=PROMPT, mask_mode="foreground")
        p = OUT_DIR / "imagen_v3.png"
        res_imagen_v3.images[0].save(p)
        print(f"Saved: {p}")
        results["Imagen v3"] = save_and_return_pil(p)
    except Exception as e:
        print(f"Imagen v3 Failed: {e}")
        results["Imagen v3"] = None

# =========================
# 3. GEMINI 3 PRO (Nano Banana)
# =========================
print("\n--- Running Gemini 3 Pro (Nano Banana Pro) ---")
MODEL_ID_NANO = "gemini-3-pro-image-preview"
gemini_out_path = OUT_DIR / "gemini_3_pro.png"

try:
    pil_img_input = PILImage.open(INPUT_IMAGE).convert("RGB")
    print(f"Sending request to {MODEL_ID_NANO}...")

    # FIX 1: Removed 'config' parameter (it was causing invalid argument error)
    response = genai_client.models.generate_content(
        model=MODEL_ID_NANO,
        contents=[pil_img_input, PROMPT],
    )

    image_saved = False
    
    # Check candidates
    if response.candidates:
        for part in response.candidates[0].content.parts:
            # Check for inline image data
            if part.inline_data:
                print(f"Found inline image data. (Format: {part.inline_data.mime_type})")
                
                # FIX 2: Handle data type correctly
                # The SDK often auto-decodes base64 into bytes for you.
                data = part.inline_data.data
                
                if isinstance(data, bytes):
                    # Already bytes - write directly
                    img_bytes = data
                else:
                    # String - likely base64, decode it
                    img_bytes = base64.b64decode(data)

                with open(gemini_out_path, "wb") as f:
                    f.write(img_bytes)
                
                image_saved = True
                break
            
            # Check if model refused and returned text
            if part.text:
                print(f"Model returned text instead: {part.text}")

    if image_saved:
        print(f"Saved: {gemini_out_path}")
        results["Gemini 3 Pro"] = save_and_return_pil(gemini_out_path)
    else:
        print("No image found in response.")
        results["Gemini 3 Pro"] = None

except Exception as e:
    print(f"Gemini 3 Pro Failed: {e}")
    results["Gemini 3 Pro"] = None

# =========================
# VISUALIZE
# =========================
print("\nDisplaying results...")
display_imgs = [PILImage.open(INPUT_IMAGE).convert("RGB")]
display_titles = ["Original"]

for k, v in results.items():
    display_imgs.append(v if v else PILImage.new('RGB', (256, 256), 'black'))
    display_titles.append(k if v else f"{k} (Failed)")

if len(display_imgs) > 0:
    plt.figure(figsize=(5*len(display_imgs), 6))
    for i, (img, t) in enumerate(zip(display_imgs, display_titles)):
        plt.subplot(1, len(display_imgs), i+1)
        plt.imshow(img)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print(f"Done. Outputs in: {OUT_DIR.absolute()}")