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

# Standard HF Config
GENERIC_CONFIG = {
  "architectures": ["CLIPModel"],
  "model_type": "clip",
  "vision_config": {
    "hidden_size": 768,
    "image_size": 224,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "patch_size": 16,
    "projection_dim": 512,
    "vocab_size": 0 
  },
  "text_config": {
    "vocab_size": 49408,
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "num_hidden_layers": 12,
    "max_position_embeddings": 77
  },
  "projection_dim": 512,
  "initializer_factor": 1.0
}

def convert_openclip_to_hf(state_dict):
    """
    Complex conversion that handles:
    1. Renaming keys
    2. Splitting fused QKV tensors into separate Q, K, V tensors
    """
    new_dict = {}
    print("  [CONVERTING] Splitting QKV tensors and remapping keys...")
    
    # We iterate over a list of keys so we can modify the dict if needed
    keys = list(state_dict.keys())
    
    for key in keys:
        val = state_dict[key]
        new_key = key

        # --- 1. Vision Model Conversion ---
        if key.startswith("visual."):
            new_key = key.replace("visual.", "vision_model.")
            
            # Embeddings & Norms
            new_key = new_key.replace("class_embedding", "embeddings.class_embedding")
            new_key = new_key.replace("positional_embedding", "embeddings.position_embedding.weight")
            new_key = new_key.replace("conv1.weight", "embeddings.patch_embedding.weight")
            new_key = new_key.replace("ln_pre.weight", "pre_layrnorm.weight")
            new_key = new_key.replace("ln_pre.bias", "pre_layrnorm.bias")
            new_key = new_key.replace("ln_post.", "post_layernorm.")
            new_key = new_key.replace("proj", "visual_projection.weight") # Fix projection mapping

            # Transformer Layers
            new_key = new_key.replace("transformer.resblocks.", "encoder.layers.")
            new_key = new_key.replace("ln_1.", "layer_norm1.")
            new_key = new_key.replace("ln_2.", "layer_norm2.")
            new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
            new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

            # --- QKV SPLITTING LOGIC ---
            # OpenCLIP uses 'attn.in_proj_weight' (combined QKV)
            # HF uses q_proj, k_proj, v_proj
            if "attn.in_proj_" in new_key:
                # Determine if it's weight or bias
                suffix = "weight" if "weight" in new_key else "bias"
                base_name = new_key.replace(f"attn.in_proj_{suffix}", "self_attn")
                
                # Split the tensor (OpenCLIP stacks them: [3*dim, ...])
                # We assume Hidden Size = 768 (standard for ViT-B/16)
                hidden_size = 768
                
                # Verify shape matches expectation
                if val.shape[0] != 3 * hidden_size:
                    print(f"    [WARNING] QKV split warning on {key}: Expected {3*hidden_size}, got {val.shape[0]}. Skipping split.")
                    new_dict[new_key] = val
                    continue

                q, k, v = torch.chunk(val, 3, dim=0)
                
                new_dict[f"{base_name}.q_proj.{suffix}"] = q
                new_dict[f"{base_name}.k_proj.{suffix}"] = k
                new_dict[f"{base_name}.v_proj.{suffix}"] = v
                continue # Skip the default add, we added 3 keys instead

            # Handle Output Projection
            new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")

        # --- 2. Text Model Conversion (Optional but cleans up logs) ---
        # Note: OpenCLIP text keys are at root, HF expects them in 'text_model'
        # We perform a rough mapping to silence warnings, though you care about Vision.
        elif not key.startswith("vision_model"): 
            # This handles 'token_embedding', 'positional_embedding', 'transformer', etc.
            new_key = "text_model." + key
            new_key = new_key.replace("text_model.token_embedding.weight", "text_model.embeddings.token_embedding.weight")
            new_key = new_key.replace("text_model.positional_embedding", "text_model.embeddings.position_embedding.weight")
            new_key = new_key.replace("text_model.ln_final.", "text_model.final_layer_norm.")
            new_key = new_key.replace("text_model.transformer.resblocks.", "text_model.encoder.layers.")
            new_key = new_key.replace("ln_1.", "layer_norm1.")
            new_key = new_key.replace("ln_2.", "layer_norm2.")
            new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
            new_key = new_key.replace("mlp.c_proj", "mlp.fc2")
            new_key = new_key.replace("text_model.text_projection", "text_projection.weight")
            
            # Text Encoder also needs QKV splitting (Hidden size 512 for Text)
            if "attn.in_proj_" in new_key:
                suffix = "weight" if "weight" in new_key else "bias"
                base_name = new_key.replace(f"attn.in_proj_{suffix}", "self_attn")
                
                hidden_size_text = 512
                if val.shape[0] == 3 * hidden_size_text:
                    q, k, v = torch.chunk(val, 3, dim=0)
                    new_dict[f"{base_name}.q_proj.{suffix}"] = q
                    new_dict[f"{base_name}.k_proj.{suffix}"] = k
                    new_dict[f"{base_name}.v_proj.{suffix}"] = v
                    continue

            new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")

        # Add the mapped key
        new_dict[new_key] = val

    return new_dict

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    print(f"=== Downloading {REPO_ID} ===")
    try:
        snapshot_download(
            repo_id=REPO_ID, 
            local_dir=OUTPUT_DIR,
            allow_patterns=["*.safetensors"],
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print(f"\n=== Processing Weights ===")
    src_path = os.path.join(OUTPUT_DIR, "open_clip_model.safetensors")
    
    # Load
    tensors = load_file(src_path)
    
    # Convert
    new_tensors = convert_openclip_to_hf(tensors)
    
    # Save
    print("  [SAVING] Writing generic 'model.safetensors'...")
    save_file(new_tensors, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    print("  [SAVING] Writing generic 'config.json'...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(GENERIC_CONFIG, f, indent=2)

    # Cleanup
    for f in os.listdir(OUTPUT_DIR):
        if "open_clip" in f:
            os.remove(os.path.join(OUTPUT_DIR, f))

    print(f"\n=== Packaging ===")
    if os.path.exists(f"{FINAL_PACKAGE_NAME}.zip"):
        os.remove(f"{FINAL_PACKAGE_NAME}.zip")
        
    shutil.make_archive(FINAL_PACKAGE_NAME, 'zip', OUTPUT_DIR)
    print(f"SUCCESS: Packaged '{FINAL_PACKAGE_NAME}.zip'")
    shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    main()