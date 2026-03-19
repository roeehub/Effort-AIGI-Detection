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

# Standard HF CLIP Config (Vision-B/16)
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
    "vocab_size": 0,
    "hidden_act": "gelu" 
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
    Robust conversion from OpenCLIP to Hugging Face CLIP.
    Includes contiguous() checks for SafeTensors compatibility.
    """
    new_dict = {}
    print("  [CONVERTING] Splitting QKV tensors and remapping keys...")
    
    for key, val in state_dict.items():
        new_key = key
        
        # --- VISION MODEL ---
        if key.startswith("visual."):
            new_key = new_key.replace("visual.", "vision_model.")
            
            # 1. Embeddings
            if "class_embedding" in key:
                new_key = "vision_model.embeddings.class_embedding"
            elif "positional_embedding" in key:
                new_key = "vision_model.embeddings.position_embedding.weight"
            elif "conv1.weight" in key:
                new_key = "vision_model.embeddings.patch_embedding.weight"
            
            # 2. Pre/Post Layer Norms
            elif "ln_pre.weight" in key:
                new_key = "vision_model.pre_layrnorm.weight"
            elif "ln_pre.bias" in key:
                new_key = "vision_model.pre_layrnorm.bias"
            elif "ln_post.weight" in key:
                new_key = "vision_model.post_layernorm.weight"
            elif "ln_post.bias" in key:
                new_key = "vision_model.post_layernorm.bias"
            
            # 3. Visual Projection
            elif "visual.proj" in key: 
                new_dict["visual_projection.weight"] = val.t().contiguous()
                continue

            # 4. Transformer Layers
            elif "transformer.resblocks." in key:
                parts = key.split(".")
                layer_idx = parts[3] 
                layer_prefix = f"vision_model.encoder.layers.{layer_idx}"
                
                if "ln_1" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    new_key = f"{layer_prefix}.layer_norm1.{suffix}"
                elif "ln_2" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    new_key = f"{layer_prefix}.layer_norm2.{suffix}"
                elif "mlp.c_fc" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    new_key = f"{layer_prefix}.mlp.fc1.{suffix}"
                elif "mlp.c_proj" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    new_key = f"{layer_prefix}.mlp.fc2.{suffix}"
                elif "attn.out_proj" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    new_key = f"{layer_prefix}.self_attn.out_proj.{suffix}"
                
                # 5. QKV Splitting
                elif "attn.in_proj_" in key:
                    suffix = "weight" if "weight" in key else "bias"
                    
                    hidden_size = 768
                    if val.shape[0] != 3 * hidden_size:
                        print(f"    [WARN] Unexpected shape for {key}: {val.shape}. Keeping original.")
                        new_dict[new_key] = val
                        continue
                    
                    q, k, v = torch.chunk(val, 3, dim=0)
                    
                    new_dict[f"{layer_prefix}.self_attn.q_proj.{suffix}"] = q.contiguous()
                    new_dict[f"{layer_prefix}.self_attn.k_proj.{suffix}"] = k.contiguous()
                    new_dict[f"{layer_prefix}.self_attn.v_proj.{suffix}"] = v.contiguous()
                    continue 

            new_dict[new_key] = val

        # --- TEXT MODEL ---
        elif key.startswith("token_embedding"):
            new_dict["text_model.embeddings.token_embedding.weight"] = val
        elif key.startswith("positional_embedding"):
            new_dict["text_model.embeddings.positional_embedding.weight"] = val
        elif key.startswith("text_projection"):
            new_dict["text_projection.weight"] = val.t().contiguous()
        elif key.startswith("ln_final"):
            suffix = "weight" if "weight" in key else "bias"
            new_dict[f"text_model.final_layer_norm.{suffix}"] = val

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
    
    tensors = load_file(src_path)
    new_tensors = convert_openclip_to_hf(tensors)
    
    print("  [SAVING] Writing clean 'model.safetensors'...")
    save_file(new_tensors, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    print("  [SAVING] Writing clean 'config.json'...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(GENERIC_CONFIG, f, indent=2)

    # --- Robust Cleanup (Fixes PermissionError) ---
    print("  [CLEANUP] Removing temp files...")
    for f in os.listdir(OUTPUT_DIR):
        if f in ["model.safetensors", "config.json"]:
            continue
        
        full_path = os.path.join(OUTPUT_DIR, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path) # Recursively delete directories
        else:
            os.remove(full_path)     # Delete files

    print(f"\n=== Packaging ===")
    if os.path.exists(f"{FINAL_PACKAGE_NAME}.zip"):
        os.remove(f"{FINAL_PACKAGE_NAME}.zip")
        
    shutil.make_archive(FINAL_PACKAGE_NAME, 'zip', OUTPUT_DIR)
    print(f"SUCCESS: Packaged '{FINAL_PACKAGE_NAME}.zip'")
    shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    main()