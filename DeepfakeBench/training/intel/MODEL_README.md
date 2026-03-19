# Custom Vision Backbone

## Overview
This package contains the pre-trained weights for the project's visual encoder. 
The architecture follows a standard Vision Transformer (ViT-B/16) topology.

## Technical Specifications
* **Architecture:** Vision Transformer (ViT)
* **Format:** Hugging Face `safetensors`
* **Input Resolution:** 224 x 224 (RGB)
* **Output Dimension:** 512
* **Model Configuration:**
  * Hidden Size: 768
  * Layers: 12
  * Attention Heads: 12
  * Patch Size: 16

## Usage
The model structure is compatible with standard transformer libraries. You can load it using the generic `AutoModel` class:

```python
from transformers import AutoModel

# Load from the local directory
model = AutoModel.from_pretrained("./")

# The optimization target is the vision graph:
vision_encoder = model.vision_model