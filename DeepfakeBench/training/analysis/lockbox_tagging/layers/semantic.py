"""Layer 4 — OpenCLIP ViT-B-16 image embed + zero-shot prompt scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Prompt groups: each key is a column suffix; value is a list of (label, prompt).
# At inference time, image-embed cos-sim to each prompt-embed → softmax → top-1 label
# stored as `clip_<group>` and the full distribution as `clip_<group>_probs`.
PROMPT_GROUPS: dict[str, list[tuple[str, str]]] = {
    "quality": [
        ("sharp", "a sharp, clear, high quality photo of a person's face"),
        ("blurry", "a blurry, low quality, out of focus photo of a person's face"),
        ("pixelated", "a pixelated, heavily compressed photo of a person's face"),
    ],
    "capture_mode": [
        ("normal_photo", "a normal photograph of a person's face"),
        ("webcam", "a person on a webcam call"),
        ("screen", "a face filmed off a computer screen"),
        ("phone_screen", "a face filmed off a phone screen"),
        ("screen_recording", "a screen recording showing a video call"),
    ],
    "occlusion": [
        ("none", "a person's face with no accessories"),
        ("headset", "a person wearing a headset"),
        ("glasses", "a person wearing glasses"),
        ("face_mask", "a person wearing a face mask"),
        ("hand", "a hand partially covering a person's face"),
    ],
    "lighting": [
        ("normal", "a person's face in normal even lighting"),
        ("dim", "a person's face in dim low lighting"),
        ("harsh", "a person's face in harsh lighting with strong shadows"),
        ("backlit", "a backlit person's face with bright background"),
    ],
}


_clip_singleton: dict | None = None


def _clip():
    """Lazy-load the OpenCLIP model + tokenizer + precomputed prompt embeddings."""
    global _clip_singleton
    if _clip_singleton is not None:
        return _clip_singleton
    import open_clip
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    # Encode all prompts once.
    prompt_embeds: dict[str, tuple[list[str], "torch.Tensor"]] = {}
    with torch.no_grad():
        for group, items in PROMPT_GROUPS.items():
            labels = [lab for lab, _ in items]
            texts = [p for _, p in items]
            t = tok(texts).to(device)
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            prompt_embeds[group] = (labels, e)

    _clip_singleton = {
        "model": model,
        "preprocess": preprocess,
        "prompt_embeds": prompt_embeds,
        "device": device,
    }
    return _clip_singleton


def compute_semantic(path: Path) -> dict:
    """Return CLIP embedding (fp16 list) + per-group top-1 label + prob."""
    out: dict = {"clip_embed": None}
    for group in PROMPT_GROUPS:
        out[f"clip_{group}"] = None
        out[f"clip_{group}_prob"] = None

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return out

    import torch

    bundle = _clip()
    model, preprocess, prompt_embeds, device = (
        bundle["model"],
        bundle["preprocess"],
        bundle["prompt_embeds"],
        bundle["device"],
    )
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
        embed_np = e.squeeze(0).cpu().numpy().astype(np.float16)
        out["clip_embed"] = embed_np.tolist()

        for group, (labels, prompt_e) in prompt_embeds.items():
            sims = (e @ prompt_e.T).squeeze(0)
            probs = torch.softmax(sims * 100.0, dim=-1)  # CLIP-style temp
            top = int(probs.argmax().item())
            out[f"clip_{group}"] = labels[top]
            out[f"clip_{group}_prob"] = float(probs[top].item())
    return out


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        r = compute_semantic(Path(arg))
        # Truncate the embed for printing.
        if r.get("clip_embed") is not None:
            r["clip_embed"] = f"<512-d float16, first 4: {r['clip_embed'][:4]}>"
        print(json.dumps({"path": arg, **r}, indent=2))
