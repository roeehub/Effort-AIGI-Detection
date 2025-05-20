from .clip_models import ClipModel
from .siglip_models import SigLIPModel
from .beitv2_models import BEiTv2Model


VALID_NAMES = {
    'CLIP:ViT-B/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch16/',
    'CLIP:ViT-B/32_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch32/',
    'CLIP:ViT-L/14_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/', 
    'SigLIP:ViT-L/16_256_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/google/siglip-large-patch16-256/',
    'BEiTv2:ViT-L/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/BEiT-v2/beitv2_large_patch16_224_pt1k_ft21k.pth',
}


def get_model(name, opt):
    assert name in VALID_NAMES.keys()
    if name.startswith("CLIP:"):
        return ClipModel(VALID_NAMES[name], opt) 
    elif name.startswith("SigLIP:"):
        return SigLIPModel(VALID_NAMES[name], opt)
    elif name.startswith("BEiTv2:"):
        return BEiTv2Model(VALID_NAMES[name], opt)
    else:
        assert False 
