import torch
from pathlib import Path
from omegaconf import OmegaConf

from models.marco import MARCO
from models.adapters import inject_adapters

DINOV2_HUB_MODELS = {
    'small': 'dinov2_vits14_reg',
    'base':  'dinov2_vitb14_reg',
    'large': 'dinov2_vitl14_reg',
    'giant': 'dinov2_vitg14_reg',
}


def build_dinov2(model_cfg, model_size):
    """Load a frozen DINOv2 backbone from torchhub and inject adapters."""
    hub_name = DINOV2_HUB_MODELS[model_size]
    model = torch.hub.load('facebookresearch/dinov2', hub_name)

    af_blocks = range(model_cfg.adaptformer_stages[0], model_cfg.adaptformer_stages[1])
    inject_adapters(model, af_blocks, model.embed_dim, model_cfg.channel_factor)

    return model


def _build_marco_from_cfg(model_cfg):
    """Core factory: instantiate MARCO from a model config node."""
    dino = build_dinov2(model_cfg, model_cfg.model_size)
    return MARCO(dino=dino, embed_dim=dino.embed_dim, model_cfg=model_cfg)


def build_marco(args=None):
    """Build MARCO model. Uses default config when args is None."""
    if args is None:
        model_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs' / 'model' / 'marco.yaml')
    else:
        model_cfg = args.model_cfg
    return _build_marco_from_cfg(model_cfg)
