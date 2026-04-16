"""AdaptFormer adapters."""

import math

import torch
from torch import nn, Tensor


class Adapter(nn.Module):
    """Bottleneck adapter: LayerNorm -> down -> ReLU -> up, scaled."""

    def __init__(self, in_channel: int, bottleneck: int = 64, scale: float = 0.1):
        super().__init__()
        self.adapter_layer_norm_before = nn.LayerNorm(in_channel)
        self.down_proj = nn.Linear(in_channel, bottleneck)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, in_channel)
        self.scale = scale

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.up_proj(self.act(self.down_proj(self.adapter_layer_norm_before(x)))) * self.scale


class AdaptedBlock(nn.Module):
    """Wraps a frozen DINOv2 block and adds a parallel adapter after self-attention."""

    def __init__(self, block: nn.Module, embed_dim: int, bottleneck: int):
        super().__init__()
        self.block = block
        self.adapter = Adapter(embed_dim, bottleneck=bottleneck)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.block.ls1(self.block.attn(self.block.norm1(x)))
        x = x + self.adapter(x) + self.block.ls2(self.block.mlp(self.block.norm2(x)))
        return x


def inject_adapters(model: nn.Module, block_indices, embed_dim: int, channel_factor: float):
    """Inject adapters into specified transformer blocks of a DINOv2 model."""
    bottleneck = int(embed_dim * channel_factor) if channel_factor <= 10 else int(channel_factor)
    for idx in block_indices:
        model.blocks[idx] = AdaptedBlock(model.blocks[idx], embed_dim, bottleneck)
