"""Feature upsampling head."""

from torch import nn
from timm.layers import LayerNorm2d


class UpsampleBlock(nn.Module):
    """2x spatial upsampling: ConvTranspose2d -> GELU -> depthwise Conv2d -> LayerNorm."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.conv2(self.act(self.conv1(x))))
