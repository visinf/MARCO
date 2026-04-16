"""MARCO model: DINOv2 backbone with AdaptFormer adapters and feature upsampling."""

import torch
import torch.nn.functional as F
import einops
from torch import nn, Tensor
from typing import Tuple

from models.upsampling_head import UpsampleBlock
from util.geometry import (
    kernel_softargmax_get_matches_logits,
    scaling_coordinates, scale_keypoints_to_featmap, normalize_coordinates,
)


class MARCO(nn.Module):
    """
    Only adapter and upscale parameters are saved/loaded by default.
    The frozen DINOv2 backbone is always loaded from torchhub.
    """
    save_keys = ['adapter', 'upscale']

    def __init__(self, dino: nn.Module, embed_dim: int, model_cfg):
        super().__init__()
        self.dino = dino
        self.embed_dim = embed_dim
        self.softmax_temp = model_cfg.softmax_temp
        self.n_upscale = model_cfg.n_upscale
        self.upscale = nn.Sequential(
            *[UpsampleBlock(embed_dim) for _ in range(self.n_upscale)]
        ) if self.n_upscale > 0 else nn.Identity()

    def forward(self, samples: Tensor, src_kps: Tensor, img_size: Tuple[int, int]) -> Tensor:
        """
        Args:
            samples: (B, 2, C, H, W) source and target images.
            src_kps: (B, N, 2) source keypoint coordinates.
            img_size: (H_img, W_img) original image size.
        Returns:
            (B, N, 2) predicted target keypoint coordinates.
        """
        fmaps = self.forward_backbone(samples)
        sim_map = self.sample_descriptors_w_sim(fmaps, src_kps, img_size)
        return self.predict_from_logits(sim_map, orig_size=img_size)
    
    def forward_backbone(self, samples: Tensor) -> Tensor:
        """Extract and upsample features for an image pair.

        Args:
            samples: (B, 2, C, H, W) pair of images.
        Returns:
            (B, 2, C, h, w) upsampled feature maps.
        """
        B = samples.shape[0]
        x = einops.rearrange(samples, 'b t c h w -> (b t) c h w')
        feats = self.dino.get_intermediate_layers(x, n=1, reshape=True)[0]
        feats = self.upscale(feats)
        return einops.rearrange(feats, '(b t) c h w -> b t c h w', b=B)

    def get_fmaps_res(self, train_res: int) -> int:
        """Compute feature map spatial resolution for a given input resolution."""
        base = train_res // self.dino.patch_size
        up = 1 if self.n_upscale == 0 else self.n_upscale * 2
        return base * up

    @staticmethod
    def sample_descriptors(fmaps: Tensor, src_kps: Tensor, img_size: Tuple[int, int]) -> Tensor:
        """Sample feature descriptors at keypoint locations via bilinear interpolation.

        Args:
            fmaps: (B, C, H_feat, W_feat) feature maps.
            src_kps: (B, N, 2) keypoints in image pixel coordinates (x, y).
            img_size: (H_img, W_img) original image size.
        Returns:
            (B, N, C) sampled descriptors.
        """
        H_feat, W_feat = fmaps.shape[-2:]
        kps_feat = scale_keypoints_to_featmap(src_kps, img_size=img_size, featmap_size=(H_feat, W_feat))
        kps_norm = normalize_coordinates(kps_feat, (H_feat, W_feat))
        grid = kps_norm.unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(fmaps, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return sampled.squeeze(3).transpose(1, 2)  # (B, N, C)

    @staticmethod
    def sample_descriptors_w_sim(fmaps: Tensor, src_kps: Tensor, img_size: Tuple[int, int]) -> Tensor:
        """Sample descriptors from source and compute similarity maps with target.

        Args:
            fmaps: (B, 2, C, H_feat, W_feat) paired feature maps.
            src_kps: (B, N, 2) source keypoints in image pixel coordinates.
            img_size: (H_img, W_img) original image size.
        Returns:
            (B, N, H_feat, W_feat) similarity maps.
        """
        feat_A, feat_B = fmaps[:, 0], fmaps[:, 1]
        H_feat, W_feat = feat_A.shape[-2:]
        descs_A = MARCO.sample_descriptors(feat_A, src_kps, img_size)  # (B, N, C)
        sim = torch.bmm(descs_A, feat_B.flatten(2))  # (B, N, H*W)
        return einops.rearrange(sim, 'b n (h w) -> b n h w', h=H_feat)

    def predict_from_logits(self, logits: Tensor, orig_size: Tuple[int, int], sigma: int = 7) -> Tensor:
        """Predict match coordinates from similarity logits via soft-argmax.

        Args:
            logits: (B, N, H, W) similarity maps.
            orig_size: (H_img, W_img) target image size for coordinate rescaling.
            sigma: Gaussian suppression kernel size.
        Returns:
            (B, N, 2) predicted keypoint coordinates in image pixel space.
        """
        h, w = logits.shape[-2:]
        matches = kernel_softargmax_get_matches_logits(logits, self.softmax_temp, sigma)
        return scaling_coordinates(matches, (h, w), orig_size)


    def state_dict(self, *args, **kwargs):
        """Return only adapter and upscale parameters."""
        state = super().state_dict(*args, **kwargs)
        if self.save_keys is None:
            return state
        return {k: v for k, v in state.items() if any(sk in k for sk in self.save_keys)}

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict, ignoring missing backbone keys."""
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        filtered_missing = [k for k in missing if any(sk in k for sk in self.save_keys)]
        if strict and (filtered_missing or unexpected):
            raise RuntimeError(
                f"Error loading state_dict for {self.__class__.__name__}:\n"
                f"Missing keys: {filtered_missing}\nUnexpected keys: {unexpected}"
            )
        return nn.modules.module._IncompatibleKeys(filtered_missing, unexpected)
