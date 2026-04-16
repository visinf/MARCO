"""Coordinate transforms, keypoint scaling, and soft-argmax matching."""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple


# ── Coordinate transforms ─────────────────────────────────────────────────────

def normalize_coordinates(coord: Tensor, coord_scale: Tuple[int, int]) -> Tensor:
    """Normalize (x, y) coords to [-1, 1].

    Args:
        coord: (B, ..., 2) in (x, y).
        coord_scale: (H, W).
    """
    H, W = coord_scale
    x = coord[..., 0] / (W - 1) * 2 - 1
    y = coord[..., 1] / (H - 1) * 2 - 1
    return torch.stack([x, y], dim=-1)


def scaling_coordinates(coord: Tensor, src_scale: Tuple[int, int], trg_scale: Tuple[int, int],
                        mode: str = 'align_corner') -> Tensor:
    """Scale (x, y) coordinates from src_scale to trg_scale.

    Args:
        coord: (B, ..., 2) in (x, y).
        src_scale: (H1, W1).
        trg_scale: (H2, W2).
        mode: 'align_corner' or 'center'.
    """
    assert mode in ('align_corner', 'center')
    H1, W1 = src_scale
    H2, W2 = trg_scale
    coord = coord.clone()
    if mode == 'align_corner':
        coord[..., 0:1] = coord[..., 0:1] / (W1 - 1) * (W2 - 1)
        coord[..., 1:2] = coord[..., 1:2] / (H1 - 1) * (H2 - 1)
    else:
        coord[..., 0:1] = (coord[..., 0:1] + 1 / 2) * W2 / W1 - 1 / 2
        coord[..., 1:2] = (coord[..., 1:2] + 1 / 2) * H2 / H1 - 1 / 2
    return coord


def regularise_coordinates(coord: Tensor, H: int, W: int, eps: float = 0) -> Tensor:
    """Clamp (x, y) coordinates to lie within image bounds [eps, dim-1-eps].

    Args:
        coord: (B, ..., 2) in (x, y).
        H, W: image dimensions.
        eps: small offset from boundaries.
    """
    coord = coord.clone()
    coord[..., 0] = coord[..., 0].clamp(eps, W - 1 - eps)
    coord[..., 1] = coord[..., 1].clamp(eps, H - 1 - eps)
    return coord


def scale_keypoints_to_featmap(kps: Tensor, img_size: Tuple[int, int],
                               featmap_size: Tuple[int, int]) -> Tensor:
    """Scale (x, y) keypoints from image size to feature map size.

    Args:
        kps: (B, N, 2).
        img_size: (H_img, W_img).
        featmap_size: (H_feat, W_feat).
    """
    H_img, W_img = img_size
    H_feat, W_feat = featmap_size
    x = kps[..., 0] * W_feat / W_img
    y = kps[..., 1] * H_feat / H_img
    return torch.stack([x, y], dim=-1)


# ── Grid and kernel utilities ─────────────────────────────────────────────────

def create_grid(H: int, W: int, gap: int = 1, device: str = 'cpu') -> Tensor:
    """Create an unnormalised (x, y) meshgrid of shape (H//gap, W//gap, 2)."""
    x = torch.linspace(0, W - 1, W // gap)
    y = torch.linspace(0, H - 1, H // gap)
    yg, xg = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xg, yg), dim=2).to(device)


def gaussian_kernel_generator(size: int, sigma: float, device) -> Tensor:
    """Generate a size×size 2D Gaussian kernel."""
    if size > 1:
        ind = size // 2
        kx = torch.linspace(-ind, ind, size, device=device)
        ky = torch.linspace(-ind, ind, size, device=device)
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        kernel = torch.sqrt(kx ** 2 + ky ** 2)
        kernel = 1 / (sigma ** 2 * 2 * np.pi) * torch.exp(-0.5 * ((kernel) / sigma) ** 2)
    else:
        kernel = torch.ones((1, 1), device=device)
    return kernel


def create_batch_kernel_grid(B: int, N: int, kernel_size: int, device) -> Tensor:
    """Create a (B, N, K, K, 2) grid of offsets for Gaussian kernel sampling."""
    ind = kernel_size // 2
    kx = torch.linspace(-ind, ind, kernel_size, device=device)
    ky = torch.linspace(-ind, ind, kernel_size, device=device)
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    kernel = torch.stack([kx, ky], dim=-1)
    return kernel[None, None].expand(B, N, -1, -1, -1)


# ── Soft-argmax matching ──────────────────────────────────────────────────────

def softmax_with_temperature(x: Tensor, beta: float, dim: int) -> Tensor:
    """Temperature-scaled softmax."""
    M, _ = x.max(dim=dim, keepdim=True)
    x = x - M
    exp_x = torch.exp(x / beta)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def apply_gaussian_kernel(scoremaps: Tensor, sigma: int = 7) -> Tensor:
    """Apply a Gaussian kernel centered at the argmax of each score map.

    Args:
        scoremaps: (B, N, H, W).
    Returns:
        Suppressed score maps (B, N, H, W).
    """
    B, N, h, w = scoremaps.shape
    device = scoremaps.device

    idx = torch.max(scoremaps.view(B, N, -1), dim=-1).indices
    idx_y = (idx // w).view(B, N, 1, 1).float()
    idx_x = (idx % w).view(B, N, 1, 1).float()

    grid = create_grid(h, w, device=device).unsqueeze(0).unsqueeze(0)
    grid = grid.expand(B, N, -1, -1, -1)
    gauss = torch.exp(-((grid[..., 0] - idx_x) ** 2 + (grid[..., 1] - idx_y) ** 2) / (2 * sigma ** 2))
    return gauss * scoremaps


def kernel_softargmax_get_matches_logits(trg_logits: Tensor, softmax_temp: float,
                                         sigma: int = 7) -> Tensor:
    """Predict match coordinates via Gaussian-suppressed soft-argmax.

    Args:
        trg_logits: (B, N, H, W) similarity logits.
        softmax_temp: softmax temperature.
        sigma: Gaussian suppression kernel size.
    Returns:
        matches: (B, N, 2) in the scale of the provided logits.
    """
    B, N, h, w = trg_logits.shape
    device = trg_logits.device

    scoremaps = apply_gaussian_kernel(trg_logits, sigma).view(B, N, -1)
    scoremaps = softmax_with_temperature(scoremaps, softmax_temp, -1)
    grid = create_grid(h, w, device=device)

    scoremaps = scoremaps.unsqueeze(-1).expand(-1, -1, -1, 2)
    grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    return (scoremaps * grid).sum(dim=2)
