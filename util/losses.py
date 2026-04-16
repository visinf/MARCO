"""Loss functions and learning rate schedulers for MARCO training."""

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple

from util.geometry import (
    scaling_coordinates, normalize_coordinates,
    create_batch_kernel_grid, gaussian_kernel_generator, softmax_with_temperature,
)


# ── Schedulers ────────────────────────────────────────────────────────────────

class WarmupCosineSchedule:
    """Cosine annealing with linear warmup."""

    def __init__(self, warmup_steps: int, start_lr: float, ref_lr: float,
                 T_max: int, final_lr: float = 0., start_step: float = 0):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = start_step

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        elif self._step < self.warmup_steps + self.T_max:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        else:
            new_lr = self.final_lr
        return new_lr


class ConstantScheduler:
    """Returns a fixed value on every step."""

    def __init__(self, lr, *args, **kwargs):
        self.lr = lr

    def step(self):
        return self.lr


# ── Loss functions ────────────────────────────────────────────────────────────

class GaussianCrossEntropyLoss:
    """Cross-entropy loss with Gaussian-smoothed ground-truth targets.

    Supports coarse-to-fine training via sigma annealing (cosine or constant schedule).
    """

    def __init__(self, ctf_cfg, softmax_temp: float = 0.04, step: int = 0):
        self.kernel_size = ctf_cfg.loss_kernel_size
        self.sigma = ctf_cfg.loss_kernel_std if ctf_cfg.loss_kernel_std is not None else self.kernel_size // 2 / 2
        self.softmax_temp = softmax_temp

        if ctf_cfg.schedule_std == 'constant':
            self.scheduler = ConstantScheduler(self.sigma)
        else:
            self.scheduler = WarmupCosineSchedule(
                warmup_steps=ctf_cfg.warmup_std, start_lr=self.sigma, ref_lr=self.sigma,
                T_max=ctf_cfg.decay_std, final_lr=ctf_cfg.end_sigma, start_step=step,
            )

    def step(self):
        self.sigma = self.scheduler.step()

    def __call__(self, trg_logits: Tensor, trg_kps: Tensor, trg_imgsize: Tuple[int, int], **kwargs):
        """Compute Gaussian CE loss.

        Args:
            trg_logits: (B, N, H, W) predicted similarity maps.
            trg_kps: (B, N, 2) ground-truth keypoints in image pixel coords.
            trg_imgsize: (H_img, W_img) original image size.
            visible_kps: optional (B, N) boolean visibility mask.
        """
        B, N, h, w = trg_logits.shape
        visible_kps = kwargs.get('visible_kps', None)
        loss = 0

        for b in range(B):
            xy = scaling_coordinates(trg_kps[b:b+1].clone(), trg_imgsize, (h, w)).to(trg_logits.device)
            corr = trg_logits[b:b+1].reshape(1, -1, h * w)
            corr = softmax_with_temperature(corr, self.softmax_temp, dim=-1).reshape(1, -1, h, w)

            if visible_kps is not None:
                mask = visible_kps[b]
                valid = corr[:, mask]
                if valid.shape[1] > 0:
                    loss += self._gaussian_ce(valid + 1e-6, xy[:, mask])
            else:
                loss += self._gaussian_ce(corr + 1e-6, xy)

        return loss

    def _gaussian_ce(self, corr: Tensor, kps: Tensor) -> Tensor:
        """Gaussian-smoothed cross-entropy between correlation maps and GT keypoints."""
        B, N, h, w = corr.shape
        device = corr.device

        kernel = create_batch_kernel_grid(B, N, self.kernel_size, device)
        kps_exp = kps.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.kernel_size, self.kernel_size, -1)
        kps_shifted = kps_exp + kernel
        valid = (kps_shifted[..., 0] > 0) & (kps_shifted[..., 0] < w - 1) & \
                (kps_shifted[..., 1] > 0) & (kps_shifted[..., 1] < h - 1)

        grid = normalize_coordinates(kps_shifted, (h, w)).view(-1, self.kernel_size, self.kernel_size, 2)
        scores = F.grid_sample(
            corr.view(-1, h, w).unsqueeze(1), grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        ).view(B, N, self.kernel_size, self.kernel_size)

        gauss = gaussian_kernel_generator(self.kernel_size, self.sigma, device)
        gauss = gauss.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        return -(gauss * torch.log(scores) * valid).sum()


class SoftArgmaxL2Loss(nn.Module):
    """Soft-argmax + L2 loss for dense match supervision.

    Args:
        H, W: feature map spatial resolution.
        tau: softmax temperature.
    """

    def __init__(self, H: int = 256, W: int = 256, tau: float = 0.02):
        super().__init__()
        self.tau = tau
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        self.register_buffer("grid", torch.stack([xs, ys], dim=-1).view(-1, 2).float())

    def step(self):
        pass

    def forward(self, logits: Tensor, gt_xy: Tensor, trg_imgsize: Tuple[int, int], **kwargs):
        """Compute soft-argmax L2 loss.

        Args:
            logits: (B, N, H, W) predicted similarity maps.
            gt_xy: (B, N, 2) ground-truth match coordinates in image pixels.
            trg_imgsize: (H_img, W_img).
        """
        B, N, H, W = logits.shape
        device = logits.device
        gt_xy = scaling_coordinates(gt_xy, trg_imgsize, (H, W)).to(device)

        prob = softmax_with_temperature(logits.view(B, N, -1), beta=self.tau, dim=-1)
        coords = self.grid.to(device)
        pred = (prob.unsqueeze(-1) * coords).sum(-2)  # (B, N, 2)

        scale = torch.tensor([W - 1, H - 1], dtype=pred.dtype, device=device)
        pred_norm = 2 * pred / scale - 1
        gt_norm = 2 * gt_xy / scale - 1

        err = torch.linalg.norm(pred_norm - gt_norm, dim=-1)
        vis = kwargs.get('visible_kps', None)
        if vis is not None:
            return (err * vis.float()).sum()
        return err.sum()
