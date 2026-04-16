"""MARCO training engine: one-epoch loop with self-distillation via dense flow."""

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.marco import MARCO
from util import flow
from util.ema import get_ema_momentum
from util.losses import GaussianCrossEntropyLoss, SoftArgmaxL2Loss
from util.geometry import scale_keypoints_to_featmap
from datasets.data_utils import batch_to_cuda
import util.misc as utils


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    models_tuple: Tuple[MARCO, MARCO],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_norm: float = 0,
    args=None,
):
    model, teacher = models_tuple
    feats_res = model.get_fmaps_res(args.train_res)

    loss_sparse = GaussianCrossEntropyLoss(
        args.coarse_to_fine, softmax_temp=model.softmax_temp,
        step=len(data_loader) * epoch,
    )
    loss_dense = SoftArgmaxL2Loss(H=feats_res, W=feats_res, tau=model.softmax_temp)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    step = len(data_loader) * epoch
    max_train_iters = int(args.get('max_train_iters', 0) or 0)

    for it, batch in enumerate(metric_logger.log_every(data_loader, 50, f'Epoch: [{epoch}]')):
        if max_train_iters and it >= max_train_iters:
            break
        model.train()
        batch = batch_to_cuda(batch)
        samples = batch['samples']
        H_img, W_img = samples.shape[-2:]

        # Student forward
        fmaps = model.forward_backbone(samples)

        # Teacher forward + dense match mining
        with torch.no_grad():
            fmaps_teacher = teacher.forward_backbone(samples)

        pair_masks_up = torch.stack([batch['sem_mask_A'], batch['sem_mask_B']], dim=1).bool()
        pair_masks = F.interpolate(
            pair_masks_up.float(), (feats_res, feats_res),
            mode='bilinear', align_corners=True,
        ).bool()
        kps_scaled = scale_keypoints_to_featmap(
            batch['keypoints'], img_size=(H_img, W_img), featmap_size=(feats_res, feats_res),
        )

        mined_matches, total_mined = mine_dense_matches(
            fmaps_teacher, pair_masks, pair_masks_up, kps_scaled,
            batch['keypoints'], batch['n_pts'], args.flow,
        )

        # ── Supervised loss (Gaussian CE on annotated keypoints) ──
        optimizer.zero_grad()
        loss_dict = {}

        src_kps, gt_kps = batch['keypoints'][:, 0], batch['keypoints'][:, 1]
        sim_map = model.sample_descriptors_w_sim(fmaps, src_kps, (H_img, W_img))
        loss_sparse.step()
        loss_sup = loss_sparse(sim_map, gt_kps, trg_imgsize=(H_img, W_img), visible_kps=batch['visibility_mask'])
        loss_dict['gaussian_ce'] = loss_sup / batch['n_pts'].sum()

        # ── Self-distillation loss (L2 on mined dense matches) ──
        loss_dict['L2_selfsup'] = torch.tensor(0.0, device=samples.device)
        for i in mined_matches:
            src_mined, trg_mined = mined_matches[i][:1], mined_matches[i][1:]
            sim_dense = model.sample_descriptors_w_sim(fmaps[i:i+1], src_mined, (H_img, W_img))
            loss_d = args.flow.weight_dense * loss_dense(sim_dense, trg_mined, trg_imgsize=(H_img, W_img))
            loss_dict['L2_selfsup'] += loss_d / total_mined

        # ── Backward + EMA update ──────────────────────────────────────────────
        loss_sum = sum(loss_dict.values())
        loss_sum.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        with torch.no_grad():
            m = get_ema_momentum(
                step, total_steps=args.ema.warmup_epochs * len(data_loader),
                m_start=args.ema.start, m_end=args.ema.end, ramp_ratio=args.ema.ramp,
            )
            # Update teacher with EMA of student parameters
            for p_s, p_t in zip(model.parameters(), teacher.parameters()):
                if p_s.requires_grad:
                    p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)

        # ── Logging ────────────────────────────────────────────────────────────
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        metric_logger.update(loss=sum(loss_dict_reduced.values()).item(), **loss_dict_reduced)
        metric_logger.update(n_points=batch['n_pts'].sum() / args.batch_size)
        metric_logger.update(loss_std=loss_sparse.sigma)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, step


# ── Flow-based match mining ──────────────────────────────────────────────────

def mine_dense_matches(fmaps_teacher, pair_masks, pair_masks_up, kps_scaled, keypoints, n_pts, flow_cfg):
    """Mine dense correspondences from teacher features via flow clustering.

    Returns:
        mined_matches: dict mapping pair index -> (2, N, 2) int32 tensor
        total_mined: total number of mined points across all pairs
    """
    H_img, W_img = pair_masks_up.shape[-2:]
    mined_matches = {}
    total_mined = 0

    for i in range(len(fmaps_teacher)):
        try:
            flow_fmap = flow.compute_dense_flow(
                fmaps_teacher[i], masks=pair_masks[i],
                keypoints=kps_scaled[i, :, :n_pts[i]],
            )
            flow_img = flow.upsample_flow(flow_fmap, (H_img, W_img))
            flow_hsv = flow.flow_to_hsv(flow_img, pair_masks_up[i, 0])
        except:
            continue

        flow_valid = flow_hsv.to(torch.float32).norm(dim=-1) > 0
        flow_feat = flow_hsv[flow_valid] / 255.0
        # if valid pixels are too few, skip this pair
        if len(flow_feat) < flow_cfg.sample_n_dense:
            continue

        cluster_labels = flow.kmeans_bic(flow_feat, k_init=flow_cfg.flow_k, k_min=3)
        valid_ids = flow.filter_clusters_by_keypoints(
            flow_img, flow_valid, cluster_labels,
            keypoints[i, :, :n_pts[i]],
        )
        dense_matches = flow.collect_matches(
            flow_img, flow_valid, cluster_labels,
            valid_cluster_ids=valid_ids, object_masks=pair_masks_up[i],
        )
        # if too few matches, skip this pair
        if dense_matches.shape[1] < 100:
            continue

        n_sample = min(flow_cfg.sample_n_dense, dense_matches.shape[1])
        idx = np.random.choice(dense_matches.shape[1], n_sample, replace=False)
        sampled = dense_matches[:, idx, :]
        mined_matches[i] = sampled
        total_mined += sampled.shape[1]

    return mined_matches, total_mined
