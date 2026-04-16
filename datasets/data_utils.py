import torch
from typing import List
from PIL import Image
import torch.nn.functional as F


# ── batch helpers ──────────────────────────────────────────────────────────────

def batch_to_cuda(d):
    """Recursively move all tensors in a dict to CUDA."""
    if isinstance(d, dict):
        return {k: batch_to_cuda(v) for k, v in d.items()}
    if isinstance(d, torch.Tensor):
        return d.cuda()
    return d


def collate_fn_eval(batch: List[dict]) -> dict:
    return batch[0]


def collate_fn_train(batch: List[dict]) -> dict:
    """Collate a list of sample dicts into a batched dict, padding keypoints to max P."""
    max_p = max(b['keypoints'].shape[1] for b in batch)

    padded_kps, padded_vis = [], []
    for b in batch:
        kps = b['keypoints']           # (2, P, 2)
        vis = b['visibility_mask']     # (P,)
        p = kps.shape[1]
        if p < max_p:
            pad = max_p - p
            kps = torch.cat([kps, kps.new_zeros(2, pad, 2)], dim=1)
            vis = torch.cat([vis, vis.new_zeros(pad, dtype=torch.bool)])
        padded_kps.append(kps)
        padded_vis.append(vis)

    return {
        'samples': torch.cat([b['samples'] for b in batch], dim=0),         # (B,2,C,H,W)
        'keypoints': torch.stack(padded_kps),                                # (B,2,max_P,2)
        'visibility_mask': torch.stack(padded_vis),                          # (B,max_P)
        'n_pts': torch.tensor([b['n_pts'] for b in batch], dtype=torch.long),  # (B,)
        'sem_mask_A': torch.stack([b['sem_mask_A'] for b in batch]),         # (B,H,W)
        'sem_mask_B': torch.stack([b['sem_mask_B'] for b in batch]),         # (B,H,W)
    }


# ── image transforms ──────────────────────────────────────────────────────────

def resize_longest_side(img_in: Image, out_size: int):
    w, h = img_in.size
    scale = out_size / max(w, h)  # scale factor based on longest side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img_in.resize((new_w, new_h), Image.BILINEAR)


def pad_tensor_no_resize(img_tensor):
    """
    Pad an image tensor [C, H, W] to a square (max(H,W) x max(H,W)),
    adding padding only to the **bottom and right**.
    """
    C, H, W = img_tensor.shape

    target_size = max(H, W)
    pad_bottom = target_size - H
    pad_right = target_size - W

    padding = (0, pad_right, 0, pad_bottom)  # pad_left, pad_right, pad_top, pad_bottom
    img_padded = F.pad(img_tensor, padding, mode='constant', value=0)

    return img_padded, (pad_bottom, pad_right)
