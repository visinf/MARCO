"""Preprocessing and visualization utilities for MARCO inference."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# ImageNet normalization
_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Palette — up to 10 keypoints; falls back to a generated colormap for more
_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8',
    '#f58231', '#911eb4', '#42d4f4',
    '#f032e6', '#bfef45', '#fabed4', '#469990',
]


def _get_colors(n: int):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    cmap = plt.cm.get_cmap('hsv', n)
    return [cmap(i) for i in range(n)]


def _resize_longest_side(img_in: Image.Image, out_size: int) -> Image.Image:
    """Resize an image so its longest side matches ``out_size``."""
    w, h = img_in.size
    scale = out_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img_in.resize((new_w, new_h), Image.BILINEAR)


def _pad_tensor_no_resize(img_tensor: torch.Tensor):
    """Pad a ``[C, H, W]`` tensor to a square using bottom/right zero padding."""
    _, h, w = img_tensor.shape
    target_size = max(h, w)
    pad_bottom = target_size - h
    pad_right = target_size - w
    padding = (0, pad_right, 0, pad_bottom)
    img_padded = F.pad(img_tensor, padding, mode='constant', value=0)
    return img_padded, (pad_bottom, pad_right)


def preprocess_data(
    src_path: str,
    trg_path: str,
    src_kps,
    inference_res: int = 840,
    device: str | torch.device = "cuda",
):
    """Load two images and prepare the model input tensor.

    Args:
        src_path: Path to the source image.
        trg_path: Path to the target image.
        src_kps: Source keypoints as a list of [x, y] pairs or a (N, 2) tensor/array,
                 in **pixel coordinates of the original source image**.
        inference_res: Longest-side resolution used for resizing (must match the
                       value used at training / the dataset config).
        device: Device on which to place the returned tensors. Defaults to `"cuda"`.

    Returns:
        A dictionary containing:
            samples:  (1, 2, C, H, W) float tensor ready for the model.
            src_kps:  (1, N, 2) float tensor of keypoints rescaled to the padded
                      image space used by the model.
            img_size: (H, W) tuple — the spatial size of the padded input images.

    Raises:
        ValueError: if any keypoint falls outside the source image bounds.
    """
    device = torch.device(device)

    src_pil = Image.open(src_path).convert("RGB")
    trg_pil = Image.open(trg_path).convert("RGB")

    # Validate keypoints against original image size
    kps = (
        torch.tensor(src_kps, dtype=torch.float32)
        if not isinstance(src_kps, torch.Tensor)
        else src_kps.float()
    )
    if kps.ndim == 1:
        kps = kps.unsqueeze(0)

    orig_w, orig_h = src_pil.size
    out_of_bounds = (
        (kps[:, 0] < 0) | (kps[:, 0] >= orig_w) |
        (kps[:, 1] < 0) | (kps[:, 1] >= orig_h)
    )
    if out_of_bounds.any():
        bad = kps[out_of_bounds].tolist()
        raise ValueError(
            f"Source image has size (w={orig_w}, h={orig_h}). "
            f"The following keypoints are out of range: {bad}. "
            f"Please provide coordinates in [0, {orig_w - 1}] x [0, {orig_h - 1}]."
        )

    # Resize longest side → tensor → pad to square
    src_resized = _resize_longest_side(src_pil, inference_res)
    trg_resized = _resize_longest_side(trg_pil, inference_res)

    src_tensor = _NORM(src_resized)  # (C, H', W')
    trg_tensor = _NORM(trg_resized)

    src_padded, _ = _pad_tensor_no_resize(src_tensor)  # (C, S, S)
    trg_padded, _ = _pad_tensor_no_resize(trg_tensor)

    H, W = src_padded.shape[-2:]

    # Scale keypoints from original image space to resized+padded image space
    scale_x = src_resized.width / orig_w
    scale_y = src_resized.height / orig_h
    kps_scaled = kps.clone()
    kps_scaled[:, 0] = kps[:, 0] * scale_x
    kps_scaled[:, 1] = kps[:, 1] * scale_y

    samples = torch.stack([src_padded, trg_padded]).unsqueeze(0).to(device)  # (1, 2, C, H, W)
    kps_scaled = kps_scaled.unsqueeze(0).to(device)                           # (1, N, 2)

    return {
        "samples": samples,
        "src_kps": kps_scaled,
        "img_size": (H, W),
    }

def visualize_prediction(
    src_path: str,
    trg_path: str,
    src_kps,
    pred_kps,
    output_path: str = "output.png",
    inference_res: int = 840,
    grid_steps: int = 6,
    visualize: bool = False,
):
    """Visualize source keypoints and predicted target keypoints.

    Args:
        src_path: Path to the source image.
        trg_path: Path to the target image.
        src_kps: (N, 2) or list of [x, y] source keypoints in original image pixels.
        pred_kps: (N, 2) or (1, N, 2) predicted target keypoints in padded/resized space.
        output_path: Where to save the output figure when visualize=False.
        inference_res: Longest-side resolution used during preprocessing.
        grid_steps: Approximate number of intervals shown on the axes.
        visualize: If True, display the figure with plt.show() instead of saving it.
    """
    src_pil = Image.open(src_path).convert("RGB")
    trg_pil = Image.open(trg_path).convert("RGB")

    # Resize for display
    orig_w_src, orig_h_src = src_pil.size
    orig_w_trg, orig_h_trg = trg_pil.size
    src_resized = _resize_longest_side(src_pil, inference_res)
    trg_resized = _resize_longest_side(trg_pil, inference_res)

    disp_w_src, disp_h_src = src_resized.width, src_resized.height
    disp_w_trg, disp_h_trg = trg_resized.width, trg_resized.height

    # Scale source keypoints to displayed source image
    scale_x_src = disp_w_src / orig_w_src
    scale_y_src = disp_h_src / orig_h_src

    kps_src = (
        torch.tensor(src_kps, dtype=torch.float32)
        if not isinstance(src_kps, torch.Tensor)
        else src_kps.float()
    )
    if kps_src.ndim == 1:
        kps_src = kps_src.unsqueeze(0)

    kps_src_disp = kps_src.clone()
    kps_src_disp[:, 0] *= scale_x_src
    kps_src_disp[:, 1] *= scale_y_src

    # Predicted keypoints are already in resized/padded image space
    kps_pred = (
        pred_kps.squeeze(0).detach().cpu().float()
        if isinstance(pred_kps, torch.Tensor)
        else torch.tensor(pred_kps, dtype=torch.float32)
    )
    if kps_pred.ndim == 1:
        kps_pred = kps_pred.unsqueeze(0)

    N = kps_src_disp.shape[0]
    colors = _get_colors(N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def _setup_axis_with_grid(
        ax,
        img,
        title,
        disp_w,
        disp_h,
        orig_w,
        orig_h,
        y_ticks_side="left",
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)

        ax.set_xlim(0, disp_w)
        ax.set_ylim(disp_h, 0)

        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h

        xticks_orig = np.linspace(0, orig_w, grid_steps + 1)
        yticks_orig = np.linspace(0, orig_h, grid_steps + 1)

        xticks_disp = xticks_orig * scale_x
        yticks_disp = yticks_orig * scale_y

        ax.set_xticks(xticks_disp)
        ax.set_yticks(yticks_disp)

        ax.set_xticklabels([f"{int(round(x))}" for x in xticks_orig], fontsize=8)
        ax.set_yticklabels([f"{int(round(y))}" for y in yticks_orig], fontsize=8)

        ax.grid(True, color="white", alpha=0.28, linewidth=0.8)

        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            length=3,
            color="white",
        )

        if y_ticks_side == "left":
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=False,
                labelleft=True,
                labelright=False,
                length=3,
                color="white",
            )
        else:
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=True,
                labelleft=False,
                labelright=True,
                length=3,
                color="white",
            )

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Source
    ax = axes[0]
    _setup_axis_with_grid(
        ax,
        src_resized,
        "Source Image + Keypoints",
        disp_w_src,
        disp_h_src,
        orig_w_src,
        orig_h_src,
        y_ticks_side="left",
    )
    for i, (x, y) in enumerate(kps_src_disp.tolist()):
        ax.plot(
            x,
            y,
            "o",
            color=colors[i],
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.4,
        )
        ax.text(
            x + 19,
            y,
            f"{i+1}",
            color=colors[i],
            fontsize=15,
            weight="bold",
            va="center",
            ha="left",
        )

    # Target
    ax = axes[1]
    _setup_axis_with_grid(
        ax,
        trg_resized,
        "Target Image + Predicted Keypoints",
        disp_w_trg,
        disp_h_trg,
        orig_w_trg,
        orig_h_trg,
        y_ticks_side="right",
    )
    for i, (x, y) in enumerate(kps_pred.tolist()):
        ax.plot(
            x,
            y,
            "o",
            color=colors[i],
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.4,
        )
        ax.text(
            x + 19,
            y,
            f"{i+1}",
            color=colors[i],
            fontsize=15,
            weight="bold",
            va="center",
            ha="left",
        )

    plt.tight_layout()

    if visualize:
        plt.show()
    else:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization → {output_path}")

    plt.close(fig)
