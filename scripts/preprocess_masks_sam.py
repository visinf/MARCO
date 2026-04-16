"""
Extract foreground masks using SAM (Segment Anything Model) for SPair-71k, AP-10K, or PF-PASCAL.

This script is adapted from the preprocessing pipeline of Geo-Aware-SC:
https://github.com/Junyi42/GeoAware-SC

For each image, the script loads the bounding box from the annotation JSON,
runs SAM with that box as a prompt, and saves the resulting binary mask as a
PNG under a `sam_masks/` directory that mirrors the original image layout.

Usage:
    python scripts/preprocess_masks_sam.py --dataset spair
    python scripts/preprocess_masks_sam.py --dataset ap-10k
    python scripts/preprocess_masks_sam.py --dataset pf-pascal

Requirements:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O pretrain/sam_vit_b_01ec64.pth
"""

# Ensure repo root is in sys.path for imports
import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
from tqdm import tqdm

from datasets.data_utils import resize_longest_side
from segment_anything import SamPredictor, sam_model_registry


# Always resolve paths relative to the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIRS = {
    "spair": os.path.join(REPO_ROOT, "data/SPair-71k/JPEGImages"),
    "ap-10k": os.path.join(REPO_ROOT, "data/ap-10k/JPEGImages"),
    "pf-pascal": os.path.join(REPO_ROOT, "data/pf-pascal/PF-dataset-PASCAL/JPEGImages"),
}


def build_pfpascal_annotation_index() -> dict:
    ann_root = os.path.join(REPO_ROOT, "data/pf-pascal/PF-dataset-PASCAL/Annotations")
    ann_index = {}
    for root, _, files in os.walk(ann_root):
        for file_name in files:
            if not file_name.endswith(".mat"):
                continue
            stem = os.path.splitext(file_name)[0]
            ann_index[stem] = os.path.join(root, file_name)
    return ann_index


def parse_bbox(annotation_path: str, dataset: str) -> np.ndarray:
    """Load bounding box and return [x1, y1, x2, y2]."""
    if dataset == "spair":
        with open(annotation_path) as f:
            data = json.load(f)
        box = data["bndbox"]
    elif dataset == "ap-10k":
        with open(annotation_path) as f:
            data = json.load(f)
        box = list(data["bbox"])
        box[2] += box[0]
        box[3] += box[1]
    elif dataset == "pf-pascal":
        box = sio.loadmat(annotation_path)["bbox"][0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return np.array(box, dtype=np.float32)


def get_annotation_path(img_path: str, dataset: str, pfpascal_ann_index: dict | None = None) -> str:
    if dataset == "spair":
        return img_path.replace(".jpg", ".json").replace(
            os.path.join(REPO_ROOT, "data/SPair-71k/JPEGImages"),
            os.path.join(REPO_ROOT, "data/SPair-71k/ImageAnnotation"),
        )
    if dataset == "ap-10k":
        return img_path.replace(".jpg", ".json").replace(
            os.path.join(REPO_ROOT, "data/ap-10k/JPEGImages"),
            os.path.join(REPO_ROOT, "data/ap-10k/ImageAnnotation"),
        )
    if dataset == "pf-pascal":
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if pfpascal_ann_index is None or stem not in pfpascal_ann_index:
            raise FileNotFoundError(f"PF-PASCAL annotation not found for image stem: {stem}")
        return pfpascal_ann_index[stem]
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_output_path(img_path: str, dataset: str, annotation_path: str | None = None) -> str:
    if dataset == "spair":
        out_dir = os.path.dirname(img_path).replace(
            os.path.join(REPO_ROOT, "data/SPair-71k/JPEGImages"),
            os.path.join(REPO_ROOT, "data/SPair-71k/sam_masks"),
        )
    elif dataset == "ap-10k":
        out_dir = os.path.dirname(img_path).replace(
            os.path.join(REPO_ROOT, "data/ap-10k/JPEGImages"),
            os.path.join(REPO_ROOT, "data/ap-10k/sam_masks"),
        )
    elif dataset == "pf-pascal":
        if annotation_path is None:
            raise ValueError("annotation_path is required for PF-PASCAL output paths")
        category = os.path.basename(os.path.dirname(annotation_path))
        out_dir = os.path.join(REPO_ROOT, "data/pf-pascal/PF-dataset-PASCAL/sam_masks", category)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(out_dir, f"{stem}_mask.png")


def scale_box(box: np.ndarray, orig_w: int, orig_h: int, target_size: int) -> np.ndarray:
    """Scale a [x1, y1, x2, y2] box to match longest-side resizing."""
    scale = target_size / max(orig_w, orig_h)
    return box * scale


def main():
    parser = argparse.ArgumentParser(description="Extract SAM masks for SPair-71k / AP-10k / PF-PASCAL")
    parser.add_argument("--dataset", choices=["spair", "ap-10k", "pf-pascal"], required=True,
                        help="Dataset name")
    parser.add_argument("--img_size", type=int, default=960,
                        help="Resize longest side to this before running SAM")
    parser.add_argument("--sam_checkpoint", default=os.path.join(REPO_ROOT, "pretrain/sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_dir = DATASET_DIRS[args.dataset]
    pfpascal_ann_index = build_pfpascal_annotation_index() if args.dataset == "pf-pascal" else None

    # ── Collect images ─────────────────────────────────────────────────────────
    image_paths = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(image_dir)
        for f in files if f.endswith(".jpg")
    )
    print(f"Found {len(image_paths)} images in {image_dir}")

    # ── Load SAM ───────────────────────────────────────────────────────────────
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)


    for img_path in tqdm(image_paths, desc="Extracting masks"):
        # Annotation path
        annotation_path = get_annotation_path(img_path, args.dataset, pfpascal_ann_index)

        # Load & resize image
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        pil_img = resize_longest_side(pil_img, args.img_size)
        image = np.array(pil_img)

        # Scale bounding box
        box = parse_bbox(annotation_path, args.dataset)
        box = scale_box(box, orig_w, orig_h, args.img_size)

        # Run SAM
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)

        # Save mask
        out_path = get_output_path(img_path, args.dataset, annotation_path)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(out_path, (np.clip(masks[0], 0, 1) * 255).astype(np.uint8))

    print("Done.")


if __name__ == "__main__":
    main()
