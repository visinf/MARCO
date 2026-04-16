"""Prepare fly_body and locust_body images for the MP-100 benchmark.

This script extracts JPEG images from the DeepPoseKit-Data HDF5 annotation files:
https://github.com/jgraving/DeepPoseKit-Data
"""

import os

import h5py
import numpy as np
from PIL import Image


def export_images(h5_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        images = f["images"]  # shape: (N, H, W, 1)
        for i in range(images.shape[0]):
            img = np.array(images[i]).squeeze(-1)
            Image.fromarray(img).save(os.path.join(out_dir, f"{i}.jpg"), quality=95)


export_images("DeepPoseKit-Data/datasets/fly/annotation_data_release.h5", "fly_body")
export_images("DeepPoseKit-Data/datasets/locust/annotation_data_release.h5", "locust_body")