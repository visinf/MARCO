"""PF-PASCAL dataset."""

import os
from typing import List
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .data_utils import pad_tensor_no_resize, resize_longest_side
from .dataset import CorrespondenceDataset


class PFPascalDataset(CorrespondenceDataset):
    def __init__(self, split: str, dataset: str, img_size: int = 768, use_sam_masks: bool = False):
        """ PF-PASCAL dataset constructor """
        super(PFPascalDataset, self).__init__(split, dataset, img_size=img_size)
        self.use_sam_masks = use_sam_masks
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        if self.split == 'trn':
            if self.use_sam_masks:
                self.sam_mask_path = os.path.abspath(os.path.join(self.img_path, os.pardir, 'sam_masks'))
                if not os.path.isdir(self.sam_mask_path):
                    raise FileNotFoundError(
                        f"SAM masks directory not found: {self.sam_mask_path}\n"
                        "Run `python scripts/preprocess_masks_sam.py --dataset pf-pascal` first."
                    )
            else:
                raise ValueError(
                    "PF-PASCAL does not provide ground-truth masks. "
                    "Set --use_sam_masks and run `python scripts/preprocess_masks_sam.py --dataset pf-pascal` first."
                )

        self.all_data = pd.read_csv(self.spt_path)

        self.src_imnames = np.array(self.all_data.iloc[:, 0])
        self.trg_imnames = np.array(self.all_data.iloc[:, 1])

        self.cls_ids = self.all_data.iloc[:, 2].values.astype('int') - 1
        self.split = split

        if split == 'trn':
            self.flip = self.all_data.iloc[:, 3].values.astype('int')
        else:
            self.flip = np.zeros((len(self.src_imnames),), dtype=np.int64)

        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.nonzero(torch.isnan(src_kk))) != 0 or \
                        len(torch.nonzero(torch.isnan(trg_kk))) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

        self.src_identifiers = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in
                                zip(self.cls_ids, self.src_imnames, self.flip)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in
                                zip(self.cls_ids, self.trg_imnames, self.flip)]

    def get_image(self, imnames: List, idx: int) -> Image.Image:
        """ Reads PIL image from path """
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def load_mask(self, sample: dict, imname: str) -> Image.Image:
        mask_path = os.path.join(self.sam_mask_path, sample['category'], imname.rsplit('.', 1)[0] + '_mask.png')
        try:
            mask = Image.open(mask_path)
            mask = resize_longest_side(mask, self.img_size)
        except FileNotFoundError:
            print(f'{mask_path} not found')
            mask = Image.fromarray(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        return mask

    def __getitem__(self, idx: int) -> dict:
        r""" Constructs and returns a batch for PF-PASCAL dataset """
        sample = super(PFPascalDataset, self).__getitem__(idx)

        h2, w2 = sample['trg_img'].shape[1:]
        trg_bbox = self.get_bbox(self.trg_bbox, idx, sample['trg_imsize'], (h2, w2))
        pckthres = self.get_pckthresh({'trg_img': sample['trg_img'], 'trg_bbox': trg_bbox}).clone()
        src_mask, trg_mask = None, None
        if self.split == 'trn' and self.use_sam_masks:
            src_mask = self.load_mask(sample, sample['src_imname'])
            trg_mask = self.load_mask(sample, sample['trg_imname'])
            src_mask = pad_tensor_no_resize((TF.to_tensor(src_mask) > 0.5).float())[0][0]
            trg_mask = pad_tensor_no_resize((TF.to_tensor(trg_mask) > 0.5).float())[0][0]

        n_pts = sample['n_pts']

        keypoints = torch.stack([sample['src_kps'], sample['trg_kps']], dim=0)  # (2, P, 2)

        return {
            'samples': torch.stack([sample['src_img'], sample['trg_img']])[None],  # (1,2,C,H,W)
            'keypoints': keypoints,
            'visibility_mask': torch.ones(n_pts, dtype=torch.bool),
            'n_pts': n_pts,
            'sem_mask_A': src_mask,
            'sem_mask_B': trg_mask,
            'pck_thresh': pckthres,
            'category': sample['category'],
        }


def read_mat(path, obj_name):
    """Read specified objects from Matlab data file (.mat)."""
    return sio.loadmat(path)[obj_name]


def build(image_set, args):
    img_size = args.train_res if image_set in ('trn', 'train') else args.dataset_cfg.inference_res
    return PFPascalDataset(image_set, 'pf-pascal', img_size=img_size,
                        use_sam_masks=args.dataset_cfg.get('use_sam_masks', False))
