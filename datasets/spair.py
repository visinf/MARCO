"""SPair-71k dataset."""

import json
import os
from glob import glob
from os.path import join
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .dataset import CorrespondenceDataset


class SPairDataset(CorrespondenceDataset):
    class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    def __init__(self, split: str, dataset: str, img_size: int = 768, use_sam_masks: bool = False):
        r""" SPair-71k dataset constructor """
        super(SPairDataset, self).__init__(split, dataset, img_size=img_size, use_original_imsize=False)
        self.use_sam_masks = use_sam_masks

        if self.use_sam_masks:
            self.sam_mask_path = os.path.abspath(os.path.join(self.img_path, os.pardir, 'sam_masks'))
            if not os.path.isdir(self.sam_mask_path):
                raise FileNotFoundError(
                    f"SAM masks directory not found: {self.sam_mask_path}\n"
                    "Run `python util/preprocess_masks_sam.py --dataset spair` first."
                )

        self.all_data = open(self.spt_path).read().split('\n')
        self.all_data = self.all_data[:len(self.all_data) - 1]

        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.all_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.all_data))
        self.seg_path = os.path.abspath(os.path.join(self.img_path, os.pardir, 'Segmentation'))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.all_data:
            anntn_files.append(glob('%s/%s.json' % (self.ann_path, data_name))[0])

        os.makedirs('.mycache', exist_ok=True)
        cache_file = '.mycache/spair.pth'
        if not os.path.isfile(cache_file) or split != 'trn':
            self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox, self.cls_ids = [], [], [], [], []
            self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []
            print("Reading SPair-71k information...")
            for anntn_file in tqdm(anntn_files, ncols=100):
                with open(anntn_file) as f:
                    anntn = json.load(f)
                self.src_kps.append(torch.tensor(anntn['src_kps']).t().float())
                self.trg_kps.append(torch.tensor(anntn['trg_kps']).t().float())
                self.src_bbox.append(torch.tensor(anntn['src_bndbox']).float())
                self.trg_bbox.append(torch.tensor(anntn['trg_bndbox']).float())
                self.cls_ids.append(self.cls.index(anntn['category']))

                self.vpvar.append(torch.tensor(anntn['viewpoint_variation']))
                self.scvar.append(torch.tensor(anntn['scale_variation']))
                self.trncn.append(torch.tensor(anntn['truncation']))
                self.occln.append(torch.tensor(anntn['occlusion']))
            all_meta = (self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox, self.cls_ids, self.vpvar, self.scvar, self.trncn, self.occln)
            if split == 'trn':
                torch.save(all_meta, cache_file)
        else:
            print('Loading SPAIR metadata from cache...')
            self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox, self.cls_ids, self.vpvar, self.scvar, self.trncn, self.occln = torch.load(cache_file, weights_only=False)

        self.src_identifiers = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]
        
    def __len__(self):
        return len(self.src_imnames)

    def __getitem__(self, idx: int) -> dict:
        """ Construct and return a batch for SPair-71k dataset """
        sample = super(SPairDataset, self).__getitem__(idx)

        h1, w1 = sample['src_img'].shape[1:]
        h2, w2 = sample['trg_img'].shape[1:]
        src_mask = self.get_mask(sample, sample['src_imname'], (h1, w1))
        trg_mask = self.get_mask(sample, sample['trg_imname'], (h2, w2))
        trg_bbox = self.get_bbox(self.trg_bbox, idx, sample['trg_imsize'], (h2, w2))
        pckthres = self.get_pckthresh({'trg_img': sample['trg_img'], 'trg_bbox': trg_bbox}).clone()

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

    def get_mask(self, sample: dict, imname: str, scaled_imsize: tuple) -> torch.Tensor:
        if self.use_sam_masks:
            mask_path = join(self.sam_mask_path, sample['category'], imname.split('.')[0] + '_mask.png')
            tensor_mask = torch.tensor(np.array(Image.open(mask_path)))
            tensor_mask = (tensor_mask > 127).float()
        else:
            mask_path = join(self.seg_path, sample['category'], imname.split('.')[0] + '.png')
            tensor_mask = torch.tensor(np.array(Image.open(mask_path)))

            class_id = self.class_dict[sample['category']] + 1
            tensor_mask[tensor_mask != class_id] = 0
            tensor_mask[tensor_mask == class_id] = 1

        tensor_mask = F.interpolate(tensor_mask.unsqueeze(0).unsqueeze(0).float(),
                                    size=(scaled_imsize[0], scaled_imsize[1]),
                                    mode='nearest').squeeze()

        return tensor_mask

    def get_image(self, img_names: List, idx: int) -> Image.Image:
        path = join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        return Image.open(path).convert('RGB')


def build(image_set, args):
    img_size = args.train_res if image_set in ('trn', 'train') else args.dataset_cfg.inference_res
    return SPairDataset(image_set, 'spair', img_size=img_size,
                        use_sam_masks=args.dataset_cfg.get('use_sam_masks', False))
