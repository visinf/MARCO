"""SPair-U dataset."""

import glob
import json
import os
from os.path import join
from typing import List
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import CorrespondenceDataset


class SPairDataset_U(CorrespondenceDataset):
    def __init__(self, split:str , dataset: str, img_size: int = 768):
        r""" SPair-71k dataset constructor """
        super(SPairDataset_U, self).__init__(split, dataset, img_size=img_size, use_original_imsize=False)

        self.all_data = open(self.spt_path).read().split('\n')
        self.all_data = self.all_data[:len(self.all_data) - 1]


        self.src_imnames = []
        self.trg_imnames = []
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.all_data:
            file = glob.glob('%s/%s.json' % (self.ann_path, data_name))
            if len(file)>0:
                anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])

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

                self.src_imnames.append(anntn['src_imname'])
                self.trg_imnames.append(anntn['trg_imname'])

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
        r""" Construct and return a batch for SPair-71k dataset """
        sample = super(SPairDataset_U, self).__getitem__(idx)

        h2, w2 = sample['trg_img'].shape[1:]
        trg_bbox = self.get_bbox(self.trg_bbox, idx, sample['trg_imsize'], (h2, w2))
        pckthres = self.get_pckthresh({'trg_img': sample['trg_img'], 'trg_bbox': trg_bbox}).clone()

        n_pts = sample['n_pts']

        keypoints = torch.stack([sample['src_kps'], sample['trg_kps']], dim=0)  # (2, P, 2)

        return {
            'samples': torch.stack([sample['src_img'], sample['trg_img']])[None],  # (1,2,C,H,W)
            'keypoints': keypoints,
            'visibility_mask': torch.ones(n_pts, dtype=torch.bool),
            'n_pts': n_pts,
            'pck_thresh': pckthres,
            'category': sample['category'],
        }

    def get_image(self, img_names: List, idx: int) -> Image.Image:
        path = join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        return Image.open(path).convert('RGB')


def build(image_set, args):
    img_size = args.dataset_cfg.inference_res
    return SPairDataset_U(image_set, 'spair-u', img_size=img_size)
