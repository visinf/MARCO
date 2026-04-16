"""AP-10k dataset."""

import glob
import json
import os
from os.path import join

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from datasets.data_utils import pad_tensor_no_resize, resize_longest_side
from datasets.dataset import CorrespondenceDataset


class AP10kDataset(CorrespondenceDataset):
    def __init__(self, image_set, dataset, args, img_size: int = 768, use_sam_masks: bool = False):
        r"""AP-10k dataset constructor."""
        super().__init__(image_set, dataset, img_size=img_size)
        self.max_pts = 30
        self.range_ts = torch.arange(self.max_pts)
        self.use_sam_masks = use_sam_masks

        if self.split == 'trn':
            if self.use_sam_masks:
                sam_dir = os.path.join('data', 'ap-10k', 'sam_masks')
                if not os.path.isdir(sam_dir):
                    raise FileNotFoundError(
                        f"SAM masks directory not found: {sam_dir}\n"
                        "Run `python util/preprocess_masks_sam.py --dataset ap10k` first."
                    )
            else:
                raise ValueError(
                    "AP-10k does not provide ground-truth masks. "
                    "Set use_sam_masks=true in configs/dataset/ap-10k.yaml and run `python scripts/preprocess_masks_sam.py --dataset ap-10k` first."
                )

        (
            self.src_imnames,
            self.trg_imnames,
            self.src_kps,
            self.trg_kps,
            self.cls_ids,
            self.cls,
            self.pckthres,
            self.src_bbox,
            self.trg_bbox,
        ) = self.load_and_prepare_data(size=img_size, split=image_set, args=args)
        self.all_data = self.src_imnames

    @staticmethod
    def load_and_prepare_data(size=840, bbox_thresh=True, split='trn', args=None):
        """Load AP-10k pair metadata and cache training split."""
        os.makedirs('.mycache', exist_ok=True)
        cache_file = '.mycache/ap10k.pth'

        if not os.path.isfile(cache_file) or split != 'trn':
            data_dir = args.dataset_cfg.data_path
            subfolders = os.listdir(join(data_dir, 'ImageAnnotation'))
            if split == 'trn':
                categories = sorted(
                    item
                    for subfolder in subfolders
                    for item in os.listdir(join(data_dir, 'ImageAnnotation', subfolder))
                )
            elif split == 'test':
                eval_subset = args.dataset_cfg.eval_subset
                if eval_subset == 'intra-species':
                    categories = sorted(
                        folder
                        for subfolder in subfolders
                        for folder in os.listdir(join(data_dir, 'ImageAnnotation', subfolder))
                    )
                elif eval_subset == 'cross-species':
                    categories = sorted(
                        subfolder
                        for subfolder in subfolders
                        if len(os.listdir(join(data_dir, 'ImageAnnotation', subfolder))) > 1
                    )
                    split += '_cross_species'
                elif eval_subset == 'cross-family':
                    categories = ['all']
                    split += '_cross_family'
                else:
                    raise ValueError(f'Unknown AP-10k eval subset: {eval_subset}')
            else:
                raise ValueError(f'Unsupported AP-10k split: {split}')

            src_imnames, trg_imnames, src_kps, trg_kps, cls_ids, pckthres, src_bbox, trg_bbox = (
                [] for _ in range(8)
            )

            for cat_idx, cat in tqdm(enumerate(categories), total=len(categories), desc="Processing Categories"):
                if cat in ['argali sheep', 'black bear', 'king cheetah'] and split == 'trn':
                    continue

                single_pairs = load_ap10k_data(data_dir, size=size, category=cat, split=split)
                for pair in single_pairs:
                    src_imnames.append(pair['src_imname'])
                    trg_imnames.append(pair['trg_imname'])
                    src_kps.append(pair['src_kps'])
                    trg_kps.append(pair['trg_kps'])
                    cls_ids.append(cat_idx)
                    src_bbox.append(pair['src_bbox'])
                    trg_bbox.append(pair['trg_bbox'])
                    if bbox_thresh:
                        pckthres.append(pair['pckthres'])

            all_meta = (src_imnames, trg_imnames, src_kps, trg_kps, cls_ids, categories, pckthres, src_bbox, trg_bbox)
            if split == 'trn':
                torch.save(all_meta, cache_file)
        else:
            print('Loading AP10k metadata from cache...')
            all_meta = torch.load(cache_file, weights_only=False)

        return all_meta

    def get_image(self, imnames, idx):
        path = imnames[idx]
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return Image.open(path).convert('RGB')

    def load_mask(self, imname: str) -> Image.Image:
        mask_path = imname.replace('JPEGImages', 'sam_masks').rsplit('.', 1)[0] + '_mask.png'
        try:
            mask = Image.open(mask_path)
            mask = resize_longest_side(mask, self.img_size)
        except FileNotFoundError:
            print(f'{mask_path} not found')
            mask = Image.fromarray(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        return mask

    def __getitem__(self, idx):
        r"""Construct and return a batch for AP-10k dataset."""
        sample = super().__getitem__(idx)

        src_mask, trg_mask = None, None
        if self.split == 'trn' and self.use_sam_masks:
            src_mask = self.load_mask(sample['src_imname'])
            trg_mask = self.load_mask(sample['trg_imname'])
            src_mask = pad_tensor_no_resize((TF.to_tensor(src_mask) > 0.5).float())[0][0]
            trg_mask = pad_tensor_no_resize((TF.to_tensor(trg_mask) > 0.5).float())[0][0]

        n_pts = sample['n_pts']
        keypoints = torch.stack([sample['src_kps'], sample['trg_kps']], dim=0)

        return {
            'samples': torch.stack([sample['src_img'], sample['trg_img']])[None],
            'keypoints': keypoints,
            'visibility_mask': torch.ones(n_pts, dtype=torch.bool),
            'n_pts': n_pts,
            'sem_mask_A': src_mask,
            'sem_mask_B': trg_mask,
            'pck_thresh': self.pckthres[idx].clone(),
            'category': sample['category'],
        }


def load_ap10k_data(path="data/ap-10k", size=840, category='cat', split='test'):
    np.random.seed(42)
    pairs = sorted(glob.glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    pair_records = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]
        src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

        with open(source_json_path) as f:
            src_file = json.load(f)
        with open(target_json_path) as f:
            trg_file = json.load(f)

        source_bbox = np.asarray(src_file["bbox"])
        target_bbox = np.asarray(trg_file["bbox"])

        source_size = np.array([src_file["width"], src_file["height"]])
        target_size = np.array([trg_file["width"], trg_file["height"]])

        source_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
        source_kps[:, -1] /= 2
        target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
        target_kps[:, -1] /= 2

        vis_mask = (source_kps[:, 2] > 0) & (target_kps[:, 2] > 0)
        idx_vis = torch.nonzero(vis_mask, as_tuple=False).flatten()
        if idx_vis.numel() == 0:
            continue

        trg_scale = size / max(target_size[0], target_size[1])
        pckthres = max(target_bbox[3], target_bbox[2]) * trg_scale

        pair_records.append({
            'src_imname': src_img_path,
            'trg_imname': trg_img_path,
            'src_kps': source_kps[idx_vis, :2].t().contiguous(),
            'trg_kps': target_kps[idx_vis, :2].t().contiguous(),
            'src_bbox': torch.tensor(source_bbox, dtype=torch.float32),
            'trg_bbox': torch.tensor(target_bbox, dtype=torch.float32),
            'pckthres': torch.tensor(pckthres, dtype=torch.float32),
        })

    return pair_records


def build(image_set, args):
    img_size = args.train_res if image_set in ('trn', 'train') else args.dataset_cfg.inference_res
    return AP10kDataset(image_set, 'ap-10k', args, img_size=img_size,
                        use_sam_masks=args.dataset_cfg.get('use_sam_masks', False))
