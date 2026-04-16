"""Base class for semantic correspondence datasets."""

import os
from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch

from .data_utils import pad_tensor_no_resize, resize_longest_side
from util.geometry import regularise_coordinates


class CorrespondenceDataset(Dataset):
    r""" Parent class of PFPascal and SPair """
    def __init__(self, split, dataset, img_size=768, thres="auto", use_original_imsize: bool=True):
        """ CorrespondenceDataset constructor """
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pf-pascal': ('pf-pascal',
                         '_pairs.csv',
                         'PF-dataset-PASCAL/JPEGImages',
                         'PF-dataset-PASCAL/Annotations',
                         'img'),
            'spair': ('SPair-71k',
                      'Layout/large',
                      'JPEGImages',
                      'PairAnnotation',
                      'bbox'),
            'spair-u': ('SPair-U',
                      'Layout/large',
                      'JPEGImages',
                      'PairAnnotation',
                      'bbox'),
            'ap-10k': ('ap-10k',
                      'PairAnnotation',
                      'JPEGImages',
                      'ImageAnnotation',
                      'bbox')
        }

        benchmark = dataset
        datapath = 'data/'
        img_size = img_size
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        self.original_imgsize = use_original_imsize

        # Directory path for train, val, or test splits
        base_path = join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pf-pascal':
            self.spt_path = join(base_path, split + '_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = join(base_path, self.metadata[benchmark][1], split + '.txt')
        elif benchmark == 'spair-u':
            self.spt_path = join(base_path, self.metadata[benchmark][1], split + '.txt')
        elif benchmark == 'ap-10k':
            self.spt_path = join(base_path, self.metadata[benchmark][1], split)
        else:
            raise ValueError(f'Unsupported benchmark: {benchmark}')

        # Directory path for images
        self.img_path = join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark in ('spair', 'spair-u'):
            self.ann_path = join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        self.max_pts = 20
        self.split = split
        self.img_size = img_size
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres
        if self.original_imgsize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])

        # To get initialized in subclass constructors
        self.all_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r""" Returns the number of pairs """
        return len(self.src_imnames)

    def __getitem__(self, idx):
        r""" Constructs and return a batch """

        # Image name
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Object category
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        if self.original_imgsize:
            batch['src_img'] = self.transform(resize_longest_side(src_pil, self.img_size))
            batch['trg_img'] = self.transform(resize_longest_side(trg_pil, self.img_size))
        else:
            batch['src_img'] = self.transform(src_pil)
            batch['trg_img'] = self.transform(trg_pil)

        h1, w1 = batch['src_img'].shape[1:]
        h2, w2 = batch['trg_img'].shape[1:]

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size, (h1, w1))
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size, (h2, w2))
        batch['n_pts'] = torch.tensor(num_pts)

        # Pad images only after keypoints are computed
        if self.original_imgsize:
            batch['src_img'] = pad_tensor_no_resize(batch['src_img'])[0]
            batch['trg_img'] = pad_tensor_no_resize(batch['trg_img'])[0]

        return batch

    def get_image(self, imnames, idx):
        r""" Reads PIL image from path """
        # Must be implemented by subclasses because image path layouts differ per dataset.
        raise NotImplementedError

    def get_pckthresh(self, batch):
        r""" Computes PCK threshold """
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            if len(bbox.shape) == 2:
                bbox = bbox.squeeze(0)
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size()
            if len(imsize_t) == 4:
                imsize_t = imsize_t[1:]
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, ori_imsize, scaled_imsize):
        r""" Returns key-points of an image """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        _, n_pts = pts_list[idx].size()
        x_crds = pts_list[idx][0] * (scaled_imsize[1] / ori_imsize[0])
        y_crds = pts_list[idx][1] * (scaled_imsize[0] / ori_imsize[1])
        kps = torch.stack([x_crds, y_crds], dim=-1)
        kps = regularise_coordinates(kps, scaled_imsize[0], scaled_imsize[1], eps=1e-4)
        kps = kps[:n_pts]

        return kps, n_pts
    
    @staticmethod
    def get_bbox(bbox_list, idx, ori_imsize, scaled_imsize):
        r""" Return object bounding-box """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (scaled_imsize[1] / ori_imsize[0])
        bbox[1::2] *= (scaled_imsize[0] / ori_imsize[1])
        return bbox
