import glob
import json
from collections import defaultdict
from typing import List, Tuple
from torchvision import transforms
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image

from util.geometry import regularise_coordinates

# ── Helpers ───────────────────────────────────────────────────────────────────

def _kp_flat_to_tensor_xy(kp_flat: List[float]) -> torch.Tensor:
    """COCO keypoints [x,y,v,...] → (N,2) float tensor of visible points (v>0)."""
    pts = []
    for i in range(0, len(kp_flat), 3):
        x, y, v = kp_flat[i:i + 3]
        if v is not None and float(v) > 0:
            pts.append((float(x), float(y)))
    if not pts:
        return torch.empty(0, 2, dtype=torch.float32)
    return torch.tensor(pts, dtype=torch.float32)


def _build_annotation_index(annotation_path: str) -> dict:
    """Load all MP-100 test annotation files and build a file_name → {wh, anns} index."""
    images = {}   # image_id → image dict
    by_img = defaultdict(list)  # image_id → [ann, ...]

    test_files = sorted(glob.glob(join(annotation_path, "*test*.json")))
    for path in test_files:
        with open(path, "r") as fj:
            coco = json.load(fj)
        for img in coco["images"]:
            images[img["id"]] = img
        for ann in coco["annotations"]:
            if "keypoints" in ann and ann.get("num_keypoints", 0) > 0:
                by_img[ann["image_id"]].append(ann)

    # index by relative file_name (matches pair JSON src_file / trg_file)
    index = {}
    for iid, im in images.items():
        anns = by_img.get(iid, [])
        if not anns:
            continue
        index[im["file_name"]] = {
            "wh": (int(im["width"]), int(im["height"])),
            "anns": anns,
        }
    return index

# ── Dataset ───────────────────────────────────────────────────────────────────

class MP100Dataset(Dataset):
    """MP-100 dataset loaded from pre-generated pair JSONs."""

    def __init__(self, split: str, data_path: str, pairs_path: str, annotation_path: str, img_size: int = 768):
        super().__init__()
        self.data_path = data_path
        pair_file = join(pairs_path, f"pairs_{split}.json")
        with open(pair_file, "r") as f:
            self.pairs = json.load(f)

        self.img_size = img_size
        self._ann_index = _build_annotation_index(annotation_path)

        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, file_rel: str) -> Tuple:
        """Load image + keypoints + bboxes for a single file_name."""
        file_abs = join(self.data_path, file_rel)
        img = Image.open(file_abs).convert("RGB")
        img_t = self.transform(img)
        h, w = img_t.shape[-2:]

        rec = self._ann_index.get(file_rel)
        if rec is None:
            return img_t, torch.empty(0, 2), torch.empty(0, 4), (h, w)

        W, H = rec["wh"]

        # keypoints
        kps_list = []
        for a in rec["anns"]:
            kp = _kp_flat_to_tensor_xy(a["keypoints"])
            if kp.numel():
                kps_list.append(kp)
        keypoints = torch.cat(kps_list, dim=0) if kps_list else torch.empty(0, 2, dtype=torch.float32)
        if keypoints.numel() and W and H:
            keypoints = keypoints.clone()
            keypoints[:, 0] *= w / float(W)
            keypoints[:, 1] *= h / float(H)

        # bboxes (scaled)
        boxes = []
        for a in rec["anns"]:
            if "bbox" in a and a["bbox"]:
                x, y, bw, bh = a["bbox"][:4]
                boxes.append(torch.tensor([x * w / W, y * h / H, bw * w / W, bh * h / H], dtype=torch.float32))
        bbox = torch.stack(boxes, dim=0) if boxes else torch.empty(0, 4, dtype=torch.float32)

        return img_t, keypoints, bbox, (h, w)

    def __getitem__(self, idx: int) -> dict:
        pair_info = self.pairs[idx]
        src_file, trg_file = pair_info["src_file"], pair_info["trg_file"]

        im_sup, kps_sup, _, (h1, w1) = self._load_image(src_file)
        im_qry, kps_qry, bbox_qry, (h2, w2) = self._load_image(trg_file)

        # regularise and align keypoints
        n = len(kps_sup)
        if kps_sup.numel():
            kps_sup = regularise_coordinates(kps_sup[:n], h1, w1, eps=1e-4)
        if kps_qry.numel():
            kps_qry = regularise_coordinates(kps_qry[:n], h2, w2, eps=1e-4)

        n_eff = min(kps_sup.shape[0], kps_qry.shape[0]) if (kps_sup.numel() and kps_qry.numel()) else 0
        keypoints = torch.stack([kps_sup[:n_eff], kps_qry[:n_eff]], dim=1) if n_eff > 0 else torch.empty(0, 2, 2)

        if bbox_qry.numel() > 0:
            longest_sides = torch.max(bbox_qry[:, 2], bbox_qry[:, 3])
            pck_thres = torch.max(longest_sides)
        else:
            pck_thres = torch.tensor(max(h2, w2), dtype=torch.float32)

        return {
            'samples': torch.stack([im_sup, im_qry])[None],        # (1,2,C,H,W)
            'keypoints': keypoints.permute(1, 0, 2) if keypoints.numel() else torch.empty(2, 0, 2),  # (2,P,2)
            'visibility_mask': torch.ones(n_eff, dtype=torch.bool) if n_eff > 0 else torch.empty(0, dtype=torch.bool),
            'n_pts': n_eff,
            'sem_mask_A': None,
            'sem_mask_B': None,
            'pck_thresh': pck_thres,
            'category': str(pair_info["category_id"]),
        }


def build(image_set, args):
    cfg = args.dataset_cfg
    img_size = cfg.inference_res
    return MP100Dataset(
        split=cfg.eval_subset,
        data_path=cfg.data_path,
        pairs_path=cfg.pairs_path,
        annotation_path=cfg.annotation_path,
        img_size=img_size,
    )
