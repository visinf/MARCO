"""PCK evaluator for semantic correspondence."""

import copy
import numpy as np
import torch
from torch import Tensor


class PCKEvaluator:
    """Percentage of Correct Keypoints (PCK) evaluator.

    Args:
        pck_by: 'image' for per-image PCK, 'point' for per-point PCK.
        avg_by: 'all' for global average, 'per_cat' for per-category average.
    """

    def __init__(self, pck_by: str = 'image', avg_by: str = 'all'):
        self.alpha = [0.01, 0.05, 0.1, 0.15, 0.20]
        self.by = pck_by
        self.avg_by = avg_by
        self.result = {}
        for alpha in self.alpha:
            self.result[f'pck{alpha}'] = {"all": []}

    def clear_result(self):
        self.result = {key: {'all': []} for key in self.result}

    def state_dict(self):
        return copy.deepcopy(self.result)

    def load_state_dict(self, state_dict):
        self.result = copy.deepcopy(state_dict)

    def merge_state_dict(self, state_dict):
        for alpha_key, alpha_result in state_dict.items():
            if alpha_key not in self.result:
                self.result[alpha_key] = {'all': []}
            for category, values in alpha_result.items():
                if category not in self.result[alpha_key]:
                    self.result[alpha_key][category] = []
                self.result[alpha_key][category].extend(values)

    def calculate_pck(self, trg_kps: Tensor, matches: Tensor, n_pts: Tensor,
                      categories, pckthres):
        """Accumulate PCK results for a batch.

        Args:
            trg_kps: (B, N, 2) ground-truth target keypoints.
            matches: (B, N, 2) predicted matches.
            n_pts: (B,) number of valid keypoints per sample.
            categories: list of category labels.
            pckthres: (B,) PCK threshold per sample.
        """
        B = trg_kps.shape[0]

        for b in range(B):
            npt = n_pts[b]
            thres = pckthres[b].item()
            category = categories[b]

            diff = torch.norm(trg_kps[b, :npt] - matches[b, :npt], dim=-1)
            for alpha in self.alpha:
                key = f'pck{alpha}'
                if category not in self.result[key]:
                    self.result[key][category] = []

                if self.by == 'image':
                    pck = (diff <= alpha * thres).float().mean().item()
                    self.result[key][category].append(pck)
                    self.result[key]["all"].append(pck)
                elif self.by == "point":
                    pck = (diff <= alpha * thres).float().tolist()
                    self.result[key][category].extend(pck)
                    self.result[key]["all"].extend(pck)
                else:
                    raise ValueError("pck_by must be 'image' or 'point'")

    def avg_result_all(self):
        out = {}
        for alpha in self.alpha:
            out[f'pck{alpha}'] = {}
            for k, v in self.result[f'pck{alpha}'].items():
                out[f'pck{alpha}'][k] = np.array(v).mean()
        return out

    def get_result(self):
        result = self.avg_result_all()
        if self.avg_by == 'all':
            return tuple(result[f'pck{a}']['all'] for a in self.alpha)
        else:
            cat_list = [cat for cat in self.result[f'pck{self.alpha[0]}'] if cat != 'all']
            per_cat_res = {}
            for alpha in self.alpha:
                per_cat_avg = np.array([result[f'pck{alpha}'][cat] for cat in cat_list])
                per_cat_res[f'pck{alpha}'] = per_cat_avg.mean()
            return tuple(per_cat_res[f'pck{a}'] for a in self.alpha)

    def print_summarize_result(self):
        result = self.avg_result_all()
        print(" " * 16 + "".join([f"{alpha:<10}" for alpha in self.alpha]))
        pcks = [f"{result[f'pck{alpha}']['all']:.4f}" for alpha in self.alpha]
        print(" " * 12 + "".join([f"{pck:<10}" for pck in pcks]))

    def save_result(self, save_file):
        result = self.avg_result_all()
        outstring = "\n"
        catstring = ""
        for alpha in self.alpha:
            cat_list = []
            pck_list = []
            for k, v in result[f'pck{alpha}'].items():
                if k != "all":
                    cat_list.append(k)
                    pck_list.append(v)
            cat_list = np.array(cat_list)
            pck_list = np.array(pck_list)
            indices = np.argsort(cat_list)
            cat_list = cat_list[indices].tolist()
            pck_list = pck_list[indices].tolist()
            pck_list = [f"{pck:.2%}" for pck in pck_list]
            cat_list.append("all")
            pck_list.append(f"{result[f'pck{alpha}']['all']:.2%}")

            if len(catstring) == 0:
                catstring += " " * 12 + "".join([f"{category:<12}" for category in cat_list]) + "\n"
                outstring += catstring
            row = f"{alpha:<12}" + "".join([f"{pck:<12}" for pck in pck_list]) + "\n"
            outstring += row

        outstring += "-----------------------------------------------------------------\n"

        with open(save_file, "w") as f:
            f.write(outstring)
