"""MARCO evaluation script for semantic correspondence."""

import argparse
import datetime
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join

import opts
import util.misc as utils
from util.misc import setup_logging, make_deterministic
from datasets import build_dataset
from datasets.data_utils import collate_fn_eval, batch_to_cuda
from models import build_marco
from util.evaluator import PCKEvaluator


def main(args):
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    setup_logging(save_dir=args.output_dir, rank=rank)
    make_deterministic(args.seed + rank)
    print(args)
    print(f'\n  Output dir   : {args.output_dir}\n')

    device = torch.device(args.device)

    # ── Loading Model ────────────────────────────────────────────
    print('\n' + '─'*60)
    print('  Model')
    print('─'*60)
    cfg = args.model_cfg
    print(f"  Size         : {cfg.get('model_size', 'N/A')}")
    print(f"  AdaptFormer  : stages {cfg.get('adaptformer_stages', 'N/A')}")
    model = build_marco(args).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        print(f"  Checkpoint   : {os.path.basename(args.checkpoint)}")
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if missing or unexpected:
            print(f'  Missing keys : {missing}')
            print(f'  Unexpected keys: {unexpected}')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}  ({n_params / 1e6:.1f}M)")

    # ── Dataset ──────────────────────────────────────────────────
    print('\n' + '─'*60)
    print('  Dataset')
    print('─'*60)
    dcfg = args.dataset_cfg
    print(f"  Dataset      : {args.dataset}")
    print(f"  Eval subset  : {dcfg.get('eval_subset', 'N/A')}")
    print(f"  Resolution   : {dcfg.get('inference_res', 'N/A')}")
    print(f"  PCK by       : {dcfg.get('pck_by', 'N/A')}")
    print('─'*60)
    dataset_test = build_dataset(args.dataset, image_set='test', args=args)

    # ── Start Inference ──────────────────────────────────────────
    print('\nStart inference ...\n')
    start = time.time()
    _ = evaluate(model, dataset_test, args)
    print(f'Total time: {time.time() - start:.1f}s')


def evaluate(model, dataset, args):
    eval_dataset = utils.shard_eval_dataset(dataset)
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn_eval)
    model.eval()
    evaluator = PCKEvaluator(pck_by=args.dataset_cfg.pck_by)

    pbar = tqdm(loader, ncols=100, desc='PCK', disable=not utils.is_main_process())
    for idx, batch in enumerate(pbar):
        batch = batch_to_cuda(batch)
        samples = batch['samples']
        h, w = samples.shape[-2:]
        src_kps, trg_kps = batch['keypoints'][:1], batch['keypoints'][1:]

        with torch.no_grad():
            pred = model(samples, src_kps, (h, w))

        evaluator.calculate_pck(
            trg_kps, pred, [batch['n_pts']], [batch['category']],
            pckthres=batch['pck_thresh'][None],
        )

        if (idx + 1) % 50 == 0:
            pcks = evaluator.get_result()
            pbar.set_description(
                f'PCK@(0.01,0.05,0.1) = {pcks[0]*100:.1f}, {pcks[1]*100:.1f}, {pcks[2]*100:.1f}'
            )

    if utils.is_dist_avail_and_initialized():
        gathered_states = [None] * utils.get_world_size()
        torch.distributed.all_gather_object(gathered_states, evaluator.state_dict())
        merged = PCKEvaluator(pck_by=args.dataset_cfg.pck_by)
        for state in gathered_states:
            merged.merge_state_dict(state)
        evaluator = merged

    metrics = None
    if utils.is_main_process():
        evaluator.print_summarize_result()
        pcks = evaluator.get_result()

        metrics = {f'PCK@{t}': float(p) for t, p in zip([0.01, 0.05, 0.1, 0.15], pcks)}
        if args.dataset in ('spair', 'spair-u'):
            evaluator.save_result(join(args.output_dir, 'per_class_score.txt'))

    if utils.is_dist_avail_and_initialized():
        shared_metrics = [metrics]
        torch.distributed.broadcast_object_list(shared_metrics, src=0)
        metrics = shared_metrics[0]

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MARCO evaluation', parents=[opts.get_args_parser()])
    args, _ = parser.parse_known_args()
    opts.load_eval_config(args)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    args.output_dir = join(
        args.output_dir, f"{args.name_exp}_{timestamp}_{random.randint(0, 999)}"
    )
    os.makedirs(args.output_dir)

    main(args)
