"""MARCO training script for semantic correspondence."""

import argparse
import datetime
import os
import random
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import opts
import util.misc as utils
from util.misc import resume_from_checkpoint, setup_logging, make_deterministic, DDPWrapper
from datasets import build_dataset
from datasets.data_utils import collate_fn_train
from models import build_marco
from engine import train_one_epoch
from util.ema import init_teacher_from_student
from evaluate import evaluate as run_evaluate


def main(args):
    if args.dataset in ('spair-u', 'mp-100'):
        raise ValueError(f"Training on '{args.dataset}' is not supported.")

    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    setup_logging(save_dir=args.output_dir, rank=rank)
    make_deterministic(args.seed + rank)
    print(args)
    print(f'\n  Output dir   : {args.output_dir}\n')

    device = torch.device(args.device)

    # ── Model setup ──────────────────────────────────────────────
    print('\n' + '─'*60)
    print('  Model')
    print('─'*60)
    cfg = args.model_cfg
    print(f"  Size         : {cfg.get('model_size', 'N/A')}")
    print(f"  AdaptFormer  : stages {cfg.get('adaptformer_stages', 'N/A')}")
    model = build_marco(args).to(device)
    model_without_ddp = model

    # Freeze backbone; only adapters and upscale head are trainable.
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'adapter' in name or 'upscale' in name:
            p.requires_grad = True

    teacher = build_marco(args).to(device)
    init_teacher_from_student(model, teacher)

    for p in teacher.parameters():
        p.requires_grad = False

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        model = DDPWrapper(model)

    trainable = [p for p in model_without_ddp.parameters() if p.requires_grad]
    frozen = [p for p in model_without_ddp.parameters() if not p.requires_grad]
    n_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"  Parameters   : {n_params:,}  ({n_params / 1e6:.1f}M)")
    print(f"  Trainable    : {sum(p.numel() for p in trainable):,}")
    print(f"  Frozen       : {sum(p.numel() for p in frozen):,}")

    if args.get('resume_train', ''):
        print(f"  Checkpoint   : {os.path.basename(args.resume_train)}")

    optimizer = torch.optim.AdamW(
        [{'params': trainable, 'initial_lr': args.lr}],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Dataset ──────────────────────────────────────────────────
    print('\n' + '─'*60)
    print('  Dataset')
    print('─'*60)
    dataset_train = build_dataset(args.dataset, image_set='trn', args=args)
    args.batch_size = args.batch_size // args.ngpu
    dcfg = args.dataset_cfg
    print(f"  Train on     : {args.dataset}")
    print(f"  Resolution   : {dcfg.get('train_res', args.get('train_res', 'N/A'))}")
    print(f"  Batch/GPU    : {args.batch_size}  (effective: {args.batch_size * args.ngpu})")
    print(f"  Epochs       : {args.epochs}")

    if args.distributed:
        sampler = DistributedSampler(dataset_train)
    else:
        sampler = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

    loader = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
    )

    output_dir = Path(args.output_dir)

    # ── Validation setup ─────────────────────────────────────────
    do_validation = bool(args.get('validate', False))
    val_dataset, val_args = None, None
    if do_validation:
        val_cfg = OmegaConf.to_container(args.dataset_cfg, resolve=True)
        if args.dataset == 'ap-10k':
            val_cfg['eval_subset'] = 'intra-species'
        val_args = OmegaConf.create({
            'num_workers': int(args.num_workers),
            'dataset': args.dataset,
            'dataset_cfg': val_cfg,
            'output_dir': str(args.output_dir),
        })
        val_dataset = build_dataset(args.dataset, image_set='test', args=val_args)
        if utils.is_main_process():
            subset = ' (intra-species)' if args.dataset == 'ap-10k' else ''
            print(f"  Val on       : {args.dataset}{subset}")
    else:
        print('  Validation   : disabled')
    print('─'*60)

    # ── Resume training ───────────────────────────────────────────
    if args.resume_train:
        model_without_ddp, optimizer = resume_from_checkpoint(
            args.resume_train, model_without_ddp, optimizer, args
        )
        teacher_ck = args.resume_train.replace('checkpoint', 'teacher')
        if os.path.isfile(teacher_ck):
            state = torch.load(teacher_ck, map_location='cpu', weights_only=False)
            teacher.load_state_dict(state['model'], strict=True)
            print(f"  Resumed teacher from {os.path.basename(teacher_ck)}")
        else:
            init_teacher_from_student(model, teacher)
            print("  Teacher checkpoint not found, initialized from student")

    # ── Start from pretrained checkpoint (model-only) ─────────────
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        print(f"  Checkpoint   : {os.path.basename(args.checkpoint)}")
        missing, unexpected = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if missing or unexpected:
            print(f'  Missing keys : {missing}')
            print(f'  Unexpected keys: {unexpected}')

    # ── Training loop ────────────────────────────────────────────
    print('\nStart training ...\n')
    start_time = time.time()
    for epoch in range(args.get('start_epoch', 0), args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        train_one_epoch(
            (model, teacher), loader, optimizer, epoch, args.clip_max_norm, args=args
        )

        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'args': args,
            }, output_dir / f'checkpoint{epoch:04}.pth')

            utils.save_on_master({
                'model': {k.replace('module.', ''): v for k, v in teacher.state_dict().items()},
                'epoch': epoch, 'args': args,
            }, output_dir / f'teacher{epoch:04}.pth')

        # Per-epoch distributed validation
        if do_validation:
            if utils.is_main_process():
                print('\n' + '─'*60)
                print(f'  Validation — epoch {epoch}')
                print('─'*60)
            metrics = run_evaluate(model_without_ddp, val_dataset, val_args)
            if utils.is_main_process():
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f'  Eval Metrics: {{{metrics_str}}}')
                print('─'*60 + '\n')

    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total}')


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('MARCO training', parents=[opts.get_args_parser()])
    args, _ = parser.parse_known_args()
    args = opts.load_train_config(args)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    args.output_dir = os.path.join(
        args.output_dir, f"{args.name_exp}_{timestamp}_{random.randint(0, 999)}"
    )

    main(args)
