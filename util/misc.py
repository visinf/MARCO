"""Training infrastructure: distributed helpers, logging, checkpointing, and DDP wrapper."""

import copy
import io
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict, deque
import datetime
from os.path import join

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Subset


# ── Distributed helpers ───────────────────────────────────────────────────────

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs, _use_new_zipfile_serialization=False)


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.ngpu = args.world_size
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.ngpu = 1
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_dict(input_dict, average=True):
    """Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_eval_indices(dataset_len, rank=0, world_size=1):
    return list(range(rank, dataset_len, world_size))


def shard_eval_dataset(dataset):
    world_size = get_world_size()
    if world_size == 1:
        return dataset
    rank = get_rank()
    indices = get_eval_indices(len(dataset), rank=rank, world_size=world_size)
    return Subset(dataset, indices)


# ── Metric logging ────────────────────────────────────────────────────────────

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg,
            max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# ── DDP wrapper ───────────────────────────────────────────────────────────────

class DDPWrapper:
    """Transparent wrapper around DistributedDataParallel.

    Proxies attribute access through to the underlying module, so that
    `model.train()`, `model.forward_backbone(...)`, etc. work seamlessly
    whether the model is wrapped in DDP or not.
    """

    def __init__(self, ddp_module):
        self.ddp_module = ddp_module

    def __call__(self, *args, **kwargs):
        return self.ddp_module(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.ddp_module, name):
            attr = getattr(self.ddp_module, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return self if result is self.ddp_module else result
                return wrapper
            return attr

        if hasattr(self.ddp_module.module, name):
            return getattr(self.ddp_module.module, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ── Setup and checkpointing ──────────────────────────────────────────────────

def setup_logging(save_dir, console="info", info_filename="info.log", redirect_std=True, rank=0):
    """Logging setup (DDP-aware).
    - Only rank 0 writes to files and console.
    - Single file: info.log (INFO and above).
    - If redirect_std=True, print()/stderr are routed into logging on rank 0,
      and discarded on other ranks.
    """
    root = logging.getLogger()
    if getattr(root, "_logging_init_done", False):
        return
    root._logging_init_done = True

    root.setLevel(logging.DEBUG)

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")

        if info_filename:
            fh = logging.FileHandler(join(save_dir, info_filename), encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            root.addHandler(fh)

        if console is not None:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG if console == "debug" else logging.INFO)
            ch.setFormatter(fmt)
            root.addHandler(ch)

        def _excepthook(tp, val, tb):
            root.error("\n" + "".join(traceback.format_exception(tp, val, tb)))
        sys.excepthook = _excepthook

        if redirect_std:
            class _StreamToLogger(io.TextIOBase):
                def __init__(self, logger, level):
                    self.logger, self.level = logger, level
                def write(self, buf):
                    for line in buf.rstrip().splitlines():
                        if line:
                            self.logger.log(self.level, line)
                    return len(buf)
                def flush(self): pass
            sys.stdout = _StreamToLogger(root, logging.INFO)
    else:
        if redirect_std:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")


def make_deterministic(seed=0):
    """Make results deterministic. May slow down training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resume_from_checkpoint(ck_path: str, model: nn.Module, optimizer=None, args=None):
    """Load model (and optionally optimizer) from a checkpoint file."""
    checkpoint = torch.load(ck_path, map_location='cpu', weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)

    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    if optimizer is None:
        return model

    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        args.start_epoch = checkpoint['epoch'] + 1

    return model, optimizer
