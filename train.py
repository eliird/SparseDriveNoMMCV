#!/usr/bin/env python
"""
Training script for SparseDrive.
Pure PyTorch implementation without mmcv dependencies.

Usage:
    # Single GPU training
    python train.py --config projects/configs/sparsedrive_small_stage1.py --work-dir work_dirs/sparsedrive_small

    # Multi-GPU distributed training (recommended)
    python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --config projects/configs/sparsedrive_small_stage1.py \
        --work-dir work_dirs/sparsedrive_small \
        --launcher pytorch

    # Resume from checkpoint
    python train.py --config projects/configs/sparsedrive_small_stage1.py \
        --work-dir work_dirs/sparsedrive_small \
        --resume-from work_dirs/sparsedrive_small/latest.pth
"""

import argparse
import os
import sys
import importlib.util

import torch
import torch.distributed as dist
import torch.nn.parallel

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from reimplementation.models import SparseDrive
from reimplementation.dataset import build_dataset, build_dataloader
from reimplementation.training import (
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    Logger,
    Trainer,
    get_dist_info
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SparseDrive')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--work-dir', required=True, help='Working directory for outputs')
    parser.add_argument('--resume-from', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch', 'slurm'],
                        help='Job launcher for distributed training')
    # Support both --local_rank and --local-rank for compatibility
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0,
                        help='Local rank for distributed training (set by torch.distributed.launch)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args


def load_config(config_path):
    """
    Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config module
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def init_dist(launcher, backend='nccl'):
    """
    Initialize distributed training.

    Args:
        launcher: Launcher type ('pytorch', 'slurm', 'none')
        backend: Backend for distributed training (default: 'nccl')
    """
    if launcher == 'pytorch':
        # Use environment variables set by torch.distributed.launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

        print(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    elif launcher == 'slurm':
        # SLURM environment
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']

        # Get local rank
        local_rank = proc_id % torch.cuda.device_count()

        dist.init_process_group(backend=backend, rank=proc_id, world_size=ntasks)
        torch.cuda.set_device(local_rank)

        print(f"Initialized SLURM distributed training: rank={proc_id}, world_size={ntasks}, local_rank={local_rank}")

    elif launcher == 'none':
        # Single GPU training
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print("Using single GPU training")
        else:
            print("WARNING: CUDA not available, using CPU")
    else:
        raise ValueError(f"Unsupported launcher: {launcher}")


def main():
    """Main training function."""
    args = parse_args()

    # Initialize distributed training
    init_dist(args.launcher)
    rank, world_size = get_dist_info()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)

    # Load config
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Loading config from {args.config}")
        print(f"{'='*80}\n")

    cfg = load_config(args.config)

    # Build model
    if rank == 0:
        print(f"{'='*80}")
        print(f"Building Model")
        print(f"{'='*80}\n")

    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')
    assert model_type == 'SparseDrive', f"Only SparseDrive model is supported, got {model_type}"

    model = SparseDrive(**model_cfg)

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # Compile model if requested (PyTorch 2.0+)
    compile_model = getattr(cfg, 'compile_model', False)
    if compile_model:
        compile_config = getattr(cfg, 'compile_config', {})
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Attempting to compile model with torch.compile")
            print(f"  Mode: {compile_config.get('mode', 'default')}")
            print(f"  Fullgraph: {compile_config.get('fullgraph', False)}")
            print(f"  Dynamic: {compile_config.get('dynamic', True)}")
            print(f"  Note: Custom CUDA ops may cause compilation to fall back")
            print(f"{'='*80}\n")

        try:
            model = torch.compile(model, **compile_config)
            if rank == 0:
                print("✓ Model compiled successfully!")
        except Exception as e:
            if rank == 0:
                print(f"⚠ Model compilation failed: {e}")
                print("  Continuing with uncompiled model...")

    # Wrap model with DistributedDataParallel for multi-GPU training
    if args.launcher != 'none':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=True  # SparseDrive may have unused params in different modes
        )
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}\n")

    # Build dataset and dataloader
    if rank == 0:
        print(f"{'='*80}")
        print(f"Building Dataset and DataLoader")
        print(f"{'='*80}\n")

    train_dataset = build_dataset(cfg.data['train'])

    # Calculate samples per GPU
    samples_per_gpu = cfg.data.get('samples_per_gpu', 1)
    workers_per_gpu = cfg.data.get('workers_per_gpu', 2)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        num_gpus=world_size,
        dist=(args.launcher != 'none'),
        seed=args.seed,
        shuffle=True
    )

    if rank == 0:
        print(f"Dataset: {len(train_dataset)} samples")
        print(f"DataLoader: {len(train_loader)} batches, {samples_per_gpu} samples per GPU")
        print(f"Total batch size: {samples_per_gpu * world_size}\n")

    # Build optimizer
    if rank == 0:
        print(f"{'='*80}")
        print(f"Building Optimizer")
        print(f"{'='*80}\n")

    optimizer = build_optimizer(model, cfg.optimizer)

    # Calculate total iterations
    # Use actual dataloader length (auto-detects GPU count) instead of config value
    num_iters_per_epoch = len(train_loader)
    max_epochs = getattr(cfg, 'num_epochs', 100)
    max_iters = max_epochs * num_iters_per_epoch

    if rank == 0:
        print(f"\nTraining schedule:")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Iterations per epoch: {num_iters_per_epoch}")
        print(f"  Total iterations: {max_iters}\n")

    # Build scheduler
    if rank == 0:
        print(f"{'='*80}")
        print(f"Building LR Scheduler")
        print(f"{'='*80}\n")

    scheduler = build_scheduler(optimizer, cfg.lr_config, max_iters)

    # Build logger
    log_dir = os.path.join(args.work_dir, 'logs')
    log_interval = cfg.log_config.get('interval', 50)
    logger = Logger(log_dir=log_dir, log_interval=log_interval, rank=rank)

    # Create trainer
    checkpoint_interval = getattr(cfg, 'checkpoint_epoch_interval', 20)
    grad_clip_config = getattr(cfg, 'optimizer_config', {})
    fp16_config = getattr(cfg, 'fp16', {})
    precision = getattr(cfg, 'precision', 'fp32')  # Default to fp32 if not specified

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        logger=logger,
        work_dir=args.work_dir,
        max_epochs=max_epochs,
        grad_clip_config=grad_clip_config,
        fp16_config=fp16_config,
        precision=precision,
        checkpoint_interval=checkpoint_interval,
        rank=rank,
        max_iters=max_iters
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from is not None:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Resuming from checkpoint")
            print(f"{'='*80}\n")

        checkpoint_info = load_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        start_epoch = checkpoint_info['epoch']
        trainer.current_iter = checkpoint_info['iteration']

    # Start training
    trainer.train(start_epoch=start_epoch)

    # Cleanup distributed training
    if args.launcher != 'none':
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
