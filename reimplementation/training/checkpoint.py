"""
Checkpoint utilities for SparseDrive training.
Save and load training state including model, optimizer, scheduler, and metadata.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    iteration: int,
    work_dir: str,
    filename: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training checkpoint.

    In distributed training, only rank 0 saves the checkpoint.

    Args:
        model: PyTorch model (can be wrapped in DistributedDataParallel)
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        iteration: Current iteration number
        work_dir: Working directory to save checkpoint
        filename: Checkpoint filename (default: 'epoch_{epoch}.pth')
        meta: Additional metadata to save

    Example:
        >>> save_checkpoint(
        ...     model, optimizer, scheduler,
        ...     epoch=10, iteration=3500,
        ...     work_dir='work_dirs/sparsedrive',
        ...     filename='epoch_10.pth'
        ... )
    """
    # Only save on rank 0 in distributed training
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # Create work directory if needed
    os.makedirs(work_dir, exist_ok=True)

    # Default filename
    if filename is None:
        filename = f'epoch_{epoch}.pth'

    checkpoint_path = os.path.join(work_dir, filename)

    # Get model state dict (unwrap DDP if needed)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Build checkpoint dict
    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }

    # Add metadata
    if meta is not None:
        checkpoint['meta'] = meta

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Also save as latest.pth for easy resuming
    latest_path = os.path.join(work_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: PyTorch optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        map_location: Device to load checkpoint to

    Returns:
        Checkpoint dict containing epoch, iteration, and metadata

    Example:
        >>> checkpoint = load_checkpoint(
        ...     'work_dirs/sparsedrive/epoch_10.pth',
        ...     model, optimizer, scheduler,
        ...     map_location='cuda'
        ... )
        >>> start_epoch = checkpoint['epoch']
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model weights
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    print(f"  Model weights loaded")

    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"  Optimizer state loaded")

    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"  Scheduler state loaded")

    # Return metadata
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'iteration': checkpoint.get('iteration', 0),
        'meta': checkpoint.get('meta', {})
    }

    print(f"  Resuming from epoch {info['epoch']}, iteration {info['iteration']}")

    return info


def get_latest_checkpoint(work_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint in work directory.

    Args:
        work_dir: Working directory

    Returns:
        Path to latest.pth if it exists, otherwise None
    """
    latest_path = os.path.join(work_dir, 'latest.pth')
    if os.path.exists(latest_path):
        return latest_path
    return None
