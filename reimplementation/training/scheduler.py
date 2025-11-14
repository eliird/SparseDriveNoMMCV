"""
Learning rate scheduler builder for SparseDrive training.
Pure PyTorch implementation with warmup support.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: Wrapped optimizer
        warmup_iters: Number of warmup iterations
        max_iters: Total number of training iterations
        warmup_ratio: LR at start of warmup as fraction of base LR (default: 1/3)
        min_lr_ratio: Minimum LR as fraction of base LR (default: 0.001)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        max_iters: int,
        warmup_ratio: float = 1.0 / 3.0,
        min_lr_ratio: float = 1e-3,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current iteration."""
        if self.last_epoch < self.warmup_iters:
            # Linear warmup phase
            progress = self.last_epoch / self.warmup_iters
            warmup_lr_ratio = self.warmup_ratio + (1 - self.warmup_ratio) * progress
            return [base_lr * warmup_lr_ratio for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            # Cosine annealing from 1.0 to min_lr_ratio
            cosine_lr_ratio = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
            return [base_lr * cosine_lr_ratio for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    """Constant learning rate (no scheduling)."""

    def get_lr(self):
        return self.base_lrs


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_config: Dict[str, Any],
    max_iters: int
) -> _LRScheduler:
    """
    Build learning rate scheduler from config dict.

    Args:
        optimizer: PyTorch optimizer
        lr_config: Config dict with format:
            {
                'policy': 'CosineAnnealing',
                'warmup': 'linear',
                'warmup_iters': 500,
                'warmup_ratio': 1.0 / 3,
                'min_lr_ratio': 1e-3,
            }
        max_iters: Maximum number of training iterations

    Returns:
        PyTorch learning rate scheduler

    Example:
        >>> lr_config = dict(
        ...     policy='CosineAnnealing',
        ...     warmup='linear',
        ...     warmup_iters=500,
        ...     warmup_ratio=1.0/3,
        ...     min_lr_ratio=1e-3,
        ... )
        >>> scheduler = build_scheduler(optimizer, lr_config, max_iters=35000)
    """
    policy = lr_config.get('policy', 'Constant')
    warmup = lr_config.get('warmup', None)

    if policy == 'CosineAnnealing':
        if warmup == 'linear':
            # CosineAnnealing with linear warmup
            warmup_iters = lr_config.get('warmup_iters', 500)
            warmup_ratio = lr_config.get('warmup_ratio', 1.0 / 3.0)
            min_lr_ratio = lr_config.get('min_lr_ratio', 1e-3)

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_iters=warmup_iters,
                max_iters=max_iters,
                warmup_ratio=warmup_ratio,
                min_lr_ratio=min_lr_ratio
            )
            print(f"  Using CosineAnnealing with linear warmup:")
            print(f"    - warmup_iters: {warmup_iters}")
            print(f"    - max_iters: {max_iters}")
            print(f"    - warmup_ratio: {warmup_ratio}")
            print(f"    - min_lr_ratio: {min_lr_ratio}")
        else:
            # CosineAnnealing without warmup
            min_lr_ratio = lr_config.get('min_lr_ratio', 1e-3)
            base_lrs = [group['lr'] for group in optimizer.param_groups]
            eta_mins = [lr * min_lr_ratio for lr in base_lrs]

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_iters,
                eta_min=min(eta_mins)  # Use minimum as eta_min
            )
            print(f"  Using CosineAnnealing without warmup:")
            print(f"    - T_max: {max_iters}")
            print(f"    - min_lr_ratio: {min_lr_ratio}")

    elif policy == 'Step':
        # StepLR scheduler
        step_size = lr_config.get('step', max_iters // 3)
        gamma = lr_config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        print(f"  Using StepLR:")
        print(f"    - step_size: {step_size}")
        print(f"    - gamma: {gamma}")

    elif policy == 'Constant' or policy is None:
        # Constant LR (no scheduling)
        scheduler = ConstantLR(optimizer)
        print(f"  Using constant learning rate (no scheduling)")

    else:
        raise ValueError(f"Unsupported LR policy: {policy}")

    return scheduler
