"""
Trainer class for SparseDrive training.
Main training loop with FP16, gradient clipping, distributed training support.
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from .logger import Logger, get_dist_info
from .checkpoint import save_checkpoint


class Trainer:
    """
    Main trainer class for SparseDrive.

    Features:
    - FP16 mixed precision training
    - Gradient clipping
    - Distributed training support
    - Automatic checkpoint saving
    - Loss logging with TensorBoard

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        logger: Logger instance
        work_dir: Working directory for checkpoints
        max_epochs: Maximum number of epochs
        grad_clip_config: Gradient clipping config dict
        fp16_config: FP16 training config dict
        checkpoint_interval: Save checkpoint every N epochs
        rank: Process rank in distributed training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        logger: Logger,
        work_dir: str,
        max_epochs: int = 100,
        grad_clip_config: Optional[Dict[str, Any]] = None,
        fp16_config: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 20,
        rank: int = 0,
        max_iters: Optional[int] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.logger = logger
        self.work_dir = work_dir
        self.max_epochs = max_epochs
        self.grad_clip_config = grad_clip_config or {}
        self.fp16_config = fp16_config or {}
        self.checkpoint_interval = checkpoint_interval
        self.rank = rank
        self.max_iters = max_iters
        self.iters_per_epoch = len(train_loader)

        # FP16 setup
        self.use_fp16 = len(self.fp16_config) > 0
        if self.use_fp16:
            init_scale = self.fp16_config.get('loss_scale', 512.0)
            self.scaler = GradScaler(init_scale=init_scale)
            if rank == 0:
                print(f"Using FP16 training with initial loss scale: {init_scale}")
        else:
            self.scaler = None

        # Gradient clipping setup
        self.max_norm = self.grad_clip_config.get('max_norm', None)
        self.norm_type = self.grad_clip_config.get('norm_type', 2)
        if self.max_norm is not None and rank == 0:
            print(f"Using gradient clipping: max_norm={self.max_norm}, norm_type={self.norm_type}")

        # Training state
        self.current_epoch = 0
        self.current_iter = 0

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of average losses for the epoch
        """
        self.model.train()
        epoch_start_time = time.time()

        # Accumulators for epoch statistics
        epoch_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()

            # Move batch to GPU if needed
            if torch.cuda.is_available():
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda()
                    elif isinstance(batch[key], list):
                        batch[key] = [
                            item.cuda() if isinstance(item, torch.Tensor) else item
                            for item in batch[key]
                        ]

            # Prepare img_metas (wrap dict in list if needed)
            img_metas = batch['img_metas']
            if isinstance(img_metas, dict):
                img_metas = [img_metas]

            # Forward pass
            with autocast(enabled=self.use_fp16):
                losses = self.model.forward_train(
                    img=batch['img'],
                    img_metas=img_metas,
                    gt_bboxes_3d=batch.get('gt_bboxes_3d'),
                    gt_labels_3d=batch.get('gt_labels_3d'),
                    gt_map_labels=batch.get('gt_map_labels'),
                    gt_map_pts=batch.get('gt_map_pts'),
                    timestamp=batch.get('timestamp'),
                    projection_mat=batch.get('projection_mat'),
                    image_wh=batch.get('image_wh'),
                    gt_depth=batch.get('gt_depth'),
                    focal=batch.get('focal'),
                )

            # Calculate total loss
            total_loss = sum([v for k, v in losses.items() if 'loss' in k.lower() and isinstance(v, torch.Tensor)])

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_fp16:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping
                if self.max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_norm,
                        norm_type=self.norm_type
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()

                # Gradient clipping
                if self.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_norm,
                        norm_type=self.norm_type
                    )

                # Optimizer step
                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Accumulate losses for epoch statistics
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    loss_val = v.item()
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += loss_val

            # Add total loss
            if 'total_loss' not in losses:
                epoch_losses['total_loss'] = epoch_losses.get('total_loss', 0.0) + total_loss.item()

            num_batches += 1

            # Logging
            batch_time = time.time() - batch_start_time
            current_lr = self.scheduler.get_last_lr()[0]

            # Prepare log losses (current batch)
            log_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            log_losses['total_loss'] = total_loss.item()

            self.logger.log_training_step(
                iteration=self.current_iter,
                epoch=epoch,
                losses=log_losses,
                lr=current_lr,
                batch_time=batch_time,
                max_iters=self.max_iters,
                iters_per_epoch=self.iters_per_epoch
            )

            self.current_iter += 1

        # Calculate epoch average losses
        avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        epoch_time = time.time() - epoch_start_time

        # Log epoch summary
        self.logger.log_epoch(epoch, avg_epoch_losses, epoch_time)

        return avg_epoch_losses

    def train(self, start_epoch: int = 0) -> None:
        """
        Run full training loop.

        Args:
            start_epoch: Starting epoch (for resuming training)
        """
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Checkpoint interval: {self.checkpoint_interval} epochs")
        print(f"{'='*80}\n")

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            epoch_losses = self.train_one_epoch(epoch)

            # Save checkpoint at intervals
            if (epoch + 1) % self.checkpoint_interval == 0 or (epoch + 1) == self.max_epochs:
                checkpoint_filename = f'epoch_{epoch+1}.pth'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    iteration=self.current_iter,
                    work_dir=self.work_dir,
                    filename=checkpoint_filename,
                    meta={'epoch_losses': epoch_losses}
                )
                self.logger.log_checkpoint(epoch + 1, checkpoint_filename)

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")

        # Close logger
        self.logger.close()
