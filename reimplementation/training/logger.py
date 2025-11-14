"""
Logger for SparseDrive training.
Supports TensorBoard and console text logging.
"""

import os
import time
from typing import Dict, Any, Optional
import torch.distributed as dist


class Logger:
    """
    Training logger with TensorBoard and console output.

    Args:
        log_dir: Directory for TensorBoard logs
        log_interval: Interval (in iterations) for logging
        rank: Process rank in distributed training (default: 0)
    """

    def __init__(self, log_dir: str, log_interval: int = 50, rank: int = 0):
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.rank = rank

        # Only rank 0 creates TensorBoard writer
        self.writer = None
        if rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logging to {log_dir}")
            except ImportError:
                print("Warning: TensorBoard not available, skipping TensorBoard logging")

        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def log_training_step(
        self,
        iteration: int,
        epoch: int,
        losses: Dict[str, float],
        lr: float,
        batch_time: Optional[float] = None,
        max_iters: Optional[int] = None,
        iters_per_epoch: Optional[int] = None
    ) -> None:
        """
        Log training step.

        Args:
            iteration: Current iteration
            epoch: Current epoch
            losses: Dict of loss values
            lr: Current learning rate
            batch_time: Time for processing current batch (optional)
            max_iters: Maximum iterations (for progress tracking)
            iters_per_epoch: Iterations per epoch (for progress tracking)
        """
        # Only log on rank 0
        if self.rank != 0:
            return

        # Only log at specified intervals
        if iteration % self.log_interval != 0:
            return

        # TensorBoard logging
        if self.writer is not None:
            # Log learning rate
            self.writer.add_scalar('train/learning_rate', lr, iteration)

            # Log individual losses
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f'train/{loss_name}', loss_value, iteration)

            # Log total loss if available
            if 'total_loss' in losses:
                self.writer.add_scalar('train/total_loss', losses['total_loss'], iteration)

            # Log batch time
            if batch_time is not None:
                self.writer.add_scalar('train/batch_time', batch_time, iteration)

        # Console logging
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_per_iter = (current_time - self.last_log_time) / self.log_interval
        self.last_log_time = current_time

        # Calculate ETA
        eta_str = ""
        if max_iters is not None and iteration > 0:
            remaining_iters = max_iters - iteration
            eta_seconds = remaining_iters * time_per_iter
            eta_hours = eta_seconds / 3600
            eta_str = f", ETA: {eta_hours:.2f}h"

        # Calculate epoch progress
        epoch_progress_str = ""
        if iters_per_epoch is not None:
            iter_in_epoch = iteration % iters_per_epoch
            epoch_progress_pct = (iter_in_epoch / iters_per_epoch) * 100
            epoch_progress_str = f" ({iter_in_epoch}/{iters_per_epoch}, {epoch_progress_pct:.1f}%)"

        # Group losses by type for better readability
        det_losses = {k: v for k, v in losses.items() if 'det_loss' in k}
        map_losses = {k: v for k, v in losses.items() if 'map_loss' in k}
        other_losses = {k: v for k, v in losses.items() if 'det_loss' not in k and 'map_loss' not in k and k != 'total_loss'}
        total_loss = losses.get('total_loss', 0.0)

        # Print header
        print(f"\n{'='*100}")
        print(f"[Epoch {epoch}{epoch_progress_str}][Iter {iteration}/{max_iters or '?'}] "
              f"lr={lr:.6f}, time={time_per_iter:.3f}s/iter{eta_str}")
        print(f"{'='*100}")

        # Print total loss prominently
        print(f"Total Loss: {total_loss:.4f}")

        # Print detection losses (summarized)
        if det_losses:
            # Group by decoder layer
            layers = {}
            for k, v in det_losses.items():
                # Extract layer number (e.g., det_loss_cls_0 -> 0)
                parts = k.split('_')
                if parts[-1].isdigit():
                    layer = int(parts[-1])
                    loss_type = '_'.join(parts[:-1])
                    if layer not in layers:
                        layers[layer] = {}
                    layers[layer][loss_type] = v

            print(f"\nDetection Losses (6 layers):")
            for layer in sorted(layers.keys()):
                layer_losses = layers[layer]
                cls = layer_losses.get('det_loss_cls', 0)
                box = layer_losses.get('det_loss_box', 0)
                cns = layer_losses.get('det_loss_cns', 0)
                yns = layer_losses.get('det_loss_yns', 0)
                print(f"  Layer {layer}: cls={cls:7.4f}, box={box:6.4f}, cns={cns:6.4f}, yns={yns:6.4f}")

        # Print map losses (summarized)
        if map_losses:
            layers = {}
            for k, v in map_losses.items():
                parts = k.split('_')
                if parts[-1].isdigit():
                    layer = int(parts[-1])
                    loss_type = '_'.join(parts[:-1])
                    if layer not in layers:
                        layers[layer] = {}
                    layers[layer][loss_type] = v

            print(f"\nMap Losses (6 layers):")
            for layer in sorted(layers.keys()):
                layer_losses = layers[layer]
                cls = layer_losses.get('map_loss_cls', 0)
                line = layer_losses.get('map_loss_line', 0)
                print(f"  Layer {layer}: cls={cls:7.4f}, line={line:6.4f}")

        # Print other losses
        if other_losses:
            print(f"\nOther Losses:")
            for k, v in other_losses.items():
                print(f"  {k}: {v:.4f}")

        print(f"{'='*100}\n")

    def log_epoch(
        self,
        epoch: int,
        epoch_losses: Dict[str, float],
        epoch_time: float
    ) -> None:
        """
        Log epoch summary.

        Args:
            epoch: Epoch number
            epoch_losses: Average losses for the epoch
            epoch_time: Time taken for the epoch
        """
        if self.rank != 0:
            return

        # TensorBoard logging
        if self.writer is not None:
            for loss_name, loss_value in epoch_losses.items():
                self.writer.add_scalar(f'epoch/{loss_name}', loss_value, epoch)

        # Console logging
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()])
        log_msg = (
            f"\n{'='*80}\n"
            f"Epoch {epoch} Summary:\n"
            f"  Losses: {loss_str}\n"
            f"  Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)\n"
            f"{'='*80}\n"
        )
        print(log_msg)

    def log_checkpoint(self, epoch: int, checkpoint_path: str) -> None:
        """Log checkpoint save event."""
        if self.rank != 0:
            return

        print(f"[Epoch {epoch}] Checkpoint saved: {checkpoint_path}")

    def close(self) -> None:
        """Close the logger and cleanup resources."""
        if self.writer is not None:
            self.writer.close()


def get_dist_info():
    """
    Get distributed training info.

    Returns:
        Tuple of (rank, world_size)
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
