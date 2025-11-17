#!/usr/bin/env python
"""
Comprehensive training debugging script.

This script performs a mini training loop with extensive logging to identify
why the loss is not decreasing.

Checks:
1. Data loading - are batches different?
2. Model parameter updates - are weights changing?
3. Gradient flow - are gradients computed and non-zero?
4. Loss computation - are all loss components reasonable?
5. Learning rate - is it in a reasonable range?
6. Numerical stability - are there NaN/Inf values?
"""

import sys
import os
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath('.'))

from reimplementation.models import SparseDrive
from reimplementation.dataset import build_dataset, build_dataloader
from reimplementation.training import build_optimizer, build_scheduler


def load_config(config_path):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def check_tensor_stats(tensor, name="tensor"):
    """Check and print tensor statistics."""
    if tensor is None:
        return f"{name}: None"

    if isinstance(tensor, (list, tuple)):
        return f"{name}: list of {len(tensor)} items"

    if not isinstance(tensor, torch.Tensor):
        return f"{name}: {type(tensor)}"

    stats = {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'mean': tensor.float().mean().item() if tensor.numel() > 0 else 0,
        'std': tensor.float().std().item() if tensor.numel() > 1 else 0,
        'min': tensor.min().item() if tensor.numel() > 0 else 0,
        'max': tensor.max().item() if tensor.numel() > 0 else 0,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
    }
    return f"{name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}, nan={stats['has_nan']}, inf={stats['has_inf']}"


def check_gradients(model, verbose=False):
    """Check gradient flow through the model."""
    grad_stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'grad_norm': 0.0,
        'zero_grads': 0,
        'none_grads': 0,
        'nan_grads': 0,
        'inf_grads': 0,
        'layer_stats': defaultdict(lambda: {'count': 0, 'has_grad': 0, 'grad_norm': 0.0})
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        grad_stats['total_params'] += 1
        layer_name = name.split('.')[0]  # Get top-level module name

        if param.grad is None:
            grad_stats['none_grads'] += 1
            grad_stats['layer_stats'][layer_name]['count'] += 1
        else:
            grad_stats['params_with_grad'] += 1
            grad_stats['layer_stats'][layer_name]['count'] += 1
            grad_stats['layer_stats'][layer_name]['has_grad'] += 1

            grad_norm = param.grad.norm().item()
            grad_stats['grad_norm'] += grad_norm ** 2
            grad_stats['layer_stats'][layer_name]['grad_norm'] += grad_norm ** 2

            if torch.isnan(param.grad).any():
                grad_stats['nan_grads'] += 1
            if torch.isinf(param.grad).any():
                grad_stats['inf_grads'] += 1
            if grad_norm < 1e-10:
                grad_stats['zero_grads'] += 1

            if verbose and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"  ⚠️  {name}: grad_norm={grad_norm:.6f}, has_nan={torch.isnan(param.grad).any()}, has_inf={torch.isinf(param.grad).any()}")

    grad_stats['grad_norm'] = np.sqrt(grad_stats['grad_norm'])

    # Compute layer-wise grad norms
    for layer_name in grad_stats['layer_stats']:
        grad_stats['layer_stats'][layer_name]['grad_norm'] = np.sqrt(
            grad_stats['layer_stats'][layer_name]['grad_norm']
        )

    return grad_stats


def check_param_updates(model, old_params):
    """Check if model parameters have actually updated."""
    update_stats = {
        'total_params': 0,
        'updated_params': 0,
        'max_change': 0.0,
        'mean_change': 0.0,
        'unchanged_params': [],
    }

    changes = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in old_params:
            continue

        update_stats['total_params'] += 1
        old_param = old_params[name]

        # Compute change
        diff = (param.data - old_param).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        changes.append(mean_diff)
        update_stats['max_change'] = max(update_stats['max_change'], max_diff)

        if max_diff > 1e-8:
            update_stats['updated_params'] += 1
        else:
            update_stats['unchanged_params'].append(name)

    if changes:
        update_stats['mean_change'] = np.mean(changes)

    return update_stats


def debug_training():
    """Run debugging training loop."""

    print("="*80)
    print("TRAINING DEBUG - COMPREHENSIVE DIAGNOSTIC")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load config
    print("\n1. Loading config...")
    cfg = load_config('projects/configs/sparsedrive_small_stage1.py')
    print("   ✓ Config loaded")

    # Build model
    print("\n2. Building model...")
    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')
    model = SparseDrive(**model_cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model built: {total_params:,} total params, {trainable_params:,} trainable")

    # Build dataset
    print("\n3. Building dataset...")
    train_dataset = build_dataset(cfg.data['train'])
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=1,  # Small batch for debugging
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        seed=42,
        shuffle=True
    )
    print(f"   ✓ Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")

    # Build optimizer
    print("\n4. Building optimizer...")
    optimizer = build_optimizer(model, cfg.optimizer)
    print(f"   ✓ Optimizer: {type(optimizer).__name__}")
    print(f"     Base LR: {cfg.optimizer['lr']}")

    # Run mini training loop
    print("\n" + "="*80)
    print("5. RUNNING MINI TRAINING LOOP (10 iterations)")
    print("="*80)

    model.train()
    losses = []
    batch_hashes = []

    for iter_idx, batch in enumerate(train_loader):
        if iter_idx >= 10:
            break

        print(f"\n{'─'*80}")
        print(f"Iteration {iter_idx}")
        print(f"{'─'*80}")

        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], list):
                batch[key] = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in batch[key]
                ]

        # Check data variety
        if 'img' in batch:
            img = batch['img']
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, torch.Tensor):
                img_hash = hash(img.cpu().numpy().tobytes())
                batch_hashes.append(img_hash)
                print(f"\n📊 Batch data:")
                print(f"   Image shape: {img.shape}")
                print(f"   Image mean: {img.float().mean():.4f}, std: {img.float().std():.4f}")
                print(f"   Unique batches so far: {len(set(batch_hashes))}/{len(batch_hashes)}")

        # Check ground truth
        if 'gt_labels_3d' in batch:
            gt_labels = batch['gt_labels_3d']
            if isinstance(gt_labels, list) and len(gt_labels) > 0:
                print(f"   GT labels: {[len(l) if hasattr(l, '__len__') else 'N/A' for l in gt_labels]}")

        # Save old parameters for comparison
        old_params = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        print(f"\n⚡ Forward pass...")
        try:
            losses_dict = model(**batch)

            print(f"   Loss components:")
            total_loss = 0
            for key, value in losses_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_val = value.item() if value.numel() == 1 else value.mean().item()
                    print(f"     {key}: {loss_val:.4f}")
                    if 'loss' in key.lower():
                        total_loss += value if value.numel() == 1 else value.mean()

            losses.append(total_loss.item())
            print(f"   Total loss: {total_loss.item():.4f}")

            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"   ❌ ERROR: Loss is {'NaN' if torch.isnan(total_loss) else 'Inf'}!")

        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Backward pass
        print(f"\n⚡ Backward pass...")
        try:
            total_loss.backward()
            print(f"   ✓ Backward complete")
        except Exception as e:
            print(f"   ❌ Backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Check gradients
        print(f"\n📈 Gradient statistics:")
        grad_stats = check_gradients(model, verbose=False)
        print(f"   Params with gradients: {grad_stats['params_with_grad']}/{grad_stats['total_params']}")
        print(f"   Global grad norm: {grad_stats['grad_norm']:.6f}")
        print(f"   Zero gradients: {grad_stats['zero_grads']}")
        print(f"   None gradients: {grad_stats['none_grads']}")
        print(f"   NaN gradients: {grad_stats['nan_grads']}")
        print(f"   Inf gradients: {grad_stats['inf_grads']}")

        if grad_stats['nan_grads'] > 0 or grad_stats['inf_grads'] > 0:
            print(f"   ❌ ERROR: Found NaN or Inf in gradients!")
            check_gradients(model, verbose=True)

        if grad_stats['grad_norm'] < 1e-7:
            print(f"   ⚠️  WARNING: Gradients are extremely small! Model may not learn.")

        # Layer-wise gradient norms
        print(f"\n   Top layers by gradient norm:")
        sorted_layers = sorted(grad_stats['layer_stats'].items(),
                             key=lambda x: x[1]['grad_norm'], reverse=True)
        for layer_name, stats in sorted_layers[:5]:
            print(f"     {layer_name}: norm={stats['grad_norm']:.6f}, params={stats['count']}, with_grad={stats['has_grad']}")

        # Optimizer step
        print(f"\n⚡ Optimizer step...")
        lr = optimizer.param_groups[0]['lr']
        print(f"   Learning rate: {lr:.6e}")

        optimizer.step()
        print(f"   ✓ Step complete")

        # Check parameter updates
        print(f"\n🔄 Parameter update statistics:")
        update_stats = check_param_updates(model, old_params)
        print(f"   Updated params: {update_stats['updated_params']}/{update_stats['total_params']}")
        print(f"   Max change: {update_stats['max_change']:.6e}")
        print(f"   Mean change: {update_stats['mean_change']:.6e}")

        if update_stats['updated_params'] == 0:
            print(f"   ❌ ERROR: No parameters were updated!")
        elif update_stats['updated_params'] < update_stats['total_params'] * 0.5:
            print(f"   ⚠️  WARNING: Less than 50% of parameters were updated")
            print(f"   Unchanged: {update_stats['unchanged_params'][:5]}")  # Show first 5

        if update_stats['max_change'] < 1e-7:
            print(f"   ⚠️  WARNING: Parameter changes are extremely small!")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n📊 Loss progression:")
    for i, loss in enumerate(losses):
        change = "" if i == 0 else f" (Δ: {loss - losses[i-1]:+.4f})"
        print(f"   Iter {i}: {loss:.4f}{change}")

    if len(losses) > 1:
        initial_loss = losses[0]
        final_loss = losses[-1]
        change = final_loss - initial_loss
        percent_change = (change / initial_loss) * 100 if initial_loss != 0 else 0

        print(f"\n   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Change: {change:+.4f} ({percent_change:+.2f}%)")

        if abs(change) < 0.01:
            print(f"   ❌ PROBLEM: Loss is virtually constant!")
        elif change > 0:
            print(f"   ⚠️  WARNING: Loss increased!")
        else:
            print(f"   ✓ Loss decreased (expected behavior)")

    print(f"\n📊 Data variety:")
    unique_batches = len(set(batch_hashes))
    print(f"   Unique batches: {unique_batches}/{len(batch_hashes)}")
    if unique_batches == 1:
        print(f"   ❌ PROBLEM: All batches are identical!")
    elif unique_batches < len(batch_hashes):
        print(f"   ⚠️  WARNING: Some batches are duplicated")
    else:
        print(f"   ✓ All batches are unique")

    print(f"\n" + "="*80)


if __name__ == '__main__':
    debug_training()
