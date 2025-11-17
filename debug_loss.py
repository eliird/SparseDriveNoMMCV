#!/usr/bin/env python
"""
Debug script to understand why losses are not decreasing.
Checks target distribution, loss values, and learning dynamics.
"""
import sys
import os
import torch
import importlib.util
sys.path.insert(0, os.path.abspath('.'))

from reimplementation.models import SparseDrive
from reimplementation.dataset import build_dataset, build_dataloader

# Load config
config_path = "projects/configs/sparsedrive_small_stage1.py"
spec = importlib.util.spec_from_file_location("config", config_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

print("\n" + "="*80)
print("DEBUG: Understanding Loss Computation")
print("="*80 + "\n")

# Build model
print("Building model...")
model_cfg = cfg.model.copy()
model_type = model_cfg.pop('type')
model = SparseDrive(**model_cfg)
model = model.cuda()
print(f"✓ Model built successfully")

# Build dataset and dataloader
print("\nBuilding dataset...")
train_dataset = build_dataset(cfg.data['train'])
train_loader = build_dataloader(
    train_dataset,
    samples_per_gpu=2,
    workers_per_gpu=0,
    num_gpus=1,
    dist=False,
    shuffle=True,
    seed=0
)
print(f"✓ Dataset: {len(train_dataset)} samples")

# Get one batch
print("\nLoading one batch...")
batch = next(iter(train_loader))

# Move to GPU
for key in batch.keys():
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].cuda()
    elif isinstance(batch[key], list):
        batch[key] = [
            item.cuda() if isinstance(item, torch.Tensor) else item
            for item in batch[key]
        ]

print("✓ Batch loaded and moved to GPU")

# Check GT labels distribution
print("\n" + "="*80)
print("CHECK 1: Ground Truth Labels Distribution")
print("="*80)
if batch.get('gt_labels_3d') is not None:
    all_labels = []
    for i, labels in enumerate(batch['gt_labels_3d']):
        if labels is not None and len(labels) > 0:
            all_labels.extend(labels.cpu().tolist() if isinstance(labels, torch.Tensor) else labels)
            print(f"Sample {i}: {len(labels)} objects, labels: {labels.cpu().tolist() if isinstance(labels, torch.Tensor) else labels}")

    if all_labels:
        import numpy as np
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nOverall label distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {int(label)}: {count} objects ({100*count/len(all_labels):.1f}%)")
    else:
        print("⚠ WARNING: No labels found in batch!")
else:
    print("⚠ WARNING: No gt_labels_3d in batch!")

# Forward pass with detailed loss analysis
print("\n" + "="*80)
print("CHECK 2: Forward Pass with Loss Analysis")
print("="*80)
model.train()

# Prepare img_metas
img_metas = batch['img_metas']
if isinstance(img_metas, dict):
    img_metas = [img_metas]

# Monkey patch the focal loss to add debug logging
original_forward = model.head.loss_cls.forward

def debug_focal_loss(pred, target, weight=None, avg_factor=None, reduction_override=None):
    """Wrapper to debug focal loss computation."""
    print(f"\n  --- Focal Loss Debug ---")
    print(f"  pred shape: {pred.shape}")
    print(f"  target shape: {target.shape}")
    print(f"  target dtype: {target.dtype}")
    print(f"  target range: [{target.min().item()}, {target.max().item()}]")

    # Count positives vs background
    num_classes = pred.size(1)
    num_positives = (target < num_classes).sum().item()
    num_background = (target >= num_classes).sum().item()
    print(f"  num_classes: {num_classes}")
    print(f"  num_positives: {num_positives} ({100*num_positives/len(target):.1f}%)")
    print(f"  num_background: {num_background} ({100*num_background/len(target):.1f}%)")

    if weight is not None:
        print(f"  weight shape: {weight.shape}")
        print(f"  weight range: [{weight.min().item():.4f}, {weight.max().item():.4f}]")

    print(f"  avg_factor: {avg_factor}")

    # Call original
    result = original_forward(pred, target, weight, avg_factor, reduction_override)
    print(f"  loss: {result.item():.4f}")

    return result

model.head.loss_cls.forward = debug_focal_loss

try:
    losses = model.forward_train(
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
    print("\n✓ Forward pass successful")
    print(f"\nFinal Losses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")

    total_loss = sum([v for k, v in losses.items() if 'loss' in k.lower() and isinstance(v, torch.Tensor)])
    print(f"\n  Total Loss: {total_loss.item():.4f}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80 + "\n")
