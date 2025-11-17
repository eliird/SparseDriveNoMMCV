#!/usr/bin/env python
"""
Diagnostic script to identify training issues.
Checks gradients, weight updates, data normalization, and loss computation.
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
print("DIAGNOSTIC: Training Issue Investigation")
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
    samples_per_gpu=2,  # Small batch for testing
    workers_per_gpu=0,  # No workers for simpler debugging
    num_gpus=1,
    dist=False,
    shuffle=True,
    seed=0
)
print(f"✓ Dataset: {len(train_dataset)} samples")
print(f"✓ DataLoader: {len(train_loader)} batches")

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

# Check 1: Input data range
print("\n" + "="*80)
print("CHECK 1: Input Data Normalization")
print("="*80)
img = batch['img']
print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image range: [{img.min().item():.3f}, {img.max().item():.3f}]")
print(f"Image mean: {img.mean().item():.3f}")
print(f"Image std: {img.std().item():.3f}")
print()
print("Expected (after ImageNet normalization):")
print("  Range: roughly [-2.0, 2.5]")
print("  Mean: roughly 0.0")
print("  Std: roughly 1.0")
if abs(img.mean().item()) > 0.5 or abs(img.std().item() - 1.0) > 0.5:
    print("⚠ WARNING: Image normalization looks wrong!")
else:
    print("✓ Image normalization looks OK")

# Check 2: GT data validity
print("\n" + "="*80)
print("CHECK 2: Ground Truth Data")
print("="*80)
if batch.get('gt_bboxes_3d') is not None:
    print(f"GT bboxes: {len(batch['gt_bboxes_3d'])} samples in batch")
    for i, boxes in enumerate(batch['gt_bboxes_3d'][:2]):  # First 2 samples
        if boxes is not None and len(boxes) > 0:
            print(f"  Sample {i}: {len(boxes)} objects, shape: {boxes.shape if hasattr(boxes, 'shape') else 'N/A'}")
        else:
            print(f"  Sample {i}: No objects")
else:
    print("⚠ WARNING: No gt_bboxes_3d in batch!")

if batch.get('gt_labels_3d') is not None:
    print(f"GT labels: {len(batch['gt_labels_3d'])} samples")
    for i, labels in enumerate(batch['gt_labels_3d'][:2]):
        if labels is not None and len(labels) > 0:
            print(f"  Sample {i}: {len(labels)} labels, unique values: {torch.unique(labels).cpu().tolist() if isinstance(labels, torch.Tensor) else 'N/A'}")
else:
    print("⚠ WARNING: No gt_labels_3d in batch!")

# Check 3: Forward pass and loss
print("\n" + "="*80)
print("CHECK 3: Forward Pass and Loss Computation")
print("="*80)
model.train()

# Prepare img_metas
img_metas = batch['img_metas']
if isinstance(img_metas, dict):
    img_metas = [img_metas]

with torch.cuda.amp.autocast(enabled=False):  # Disable FP16 for clearer debugging
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
        print("✓ Forward pass successful")
        print(f"\nLosses computed:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")

        total_loss = sum([v for k, v in losses.items() if 'loss' in k.lower() and isinstance(v, torch.Tensor)])
        print(f"\n  Total Loss: {total_loss.item():.4f}")

        # Check if losses are reasonable
        if total_loss.item() > 500:
            print("\n⚠ WARNING: Total loss is very high (>500)!")
            print("  This suggests the model might not be initialized properly")
            print("  or there's an issue with the loss computation.")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Check 4: Gradient flow
print("\n" + "="*80)
print("CHECK 4: Gradient Flow")
print("="*80)

# Backward pass
total_loss.backward()

# Check gradients
param_groups = [
    ("Backbone conv1", model.img_backbone.conv1.weight),
    ("Backbone layer1", list(model.img_backbone.layer1.parameters())[0]),
    ("Neck", list(model.img_neck.parameters())[0]),
    ("Head", list(model.head.parameters())[0]),
]

print("Gradient statistics:")
for name, param in param_groups:
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_max = param.grad.abs().max().item()
        print(f"  {name:20s}: norm={grad_norm:10.6f}, mean={grad_mean:10.6f}, max={grad_max:10.6f}")
        if grad_norm == 0:
            print(f"    ⚠ WARNING: Zero gradient!")
    else:
        print(f"  {name:20s}: NO GRADIENT!")

# Check 5: Weight update simulation
print("\n" + "="*80)
print("CHECK 5: Weight Update Simulation")
print("="*80)

# Save initial weight
initial_weight = model.img_backbone.conv1.weight.data.clone()

# Create simple optimizer and do one step
optimizer = torch.optim.AdamW([model.img_backbone.conv1.weight], lr=4e-4)
optimizer.step()

# Check if weight changed
weight_diff = (model.img_backbone.conv1.weight.data - initial_weight).abs().max().item()
print(f"Max weight change after optimizer step: {weight_diff:.8f}")
if weight_diff == 0:
    print("✗ CRITICAL: Weights did NOT change after optimizer.step()!")
    print("  This indicates optimizer is not updating weights.")
elif weight_diff < 1e-8:
    print("⚠ WARNING: Weight change is very small")
else:
    print(f"✓ Weights are being updated (max change: {weight_diff:.8f})")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80 + "\n")
