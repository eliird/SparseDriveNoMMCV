#!/usr/bin/env python
"""
Diagnostic script to check data loading issues.

This script will:
1. Load a few batches and check if data is actually different
2. Verify ground truth labels are present and valid
3. Check if augmentation is working
4. Verify images are being loaded correctly
"""

import sys
import os
import importlib.util
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))

from reimplementation.dataset import build_dataset, build_dataloader


def load_config(config_path):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def check_data_loading():
    """Check if data loading is working correctly."""

    print("="*80)
    print("DATA LOADING DIAGNOSTIC")
    print("="*80)

    # Load config
    cfg = load_config('projects/configs/sparsedrive_small_stage1.py')

    # Build dataset
    print("\n1. Building dataset...")
    train_dataset = build_dataset(cfg.data['train'])
    print(f"   ✓ Dataset size: {len(train_dataset)}")

    # Build dataloader with small batch size for testing
    print("\n2. Building dataloader...")
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,  # No multiprocessing for easier debugging
        num_gpus=1,
        dist=False,
        seed=0,
        shuffle=True
    )
    print(f"   ✓ DataLoader created: {len(train_loader)} batches")

    # Load a few batches
    print("\n3. Checking first few batches...")
    print("-"*80)

    batch_stats = []
    num_batches_to_check = 5

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_check:
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  Keys in batch: {list(batch.keys())}")

        # Check images
        if 'img' in batch:
            imgs = batch['img']
            if isinstance(imgs, list):
                imgs = imgs[0]  # Take first timestep if temporal
            if isinstance(imgs, torch.Tensor):
                print(f"  Images shape: {imgs.shape}")
                print(f"  Images dtype: {imgs.dtype}")
                print(f"  Images mean: {imgs.float().mean():.4f}")
                print(f"  Images std: {imgs.float().std():.4f}")
                print(f"  Images min/max: {imgs.min():.4f} / {imgs.max():.4f}")

                # Store hash to check if batches are different
                img_hash = hash(imgs.cpu().numpy().tobytes())
                batch_stats.append({
                    'batch_idx': batch_idx,
                    'img_hash': img_hash,
                    'img_mean': imgs.float().mean().item(),
                })
            else:
                print(f"  Images type: {type(imgs)}")

        # Check ground truth bounding boxes
        if 'gt_bboxes_3d' in batch:
            gt_bboxes = batch['gt_bboxes_3d']
            if isinstance(gt_bboxes, list):
                print(f"  GT bboxes: {len(gt_bboxes)} samples in batch")
                for i, boxes in enumerate(gt_bboxes[:2]):  # Show first 2 samples
                    if isinstance(boxes, torch.Tensor):
                        print(f"    Sample {i}: {boxes.shape[0]} boxes, shape={boxes.shape}")
                    else:
                        print(f"    Sample {i}: {type(boxes)}")
            else:
                print(f"  GT bboxes shape: {gt_bboxes.shape if hasattr(gt_bboxes, 'shape') else type(gt_bboxes)}")

        # Check ground truth labels
        if 'gt_labels_3d' in batch:
            gt_labels = batch['gt_labels_3d']
            if isinstance(gt_labels, list):
                print(f"  GT labels: {len(gt_labels)} samples")
                for i, labels in enumerate(gt_labels[:2]):
                    if isinstance(labels, torch.Tensor):
                        unique_labels = torch.unique(labels)
                        print(f"    Sample {i}: {len(labels)} labels, unique: {unique_labels.tolist()}")
                    else:
                        print(f"    Sample {i}: {type(labels)}")
            else:
                print(f"  GT labels: {type(gt_labels)}")

        # Check map ground truth
        if 'gt_bboxes' in batch:
            gt_map = batch['gt_bboxes']
            if isinstance(gt_map, list):
                print(f"  GT map elements: {len(gt_map)} samples")
            else:
                print(f"  GT map: {type(gt_map)}")

    # Check if batches are actually different
    print("\n" + "="*80)
    print("4. Checking if batches are different (data variety)...")
    print("-"*80)

    if len(batch_stats) > 1:
        unique_hashes = len(set(s['img_hash'] for s in batch_stats))
        print(f"  Number of unique image batches: {unique_hashes}/{len(batch_stats)}")

        if unique_hashes == 1:
            print("  ❌ WARNING: All batches have identical images! Data may not be loading correctly.")
        elif unique_hashes < len(batch_stats) * 0.8:
            print(f"  ⚠️  WARNING: Only {unique_hashes} unique batches out of {len(batch_stats)}. Suspiciously low variety.")
        else:
            print("  ✓ Batches appear to have different data")

        # Check if image statistics vary (suggests augmentation is working)
        means = [s['img_mean'] for s in batch_stats]
        mean_std = np.std(means)
        print(f"\n  Image mean variation across batches:")
        print(f"    Means: {[f'{m:.4f}' for m in means]}")
        print(f"    Std dev: {mean_std:.6f}")

        if mean_std < 0.001:
            print("  ⚠️  WARNING: Very low variation in image statistics. Augmentation may not be working.")
        else:
            print("  ✓ Image statistics vary across batches (augmentation likely working)")

    # Sample one item directly from dataset to check raw data
    print("\n" + "="*80)
    print("5. Checking raw dataset sample (bypassing DataLoader)...")
    print("-"*80)

    try:
        sample = train_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")

        if 'img' in sample:
            img = sample['img']
            if isinstance(img, torch.Tensor):
                print(f"  Raw image shape: {img.shape}")
                print(f"  Raw image mean: {img.float().mean():.4f}")
            elif isinstance(img, list):
                print(f"  Raw image is a list with {len(img)} elements")
                if len(img) > 0 and isinstance(img[0], torch.Tensor):
                    print(f"    First element shape: {img[0].shape}")

        if 'gt_bboxes_3d' in sample:
            boxes = sample['gt_bboxes_3d']
            print(f"  Raw gt_bboxes_3d: {type(boxes)}, shape={boxes.shape if hasattr(boxes, 'shape') else 'N/A'}")

        if 'gt_labels_3d' in sample:
            labels = sample['gt_labels_3d']
            print(f"  Raw gt_labels_3d: {type(labels)}, len={len(labels) if hasattr(labels, '__len__') else 'N/A'}")
            if isinstance(labels, torch.Tensor):
                print(f"    Unique labels: {torch.unique(labels).tolist()}")
                print(f"    Label range: {labels.min().item()} to {labels.max().item()}")

    except Exception as e:
        print(f"  ❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()

    # Check if labels contain valid class indices
    print("\n" + "="*80)
    print("6. Label validation...")
    print("-"*80)

    num_classes = len(cfg.class_names)
    print(f"  Number of classes in config: {num_classes}")
    print(f"  Classes: {cfg.class_names}")

    # Sample a few more items to check label distribution
    print("\n  Checking label distribution across 20 samples...")
    all_labels = []
    for i in range(min(20, len(train_dataset))):
        try:
            sample = train_dataset[i]
            if 'gt_labels_3d' in sample:
                labels = sample['gt_labels_3d']
                if isinstance(labels, torch.Tensor):
                    all_labels.extend(labels.cpu().numpy().tolist())
        except Exception as e:
            print(f"  Warning: Could not load sample {i}: {e}")

    if all_labels:
        unique_labels = np.unique(all_labels)
        print(f"  Unique labels found: {unique_labels}")
        print(f"  Label counts: {np.bincount(np.array(all_labels).astype(int))}")

        # Check if any labels are out of range
        invalid_labels = [l for l in all_labels if l < 0 or l >= num_classes]
        if invalid_labels:
            print(f"  ❌ ERROR: Found invalid labels: {set(invalid_labels)}")
            print(f"     Valid range is 0 to {num_classes-1}")
        else:
            print(f"  ✓ All labels are valid (in range 0-{num_classes-1})")
    else:
        print("  ⚠️  WARNING: No labels found in sampled data")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    check_data_loading()
