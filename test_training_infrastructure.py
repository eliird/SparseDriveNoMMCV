#!/usr/bin/env python
"""
Test script to verify training infrastructure works correctly.
This runs a quick sanity check without full training.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from reimplementation.training import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    Logger
)
from reimplementation.models import SparseDrive

print("=" * 80)
print("Testing Training Infrastructure")
print("=" * 80)

# Test 1: Build optimizer
print("\n1. Testing optimizer builder...")
try:
    # Create a simple model
    model = SparseDrive(
        use_grid_mask=False,
        use_deformable_func=False,
        img_backbone=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            frozen_stages=-1,
            bn_eval=False,
            bn_frozen=False,
            style="pytorch",
            with_cp=False,
            out_indices=(0, 1, 2, 3),
        ),
        img_neck=dict(
            type="FPN",
            num_outs=4,
            start_level=0,
            out_channels=256,
            add_extra_convs="on_output",
            relu_before_extra_convs=True,
            in_channels=[256, 512, 1024, 2048],
        ),
        depth_branch=None,  # Skip for quick test
        head=dict(
            type="SparseDriveHead",
            task_config=dict(with_det=True, with_map=False, with_motion_plan=False),
            det_head=dict(
                type="Sparse4DHead",
                cls_threshold_to_reg=0.05,
                decouple_attn=True,
                instance_bank=dict(
                    type="InstanceBank",
                    num_anchor=100,  # Reduced for testing
                    embed_dims=256,
                    anchor="data/kmeans/kmeans_det_900.npy",
                    anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                    num_temp_instances=-1,
                ),
                anchor_encoder=dict(
                    type="SparseBox3DEncoder",
                    vel_dims=3,
                    embed_dims=[128, 32, 32, 64],
                    mode="cat",
                    output_fc=False,
                    in_loops=1,
                    out_loops=4,
                ),
                num_single_frame_decoder=1,
                operation_order=["gnn", "norm", "deformable", "ffn", "norm", "refine"],
                graph_model=dict(
                    type="MultiheadFlashAttention",
                    embed_dims=512,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                norm_layer=dict(type="LN", normalized_shape=256),
                ffn=dict(
                    type="AsymmetricFFN",
                    in_channels=512,
                    pre_norm=dict(type="LN"),
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
                deformable_model=dict(
                    type="DeformableFeatureAggregation",
                    embed_dims=256,
                    num_groups=8,
                    num_levels=4,
                    num_cams=6,
                    attn_drop=0.15,
                    use_deformable_func=False,
                    use_camera_embed=True,
                    residual_mode="cat",
                    kps_generator=dict(
                        type="SparseBox3DKeyPointsGenerator",
                        num_learnable_pts=6,
                        fix_scale=[
                            [0, 0, 0],
                            [0.45, 0, 0],
                            [-0.45, 0, 0],
                            [0, 0.45, 0],
                            [0, -0.45, 0],
                            [0, 0, 0.45],
                            [0, 0, -0.45],
                        ],
                    ),
                ),
                refine_layer=dict(
                    type="SparseBox3DRefinementModule",
                    embed_dims=256,
                    num_cls=10,
                    refine_yaw=True,
                ),
                sampler=dict(type="SparseBox3DTarget"),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                loss_reg=dict(
                    type="SparseBox3DLoss",
                    loss_box=dict(type="L1Loss", loss_weight=0.25),
                    loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                    loss_yawness=dict(type="GaussianFocalLoss"),
                ),
                decoder=dict(type="SparseBox3DDecoder"),
            ),
        ),
    )

    optimizer_config = dict(
        type='AdamW',
        lr=4e-4,
        weight_decay=0.001,
        paramwise_cfg=dict(
            custom_keys={
                'img_backbone': dict(lr_mult=0.5),
            }
        )
    )

    optimizer = build_optimizer(model, optimizer_config)
    print("  ✓ Optimizer built successfully")
    print(f"    Number of parameter groups: {len(optimizer.param_groups)}")

except Exception as e:
    print(f"  ✗ Failed to build optimizer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Build scheduler
print("\n2. Testing scheduler builder...")
try:
    lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        min_lr_ratio=1e-3,
    )

    scheduler = build_scheduler(optimizer, lr_config, max_iters=10000)
    print("  ✓ Scheduler built successfully")

    # Test a few scheduler steps
    initial_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    after_step_lr = scheduler.get_last_lr()[0]
    print(f"    Initial LR: {initial_lr:.6f}")
    print(f"    After 1 step: {after_step_lr:.6f}")

except Exception as e:
    print(f"  ✗ Failed to build scheduler: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Checkpoint saving/loading
print("\n3. Testing checkpoint save/load...")
try:
    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        iteration=1000,
        work_dir=temp_dir,
        filename='test_checkpoint.pth',
        meta={'test': 'metadata'}
    )
    print("  ✓ Checkpoint saved successfully")

    # Load checkpoint
    checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
    info = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location='cpu'
    )
    print("  ✓ Checkpoint loaded successfully")
    print(f"    Resumed from epoch {info['epoch']}, iteration {info['iteration']}")

    # Cleanup
    shutil.rmtree(temp_dir)

except Exception as e:
    print(f"  ✗ Failed checkpoint save/load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Logger
print("\n4. Testing logger...")
try:
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    logger = Logger(log_dir=temp_dir, log_interval=10, rank=0)

    # Test logging
    losses = {'loss_cls': 1.5, 'loss_box': 0.8, 'total_loss': 2.3}
    logger.log_training_step(
        iteration=10,
        epoch=1,
        losses=losses,
        lr=0.0004,
        batch_time=0.5
    )
    print("  ✓ Logger working successfully")

    logger.close()
    shutil.rmtree(temp_dir)

except Exception as e:
    print(f"  ✗ Failed to test logger: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All Training Infrastructure Tests Passed!")
print("=" * 80)
print("\nYou can now use train.py to start training:")
print("  python train.py --config projects/configs/sparsedrive_small_stage1.py --work-dir work_dirs/test")
print("\nOr for multi-GPU training:")
print("  python -m torch.distributed.launch --nproc_per_node=8 train.py \\")
print("    --config projects/configs/sparsedrive_small_stage1.py \\")
print("    --work-dir work_dirs/sparsedrive_small \\")
print("    --launcher pytorch")
print("=" * 80)
