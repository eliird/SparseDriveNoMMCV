#!/usr/bin/env python
"""
Test SparseDrive model with actual dataloader.
Tests forward and backward passes with real data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np

from reimplementation.models import SparseDrive
from reimplementation.dataset import build_dataset, build_dataloader

# Import pipeline transforms to ensure they're registered
from reimplementation.dataset import pipelines
from reimplementation.dataset import vectorize

print("=" * 80)
print("Testing SparseDrive Model with Real Data")
print("=" * 80)

# ============================================================================
# Configuration from sparsedrive_small_stage1.py
# ============================================================================

class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
map_class_names = ['ped_crossing', 'divider', 'boundary']
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = False  # Set to False to avoid needing compiled ops
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True
input_shape = (704, 256)
batch_size = 1  # Use batch size 1 for testing

# Load anchors
det_anchor = np.load('data/kmeans/kmeans_det_900.npy').astype(np.float32)
map_anchor = np.load('data/kmeans/kmeans_map_100.npy').astype(np.float32)

# ============================================================================
# Dataset Configuration
# ============================================================================

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_map_labels',
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
anno_root = 'data/infos/'

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

train_data_config = dict(
    **data_basic_config,
    ann_file=anno_root + "nuscenes_infos_train.pkl",
    pipeline=train_pipeline,
    test_mode=False,
    data_aug_conf=data_aug_conf,
)

# ============================================================================
# Model Configuration (Stage 1: Det + Map)
# ============================================================================

task_config_stage1 = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=False,
)

model_config = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        bn_eval=False,  # Use bn_eval instead of norm_eval
        bn_frozen=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        # Note: norm_cfg and pretrained are not supported in this ResNet implementation
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseDriveHead",
        task_config=task_config_stage1,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=det_anchor,
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder +
                ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"] * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ) if temporal else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                        [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor=map_anchor,
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder_map +
                ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"] * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ) if temporal_map else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
        ),
    ),
)

# ============================================================================
# Test 1: Build Dataset and DataLoader
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Build Dataset and DataLoader")
print("=" * 80)

try:
    # Build dataset
    train_dataset = build_dataset(train_data_config)
    print(f"✓ Train dataset built successfully")
    print(f"  - Number of samples: {len(train_dataset)}")

    # Build dataloader
    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,  # Use 0 for debugging
        shuffle=False,
        dist=False
    )
    print(f"✓ DataLoader built successfully")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {len(train_dataloader)}")

except Exception as e:
    print(f"✗ Failed to build dataset/dataloader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 2: Build Model
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Build SparseDrive Model")
print("=" * 80)

try:
    config = model_config.copy()
    config.pop('type')

    model = SparseDrive(**config)
    model.train()  # Set to training mode

    # Move model to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Model moved to CUDA")
    else:
        print(f"⚠ CUDA not available, running on CPU")

    print(f"✓ SparseDrive model instantiated successfully")
    print(f"  - Has backbone: {hasattr(model, 'img_backbone')}")
    print(f"  - Has neck: {hasattr(model, 'img_neck')}")
    print(f"  - Has head: {hasattr(model, 'head')}")
    print(f"  - Has depth_branch: {model.depth_branch is not None}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Statistics:")
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"✗ Failed to build model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Forward Pass with Real Data
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Forward Pass with Real Data")
print("=" * 80)

try:
    # Get one batch of data
    print("Loading one batch of data...")
    batch = next(iter(train_dataloader))

    print(f"✓ Batch loaded successfully")
    print(f"\n  Batch contents:")
    for key in batch.keys():
        if hasattr(batch[key], 'shape'):
            print(f"    - {key}: shape {batch[key].shape}, dtype {batch[key].dtype}")
        elif isinstance(batch[key], list):
            if len(batch[key]) > 0 and hasattr(batch[key][0], 'shape'):
                print(f"    - {key}: list of {len(batch[key])} items")
            else:
                print(f"    - {key}: list of length {len(batch[key])}")
        else:
            print(f"    - {key}: type {type(batch[key])}")

    # Move batch to CUDA if available
    if torch.cuda.is_available():
        print(f"\n  Moving batch to CUDA...")
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
            elif isinstance(batch[key], list):
                batch[key] = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch[key]]
        print(f"  ✓ Batch moved to CUDA")

    # Forward pass
    print(f"\nRunning forward pass...")

    # Debug: Check img_metas structure
    print(f"\n  Debug: Checking img_metas structure...")
    img_metas = batch['img_metas']
    print(f"    - Type: {type(img_metas)}")
    if isinstance(img_metas, dict):
        print(f"    - Keys: {list(img_metas.keys())}")
        print(f"    - It's a dict, needs to be wrapped in a list for batch processing")
        img_metas = [img_metas]
    elif isinstance(img_metas, list):
        print(f"    - Length: {len(img_metas)}")
        if len(img_metas) > 0:
            print(f"    - First element type: {type(img_metas[0])}")
            if isinstance(img_metas[0], dict):
                print(f"    - First element keys: {list(img_metas[0].keys())}")

    # Prepare inputs for model.forward_train
    # The model expects: forward_train(img, img_metas, **kwargs)
    outputs = model.forward_train(
        img=batch['img'],
        img_metas=img_metas,  # Use the processed img_metas (wrapped in list if needed)
        gt_bboxes_3d=batch['gt_bboxes_3d'],
        gt_labels_3d=batch['gt_labels_3d'],
        gt_map_labels=batch['gt_map_labels'],
        gt_map_pts=batch['gt_map_pts'],
        timestamp=batch['timestamp'],
        projection_mat=batch['projection_mat'],
        image_wh=batch['image_wh'],
        gt_depth=batch.get('gt_depth'),
        focal=batch.get('focal'),
    )

    print(f"✓ Forward pass completed successfully")
    print(f"\n  Outputs:")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"    - {key}: {type(value)} = {value}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Backward Pass
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: Backward Pass")
print("=" * 80)

try:
    # Calculate total loss
    if isinstance(outputs, dict):
        # Sum all loss values
        total_loss = sum([v for k, v in outputs.items() if 'loss' in k.lower() and isinstance(v, torch.Tensor)])
        print(f"✓ Total loss calculated: {total_loss.item():.4f}")

        # Individual losses
        print(f"\n  Individual losses:")
        for key, value in outputs.items():
            if 'loss' in key.lower() and isinstance(value, torch.Tensor):
                print(f"    - {key}: {value.item():.4f}")

        # Backward pass
        print(f"\nRunning backward pass...")
        total_loss.backward()

        print(f"✓ Backward pass completed successfully")

        # Check gradients
        has_grad = False
        no_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                else:
                    no_grad = True

        print(f"\n  Gradient check:")
        print(f"    - Parameters with gradients: {'Yes' if has_grad else 'No'}")
        print(f"    - Parameters without gradients: {'Yes' if no_grad else 'No'}")

        if has_grad:
            # Show some gradient stats
            grad_norms = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norms.append(param.grad.norm().item())

            if grad_norms:
                print(f"    - Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")

except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test Summary
# ============================================================================
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

print("\n✓ ALL TESTS PASSED!")
print("  1. Dataset and DataLoader built successfully")
print("  2. SparseDrive model with ResNet50 + FPN instantiated")
print("  3. Forward pass with real data completed")
print("  4. Backward pass and gradient computation succeeded")

print("\n✓ Model is ready for training!")
print("  - All components working correctly")
print("  - Data pipeline functioning properly")
print("  - Forward and backward passes validated")

print("\n" + "=" * 80)
print("SparseDrive Model + Data Integration Test Complete!")
print("=" * 80)
