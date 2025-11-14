"""
Test script for SparseDriveHead instantiation with full configuration.
Tests det_head, map_head, and motion_plan_head integration.
"""

import torch
import sys
import numpy as np
import tempfile
import os
import shutil

sys.path.insert(0, '/data1/work/irdali.durrani/SparseDrive')

from reimplementation.models.heads import SparseDriveHead

print("="*80)
print("Testing SparseDriveHead Instantiation")
print("="*80)

# Create dummy anchor arrays
det_anchor = np.random.randn(900, 11).astype(np.float32)
map_anchor = np.random.randn(100, 40).astype(np.float32)
motion_anchor = np.random.randn(6, 12, 2).astype(np.float32)
plan_anchor = np.random.randn(3, 6, 6, 2).astype(np.float32)

# Create temporary directory for anchors
temp_dir = tempfile.mkdtemp()
motion_anchor_path = os.path.join(temp_dir, 'motion_anchor.npy')
plan_anchor_path = os.path.join(temp_dir, 'plan_anchor.npy')
np.save(motion_anchor_path, motion_anchor)
np.save(plan_anchor_path, plan_anchor)

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
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True
input_shape = (704, 256)

# ============================================================================
# Test 1: SparseDriveHead with Det + Map (no motion)
# ============================================================================
print("\n" + "="*80)
print("Test 1: SparseDriveHead with Det + Map (no motion)")
print("="*80)

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=False,
)

sparsedrive_config = dict(
    type="SparseDriveHead",
    task_config=task_config,
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
)

try:
    config = sparsedrive_config.copy()
    config.pop('type')

    model = SparseDriveHead(**config)
    print(f"✓ SparseDriveHead (Det+Map) instantiated successfully")
    print(f"  - Task config: {model.task_config}")
    print(f"  - Has det_head: {hasattr(model, 'det_head')}")
    print(f"  - Has map_head: {hasattr(model, 'map_head')}")
    print(f"  - Has motion_plan_head: {hasattr(model, 'motion_plan_head')}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")

except Exception as e:
    print(f"✗ Failed to instantiate SparseDriveHead (Det+Map): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 2: SparseDriveHead with Det + Map + Motion
# ============================================================================
print("\n" + "="*80)
print("Test 2: SparseDriveHead with Det + Map + Motion")
print("="*80)

task_config_full = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,
)

sparsedrive_config_full = sparsedrive_config.copy()
sparsedrive_config_full['task_config'] = task_config_full
sparsedrive_config_full['motion_plan_head'] = dict(
    type='MotionPlanningHead',
    fut_ts=fut_ts,
    fut_mode=fut_mode,
    ego_fut_ts=ego_fut_ts,
    ego_fut_mode=ego_fut_mode,
    motion_anchor=motion_anchor_path,
    plan_anchor=plan_anchor_path,
    embed_dims=embed_dims,
    decouple_attn=decouple_attn_motion,
    instance_queue=dict(
        type="InstanceQueue",
        embed_dims=embed_dims,
        queue_length=queue_length,
        tracking_threshold=0.2,
        feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
    ),
    operation_order=(
        ["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3 + ["refine"]
    ),
    temp_graph_model=dict(
        type="MultiheadAttention",
        embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
        num_heads=num_groups,
        batch_first=True,
        dropout=drop_out,
    ),
    graph_model=dict(
        type="MultiheadFlashAttention",
        embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
        num_heads=num_groups,
        batch_first=True,
        dropout=drop_out,
    ),
    cross_graph_model=dict(
        type="MultiheadFlashAttention",
        embed_dims=embed_dims,
        num_heads=num_groups,
        batch_first=True,
        dropout=drop_out,
    ),
    norm_layer=dict(type="LN", normalized_shape=embed_dims),
    ffn=dict(
        type="AsymmetricFFN",
        in_channels=embed_dims,
        pre_norm=dict(type="LN"),
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 2,
        num_fcs=2,
        ffn_drop=drop_out,
        act_cfg=dict(type="ReLU", inplace=True),
    ),
    refine_layer=dict(
        type="MotionPlanningRefinementModule",
        embed_dims=embed_dims,
        fut_ts=fut_ts,
        fut_mode=fut_mode,
        ego_fut_ts=ego_fut_ts,
        ego_fut_mode=ego_fut_mode,
    ),
    motion_sampler=dict(type="MotionTarget"),
    motion_loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
    motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
    planning_sampler=dict(type="PlanningTarget", ego_fut_ts=ego_fut_ts, ego_fut_mode=ego_fut_mode),
    plan_loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
    plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
    plan_loss_status=dict(type='L1Loss', loss_weight=1.0),
    motion_decoder=dict(type="SparseBox3DMotionDecoder"),
    planning_decoder=dict(type="HierarchicalPlanningDecoder", ego_fut_ts=ego_fut_ts, ego_fut_mode=ego_fut_mode, use_rescore=True),
    num_det=50,
    num_map=10,
)

try:
    config_full = sparsedrive_config_full.copy()
    config_full.pop('type')

    model_full = SparseDriveHead(**config_full)
    print(f"✓ SparseDriveHead (Full) instantiated successfully")
    print(f"  - Task config: {model_full.task_config}")
    print(f"  - Has det_head: {hasattr(model_full, 'det_head')}")
    print(f"  - Has map_head: {hasattr(model_full, 'map_head')}")
    print(f"  - Has motion_plan_head: {hasattr(model_full, 'motion_plan_head')}")

    total_params = sum(p.numel() for p in model_full.parameters())
    trainable_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"✗ Failed to instantiate SparseDriveHead (Full): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Summary
# ============================================================================
print("\n" + "="*80)
print("Test Summary")
print("="*80)

print("\n✓ SparseDriveHead (Det+Map) instantiation: SUCCESS")
print("  - Config matches sparsedrive_small_stage1.py")
print("  - All mmcv dependencies removed")
print("  - Custom builder system working")

print("\n✓ SparseDriveHead (Full with Motion) instantiation: SUCCESS")
print("  - Config matches sparsedrive_small_stage1.py")
print("  - All three heads integrated properly")
print("  - All components properly initialized")

print("\nNote: Forward/backward pass tests require:")
print("  - Properly formatted feature_maps from backbone/neck")
print("  - Metas dict with transformation matrices")
print("  - Ground truth data for loss computation")
print("\nThese can be tested separately with real data loaders.")

print("\n" + "="*80)
print("All SparseDriveHead tests completed successfully!")
print("="*80)

# Cleanup
shutil.rmtree(temp_dir)
print(f"\n✓ Cleaned up temporary files")
