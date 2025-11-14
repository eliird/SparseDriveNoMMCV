#!/usr/bin/env python
"""
Test script for NuScenes3DDataset with SparseDrive config.

This script tests the dataset loading and pipeline processing
without requiring the full mmdetection/mmcv training framework.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from reimplementation.dataset.nuscenes_3d_dataset import NuScenes3DDataset

# Import pipeline transforms to ensure they're registered
from reimplementation.dataset import pipelines
from reimplementation.dataset import vectorize

# Config parameters from sparsedrive_small_stage1.py
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]

input_shape = (704, 256)
roi_size = (30, 60)
num_sample = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

file_client_args = dict(backend="disk")

# Training pipeline from config
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

# Test pipeline (simpler)
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'ego_status',
            'gt_ego_fut_cmd',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
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


def test_dataset_init():
    """Test dataset initialization."""
    print("=" * 80)
    print("TEST 1: Dataset Initialization")
    print("=" * 80)

    try:
        dataset = NuScenes3DDataset(
            ann_file='data/infos/nuscenes_infos_train.pkl',  # Using train data since val is empty
            pipeline=test_pipeline,
            data_root='data/nuscenes/',
            classes=class_names,
            map_classes=map_class_names,
            modality=input_modality,
            test_mode=False,  # Use train mode
            version='v1.0-trainval',
            data_aug_conf=data_aug_conf,
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Classes: {dataset.CLASSES}")
        print(f"  - Map classes: {dataset.MAP_CLASSES}")
        print(f"  - Version: {dataset.version}")
        return dataset
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_sample(dataset, idx=0):
    """Test loading a single sample."""
    print("\n" + "=" * 80)
    print(f"TEST 2: Loading Sample {idx}")
    print("=" * 80)

    try:
        data = dataset[idx]
        print(f"✓ Sample {idx} loaded successfully")
        print(f"\n  Keys in data:")
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                print(f"    - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
            elif hasattr(data[key], '__len__') and not isinstance(data[key], str):
                print(f"    - {key}: len {len(data[key])}, type {type(data[key])}")
            else:
                print(f"    - {key}: {type(data[key])}")

        return True
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_registration():
    """Test that all pipeline transforms are registered."""
    print("\n" + "=" * 80)
    print("TEST 3: Pipeline Registration")
    print("=" * 80)

    from reimplementation.dataset.compose import PIPELINES

    expected_transforms = [
        'LoadMultiViewImageFromFiles',
        'LoadPointsFromFile',
        'ResizeCropFlipImage',
        'MultiScaleDepthMapGenerator',
        'BBoxRotation',
        'PhotoMetricDistortionMultiViewImage',
        'NormalizeMultiviewImage',
        'CircleObjectRangeFilter',
        'InstanceNameFilter',
        'NuScenesSparse4DAdaptor',
        'Collect',
        'VectorizeMap',
    ]

    print(f"Registered pipeline transforms:")
    for name in PIPELINES.keys():
        print(f"  - {name}")

    print(f"\nChecking expected transforms:")
    all_registered = True
    for transform in expected_transforms:
        if transform in PIPELINES:
            print(f"  ✓ {transform}")
        else:
            print(f"  ✗ {transform} (MISSING)")
            all_registered = False

    return all_registered


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("NuScenes3DDataset Test Suite")
    print("=" * 80 + "\n")

    # Test 1: Pipeline registration
    pipeline_ok = test_pipeline_registration()

    # Test 2: Dataset initialization
    dataset = test_dataset_init()

    if dataset is None:
        print("\n" + "=" * 80)
        print("TESTS FAILED: Could not initialize dataset")
        print("=" * 80)
        return 1

    # Test 3: Load a single sample
    sample_ok = test_single_sample(dataset, idx=0)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Pipeline registration: {'✓ PASS' if pipeline_ok else '✗ FAIL'}")
    print(f"  Dataset initialization: {'✓ PASS' if dataset is not None else '✗ FAIL'}")
    print(f"  Sample loading: {'✓ PASS' if sample_ok else '✗ FAIL'}")

    all_passed = pipeline_ok and dataset is not None and sample_ok
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
