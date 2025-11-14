#!/usr/bin/env python
"""
Example/test script for using build_dataset and build_dataloader functions.

This demonstrates the mmdet-style API for building datasets and dataloaders
from config dictionaries.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from reimplementation.dataset import build_dataset, build_dataloader

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
batch_size = 2

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

# Define dataset configs in mmdet style
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

eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'nuscenes_infos_train.pkl',
    pipeline=test_pipeline,
    test_mode=True,
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)


def test_build_dataset():
    """Test building dataset from config."""
    print("=" * 80)
    print("TEST 1: Build Dataset from Config")
    print("=" * 80)

    try:
        # Build train dataset
        train_dataset = build_dataset(data['train'])
        print(f"✓ Train dataset built successfully")
        print(f"  - Type: {type(train_dataset).__name__}")
        print(f"  - Number of samples: {len(train_dataset)}")
        print(f"  - Classes: {train_dataset.CLASSES[:3]}... ({len(train_dataset.CLASSES)} total)")

        # Build val dataset
        val_dataset = build_dataset(data['val'])
        print(f"\n✓ Val dataset built successfully")
        print(f"  - Type: {type(val_dataset).__name__}")
        print(f"  - Number of samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    except Exception as e:
        print(f"✗ Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_build_dataloader(dataset):
    """Test building dataloader from dataset."""
    print("\n" + "=" * 80)
    print("TEST 2: Build DataLoader")
    print("=" * 80)

    try:
        # Build dataloader with mmdet-style API
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,
            shuffle=False,
            dist=False
        )

        print(f"✓ DataLoader built successfully")
        print(f"  - Batch size: {dataloader.batch_size}")
        print(f"  - Number of workers: {dataloader.num_workers}")
        print(f"  - Number of batches: {len(dataloader)}")

        # Test loading a batch
        print(f"\n  Loading first batch...")
        batch = next(iter(dataloader))

        print(f"✓ First batch loaded successfully")
        print(f"\n  Batch contents:")
        for key in batch.keys():
            if hasattr(batch[key], 'shape'):
                print(f"    - {key}: shape {batch[key].shape}, dtype {batch[key].dtype}")
            elif isinstance(batch[key], list):
                if len(batch[key]) > 0 and hasattr(batch[key][0], 'shape'):
                    print(f"    - {key}: list of {len(batch[key])} items, first shape {batch[key][0].shape}")
                else:
                    print(f"    - {key}: list of length {len(batch[key])}")
            else:
                print(f"    - {key}: type {type(batch[key])}")

        return True

    except Exception as e:
        print(f"✗ DataLoader building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_gpu_config():
    """Test multi-GPU dataloader configuration."""
    print("\n" + "=" * 80)
    print("TEST 3: Multi-GPU DataLoader Configuration")
    print("=" * 80)

    try:
        train_dataset = build_dataset(data['train'])

        # Build dataloader with multi-GPU settings (but not actually distributed)
        dataloader = build_dataloader(
            train_dataset,
            samples_per_gpu=data['samples_per_gpu'],
            workers_per_gpu=data['workers_per_gpu'],
            num_gpus=1,
            shuffle=True,
            dist=False,
            drop_last=True,
            pin_memory=True,
        )

        print(f"✓ Multi-GPU style dataloader built successfully")
        print(f"  - Batch size: {dataloader.batch_size}")
        print(f"  - Number of workers: {dataloader.num_workers}")
        print(f"  - Shuffle: {dataloader.sampler is None}")
        print(f"  - Drop last: {dataloader.drop_last}")
        print(f"  - Pin memory: {dataloader.pin_memory}")

        return True

    except Exception as e:
        print(f"✗ Multi-GPU config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Dataset & DataLoader Builder Test Suite")
    print("=" * 80 + "\n")

    # Test 1: Build datasets
    train_dataset, val_dataset = test_build_dataset()

    if train_dataset is None:
        print("\n" + "=" * 80)
        print("TESTS FAILED: Could not build dataset")
        print("=" * 80)
        return 1

    # Test 2: Build dataloader
    dataloader_ok = test_build_dataloader(train_dataset)

    # Test 3: Multi-GPU config
    multi_gpu_ok = test_multi_gpu_config()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Build dataset: {'✓ PASS' if train_dataset is not None else '✗ FAIL'}")
    print(f"  Build dataloader: {'✓ PASS' if dataloader_ok else '✗ FAIL'}")
    print(f"  Multi-GPU config: {'✓ PASS' if multi_gpu_ok else '✗ FAIL'}")

    all_passed = train_dataset is not None and dataloader_ok and multi_gpu_ok
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())