"""
Custom collate function for SparseDrive dataset.
Handles variable-length sequences like bboxes, labels, etc.
"""

import torch
from typing import List, Dict, Any


def sparse_drive_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for SparseDrive dataset.

    Handles variable-length tensors that cannot be stacked (e.g., different
    numbers of bboxes per sample).

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Collated batch dict
    """
    if len(batch) == 0:
        return {}

    # Keys that should be stacked (fixed size across samples)
    stackable_keys = [
        'img',  # Images: [B, N, C, H, W]
        'timestamp',  # Timestamps: [B, N]
        'projection_mat',  # Projection matrices: [B, N, 4, 4]
        'image_wh',  # Image width/height: [B, N, 2]
        'focal',  # Focal length
    ]

    # Keys that should be kept as lists (variable length or variable size)
    list_keys = [
        'gt_bboxes_3d',  # 3D bounding boxes (different #objects per sample)
        'gt_labels_3d',  # 3D labels (different #objects per sample)
        'gt_map_labels',  # Map labels (different #map elements per sample)
        'gt_map_pts',  # Map points (different #map elements per sample)
        'gt_agent_fut_trajs',  # Future trajectories
        'gt_agent_fut_masks',  # Future masks
        'gt_ego_fut_trajs',  # Ego future trajectories
        'gt_ego_fut_masks',  # Ego future masks
        'gt_ego_fut_cmd',  # Ego commands
        'ego_status',  # Ego status
        # Note: gt_depth is handled specially (transposed list of scales)
    ]

    # Keys that should be kept as-is (metadata)
    meta_keys = ['img_metas']

    collated = {}
    sample_keys = batch[0].keys()

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        # Special handling for gt_depth (list of scales, need to transpose)
        if key == 'gt_depth' and all(isinstance(v, list) for v in values):
            # gt_depth is a list of tensors (one per scale)
            # Each sample: [scale0, scale1, scale2, ...]
            # We need to transpose to: [batch_scale0, batch_scale1, batch_scale2, ...]
            try:
                num_scales = len(values[0])
                transposed = []
                for scale_idx in range(num_scales):
                    scale_tensors = [v[scale_idx] for v in values]
                    # Stack tensors for this scale
                    if all(isinstance(t, torch.Tensor) for t in scale_tensors):
                        transposed.append(torch.stack(scale_tensors, dim=0))
                    else:
                        # Convert numpy arrays to tensors first
                        transposed.append(torch.stack([torch.as_tensor(t) for t in scale_tensors], dim=0))
                collated[key] = transposed
                continue
            except Exception as e:
                print(f"Warning: Could not transpose gt_depth, keeping as list. Error: {e}")
                collated[key] = values
                continue

        if key in stackable_keys:
            # Stack tensors (all should have same shape)
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    collated[key] = torch.stack(values, dim=0)
                except RuntimeError as e:
                    # If stacking fails, keep as list
                    print(f"Warning: Could not stack {key}, keeping as list. Error: {e}")
                    collated[key] = values
            elif all(v is not None for v in values):
                # Try to convert to tensor and stack
                try:
                    collated[key] = torch.stack([torch.as_tensor(v) for v in values], dim=0)
                except (RuntimeError, ValueError) as e:
                    print(f"Warning: Could not stack {key}, keeping as list. Error: {e}")
                    collated[key] = values
            else:
                # Some values are None, keep as list
                collated[key] = values

        elif key in list_keys:
            # Keep as list (variable length across samples)
            collated[key] = values

        elif key in meta_keys:
            # Special handling for img_metas
            if key == 'img_metas':
                # img_metas is a list of dicts, keep as list
                collated[key] = values
            else:
                collated[key] = values

        else:
            # Unknown key - try to stack, fallback to list
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    collated[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    collated[key] = values
            else:
                collated[key] = values

    return collated
