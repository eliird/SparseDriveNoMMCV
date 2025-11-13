"""
CUDA operations for efficient deformable aggregation.
Pure PyTorch/CUDA implementation without mmcv dependencies.
"""
import torch

from .deformable_aggregation import DeformableAggregationFunction, CUDA_EXT_AVAILABLE

__all__ = [
    'deformable_aggregation_function',
    'feature_maps_format',
    'DeformableAggregationFunction',
    'CUDA_EXT_AVAILABLE',
]


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """Apply deformable aggregation using CUDA extension.
    
    This is a convenience wrapper around DeformableAggregationFunction.apply().
    
    Args:
        feature_maps (Tensor): Flattened multi-camera multi-scale features
        spatial_shape (Tensor): Spatial dimensions of each feature map
        scale_start_index (Tensor): Starting indices for each feature map
        sampling_location (Tensor): 2D sampling locations
        weights (Tensor): Attention weights for aggregation
    
    Returns:
        Tensor: Aggregated features
    
    Raises:
        RuntimeError: If CUDA extension is not available
    """
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError(
            "Deformable aggregation CUDA extension is not available. "
            "Please compile the extension using: cd reimplementation/ops && python setup.py install"
        )
    
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False):
    """Format or un-format multi-camera multi-scale feature maps.
    
    This function converts between two representations:
    1. List of feature maps: [[cam1_level1, cam1_level2, ...], [cam2_level1, ...], ...]
       Each feature map has shape (bs, num_cams, C, H, W)
    2. Flattened representation: (col_feats, spatial_shape, scale_start_index)
       - col_feats: (bs, num_total_pixels, C) - all feature maps flattened
       - spatial_shape: (num_cams, num_levels, 2) - (H, W) for each feature map
       - scale_start_index: (num_cams, num_levels) - starting pixel index
    
    Args:
        feature_maps: Either list of feature maps or tuple of (col_feats, spatial_shape, scale_start_index)
        inverse (bool): If True, convert from flattened to list format. Default: False
    
    Returns:
        If inverse=False: [col_feats, spatial_shape, scale_start_index]
        If inverse=True: List of feature maps
    """
    if inverse:
        # Convert from flattened format back to list of feature maps
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        # Calculate split sizes for each camera's feature maps
        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        # Group cameras with identical spatial shapes
        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        
        # Split by camera groups
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        # Reshape each feature map to (bs, num_cams, C, H, W)
        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    # Convert from list of feature maps to flattened format
    if isinstance(feature_maps[0], (list, tuple)):
        # Handle nested lists (multiple camera groups)
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    # Single list of feature maps
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    # Flatten each feature map's spatial dimensions
    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])  # (H, W)
        # Reshape to (bs, num_cams, C, H*W)
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    # Concatenate all levels and reshape to (bs, num_total_pixels, C)
    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    
    # Create spatial_shape tensor: (num_cams, num_levels, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    
    # Calculate starting index for each feature map
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
