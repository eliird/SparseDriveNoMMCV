"""
Custom builder system to replace mmcv's build_from_cfg and registries.

This module provides a simple component registry and builder function
that works without mmcv dependencies. Uses lazy imports to avoid circular dependencies.
"""

import torch.nn as nn
from typing import Dict, Any, Optional, Callable


def _get_component_class(module_type: str):
    """Lazy import and return component class to avoid circular imports."""

    # ========== Normalization layers ==========
    if module_type in ['LN', 'LayerNorm']:
        return nn.LayerNorm

    # ========== Attention modules ==========
    elif module_type in ['MultiheadAttention', 'MultiheadFlashAttention']:
        from ..common.attention import MultiheadFlashAttention
        return MultiheadFlashAttention
    elif module_type == 'DeformableFeatureAggregation':
        from ..deformable.deformable_feature_aggregation import DeformableFeatureAggregation
        return DeformableFeatureAggregation

    # ========== Feedforward networks ==========
    elif module_type in ['AsymmetricFFN', 'FFN']:
        from ..common.asym_ffn import AsymmetricFFN
        return AsymmetricFFN

    # ========== Plugin layers ==========
    elif module_type == 'InstanceBank':
        from ..common.instance_bank import InstanceBank
        return InstanceBank
    elif module_type == 'InstanceQueue':
        from ..motion.instance_queue import InstanceQueue
        return InstanceQueue
    elif module_type == 'SparseBox3DRefinementModule':
        from ..common.sparse_3d_refinement import SparseBox3DRefinementModule
        return SparseBox3DRefinementModule
    elif module_type == 'SparsePoint3DRefinementModule':
        from ..common.sparse_3d_refinement import SparsePoint3DRefinementModule
        return SparsePoint3DRefinementModule
    elif module_type == 'MotionPlanningRefinementModule':
        from ..motion.refinement import MotionPlanningRefinementModule
        return MotionPlanningRefinementModule
    elif module_type == 'SparseBox3DKeyPointsGenerator':
        from ..deformable.sparse_box_3d_key_point_gen import SparseBox3DKeyPointsGenerator
        return SparseBox3DKeyPointsGenerator
    elif module_type == 'SparsePoint3DKeyPointsGenerator':
        from ..deformable.sparse_point_3d_key_point_gen import SparsePoint3DKeyPointsGenerator
        return SparsePoint3DKeyPointsGenerator

    # ========== Positional encodings ==========
    elif module_type == 'SparseBox3DEncoder':
        from ..common.sparse_box_3d_encoder import SparseBox3DEncoder
        return SparseBox3DEncoder
    elif module_type == 'SparsePoint3DEncoder':
        from ..common.sparse_point_3d_encoder import SparsePoint3DEncoder
        return SparsePoint3DEncoder

    # ========== Samplers/Targets ==========
    elif module_type == 'SparseBox3DTarget':
        from ..common.target import SparseBox3DTarget
        return SparseBox3DTarget
    elif module_type == 'SparsePoint3DTarget':
        from ..common.assigner import SparsePoint3DTarget
        return SparsePoint3DTarget
    elif module_type == 'MotionTarget':
        from ..motion.target import MotionTarget
        return MotionTarget
    elif module_type == 'PlanningTarget':
        from ..motion.target import PlanningTarget
        return PlanningTarget

    # ========== Decoders ==========
    elif module_type == 'SparseBox3DDecoder':
        from ..decoders.sparse_box_decoder import SparseBox3DDecoder
        return SparseBox3DDecoder
    elif module_type == 'SparsePoint3DDecoder':
        from ..decoders.sparse_point_decoder import SparsePoint3DDecoder
        return SparsePoint3DDecoder
    elif module_type == 'SparseBox3DMotionDecoder':
        from ..motion.decoder import SparseBox3DMotionDecoder
        return SparseBox3DMotionDecoder
    elif module_type == 'HierarchicalPlanningDecoder':
        from ..motion.decoder import HierarchicalPlanningDecoder
        return HierarchicalPlanningDecoder

    # ========== Losses ==========
    elif module_type == 'FocalLoss':
        from ..losses.focal_loss import FocalLoss
        return FocalLoss
    elif module_type == 'SparseBox3DLoss':
        from ..losses.sparse_box import SparseBox3DLoss
        return SparseBox3DLoss
    elif module_type == 'SparseLineLoss':
        from ..losses.sparse_line import SparseLineLoss
        return SparseLineLoss
    elif module_type == 'L1Loss':
        from ..losses.l1_loss import L1Loss
        return L1Loss
    elif module_type == 'SmoothL1Loss':
        from ..losses.l1_loss import SmoothL1Loss
        return SmoothL1Loss
    elif module_type == 'GaussianFocalLoss':
        from ..losses.guassian import GaussianFocalLoss
        return GaussianFocalLoss
    elif module_type == 'CrossEntropyLoss':
        from ..losses.cross_entropy import CrossEntropyLoss
        return CrossEntropyLoss

    # ========== Heads ==========
    elif module_type == 'Sparse4DHead':
        from ..heads.sparse_4d import Sparse4DHead
        return Sparse4DHead
    elif module_type == 'MotionPlanningHead':
        from ..motion.motion_plaaning_head import MotionPlanningHead
        return MotionPlanningHead
    elif module_type == 'SparseDriveHead':
        from ..heads.sparse_drive_head import SparseDriveHead
        return SparseDriveHead

    # ========== Full Models ==========
    elif module_type == 'SparseDrive':
        from ..sparse_drive import SparseDrive
        return SparseDrive

    else:
        raise KeyError(
            f'Unrecognized module type: {module_type}. '
            f'Please check the type name or register it using register_component.'
        )


# For backwards compatibility, maintain a registry dict
COMPONENT_REGISTRY: Dict[str, type] = {}


def build_from_cfg(cfg: Optional[Dict[str, Any]], registry_name: Optional[str] = None):
    """Build a module from config dict.

    This function replaces mmcv's build_from_cfg. It looks up the component
    type and instantiates it with the provided kwargs. Uses lazy imports
    to avoid circular dependencies.

    Args:
        cfg (dict or None): Configuration dict with 'type' key, or None
        registry_name (str, optional): Name of registry (ignored, for compatibility with mmcv API)

    Returns:
        nn.Module or object: Built module instance, or None if cfg is None

    Raises:
        TypeError: If cfg is not a dict
        KeyError: If 'type' is missing or unrecognized

    Examples:
        >>> # Build a LayerNorm
        >>> ln = build_from_cfg({'type': 'LN', 'normalized_shape': 256})

        >>> # Build an attention module
        >>> attn = build_from_cfg({
        ...     'type': 'MultiheadFlashAttention',
        ...     'embed_dims': 256,
        ...     'num_heads': 8
        ... })

        >>> # Build with nested configs
        >>> dfg = build_from_cfg({
        ...     'type': 'DeformableFeatureAggregation',
        ...     'embed_dims': 256,
        ...     'kps_generator': {
        ...         'type': 'SparseBox3DKeyPointsGenerator',
        ...         'embed_dims': 256
        ...     }
        ... })
    """
    if cfg is None:
        return None

    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError('cfg must contain the key "type"')

    # Make a copy to avoid modifying the original config
    cfg = cfg.copy()
    module_type = cfg.pop('type')

    # Check if it's in the manually registered components first
    if module_type in COMPONENT_REGISTRY:
        module_class = COMPONENT_REGISTRY[module_type]
    else:
        # Otherwise, use lazy import
        module_class = _get_component_class(module_type)

    # Special handling for LayerNorm which requires normalized_shape as first positional arg
    if module_type in ['LN', 'LayerNorm']:
        if 'normalized_shape' in cfg:
            normalized_shape = cfg.pop('normalized_shape')
            return module_class(normalized_shape, **cfg)
        elif 'embed_dims' in cfg:
            # Use embed_dims as normalized_shape if provided
            normalized_shape = cfg.pop('embed_dims')
            return module_class(normalized_shape, **cfg)
        else:
            raise ValueError("LayerNorm requires 'normalized_shape' or 'embed_dims'")

    # Build and return the module
    # NOTE: We do NOT recursively build nested configs here.
    # Each component is responsible for building its own nested configs
    # using build_from_cfg as needed.
    return module_class(**cfg)


def register_component(name: str, component_class: type):
    """Register a new component type.

    This allows extending the registry with custom components.

    Args:
        name (str): Type name for the component
        component_class (type): The class to register

    Example:
        >>> class MyCustomModule(nn.Module):
        ...     def __init__(self, param1, param2):
        ...         super().__init__()
        ...         self.param1 = param1
        ...         self.param2 = param2
        >>>
        >>> register_component('MyCustomModule', MyCustomModule)
        >>> module = build_from_cfg({'type': 'MyCustomModule', 'param1': 1, 'param2': 2})
    """
    COMPONENT_REGISTRY[name] = component_class


__all__ = ['build_from_cfg', 'register_component', 'COMPONENT_REGISTRY']
