"""
Custom builder system to replace mmcv's build_from_cfg and registries.

This module provides a simple component registry and builder function
that works without mmcv dependencies.
"""

import torch.nn as nn
from typing import Dict, Any, Optional

# Import all components that can be built
from ..common.attention import MultiheadFlashAttention
from ..common.asym_ffn import AsymmetricFFN
from ..common.instance_bank import InstanceBank
from ..common.sparse_box_3d_encoder import SparseBox3DEncoder
from ..common.sparse_point_3d_encoder import SparsePoint3DEncoder
from ..common.sparse_3d_refinement import (
    SparseBox3DRefinementModule,
    SparsePoint3DRefinementModule,
)
from ..common.target import SparseBox3DTarget
from ..common.assigner import SparsePoint3DTarget
from ..deformable.deformable_feature_aggregation import DeformableFeatureAggregation
from ..deformable.sparse_box_3d_key_point_gen import SparseBox3DKeyPointsGenerator
from ..deformable.sparse_point_3d_key_point_gen import SparsePoint3DKeyPointsGenerator
from ..decoders.sparse_box_decoder import SparseBox3DDecoder
from ..decoders.sparse_point_decoder import SparsePoint3DDecoder
from ..losses.focal_loss import FocalLoss
from ..losses.sparse_box import SparseBox3DLoss
from ..losses.sparse_line import SparseLineLoss
from ..losses.l1_loss import L1Loss, SmoothL1Loss
from ..losses.guassian import GaussianFocalLoss
from ..losses.cross_entropy import CrossEntropyLoss

# Import motion/planning components
from ..motion.instance_queue import InstanceQueue
from ..motion.refinement import MotionPlanningRefinementModule
from ..motion.target import MotionTarget, PlanningTarget
from ..motion.decoder import SparseBox3DMotionDecoder, HierarchicalPlanningDecoder
from ..motion.motion_plaaning_head import MotionPlanningHead


# Component registry mapping type strings to classes
COMPONENT_REGISTRY: Dict[str, type] = {
    # ========== Attention modules ==========
    'MultiheadAttention': MultiheadFlashAttention,
    'MultiheadFlashAttention': MultiheadFlashAttention,
    'DeformableFeatureAggregation': DeformableFeatureAggregation,

    # ========== Feedforward networks ==========
    'AsymmetricFFN': AsymmetricFFN,
    'FFN': AsymmetricFFN,  # Alias

    # ========== Normalization layers ==========
    'LN': nn.LayerNorm,
    'LayerNorm': nn.LayerNorm,

    # ========== Plugin layers ==========
    'InstanceBank': InstanceBank,
    'InstanceQueue': InstanceQueue,
    'SparseBox3DRefinementModule': SparseBox3DRefinementModule,
    'SparsePoint3DRefinementModule': SparsePoint3DRefinementModule,
    'MotionPlanningRefinementModule': MotionPlanningRefinementModule,
    'SparseBox3DKeyPointsGenerator': SparseBox3DKeyPointsGenerator,
    'SparsePoint3DKeyPointsGenerator': SparsePoint3DKeyPointsGenerator,

    # ========== Positional encodings ==========
    'SparseBox3DEncoder': SparseBox3DEncoder,
    'SparsePoint3DEncoder': SparsePoint3DEncoder,

    # ========== Samplers/Targets ==========
    'SparseBox3DTarget': SparseBox3DTarget,
    'SparsePoint3DTarget': SparsePoint3DTarget,
    'MotionTarget': MotionTarget,
    'PlanningTarget': PlanningTarget,

    # ========== Decoders ==========
    'SparseBox3DDecoder': SparseBox3DDecoder,
    'SparsePoint3DDecoder': SparsePoint3DDecoder,
    'SparseBox3DMotionDecoder': SparseBox3DMotionDecoder,
    'HierarchicalPlanningDecoder': HierarchicalPlanningDecoder,

    # ========== Losses ==========
    'FocalLoss': FocalLoss,
    'SparseBox3DLoss': SparseBox3DLoss,
    'SparseLineLoss': SparseLineLoss,
    'L1Loss': L1Loss,
    'SmoothL1Loss': SmoothL1Loss,
    'GaussianFocalLoss': GaussianFocalLoss,
    'CrossEntropyLoss': CrossEntropyLoss,

    # ========== Heads ==========
    'MotionPlanningHead': MotionPlanningHead,
}


def build_from_cfg(cfg: Optional[Dict[str, Any]], registry_name: Optional[str] = None):
    """Build a module from config dict.

    This function replaces mmcv's build_from_cfg. It looks up the component
    type in COMPONENT_REGISTRY and instantiates it with the provided kwargs.

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

    if module_type not in COMPONENT_REGISTRY:
        raise KeyError(
            f'Unrecognized module type: {module_type}. '
            f'Available types: {list(COMPONENT_REGISTRY.keys())}'
        )

    module_class = COMPONENT_REGISTRY[module_type]

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
