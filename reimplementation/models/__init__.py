"""
Pure PyTorch models for SparseDrive.
All mmcv/mmdet dependencies removed.
"""

from .sparse_drive import SparseDrive
from .heads import Sparse4DHead, SparseDriveHead
from .motion import MotionPlanningHead

# Import backbones and necks to register them
from .backbones import ResNet
from .necks import FPN

__all__ = [
    'SparseDrive',
    'Sparse4DHead',
    'SparseDriveHead',
    'MotionPlanningHead',
    'ResNet',
    'FPN',
]
