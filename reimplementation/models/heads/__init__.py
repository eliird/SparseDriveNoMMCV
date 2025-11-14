"""
Task-specific prediction heads.
"""

from .sparse_4d import Sparse4DHead
from .sparse_drive_head import SparseDriveHead

__all__ = [
    'Sparse4DHead',
    'SparseDriveHead',
]
