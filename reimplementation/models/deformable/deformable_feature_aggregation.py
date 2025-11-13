"""
DeformableFeatureAggregation: Aggregates multi-view multi-scale features using deformable attention.
Pure PyTorch implementation without mmcv dependencies.
"""
from typing import List, Optional
import torch
import torch.nn as nn
from ..utils import xavier_init, constant_init
from .sparse_box_3d_key_point_gen import SparseBox3DKeyPointsGenerator

__all__ = ['DeformableFeatureAggregation']

# Try to import deformable aggregation CUDA function
try:
    from reimplementation.ops import deformable_aggregation_function as DAF, CUDA_EXT_AVAILABLE
    DAF_AVAILABLE = CUDA_EXT_AVAILABLE
except ImportError:
    DAF = None
    DAF_AVAILABLE = False


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """Create a sequence of Linear-ReLU-LayerNorm layers.

    Args:
        embed_dims (int): Output embedding dimension
        in_loops (int): Number of Linear-ReLU pairs per outer loop
        out_loops (int): Number of outer loops (each adds LayerNorm)
        input_dims (int, optional): Input dimension. Default: embed_dims

    Returns:
        list: List of nn.Module layers
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class DeformableFeatureAggregation(nn.Module):
    """Deformable Feature Aggregation for multi-view 3D object detection.

    This module aggregates features from multiple camera views and feature pyramid levels
    by sampling at 3D keypoints projected onto 2D image planes. It uses learnable attention
    weights to fuse features across views, levels, and keypoints.

    The module can optionally use a CUDA-accelerated deformable aggregation function for
    efficiency, or fall back to PyTorch's grid_sample.

    Args:
        embed_dims (int): Embedding dimension. Default: 256
        num_groups (int): Number of attention groups. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_cams (int): Number of camera views. Default: 6
        proj_drop (float): Dropout rate for output projection. Default: 0.0
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        kps_generator (dict): Config for keypoint generator. Must include 'type' and
            other parameters for SparseBox3DKeyPointsGenerator.
        temporal_fusion_module (dict, optional): Config for temporal fusion. Default: None
        use_temporal_anchor_embed (bool): Whether to use temporal anchor embeddings. Default: True
        use_deformable_func (bool): Whether to use CUDA deformable aggregation function. Default: False
        use_camera_embed (bool): Whether to use per-camera embeddings. Default: False
        residual_mode (str): How to combine aggregated features with input. 'add' or 'cat'. Default: 'add'
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()

        # Validate parameters
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )

        # Store configuration
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)

        # Setup deformable aggregation function
        if use_deformable_func:
            assert DAF_AVAILABLE, "deformable_aggregation CUDA op needs to be compiled."
        self.use_deformable_func = use_deformable_func

        # Build keypoint generator
        if kps_generator is None:
            raise ValueError("kps_generator config must be provided")

        kps_cfg = kps_generator.copy()
        kps_type = kps_cfg.pop('type', 'SparseBox3DKeyPointsGenerator')
        if kps_type != 'SparseBox3DKeyPointsGenerator':
            raise ValueError(f"Only SparseBox3DKeyPointsGenerator is supported, got {kps_type}")

        kps_cfg['embed_dims'] = embed_dims
        self.kps_generator = SparseBox3DKeyPointsGenerator(**kps_cfg)
        self.num_pts = self.kps_generator.num_pts

        # Build temporal fusion module (if provided) #TODO different from original
        if temporal_fusion_module is not None:
            raise NotImplementedError("Temporal fusion module not yet implemented")
        self.temp_module = None

        # Output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # Camera-specific encoding and attention weights
        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

        self.init_weight()

    def init_weight(self):
        """Initialize module weights."""
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """Forward pass for deformable feature aggregation."""
        bs, num_anchor = instance_feature.shape[:2]

        # Generate 3D keypoints from anchors and instance features
        key_points = self.kps_generator(anchor, instance_feature)

        # Compute attention weights for feature aggregation
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        # Aggregate features
        if self.use_deformable_func:
            # Use CUDA deformable aggregation function
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            # Use PyTorch grid_sample
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # Fuse multi-point features

        # Project and apply dropout
        output = self.proj_drop(self.output_proj(features))

        # Add residual connection
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        """Compute attention weights for feature aggregation."""
        bs, num_anchor = instance_feature.shape[:2]

        # Combine instance features with anchor embeddings
        feature = instance_feature + anchor_embed

        # Add camera-specific embeddings if enabled
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        # Predict attention weights
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )

        # Apply attention dropout during training
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )

        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """Project 3D keypoints to 2D image coordinates."""
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Convert to homogeneous coordinates
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )

        # Project using camera matrices
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)

        # Perspective division
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )

        # Normalize to [0, 1] if image dimensions provided
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]

        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample features from multi-scale feature maps at 3D keypoint locations."""
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Project 3D points to 2D
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )

        # Convert from [0, 1] to [-1, 1] for grid_sample
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        # Sample features at projected locations
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1),
                    points_2d,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )
            )

        # Stack and reshape
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(0, 4, 1, 2, 5, 3)

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Fuse features across views and levels using attention weights."""
        bs, num_anchor = weights.shape[:2]

        # Reshape features into groups for group-wise attention
        features = features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )

        # Apply attention weights and sum across views and levels
        features = weights[..., None] * features
        features = features.sum(dim=2).sum(dim=2)

        # Reshape back to full embedding dimension
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )

        return features


def test_deformable_feature_aggregation():
    """Test DeformableFeatureAggregation implementation."""
    print("Testing DeformableFeatureAggregation...")

    # Test 1: Basic forward pass without camera embed
    print("\n1. Testing basic forward pass...")
    embed_dims = 256
    num_groups = 8
    num_levels = 4
    num_cams = 6
    batch_size = 2
    num_anchor = 10

    model = DeformableFeatureAggregation(
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=num_cams,
        use_camera_embed=False,
        residual_mode='add',
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            num_learnable_pts=6,
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0]]
        )
    )

    # Create dummy inputs
    instance_feature = torch.randn(batch_size, num_anchor, embed_dims)
    anchor = torch.randn(batch_size, num_anchor, 11)
    anchor_embed = torch.randn(batch_size, num_anchor, embed_dims)

    # Create multi-scale feature maps
    feature_maps = [
        torch.randn(batch_size, num_cams, embed_dims, 32, 32),
        torch.randn(batch_size, num_cams, embed_dims, 16, 16),
        torch.randn(batch_size, num_cams, embed_dims, 8, 8),
        torch.randn(batch_size, num_cams, embed_dims, 4, 4),
    ]

    # Create metadata
    metas = {
        'projection_mat': torch.randn(batch_size, num_cams, 4, 4),
        'image_wh': torch.tensor([[800, 450]] * num_cams).unsqueeze(0).repeat(batch_size, 1, 1).float()
    }

    output = model(instance_feature, anchor, anchor_embed, feature_maps, metas)

    print(f"   Input shape: {instance_feature.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, num_anchor, embed_dims)
    print("   ✓ Basic forward pass works")

    # Test 2: With camera embeddings
    print("\n2. Testing with camera embeddings...")
    model_cam = DeformableFeatureAggregation(
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=num_cams,
        use_camera_embed=True,
        residual_mode='add',
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            num_learnable_pts=3,
            fix_scale=[[0, 0, 0]]
        )
    )

    output_cam = model_cam(instance_feature, anchor, anchor_embed, feature_maps, metas)
    print(f"   Output shape with camera embed: {output_cam.shape}")
    assert output_cam.shape == (batch_size, num_anchor, embed_dims)
    print("   ✓ Camera embeddings work")

    # Test 3: With concatenation residual
    print("\n3. Testing with concatenation residual...")
    model_cat = DeformableFeatureAggregation(
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=num_cams,
        residual_mode='cat',
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            num_learnable_pts=0,
            fix_scale=[[0, 0, 0]]
        )
    )

    output_cat = model_cat(instance_feature, anchor, anchor_embed, feature_maps, metas)
    print(f"   Output shape with cat residual: {output_cat.shape}")
    assert output_cat.shape == (batch_size, num_anchor, embed_dims * 2)
    print("   ✓ Concatenation residual works")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    instance_feature_grad = torch.randn(batch_size, num_anchor, embed_dims, requires_grad=True)
    anchor_grad = torch.randn(batch_size, num_anchor, 11, requires_grad=True)

    output_grad = model(instance_feature_grad, anchor_grad, anchor_embed, feature_maps, metas)
    loss = output_grad.sum()
    loss.backward()

    print(f"   Instance feature gradients: {instance_feature_grad.grad is not None}")
    print(f"   Anchor gradients: {anchor_grad.grad is not None}")
    assert instance_feature_grad.grad is not None
    assert anchor_grad.grad is not None
    print("   ✓ Gradient flow works")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.deformable.deformable_feature_aggregation
    test_deformable_feature_aggregation()
