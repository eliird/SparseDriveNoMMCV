"""
SparseBox3DKeyPointsGenerator: Generates 3D keypoints from box anchors.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
import torch.nn as nn
from ..utils import xavier_init
from ..common.box3d import *

__all__ = ['SparseBox3DKeyPointsGenerator']


class SparseBox3DKeyPointsGenerator(nn.Module):
    """Generate 3D keypoints from box anchors.

    This module generates keypoints around 3D bounding boxes using a combination of:
    1. Fixed keypoints at predetermined scales (e.g., center, corners, face centers)
    2. Learnable keypoints predicted from instance features

    The keypoints are rotated and translated according to the box pose, and can
    optionally be transformed to previous timestamps for temporal modeling.

    Args:
        embed_dims (int): Embedding dimension for instance features. Default: 256
        num_learnable_pts (int): Number of learnable keypoints to predict. Default: 0
        fix_scale (list[list] | tuple[tuple], optional): Fixed keypoint positions as
            [x, y, z] offsets relative to box size. Example:
            [[0, 0, 0],          # center
             [0.45, 0, 0],       # front
             [-0.45, 0, 0],      # back
             [0, 0.45, 0],       # right
             [0, -0.45, 0],      # left
             [0, 0, 0.45],       # top
             [0, 0, -0.45]]      # bottom
            Default: [[0, 0, 0]] (center only)

    Example config:
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0],
                [-0.45, 0, 0],
                [0, 0.45, 0],
                [0, -0.45, 0],
                [0, 0, 0.45],
                [0, 0, -0.45],
            ],
        )
    """

    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts

        # Setup fixed keypoints
        if fix_scale is None:
            fix_scale = [[0.0, 0.0, 0.0]]
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale, dtype=torch.float32),
            requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts

        # Setup learnable keypoints
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)
            self.init_weight()

    def init_weight(self):
        """Initialize learnable keypoint predictor."""
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        """Generate keypoints from anchors.

        Args:
            anchor (Tensor): Box anchors of shape (bs, num_anchor, 11)
                Format: [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
            instance_feature (Tensor, optional): Instance features of shape
                (bs, num_anchor, embed_dims) for predicting learnable keypoints.
                Default: None
            T_cur2temp_list (list[Tensor], optional): List of transformation matrices
                from current frame to temporal frames, each of shape (bs, 4, 4).
                Default: None
            cur_timestamp (Tensor, optional): Current timestamp of shape (bs,).
                Default: None
            temp_timestamps (list[Tensor], optional): List of temporal timestamps,
                each of shape (bs,). Default: None

        Returns:
            Tensor | tuple:
                - If temporal info not provided: keypoints of shape
                  (bs, num_anchor, num_pts, 3)
                - If temporal info provided: tuple of (key_points, temp_key_points_list)
                  where temp_key_points_list is a list of keypoints transformed to
                  previous frames
        """
        bs, num_anchor = anchor.shape[:2]

        # Get box size (W, L, H) and expand for keypoint scaling
        size = anchor[..., None, [W, L, H]].exp()  # (bs, num_anchor, 1, 3)

        # Generate fixed keypoints scaled by box size
        key_points = self.fix_scale * size  # (bs, num_anchor, num_fix_pts, 3)

        # Add learnable keypoints if enabled
        if self.num_learnable_pts > 0 and instance_feature is not None:
            # Predict learnable offsets from instance features
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid() - 0.5  # Range: [-0.5, 0.5]
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        # Build rotation matrix from yaw angle
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        # Rotate keypoints according to box orientation
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)

        # Translate keypoints to box position
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        # Return early if no temporal modeling needed
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        # Transform keypoints to previous frames for temporal modeling
        temp_key_points_list = []
        velocity = anchor[..., VX:]  # (bs, num_anchor, 3)

        for i, t_time in enumerate(temp_timestamps):
            # Compute time difference and velocity-based translation
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )

            # Move keypoints back in time using velocity
            temp_key_points = key_points - translation[:, :, None]

            # Apply coordinate transformation to previous frame
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)

        return key_points, temp_key_points_list


def test_sparse_box_3d_key_points_generator():
    """Test SparseBox3DKeyPointsGenerator implementation."""
    print("Testing SparseBox3DKeyPointsGenerator...")

    # Test 1: Fixed keypoints only
    print("\n1. Testing with fixed keypoints only...")
    fix_scale = [
        [0, 0, 0],       # center
        [0.45, 0, 0],    # front
        [-0.45, 0, 0],   # back
        [0, 0.45, 0],    # right
        [0, -0.45, 0],   # left
        [0, 0, 0.45],    # top
        [0, 0, -0.45],   # bottom
    ]

    generator = SparseBox3DKeyPointsGenerator(
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=fix_scale
    )

    batch_size = 2
    num_anchor = 10
    # Create dummy anchors: [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
    anchor = torch.randn(batch_size, num_anchor, 11)

    key_points = generator(anchor)
    print(f"   Anchor shape: {anchor.shape}")
    print(f"   Keypoints shape: {key_points.shape}")
    print(f"   Expected: ({batch_size}, {num_anchor}, {len(fix_scale)}, 3)")
    assert key_points.shape == (batch_size, num_anchor, len(fix_scale), 3)

    # Test 2: Fixed + learnable keypoints
    print("\n2. Testing with fixed + learnable keypoints...")
    num_learnable = 6
    generator_learn = SparseBox3DKeyPointsGenerator(
        embed_dims=256,
        num_learnable_pts=num_learnable,
        fix_scale=fix_scale
    )

    instance_feature = torch.randn(batch_size, num_anchor, 256)
    key_points_learn = generator_learn(anchor, instance_feature=instance_feature)

    expected_pts = len(fix_scale) + num_learnable
    print(f"   Keypoints shape: {key_points_learn.shape}")
    print(f"   Expected: ({batch_size}, {num_anchor}, {expected_pts}, 3)")
    assert key_points_learn.shape == (batch_size, num_anchor, expected_pts, 3)

    # Test 3: With temporal modeling
    print("\n3. Testing with temporal modeling...")
    cur_timestamp = torch.tensor([1.0, 1.0])
    temp_timestamps = [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 0.0])]
    T_cur2temp_list = [
        torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
        torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    ]

    key_points_cur, temp_key_points_list = generator(
        anchor,
        cur_timestamp=cur_timestamp,
        temp_timestamps=temp_timestamps,
        T_cur2temp_list=T_cur2temp_list
    )

    print(f"   Current keypoints shape: {key_points_cur.shape}")
    print(f"   Number of temporal frames: {len(temp_key_points_list)}")
    print(f"   Temporal keypoints[0] shape: {temp_key_points_list[0].shape}")
    assert key_points_cur.shape == (batch_size, num_anchor, len(fix_scale), 3)
    assert len(temp_key_points_list) == 2
    assert temp_key_points_list[0].shape == (batch_size, num_anchor, len(fix_scale), 3)

    # Test 4: Check rotation works correctly
    print("\n4. Testing rotation...")
    # Create a box at origin with yaw=0
    anchor_test = torch.zeros(1, 1, 11)
    anchor_test[0, 0, W:H+1] = 0.0  # log(1) = 0, so size = 1x1x1
    anchor_test[0, 0, COS_YAW] = 1.0  # cos(0) = 1
    anchor_test[0, 0, SIN_YAW] = 0.0  # sin(0) = 0

    generator_simple = SparseBox3DKeyPointsGenerator(
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=[[1.0, 0.0, 0.0]]  # 1 unit in x direction
    )

    kpts = generator_simple(anchor_test)
    print(f"   Keypoint at yaw=0: {kpts[0, 0, 0]}")
    print(f"   Expected: approximately [1, 0, 0]")
    assert torch.allclose(kpts[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)

    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    generator_grad = SparseBox3DKeyPointsGenerator(
        embed_dims=256,
        num_learnable_pts=3,
        fix_scale=[[0, 0, 0]]
    )

    anchor_grad = torch.randn(2, 5, 11, requires_grad=True)
    instance_feat_grad = torch.randn(2, 5, 256, requires_grad=True)

    kpts_grad = generator_grad(anchor_grad, instance_feature=instance_feat_grad)
    loss = kpts_grad.sum()
    loss.backward()

    print(f"   Anchor gradients: {anchor_grad.grad is not None}")
    print(f"   Instance feature gradients: {instance_feat_grad.grad is not None}")
    assert anchor_grad.grad is not None
    assert instance_feat_grad.grad is not None

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.deformable.sparse_box_3d_key_point_gen
    test_sparse_box_3d_key_points_generator()
