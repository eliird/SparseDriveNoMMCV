"""
SparseBox3DEncoder: Encodes 3D bounding box parameters into feature embeddings.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
import torch.nn as nn
from .box3d import *

__all__ = ['SparseBox3DEncoder']


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
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


class SparseBox3DEncoder(nn.Module):
    """Encoder for 3D bounding box parameters.

    This module encodes different components of a 3D bounding box (position, size,
    yaw angle, velocity) into feature embeddings. The encoded features can be either
    added or concatenated depending on the mode.

    Args:
        embed_dims (int | list): Embedding dimensions. If int, same dimension is used
            for all components. If list, should have 5 elements: [pos, size, yaw, vel, output].
        vel_dims (int): Number of velocity dimensions (0, 1, 2, or 3). Default: 3
        mode (str): How to combine features. 'add' or 'cat'. Default: 'add'
        output_fc (bool): Whether to apply output FC layer. Default: True
        in_loops (int): Number of inner Linear-ReLU loops. Default: 1
        out_loops (int): Number of outer loops (each adds LayerNorm). Default: 2

    Box parameter order (undecoded):
        [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
        - Position: [X, Y, Z]
        - Size: [W, L, H]
        - Yaw: [SIN_YAW, COS_YAW]
        - Velocity: [VX, VY, VZ] (optional, controlled by vel_dims)
    """

    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


def test_sparse_box_3d_encoder():
    """Test SparseBox3DEncoder implementation."""
    import torch

    print("Testing SparseBox3DEncoder...")

    # Test 1: mode='add' (all features added together)
    print("\n1. Testing mode='add'...")
    encoder_add = SparseBox3DEncoder(
        embed_dims=256,
        vel_dims=3,
        mode='add',
        output_fc=True,
        in_loops=1,
        out_loops=2
    )

    batch_size = 2
    num_anchors = 900
    # Box format: [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
    box_3d = torch.randn(batch_size, num_anchors, 11)

    output_add = encoder_add(box_3d)
    print(f"   Input shape: {box_3d.shape}")
    print(f"   Output shape (add mode): {output_add.shape}")
    assert output_add.shape == (batch_size, num_anchors, 256)

    # Test 2: mode='cat' (features concatenated)
    print("\n2. Testing mode='cat'...")
    encoder_cat = SparseBox3DEncoder(
        embed_dims=[128, 32, 32, 64, 256],  # Different dims for each component
        vel_dims=3,
        mode='cat',
        output_fc=True,
        in_loops=1,
        out_loops=2
    )

    output_cat = encoder_cat(box_3d)
    print(f"   Output shape (cat mode): {output_cat.shape}")
    # Cat mode: 128 + 32 + 32 + 64 = 256 before output_fc
    assert output_cat.shape == (batch_size, num_anchors, 256)

    # Test 3: Without velocity
    print("\n3. Testing without velocity (vel_dims=0)...")
    encoder_no_vel = SparseBox3DEncoder(
        embed_dims=256,
        vel_dims=0,
        mode='add',
        output_fc=True
    )

    output_no_vel = encoder_no_vel(box_3d)
    print(f"   Output shape (no velocity): {output_no_vel.shape}")
    assert output_no_vel.shape == (batch_size, num_anchors, 256)

    # Test 4: Without output FC
    print("\n4. Testing without output_fc...")
    encoder_no_fc = SparseBox3DEncoder(
        embed_dims=256,
        vel_dims=3,
        mode='add',
        output_fc=False
    )

    output_no_fc = encoder_no_fc(box_3d)
    print(f"   Output shape (no output_fc): {output_no_fc.shape}")
    assert output_no_fc.shape == (batch_size, num_anchors, 256)

    # Test 5: Different in_loops and out_loops
    print("\n5. Testing with different loop counts...")
    encoder_loops = SparseBox3DEncoder(
        embed_dims=256,
        vel_dims=3,
        mode='add',
        output_fc=True,
        in_loops=2,
        out_loops=3
    )

    output_loops = encoder_loops(box_3d)
    print(f"   Output shape (in_loops=2, out_loops=3): {output_loops.shape}")
    assert output_loops.shape == (batch_size, num_anchors, 256)

    # Test 6: Gradient flow
    print("\n6. Testing gradient flow...")
    encoder_grad = SparseBox3DEncoder(embed_dims=256, vel_dims=3, mode='add')
    box_3d_grad = torch.randn(2, 10, 11, requires_grad=True)
    output_grad = encoder_grad(box_3d_grad)
    loss = output_grad.sum()
    loss.backward()
    print(f"   Gradients computed: {box_3d_grad.grad is not None}")
    assert box_3d_grad.grad is not None

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.sparse_box_3d_encoder
    test_sparse_box_3d_encoder()