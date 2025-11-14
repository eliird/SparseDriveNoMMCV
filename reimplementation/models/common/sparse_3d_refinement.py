import torch
import torch.nn as nn
from .box3d import *
from ..utils.model_utils import linear_relu_ln, bias_init_with_prob, Scale


class SparsePoint3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
        else:
            cls = None
        qt = None
        return output, cls, qt
    

class SparseBox3DRefinementModule(nn.Module):
    """3D bounding box refinement module for sparse detection.
    
    This module refines 3D bounding box predictions by predicting deltas
    for position, size, rotation, and optionally velocity. It also provides
    classification scores and quality estimation if enabled.
    
    Args:
        embed_dims (int): Embedding dimension for input features. Default: 256
        output_dim (int): Output dimension for box parameters. Default: 11
            - 8 for [x, y, z, w, l, h, sin_yaw, cos_yaw]  
            - 11 for [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]
        num_cls (int): Number of object classes. Default: 10
        normalize_yaw (bool): Whether to normalize yaw prediction. Default: False
        refine_yaw (bool): Whether to refine yaw angle. Default: False
        with_cls_branch (bool): Whether to include classification branch. Default: True
        with_quality_estimation (bool): Whether to include quality estimation. Default: False
    """
    
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        # Define which box parameters to refine
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        # Main refinement layers
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),  # 2 layers, 2 iterations
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),  # Learnable scaling
        )
        
        # Optional classification branch
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),  # 1 layer, 2 iterations
                nn.Linear(self.embed_dims, self.num_cls),
            )
            
        # Optional quality estimation branch
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),  # 1 layer, 2 iterations
                nn.Linear(self.embed_dims, 2),  # 2D quality score
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        """Forward pass for 3D bounding box refinement.
        
        Args:
            instance_feature (torch.Tensor): Instance features of shape (B, N, embed_dims)
            anchor (torch.Tensor): Anchor box parameters of shape (B, N, output_dim)
                - Format: [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]
            anchor_embed (torch.Tensor): Anchor embeddings of shape (B, N, embed_dims)  
            time_interval (torch.Tensor): Time interval for velocity computation. Default: 1.0
            return_cls (bool): Whether to return classification scores. Default: True
            
        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Refined box parameters (B, N, output_dim)
                - cls (torch.Tensor): Classification scores (B, N, num_cls) or None
                - quality (torch.Tensor): Quality scores (B, N, 2) or None
        """
        # Combine instance and anchor features
        feature = instance_feature + anchor_embed
        
        # Predict refinement deltas
        output = self.layers(feature)
        
        # Add deltas to anchor values for refined predictions
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        
        # Normalize yaw angles if enabled
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
            
        # Handle velocity prediction (for output_dim > 8)
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            # Compute velocity from translation and time interval
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        # Classification prediction
        if return_cls:
            assert self.with_cls_branch, "Classification branch not enabled!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
            
        # Quality estimation prediction
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
            
        return output, cls, quality


def test_sparse_box_3d_refinement():
    """Test SparseBox3DRefinementModule implementation."""
    print("Testing SparseBox3DRefinementModule...")
    
    # Test parameters
    batch_size = 2
    num_instances = 100
    embed_dims = 256
    num_cls = 10
    
    # Test 1: Basic refinement (8D output without velocity)
    print("\n1. Testing basic refinement (8D output)...")
    model_8d = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=8,
        num_cls=num_cls,
        normalize_yaw=False,
        refine_yaw=True,
        with_cls_branch=True,
        with_quality_estimation=False
    )
    
    # Create dummy inputs
    instance_feature = torch.randn(batch_size, num_instances, embed_dims)
    anchor = torch.randn(batch_size, num_instances, 8)  # [x,y,z,w,l,h,sin_yaw,cos_yaw]
    anchor_embed = torch.randn(batch_size, num_instances, embed_dims)
    
    output, cls, quality = model_8d(instance_feature, anchor, anchor_embed)
    
    print(f"   Input shape: {instance_feature.shape}")
    print(f"   Anchor shape: {anchor.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Classification shape: {cls.shape}")
    assert output.shape == (batch_size, num_instances, 8)
    assert cls.shape == (batch_size, num_instances, num_cls)
    assert quality is None
    print("   ✓ Basic refinement works")
    
    # Test 2: With velocity (11D output)
    print("\n2. Testing with velocity (11D output)...")
    model_11d = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=11,
        num_cls=num_cls,
        normalize_yaw=True,
        refine_yaw=True,
        with_cls_branch=True,
        with_quality_estimation=True
    )
    
    anchor_11d = torch.randn(batch_size, num_instances, 11)  # [x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy,vz]
    time_interval = torch.tensor(0.1)  # 100ms
    
    output, cls, quality = model_11d(instance_feature, anchor_11d, anchor_embed, time_interval)
    
    print(f"   Output shape with velocity: {output.shape}")
    print(f"   Quality shape: {quality.shape}")
    assert output.shape == (batch_size, num_instances, 11)
    assert cls.shape == (batch_size, num_instances, num_cls)
    assert quality.shape == (batch_size, num_instances, 2)
    print("   ✓ Velocity prediction works")
    
    # Test 3: Yaw normalization
    print("\n3. Testing yaw normalization...")
    output_normalized, _, _ = model_11d(instance_feature, anchor_11d, anchor_embed)
    
    # Check that sin_yaw^2 + cos_yaw^2 ≈ 1
    sin_yaw = output_normalized[..., SIN_YAW]
    cos_yaw = output_normalized[..., COS_YAW]
    yaw_norm = sin_yaw**2 + cos_yaw**2
    
    print(f"   Yaw normalization check: {yaw_norm.mean():.6f} ≈ 1.0")
    assert torch.allclose(yaw_norm, torch.ones_like(yaw_norm), atol=1e-6)
    print("   ✓ Yaw normalization works")
    
    # Test 4: Without classification
    print("\n4. Testing without classification...")
    model_no_cls = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=8,
        num_cls=num_cls,
        with_cls_branch=False
    )
    
    output, cls, quality = model_no_cls(instance_feature, anchor, anchor_embed, return_cls=False)
    
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, num_instances, 8)
    assert cls is None
    assert quality is None
    print("   ✓ No classification branch works")
    
    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    instance_feature_grad = torch.randn(batch_size, num_instances, embed_dims, requires_grad=True)
    anchor_grad = torch.randn(batch_size, num_instances, 8, requires_grad=True)
    anchor_embed_grad = torch.randn(batch_size, num_instances, embed_dims, requires_grad=True)
    
    output_grad, cls_grad, _ = model_8d(instance_feature_grad, anchor_grad, anchor_embed_grad)
    loss = (output_grad.sum() + cls_grad.sum())
    loss.backward()
    
    print(f"   Instance feature gradients: {instance_feature_grad.grad is not None}")
    print(f"   Anchor gradients: {anchor_grad.grad is not None}")
    print(f"   Anchor embed gradients: {anchor_embed_grad.grad is not None}")
    assert instance_feature_grad.grad is not None
    assert anchor_grad.grad is not None
    assert anchor_embed_grad.grad is not None
    print("   ✓ Gradient flow works")
    
    # Test 6: Weight initialization
    print("\n6. Testing weight initialization...")
    model_init = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=8,
        num_cls=num_cls,
        with_cls_branch=True
    )
    model_init.init_weight()
    
    # Check that classification bias is properly initialized
    cls_bias = model_init.cls_layers[-1].bias
    expected_bias = bias_init_with_prob(0.01)
    
    print(f"   Classification bias: {cls_bias[0].item():.6f}")
    print(f"   Expected bias: {expected_bias:.6f}")
    assert torch.allclose(cls_bias, torch.full_like(cls_bias, expected_bias), atol=1e-6)
    print("   ✓ Weight initialization works")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.sparse_3d_refinement
    test_sparse_box_3d_refinement()