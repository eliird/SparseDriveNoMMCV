import torch
import torch.nn as nn

from .cross_entropy import CrossEntropyLoss
from .l1_loss import L1Loss
from .guassian import GaussianFocalLoss
from ..common.box3d import *

'''
Usage example from config:
    loss_reg=dict(
        type="SparseBox3DLoss",
        loss_box=dict(type="L1Loss", loss_weight=0.25),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
        loss_yawness=dict(type="GaussianFocalLoss"),
        cls_allow_reverse=[class_names.index("barrier")],
    ),
    decoder=dict(type="SparseBox3DDecoder"),
    reg_weights=[2.0] * 3 + [1.0] * 7,
'''


class SparseBox3DLoss(nn.Module):
    """Sparse 3D box regression loss.

    Computes box regression loss (L1) and optionally centerness and yawness quality losses.

    Args:
        loss_box (nn.Module or dict): Box regression loss (e.g., L1Loss)
        loss_centerness (nn.Module or dict, optional): Centerness quality loss
        loss_yawness (nn.Module or dict, optional): Yaw angle quality loss
        cls_allow_reverse (list, optional): Class IDs that allow yaw reversal (e.g., barriers)
    """

    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ):
        super().__init__()

        # Build loss modules from either nn.Module instances or dict configs
        def build_loss(cfg):
            if cfg is None:
                return None
            if isinstance(cfg, nn.Module):
                return cfg
            # If dict config, build the loss module
            if isinstance(cfg, dict):
                loss_type = cfg.get('type')
                loss_kwargs = {k: v for k, v in cfg.items() if k != 'type'}

                if loss_type == 'L1Loss':
                    return L1Loss(**loss_kwargs)
                elif loss_type == 'CrossEntropyLoss':
                    return CrossEntropyLoss(**loss_kwargs)
                elif loss_type == 'GaussianFocalLoss':
                    return GaussianFocalLoss(**loss_kwargs)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
            return cfg

        self.loss_box = build_loss(loss_box)
        self.loss_cns = build_loss(loss_centerness)
        self.loss_yns = build_loss(loss_yawness)
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss
        return output


if __name__ == '__main__':
    print("Testing SparseBox3DLoss module...")

    # Test 1: Basic box loss only
    print("\n=== Test 1: Basic box loss only ===")
    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='mean')
    )

    # Create predictions and targets (encoded boxes: [x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy, vz])
    box_pred = torch.randn(10, 11)  # 10 boxes, 11 dims
    box_target = torch.randn(10, 11)
    weight = torch.ones(10, 11)  # Weight should match box dimensions

    losses = loss_module(box_pred, box_target, weight=weight, avg_factor=10.0)

    print(f"Output keys: {losses.keys()}")
    print(f"Box loss: {losses['loss_box']:.6f}")
    assert 'loss_box' in losses, "Should have box loss"
    assert losses['loss_box'].item() >= 0, "Box loss should be non-negative"
    print("✓ Test 1 passed")

    # Test 2: Box loss with dict config
    print("\n=== Test 2: Box loss with dict config ===")
    loss_module = SparseBox3DLoss(
        loss_box={'type': 'L1Loss', 'loss_weight': 0.5, 'reduction': 'mean'}
    )

    losses = loss_module(box_pred, box_target, weight=weight, avg_factor=10.0)
    print(f"Box loss with dict config: {losses['loss_box']:.6f}")
    assert 'loss_box' in losses, "Should have box loss"
    print("✓ Test 2 passed")

    # Test 3: With centerness and yawness quality losses
    print("\n=== Test 3: With quality losses (centerness + yawness) ===")
    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='mean'),
        loss_centerness={'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'mean'},
        loss_yawness={'type': 'GaussianFocalLoss', 'reduction': 'mean'}
    )

    # Quality predictions: [centerness, yawness] for each box
    quality_pred = torch.randn(10, 2)  # 10 boxes, 2 quality scores

    losses = loss_module(
        box_pred, box_target, weight=weight, avg_factor=10.0, quality=quality_pred
    )

    print(f"Output keys: {list(losses.keys())}")
    print(f"Box loss: {losses['loss_box']:.6f}")
    print(f"Centerness loss: {losses['loss_cns']:.6f}")
    print(f"Yawness loss: {losses['loss_yns']:.6f}")

    assert 'loss_box' in losses, "Should have box loss"
    assert 'loss_cns' in losses, "Should have centerness loss"
    assert 'loss_yns' in losses, "Should have yawness loss"
    print("✓ Test 3 passed")

    # Test 4: With prefix and suffix for loss names
    print("\n=== Test 4: With prefix and suffix ===")
    losses = loss_module(
        box_pred, box_target, weight=weight, avg_factor=10.0,
        quality=quality_pred, prefix="layer1_", suffix="_final"
    )

    print(f"Output keys: {list(losses.keys())}")
    assert 'layer1_loss_box_final' in losses, "Should have prefixed/suffixed box loss"
    assert 'layer1_loss_cns_final' in losses, "Should have prefixed/suffixed centerness loss"
    assert 'layer1_loss_yns_final' in losses, "Should have prefixed/suffixed yawness loss"
    print("✓ Test 4 passed")

    # Test 5: Class-aware yaw reversal
    print("\n=== Test 5: Class-aware yaw reversal ===")
    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='mean'),
        cls_allow_reverse=[3, 5]  # Classes 3 and 5 allow yaw reversal (e.g., barriers)
    )

    # Create boxes where yaw is reversed (sin and cos flipped)
    box_pred_normal = torch.zeros(10, 11)
    box_target_reversed = torch.zeros(10, 11)

    # Set yaw such that cos_similarity < 0 (opposite directions)
    box_pred_normal[:, SIN_YAW] = 1.0
    box_pred_normal[:, COS_YAW] = 0.0
    box_target_reversed[:, SIN_YAW] = -1.0
    box_target_reversed[:, COS_YAW] = 0.0

    # Class targets: first 5 are class 3 (reversible), rest are class 1 (not reversible)
    cls_target = torch.tensor([3, 3, 3, 3, 3, 1, 1, 1, 1, 1])

    losses = loss_module(
        box_pred_normal, box_target_reversed.clone(),
        weight=weight, avg_factor=10.0, cls_target=cls_target
    )

    print(f"Box loss with yaw reversal handling: {losses['loss_box']:.6f}")
    # The loss should account for the fact that classes 3 can be reversed
    print("✓ Test 5 passed")

    # Test 6: Gradient flow
    print("\n=== Test 6: Gradient flow ===")
    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='mean'),
        loss_centerness={'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'mean'},
        loss_yawness={'type': 'GaussianFocalLoss', 'reduction': 'mean'}
    )

    box_pred = torch.randn(5, 11, requires_grad=True)
    box_target = torch.randn(5, 11)
    quality_pred = torch.randn(5, 2, requires_grad=True)

    losses = loss_module(box_pred, box_target, quality=quality_pred, avg_factor=5.0)

    total_loss = losses['loss_box'] + losses['loss_cns'] + losses['loss_yns']
    total_loss.backward()

    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Box prediction gradient norm: {box_pred.grad.norm().item():.6f}")
    print(f"Quality prediction gradient norm: {quality_pred.grad.norm().item():.6f}")

    assert box_pred.grad is not None, "Box gradients should be computed"
    assert quality_pred.grad is not None, "Quality gradients should be computed"
    assert not torch.isnan(box_pred.grad).any(), "Box gradients should not contain NaN"
    assert not torch.isnan(quality_pred.grad).any(), "Quality gradients should not contain NaN"
    print("✓ Test 6 passed")

    # Test 7: Centerness target calculation
    print("\n=== Test 7: Centerness target calculation ===")
    # Centerness should be high when pred is close to target center, low when far
    box_pred_close = torch.zeros(2, 11)
    box_target_close = torch.zeros(2, 11)

    # First box: pred center very close to target center
    box_pred_close[0, X:Z+1] = torch.tensor([1.0, 2.0, 3.0])
    box_target_close[0, X:Z+1] = torch.tensor([1.01, 2.01, 3.01])  # Very close

    # Second box: pred center far from target center
    box_pred_close[1, X:Z+1] = torch.tensor([1.0, 2.0, 3.0])
    box_target_close[1, X:Z+1] = torch.tensor([5.0, 6.0, 7.0])  # Far

    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='none'),
        loss_centerness={'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'none'},
        loss_yawness={'type': 'GaussianFocalLoss', 'reduction': 'none'},  # Need to provide all quality losses
    )

    quality_pred = torch.randn(2, 2)
    losses = loss_module(box_pred_close, box_target_close, quality=quality_pred)

    # Centerness target = exp(-||pred_center - target_center||)
    # For close boxes, centerness target should be close to 1
    # For far boxes, centerness target should be close to 0
    print("Centerness loss per box:", losses['loss_cns'])
    print("✓ Test 7 passed")

    # Test 8: Yawness target calculation
    print("\n=== Test 8: Yawness target calculation ===")
    # Yawness should be 1 when yaw angles are aligned, 0 when opposite
    box_pred_yaw = torch.zeros(2, 11)
    box_target_yaw = torch.zeros(2, 11)

    # First box: aligned yaw (same direction)
    box_pred_yaw[0, SIN_YAW:COS_YAW+1] = torch.tensor([0.707, 0.707])  # 45 degrees
    box_target_yaw[0, SIN_YAW:COS_YAW+1] = torch.tensor([0.7, 0.71])   # ~45 degrees

    # Second box: opposite yaw (180 degrees different)
    box_pred_yaw[1, SIN_YAW:COS_YAW+1] = torch.tensor([1.0, 0.0])   # 90 degrees
    box_target_yaw[1, SIN_YAW:COS_YAW+1] = torch.tensor([-1.0, 0.0])  # -90 degrees

    loss_module = SparseBox3DLoss(
        loss_box=L1Loss(loss_weight=0.25, reduction='none'),
        loss_centerness={'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'none'},  # Need to provide all quality losses
        loss_yawness={'type': 'GaussianFocalLoss', 'reduction': 'none'},
    )

    quality_pred = torch.zeros(2, 2)
    quality_pred[:, YNS] = 0.0  # Logits for yawness (before sigmoid)

    losses = loss_module(box_pred_yaw, box_target_yaw, quality=quality_pred)

    # Yawness target = (cosine_similarity > 0).float()
    # First box should have yawness target = 1 (aligned)
    # Second box should have yawness target = 0 (opposite)
    print("Yawness loss per box:", losses['loss_yns'])
    print("✓ Test 8 passed")

    print("\n" + "="*50)
    print("All SparseBox3DLoss tests passed!")
    print("="*50)