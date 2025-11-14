import torch
import torch.nn as nn
from .l1_loss import smooth_l1_loss, l1_loss, L1Loss, SmoothL1Loss

'''
  loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),

'''

class SparseLineLoss(nn.Module):
    def __init__(
        self,
        loss_line,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super().__init__()

        # Build loss module from either nn.Module instance or dict config
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
                elif loss_type == 'SmoothL1Loss':
                    return SmoothL1Loss(**loss_kwargs)
                elif loss_type == 'LinesL1Loss':
                    return LinesL1Loss(**loss_kwargs)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
            return cfg

        self.loss_line = build_loss(loss_line)
        self.num_sample = num_sample
        self.roi_size = roi_size

    def forward(
        self,
        line,
        line_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):

        output = {}
        line = self.normalize_line(line)
        line_target = self.normalize_line(line_target)
        line_loss = self.loss_line(
            line, line_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_line{suffix}"] = line_loss

        return output

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line



class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


if __name__ == '__main__':
    print("Testing sparse_line module...")

    # Test 1: LinesL1Loss basic functionality
    print("\n=== Test 1: LinesL1Loss basic ===")
    loss_module = LinesL1Loss(reduction='mean', loss_weight=1.0, beta=0.5)

    # 10 lines, 40 coordinates (20 points)
    pred = torch.randn(10, 40)
    target = torch.randn(10, 40)

    loss = loss_module(pred, target)
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")

    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 1 passed")

    # Test 2: LinesL1Loss with weight
    print("\n=== Test 2: LinesL1Loss with weight ===")
    pred = torch.randn(8, 40)
    target = torch.randn(8, 40)
    weight = torch.rand(8, 40)

    loss_weighted = loss_module(pred, target, weight=weight)
    loss_unweighted = loss_module(pred, target)

    print(f"Weighted loss: {loss_weighted.item():.6f}")
    print(f"Unweighted loss: {loss_unweighted.item():.6f}")
    print("✓ Test 2 passed")

    # Test 3: LinesL1Loss with avg_factor
    print("\n=== Test 3: LinesL1Loss with avg_factor ===")
    pred = torch.randn(5, 40)
    target = torch.randn(5, 40)
    avg_factor = 10.0

    loss = loss_module(pred, target, avg_factor=avg_factor)
    print(f"Loss with avg_factor={avg_factor}: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 3 passed")

    # Test 4: LinesL1Loss with beta=0 (pure L1)
    print("\n=== Test 4: LinesL1Loss with beta=0 (pure L1) ===")
    loss_module_l1 = LinesL1Loss(reduction='mean', loss_weight=1.0, beta=0.0)

    pred = torch.randn(6, 40)
    target = torch.randn(6, 40)

    loss = loss_module_l1(pred, target)
    print(f"Pure L1 loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 4 passed")

    # Test 5: LinesL1Loss gradient flow
    print("\n=== Test 5: LinesL1Loss gradient flow ===")
    pred = torch.randn(5, 40, requires_grad=True)
    target = torch.randn(5, 40)

    loss = loss_module(pred, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred.grad.shape}")
    assert pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 5 passed")

    # Test 6: SparseLineLoss initialization with dict config
    print("\n=== Test 6: SparseLineLoss with dict config ===")
    sparse_loss = SparseLineLoss(
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=20,
        roi_size=(30, 60),
    )

    assert isinstance(sparse_loss.loss_line, LinesL1Loss), "Should build LinesL1Loss"
    assert sparse_loss.num_sample == 20, "num_sample should be set correctly"
    assert sparse_loss.roi_size == (30, 60), "roi_size should be set correctly"
    print("✓ Test 6 passed")

    # Test 7: SparseLineLoss initialization with instance
    print("\n=== Test 7: SparseLineLoss with loss instance ===")
    loss_inst = LinesL1Loss(loss_weight=5.0, beta=0.5)
    sparse_loss = SparseLineLoss(
        loss_line=loss_inst,
        num_sample=20,
        roi_size=(30, 60),
    )

    assert sparse_loss.loss_line is loss_inst, "Should use the provided instance"
    print("✓ Test 7 passed")

    # Test 8: SparseLineLoss normalize_line
    print("\n=== Test 8: SparseLineLoss normalize_line ===")
    sparse_loss = SparseLineLoss(
        loss_line=dict(type='LinesL1Loss', loss_weight=1.0),
        num_sample=20,
        roi_size=(30, 60),
    )

    # Create lines in BEV coordinates (center at origin)
    # roi_size = (30, 60) means x in [-15, 15], y in [-30, 30]
    lines = torch.tensor([[0.0, 0.0] * 20])  # Center point repeated (batch=1)
    normalized = sparse_loss.normalize_line(lines)

    print(f"Original line shape: {lines.shape}")
    print(f"Normalized line shape: {normalized.shape}")
    print(f"Center point (0,0) normalized to: ({normalized[0, 0]:.3f}, {normalized[0, 1]:.3f})")

    # Center (0, 0) should normalize to approximately (0.5, 0.5)
    assert normalized.shape == lines.shape, "Shape should be preserved"
    assert 0.4 < normalized[0, 0] < 0.6, "x should be around 0.5"
    assert 0.4 < normalized[0, 1] < 0.6, "y should be around 0.5"
    print("✓ Test 8 passed")

    # Test 9: SparseLineLoss forward pass
    print("\n=== Test 9: SparseLineLoss forward pass ===")
    sparse_loss = SparseLineLoss(
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=20,
        roi_size=(30, 60),
    )

    # Batch of 4 lines, 40 coordinates (20 points x 2)
    line_pred = torch.randn(4, 40)
    line_target = torch.randn(4, 40)

    output = sparse_loss(line_pred, line_target, avg_factor=4.0)

    print(f"Output keys: {list(output.keys())}")
    print(f"Loss value: {output['loss_line']:.6f}")

    assert 'loss_line' in output, "Should have loss_line in output"
    assert output['loss_line'].item() >= 0, "Loss should be non-negative"
    print("✓ Test 9 passed")

    # Test 10: SparseLineLoss with prefix and suffix
    print("\n=== Test 10: SparseLineLoss with prefix/suffix ===")
    line_pred = torch.randn(3, 40)
    line_target = torch.randn(3, 40)

    output = sparse_loss(line_pred, line_target, prefix="map_", suffix="_final")

    print(f"Output keys: {list(output.keys())}")
    assert 'map_loss_line_final' in output, "Should have prefixed/suffixed key"
    print("✓ Test 10 passed")

    # Test 11: SparseLineLoss with empty lines
    print("\n=== Test 11: SparseLineLoss with empty lines ===")
    line_pred = torch.empty(0, 40)
    line_target = torch.empty(0, 40)

    output = sparse_loss(line_pred, line_target)

    print(f"Loss with empty lines: {output['loss_line'].item():.6f}")
    assert output['loss_line'].item() == 0.0, "Loss should be 0 for empty lines"
    print("✓ Test 11 passed")

    # Test 12: Normalization correctness
    print("\n=== Test 12: Normalization corner cases ===")
    sparse_loss = SparseLineLoss(
        loss_line=dict(type='LinesL1Loss', loss_weight=1.0),
        num_sample=20,
        roi_size=(30, 60),
    )

    # Test corners of ROI
    # Top-left: (-15, -30), Top-right: (15, -30)
    # Bottom-left: (-15, 30), Bottom-right: (15, 30)
    corners = torch.tensor([
        [-15.0, -30.0] * 20,  # Top-left (should -> ~0, ~0)
        [15.0, 30.0] * 20,    # Bottom-right (should -> ~1, ~1)
    ])

    normalized = sparse_loss.normalize_line(corners)

    print(f"Top-left (-15, -30) -> ({normalized[0, 0]:.3f}, {normalized[0, 1]:.3f})")
    print(f"Bottom-right (15, 30) -> ({normalized[1, 0]:.3f}, {normalized[1, 1]:.3f})")

    # Top-left should be close to (0, 0)
    assert -0.1 < normalized[0, 0] < 0.1, "Top-left x should be near 0"
    assert -0.1 < normalized[0, 1] < 0.1, "Top-left y should be near 0"

    # Bottom-right should be close to (1, 1)
    assert 0.9 < normalized[1, 0] < 1.1, "Bottom-right x should be near 1"
    assert 0.9 < normalized[1, 1] < 1.1, "Bottom-right y should be near 1"
    print("✓ Test 12 passed")

    # Test 13: Loss weight scaling
    print("\n=== Test 13: Loss weight scaling ===")
    pred = torch.randn(5, 40)
    target = torch.randn(5, 40)

    loss_w1 = LinesL1Loss(loss_weight=1.0, beta=0.5)(pred, target)
    loss_w2 = LinesL1Loss(loss_weight=2.0, beta=0.5)(pred, target)
    loss_w05 = LinesL1Loss(loss_weight=0.5, beta=0.5)(pred, target)

    print(f"Loss weight 1.0: {loss_w1.item():.6f}")
    print(f"Loss weight 2.0: {loss_w2.item():.6f}")
    print(f"Loss weight 0.5: {loss_w05.item():.6f}")

    assert torch.allclose(loss_w2, loss_w1 * 2.0, rtol=1e-5), "Weight should scale loss"
    assert torch.allclose(loss_w05, loss_w1 * 0.5, rtol=1e-5), "Weight should scale loss"
    print("✓ Test 13 passed")

    print("\n" + "="*50)
    print("All sparse_line tests passed!")
    print("="*50)