# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..utils.loss_utils import weighted_loss


@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss



class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg


if __name__ == '__main__':
    print("Testing GaussianFocalLoss module...")

    # Test 1: Basic Gaussian focal loss
    print("\n=== Test 1: Basic Gaussian focal loss ===")
    criterion = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='mean')

    # Create predictions (heatmap values between 0 and 1)
    pred = torch.sigmoid(torch.randn(1, 10, 32, 32))  # Heatmap predictions
    # Create Gaussian target heatmap (peaks at 1, background near 0)
    target = torch.zeros(1, 10, 32, 32)
    target[0, 3, 16, 16] = 1.0  # Peak at center
    target[0, 7, 8, 24] = 1.0   # Another peak

    loss = criterion(pred, target)
    print(f"Input shape: pred={pred.shape}, target={target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.shape == torch.Size([]), "Loss should be scalar with mean reduction"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 1 passed")

    # Test 2: Different reduction modes
    print("\n=== Test 2: Different reduction modes ===")
    pred = torch.sigmoid(torch.randn(2, 5, 16, 16))
    target = torch.rand(2, 5, 16, 16)  # Random Gaussian targets

    criterion_none = GaussianFocalLoss(reduction='none')
    loss_none = criterion_none(pred, target)
    print(f"Reduction='none': shape={loss_none.shape}")
    assert loss_none.shape == pred.shape, "Loss should have same shape as input with none reduction"

    criterion_sum = GaussianFocalLoss(reduction='sum')
    loss_sum = criterion_sum(pred, target)
    print(f"Reduction='sum': value={loss_sum.item():.6f}")
    assert torch.allclose(loss_sum, loss_none.sum(), rtol=1e-5), "Sum reduction should equal sum of none reduction"

    criterion_mean = GaussianFocalLoss(reduction='mean')
    loss_mean = criterion_mean(pred, target)
    print(f"Reduction='mean': value={loss_mean.item():.6f}")
    assert torch.allclose(loss_mean, loss_none.mean(), rtol=1e-5), "Mean reduction should equal mean of none reduction"
    print("✓ Test 2 passed")

    # Test 3: Gradient flow
    print("\n=== Test 3: Gradient flow ===")
    pred_raw = torch.randn(1, 3, 8, 8, requires_grad=True)
    pred = torch.sigmoid(pred_raw)
    target = torch.rand(1, 3, 8, 8)
    criterion = GaussianFocalLoss(reduction='mean')
    loss = criterion(pred, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred_raw.grad.shape}")
    print(f"Gradient norm: {pred_raw.grad.norm().item():.6f}")
    assert pred_raw.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred_raw.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 3 passed")

    # Test 4: Loss behavior with positive samples (target=1)
    print("\n=== Test 4: Loss behavior with positive samples ===")
    criterion = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='none')

    # High confidence correct prediction (pred close to 1 where target=1)
    pred_correct = torch.tensor([[0.95]])
    target_pos = torch.tensor([[1.0]])
    loss_correct = criterion(pred_correct, target_pos)

    # Low confidence wrong prediction (pred close to 0 where target=1)
    pred_wrong = torch.tensor([[0.05]])
    loss_wrong = criterion(pred_wrong, target_pos)

    print(f"Loss when pred=0.95, target=1.0: {loss_correct.item():.6f}")
    print(f"Loss when pred=0.05, target=1.0: {loss_wrong.item():.6f}")
    assert loss_wrong.item() > loss_correct.item(), "Wrong prediction should have higher loss"
    print("✓ Test 4 passed")

    # Test 5: Loss behavior with negative samples (target≈0)
    print("\n=== Test 5: Loss behavior with negative samples ===")
    criterion = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='none')

    # Background: target close to 0
    target_neg = torch.tensor([[0.01]])

    # Good background prediction (pred close to 0)
    pred_good_bg = torch.tensor([[0.05]])
    loss_good_bg = criterion(pred_good_bg, target_neg)

    # Bad background prediction (pred close to 1)
    pred_bad_bg = torch.tensor([[0.95]])
    loss_bad_bg = criterion(pred_bad_bg, target_neg)

    print(f"Loss when pred=0.05, target≈0: {loss_good_bg.item():.6f}")
    print(f"Loss when pred=0.95, target≈0: {loss_bad_bg.item():.6f}")
    assert loss_bad_bg.item() > loss_good_bg.item(), "Bad background prediction should have higher loss"
    print("✓ Test 5 passed")

    # Test 6: Different alpha and gamma values
    print("\n=== Test 6: Different alpha and gamma values ===")
    pred = torch.sigmoid(torch.randn(2, 4, 16, 16))
    target = torch.rand(2, 4, 16, 16)

    loss_default = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='mean')(pred, target)
    loss_alpha_3 = GaussianFocalLoss(alpha=3.0, gamma=4.0, reduction='mean')(pred, target)
    loss_gamma_2 = GaussianFocalLoss(alpha=2.0, gamma=2.0, reduction='mean')(pred, target)

    print(f"Loss (α=2.0, γ=4.0): {loss_default.item():.6f}")
    print(f"Loss (α=3.0, γ=4.0): {loss_alpha_3.item():.6f}")
    print(f"Loss (α=2.0, γ=2.0): {loss_gamma_2.item():.6f}")
    print("✓ Test 6 passed")

    # Test 7: With sample weights
    print("\n=== Test 7: With sample weights ===")
    pred = torch.sigmoid(torch.randn(1, 2, 8, 8))
    target = torch.rand(1, 2, 8, 8)
    weight = torch.rand(1, 2, 8, 8)  # Spatial weights

    criterion = GaussianFocalLoss(reduction='mean')
    loss_weighted = criterion(pred, target, weight=weight)
    loss_unweighted = criterion(pred, target)

    print(f"Weighted loss: {loss_weighted.item():.6f}")
    print(f"Unweighted loss: {loss_unweighted.item():.6f}")
    print("✓ Test 7 passed")

    # Test 8: Loss weight parameter
    print("\n=== Test 8: Loss weight parameter ===")
    pred = torch.sigmoid(torch.randn(1, 3, 8, 8))
    target = torch.rand(1, 3, 8, 8)

    loss_weight_1 = GaussianFocalLoss(loss_weight=1.0, reduction='mean')(pred, target)
    loss_weight_2 = GaussianFocalLoss(loss_weight=2.0, reduction='mean')(pred, target)
    loss_weight_05 = GaussianFocalLoss(loss_weight=0.5, reduction='mean')(pred, target)

    assert torch.allclose(loss_weight_2, loss_weight_1 * 2.0, rtol=1e-5), "Loss weight should scale the loss"
    assert torch.allclose(loss_weight_05, loss_weight_1 * 0.5, rtol=1e-5), "Loss weight should scale the loss"
    print(f"Loss weight 1.0: {loss_weight_1.item():.6f}")
    print(f"Loss weight 2.0: {loss_weight_2.item():.6f}")
    print(f"Loss weight 0.5: {loss_weight_05.item():.6f}")
    print("✓ Test 8 passed")

    # Test 9: With avg_factor
    print("\n=== Test 9: With avg_factor ===")
    pred = torch.sigmoid(torch.randn(1, 2, 8, 8))
    target = torch.rand(1, 2, 8, 8)
    avg_factor = 50.0

    criterion = GaussianFocalLoss(reduction='mean')
    loss = criterion(pred, target, avg_factor=avg_factor)

    # With avg_factor, loss = sum / avg_factor
    criterion_none = GaussianFocalLoss(reduction='none')
    loss_none = criterion_none(pred, target)
    expected_loss = loss_none.sum() / avg_factor
    assert torch.allclose(loss, expected_loss, rtol=1e-5), "Loss with avg_factor should match manual calculation"
    print(f"Loss with avg_factor={avg_factor}: {loss.item():.6f}")
    print("✓ Test 9 passed")

    # Test 10: Realistic heatmap scenario
    print("\n=== Test 10: Realistic heatmap scenario ===")
    # Simulate a keypoint heatmap detection scenario
    batch_size, num_keypoints, H, W = 2, 10, 64, 64
    pred = torch.sigmoid(torch.randn(batch_size, num_keypoints, H, W))

    # Create Gaussian target heatmap with a few keypoints
    target = torch.zeros(batch_size, num_keypoints, H, W)
    # Add Gaussian peaks for keypoints
    def add_gaussian_peak(heatmap, center_x, center_y, sigma=2.0):
        x = torch.arange(0, W, dtype=torch.float32)
        y = torch.arange(0, H, dtype=torch.float32)
        y = y.unsqueeze(1)
        gaussian = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return torch.maximum(heatmap, gaussian)

    target[0, 0] = add_gaussian_peak(target[0, 0], 32, 32)  # Center keypoint
    target[0, 3] = add_gaussian_peak(target[0, 3], 16, 48)  # Off-center keypoint
    target[1, 5] = add_gaussian_peak(target[1, 5], 48, 16)  # Another keypoint

    criterion = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='mean')
    loss = criterion(pred, target)

    print(f"Realistic heatmap loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Test 10 passed")

    print("\n" + "="*50)
    print("All GaussianFocalLoss tests passed!")
    print("="*50)
