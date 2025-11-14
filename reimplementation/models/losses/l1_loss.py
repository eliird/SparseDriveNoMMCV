# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    Args:
        loss_func (function): A function that computes element-wise loss.

    Returns:
        function: A wrapped function with weight and reduction.
    """
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        # Get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        # Apply weighting and reduction
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
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
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox




class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
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
            target (torch.Tensor): The learning target of the prediction.
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
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


if __name__ == '__main__':
    print("Testing L1Loss and SmoothL1Loss modules...")

    # Test 1: Basic L1 loss
    print("\n=== Test 1: Basic L1 loss ===")
    criterion = L1Loss(reduction='mean', loss_weight=1.0)
    pred = torch.randn(10, 5)
    target = torch.randn(10, 5)
    loss = criterion(pred, target)
    print(f"Input shape: pred={pred.shape}, target={target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.shape == torch.Size([]), "Loss should be scalar with mean reduction"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify it's actually L1
    expected_loss = torch.abs(pred - target).mean()
    assert torch.allclose(loss, expected_loss), "L1 loss should match manual calculation"
    print("✓ Test 1 passed")

    # Test 2: L1 loss with different reductions
    print("\n=== Test 2: L1 loss with different reductions ===")
    pred = torch.randn(8, 3)
    target = torch.randn(8, 3)

    criterion_none = L1Loss(reduction='none')
    loss_none = criterion_none(pred, target)
    print(f"Reduction='none': shape={loss_none.shape}")
    assert loss_none.shape == pred.shape, "Loss should have same shape as input with none reduction"

    criterion_sum = L1Loss(reduction='sum')
    loss_sum = criterion_sum(pred, target)
    print(f"Reduction='sum': value={loss_sum.item():.6f}")
    assert torch.allclose(loss_sum, loss_none.sum()), "Sum reduction should equal sum of none reduction"

    criterion_mean = L1Loss(reduction='mean')
    loss_mean = criterion_mean(pred, target)
    print(f"Reduction='mean': value={loss_mean.item():.6f}")
    assert torch.allclose(loss_mean, loss_none.mean()), "Mean reduction should equal mean of none reduction"
    print("✓ Test 2 passed")

    # Test 3: L1 loss with weights
    print("\n=== Test 3: L1 loss with weights ===")
    pred = torch.randn(5, 4)
    target = torch.randn(5, 4)
    weight = torch.rand(5, 4)

    criterion = L1Loss(reduction='mean')
    loss_weighted = criterion(pred, target, weight=weight)
    loss_unweighted = criterion(pred, target)

    print(f"Weighted loss: {loss_weighted.item():.6f}")
    print(f"Unweighted loss: {loss_unweighted.item():.6f}")

    # Manual calculation
    expected_weighted = (torch.abs(pred - target) * weight).mean()
    assert torch.allclose(loss_weighted, expected_weighted), "Weighted loss should match manual calculation"
    print("✓ Test 3 passed")

    # Test 4: L1 loss with avg_factor
    print("\n=== Test 4: L1 loss with avg_factor ===")
    pred = torch.randn(10, 3)
    target = torch.randn(10, 3)
    avg_factor = 15.0

    criterion = L1Loss(reduction='mean')
    loss = criterion(pred, target, avg_factor=avg_factor)

    # With avg_factor, loss = sum / avg_factor
    expected_loss = torch.abs(pred - target).sum() / avg_factor
    assert torch.allclose(loss, expected_loss), "Loss with avg_factor should match manual calculation"
    print(f"Loss with avg_factor={avg_factor}: {loss.item():.6f}")
    print("✓ Test 4 passed")

    # Test 5: L1 loss gradient flow
    print("\n=== Test 5: L1 loss gradient flow ===")
    pred = torch.randn(6, 4, requires_grad=True)
    target = torch.randn(6, 4)
    criterion = L1Loss(reduction='mean')
    loss = criterion(pred, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient norm: {pred.grad.norm().item():.6f}")
    assert pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 5 passed")

    # Test 6: Basic Smooth L1 loss
    print("\n=== Test 6: Basic Smooth L1 loss ===")
    criterion = SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)
    pred = torch.randn(10, 5)
    target = torch.randn(10, 5)
    loss = criterion(pred, target)
    print(f"Input shape: pred={pred.shape}, target={target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    assert loss.shape == torch.Size([]), "Loss should be scalar with mean reduction"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 6 passed")

    # Test 7: Smooth L1 loss behavior (quadratic vs linear)
    print("\n=== Test 7: Smooth L1 loss behavior ===")
    criterion = SmoothL1Loss(beta=1.0, reduction='none')

    # Small difference (< beta): quadratic
    pred_small = torch.tensor([[0.0, 0.0]])
    target_small = torch.tensor([[0.3, 0.5]])  # diff = 0.3, 0.5 (both < 1.0)
    loss_small = criterion(pred_small, target_small)

    # Manual calculation for diff < beta: 0.5 * diff^2 / beta
    diff_small = torch.abs(pred_small - target_small)
    expected_small = 0.5 * diff_small * diff_small / 1.0
    assert torch.allclose(loss_small, expected_small), "Smooth L1 should be quadratic for small differences"
    print(f"Small diff (< beta): {diff_small.numpy()}, loss: {loss_small.numpy()}")

    # Large difference (>= beta): linear
    pred_large = torch.tensor([[0.0, 0.0]])
    target_large = torch.tensor([[2.0, 3.0]])  # diff = 2.0, 3.0 (both >= 1.0)
    loss_large = criterion(pred_large, target_large)

    # Manual calculation for diff >= beta: diff - 0.5 * beta
    diff_large = torch.abs(pred_large - target_large)
    expected_large = diff_large - 0.5 * 1.0
    assert torch.allclose(loss_large, expected_large), "Smooth L1 should be linear for large differences"
    print(f"Large diff (>= beta): {diff_large.numpy()}, loss: {loss_large.numpy()}")
    print("✓ Test 7 passed")

    # Test 8: Smooth L1 loss with different beta values
    print("\n=== Test 8: Smooth L1 loss with different beta ===")
    pred = torch.randn(5, 3)
    target = torch.randn(5, 3)

    loss_beta_05 = SmoothL1Loss(beta=0.5, reduction='mean')(pred, target)
    loss_beta_10 = SmoothL1Loss(beta=1.0, reduction='mean')(pred, target)
    loss_beta_20 = SmoothL1Loss(beta=2.0, reduction='mean')(pred, target)

    print(f"Beta=0.5: {loss_beta_05.item():.6f}")
    print(f"Beta=1.0: {loss_beta_10.item():.6f}")
    print(f"Beta=2.0: {loss_beta_20.item():.6f}")
    print("✓ Test 8 passed")

    # Test 9: Smooth L1 loss gradient flow
    print("\n=== Test 9: Smooth L1 loss gradient flow ===")
    pred = torch.randn(8, 4, requires_grad=True)
    target = torch.randn(8, 4)
    criterion = SmoothL1Loss(beta=1.0, reduction='mean')
    loss = criterion(pred, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient norm: {pred.grad.norm().item():.6f}")
    assert pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 9 passed")

    # Test 10: Loss weight parameter
    print("\n=== Test 10: Loss weight parameter ===")
    pred = torch.randn(5, 3)
    target = torch.randn(5, 3)

    loss_weight_1 = L1Loss(loss_weight=1.0, reduction='mean')(pred, target)
    loss_weight_2 = L1Loss(loss_weight=2.0, reduction='mean')(pred, target)
    loss_weight_05 = L1Loss(loss_weight=0.5, reduction='mean')(pred, target)

    assert torch.allclose(loss_weight_2, loss_weight_1 * 2.0), "Loss weight should scale the loss"
    assert torch.allclose(loss_weight_05, loss_weight_1 * 0.5), "Loss weight should scale the loss"
    print(f"Loss weight 1.0: {loss_weight_1.item():.6f}")
    print(f"Loss weight 2.0: {loss_weight_2.item():.6f}")
    print(f"Loss weight 0.5: {loss_weight_05.item():.6f}")
    print("✓ Test 10 passed")

    # Test 11: Empty target handling
    print("\n=== Test 11: Empty target handling ===")
    pred = torch.randn(5, 3)
    target = torch.randn(0, 3)  # Empty

    criterion_l1 = L1Loss(reduction='mean')
    loss_l1 = criterion_l1(pred[:0], target)
    print(f"L1 loss with empty target: {loss_l1.item():.6f}")
    assert loss_l1.item() == 0.0, "Loss should be 0 for empty target"

    criterion_smooth = SmoothL1Loss(beta=1.0, reduction='mean')
    loss_smooth = criterion_smooth(pred[:0], target)
    print(f"Smooth L1 loss with empty target: {loss_smooth.item():.6f}")
    assert loss_smooth.item() == 0.0, "Loss should be 0 for empty target"
    print("✓ Test 11 passed")

    print("\n" + "="*50)
    print("All L1Loss and SmoothL1Loss tests passed!")
    print("="*50)
