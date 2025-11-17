# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA focal loss
try:
    from ...ops import sigmoid_focal_loss as _sigmoid_focal_loss, FOCAL_LOSS_AVAILABLE
except ImportError:
    _sigmoid_focal_loss = None
    FOCAL_LOSS_AVAILABLE = False


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
        avg_factor (float): Avarage factor when computing the mean of losses.

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

# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of CUDA version `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Falls back to PyTorch version if CUDA extension is not available or input is on CPU.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Check if we can use CUDA version
    # CUDA kernel doesn't support BFloat16, so fall back for bfloat16
    use_cuda = (FOCAL_LOSS_AVAILABLE and
                pred.is_cuda and
                _sigmoid_focal_loss is not None and
                pred.dtype != torch.bfloat16)

    if use_cuda:
        # Function.apply does not accept keyword arguments, so the decorator
        # "weighted_loss" is not applicable
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                                   alpha, None, 'none')
    else:
        # Fall back to PyTorch version
        if not FOCAL_LOSS_AVAILABLE:
            import warnings
            warnings.warn(
                "CUDA focal loss is not available, using PyTorch fallback. "
                "For better performance, compile CUDA extension: "
                "cd reimplementation/ops && python setup.py install",
                UserWarning
            )
        elif pred.dtype == torch.bfloat16:
            # Note: Using PyTorch fallback for BFloat16 (CUDA kernel doesn't support it)
            pass

        # PyTorch fallback expects one-hot encoded targets
        # CUDA version uses class indices, so convert if needed
        if target.dtype == torch.long and target.dim() == 1:
            num_classes = pred.size(1)
            target_one_hot = torch.zeros_like(pred)
            # Only create one-hot for valid classes (< num_classes)
            # Background class (>= num_classes) stays as all-zeros
            valid_mask = target < num_classes
            if valid_mask.any():
                target_one_hot[valid_mask] = F.one_hot(
                    target[valid_mask], num_classes=num_classes
                ).to(pred.dtype)  # Match pred dtype (BF16, FP16, or FP32)
            target = target_one_hot

        return py_sigmoid_focal_loss(pred, target, weight, gamma, alpha, reduction, avg_factor)

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss




class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
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
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                # BUG FIX: Correctly handle background class (target >= num_classes)
                # Background queries should not be positive for any class, not all-zeros
                num_classes = pred.size(1)
                target_one_hot = torch.zeros_like(pred)
                # Only create one-hot for valid classes (< num_classes)
                # Background class (>= num_classes) stays as all-zeros, which is correct
                valid_mask = target < num_classes
                if valid_mask.any():
                    target_one_hot[valid_mask] = F.one_hot(
                        target[valid_mask], num_classes=num_classes
                    ).float()
                target = target_one_hot
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


if __name__ == '__main__':
    print("Testing FocalLoss module...")

    # Test 1: Basic focal loss computation
    print("\n=== Test 1: Basic focal loss (CPU) ===")
    criterion = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean')
    pred = torch.randn(100, 10)  # (N, num_classes)
    target = torch.randint(0, 10, (100,))  # (N,)
    loss = criterion(pred, target)
    print(f"Input shape: pred={pred.shape}, target={target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.shape == torch.Size([]), "Loss should be scalar with mean reduction"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test 1 passed")

    # Test 2: Gradient flow
    print("\n=== Test 2: Gradient flow ===")
    pred = torch.randn(50, 5, requires_grad=True)
    target = torch.randint(0, 5, (50,))
    criterion = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean')
    loss = criterion(pred, target)
    loss.backward()
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient norm: {pred.grad.norm().item():.6f}")
    assert pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 2 passed")

    # Test 3: Different reduction modes
    print("\n=== Test 3: Different reduction modes ===")
    pred = torch.randn(20, 3)
    target = torch.randint(0, 3, (20,))

    criterion_none = FocalLoss(use_sigmoid=True, reduction='none')
    loss_none = criterion_none(pred, target)
    print(f"Reduction='none': shape={loss_none.shape}, mean={loss_none.mean().item():.6f}")
    # Focal loss with 'none' reduction returns (N, C) shape (per-sample, per-class loss)
    assert loss_none.shape == torch.Size([20, 3]), "Loss should have shape (N, C) with none reduction"

    criterion_sum = FocalLoss(use_sigmoid=True, reduction='sum')
    loss_sum = criterion_sum(pred, target)
    print(f"Reduction='sum': shape={loss_sum.shape}, value={loss_sum.item():.6f}")
    assert loss_sum.shape == torch.Size([]), "Loss should be scalar with sum reduction"

    criterion_mean = FocalLoss(use_sigmoid=True, reduction='mean')
    loss_mean = criterion_mean(pred, target)
    print(f"Reduction='mean': shape={loss_mean.shape}, value={loss_mean.item():.6f}")
    assert loss_mean.shape == torch.Size([]), "Loss should be scalar with mean reduction"

    # Verify relationships
    assert torch.allclose(loss_sum, loss_none.sum(), rtol=1e-5), "Sum reduction should equal sum of none reduction"
    print("✓ Test 3 passed")

    # Test 4: With sample weights
    print("\n=== Test 4: With sample weights ===")
    pred = torch.randn(30, 4)
    target = torch.randint(0, 4, (30,))
    weight = torch.rand(30)  # Sample-wise weights

    criterion = FocalLoss(use_sigmoid=True, reduction='mean')
    loss_weighted = criterion(pred, target, weight=weight)
    loss_unweighted = criterion(pred, target)

    print(f"Weighted loss: {loss_weighted.item():.6f}")
    print(f"Unweighted loss: {loss_unweighted.item():.6f}")
    assert loss_weighted.item() != loss_unweighted.item(), "Weighted and unweighted losses should differ"
    print("✓ Test 4 passed")

    # Test 5: CUDA version (if available)
    if torch.cuda.is_available() and FOCAL_LOSS_AVAILABLE:
        print("\n=== Test 5: CUDA version ===")
        pred_cuda = torch.randn(100, 10, device='cuda')
        target_cuda = torch.randint(0, 10, (100,), device='cuda')

        criterion = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean')
        loss_cuda = criterion(pred_cuda, target_cuda)

        # Compare with CPU version
        pred_cpu = pred_cuda.cpu()
        target_cpu = target_cuda.cpu()
        loss_cpu = criterion(pred_cpu, target_cpu)

        print(f"CUDA loss: {loss_cuda.item():.6f}")
        print(f"CPU loss: {loss_cpu.item():.6f}")
        print(f"Difference: {abs(loss_cuda.item() - loss_cpu.item()):.6e}")

        # Results should be close but may have small numerical differences
        assert torch.allclose(loss_cuda.cpu(), loss_cpu, rtol=1e-3, atol=1e-5), \
            "CUDA and CPU versions should produce similar results"
        print("✓ Test 5 passed (CUDA extension working)")
    else:
        if not torch.cuda.is_available():
            print("\n=== Test 5: CUDA not available, skipping ===")
        else:
            print("\n=== Test 5: CUDA focal loss extension not compiled ===")
            print("Using PyTorch fallback (expect warning above)")
            pred = torch.randn(100, 10)
            target = torch.randint(0, 10, (100,))
            criterion = FocalLoss(use_sigmoid=True, reduction='mean')
            loss = criterion(pred, target)
            print(f"Fallback loss: {loss.item():.6f}")
            print("✓ Test 5 passed (fallback working)")

    # Test 6: Edge cases
    print("\n=== Test 6: Edge cases ===")

    # All correct predictions (high confidence)
    pred_correct = torch.zeros(10, 3)
    target_correct = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    for i, t in enumerate(target_correct):
        pred_correct[i, t] = 10.0  # High confidence

    criterion = FocalLoss(use_sigmoid=True, reduction='mean')
    loss_correct = criterion(pred_correct, target_correct)
    print(f"Loss for correct predictions: {loss_correct.item():.6f}")

    # All wrong predictions (high confidence)
    pred_wrong = torch.zeros(10, 3)
    target_wrong = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    for i, t in enumerate(target_wrong):
        wrong_class = (t + 1) % 3
        pred_wrong[i, wrong_class] = 10.0  # High confidence on wrong class

    loss_wrong = criterion(pred_wrong, target_wrong)
    print(f"Loss for wrong predictions: {loss_wrong.item():.6f}")

    assert loss_correct < loss_wrong, "Correct predictions should have lower loss"
    print("✓ Test 6 passed")

    print("\n" + "="*50)
    print("All FocalLoss tests passed!")
    print("="*50)

    if FOCAL_LOSS_AVAILABLE:
        print("\n✓ CUDA focal loss extension is available and working")
    else:
        print("\n⚠ CUDA focal loss extension not available, using PyTorch fallback")
        print("  To enable CUDA acceleration:")
        print("  cd reimplementation/ops && python setup.py install")
