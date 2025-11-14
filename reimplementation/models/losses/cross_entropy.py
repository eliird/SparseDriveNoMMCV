# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.loss_utils import weight_reduce_loss

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),  # only accepts float type tensor
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert not (
            self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


if __name__ == '__main__':
    print("Testing CrossEntropyLoss module...")

    # Test 1: Basic cross entropy loss
    print("\n=== Test 1: Basic cross entropy loss ===")
    criterion = CrossEntropyLoss(use_sigmoid=False, reduction='mean')
    pred = torch.randn(10, 5)  # (N, num_classes)
    label = torch.randint(0, 5, (10,))  # (N,)
    loss = criterion(pred, label)
    print(f"Input shape: pred={pred.shape}, label={label.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.shape == torch.Size([]), "Loss should be scalar with mean reduction"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify it matches PyTorch's cross entropy
    expected_loss = F.cross_entropy(pred, label, reduction='mean')
    assert torch.allclose(loss, expected_loss), "Cross entropy should match PyTorch"
    print("✓ Test 1 passed")

    # Test 2: Cross entropy with different reductions
    print("\n=== Test 2: Cross entropy with different reductions ===")
    pred = torch.randn(8, 4)
    label = torch.randint(0, 4, (8,))

    criterion_none = CrossEntropyLoss(reduction='none')
    loss_none = criterion_none(pred, label)
    print(f"Reduction='none': shape={loss_none.shape}")
    assert loss_none.shape == torch.Size([8]), "Loss should have shape (N,) with none reduction"

    criterion_sum = CrossEntropyLoss(reduction='sum')
    loss_sum = criterion_sum(pred, label)
    print(f"Reduction='sum': value={loss_sum.item():.6f}")
    assert torch.allclose(loss_sum, loss_none.sum()), "Sum reduction should equal sum of none reduction"

    criterion_mean = CrossEntropyLoss(reduction='mean')
    loss_mean = criterion_mean(pred, label)
    print(f"Reduction='mean': value={loss_mean.item():.6f}")
    assert torch.allclose(loss_mean, loss_none.mean()), "Mean reduction should equal mean of none reduction"
    print("✓ Test 2 passed")

    # Test 3: Cross entropy with class weights
    print("\n=== Test 3: Cross entropy with class weights ===")
    pred = torch.randn(10, 3)
    label = torch.randint(0, 3, (10,))
    class_weight = [1.0, 2.0, 3.0]  # Different weights for each class

    criterion_weighted = CrossEntropyLoss(reduction='none', class_weight=class_weight)
    loss_weighted = criterion_weighted(pred, label)

    criterion_unweighted = CrossEntropyLoss(reduction='none')
    loss_unweighted = criterion_unweighted(pred, label)

    print(f"Weighted loss (sample 0): {loss_weighted[0].item():.6f}")
    print(f"Unweighted loss (sample 0): {loss_unweighted[0].item():.6f}")

    # With reduction='none', verify individual samples match
    expected_loss = F.cross_entropy(pred, label, weight=pred.new_tensor(class_weight), reduction='none')
    assert torch.allclose(loss_weighted, expected_loss, rtol=1e-5), "Weighted cross entropy should match PyTorch"

    # Verify class weighting works: weighted_loss[i] = unweighted_loss[i] * class_weight[label[i]]
    for i in range(len(label)):
        expected_weighted = loss_unweighted[i] * class_weight[label[i]]
        assert torch.allclose(loss_weighted[i], expected_weighted, rtol=1e-5), f"Sample {i} should be weighted correctly"
    print("✓ Test 3 passed")

    # Test 4: Cross entropy with sample weights
    print("\n=== Test 4: Cross entropy with sample weights ===")
    pred = torch.randn(6, 4)
    label = torch.randint(0, 4, (6,))
    weight = torch.rand(6)  # Sample-wise weights

    criterion = CrossEntropyLoss(reduction='mean')
    loss_weighted = criterion(pred, label, weight=weight)
    loss_unweighted = criterion(pred, label)

    print(f"Sample-weighted loss: {loss_weighted.item():.6f}")
    print(f"Unweighted loss: {loss_unweighted.item():.6f}")
    print("✓ Test 4 passed")

    # Test 5: Cross entropy gradient flow
    print("\n=== Test 5: Cross entropy gradient flow ===")
    pred = torch.randn(5, 3, requires_grad=True)
    label = torch.randint(0, 3, (5,))
    criterion = CrossEntropyLoss(reduction='mean')
    loss = criterion(pred, label)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient norm: {pred.grad.norm().item():.6f}")
    assert pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 5 passed")

    # Test 6: Soft cross entropy (for mixup)
    print("\n=== Test 6: Soft cross entropy ===")
    criterion = CrossEntropyLoss(use_soft=True, reduction='mean')
    pred = torch.randn(8, 4)
    label = torch.randn(8, 4).softmax(dim=-1)  # Soft labels (probabilities)

    loss = criterion(pred, label)
    print(f"Soft cross entropy loss: {loss.item():.6f}")
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Manual calculation: -sum(label * log_softmax(pred))
    expected_loss = -(label * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(loss, expected_loss), "Soft cross entropy should match manual calculation"
    print("✓ Test 6 passed")

    # Test 7: Binary cross entropy (use_sigmoid=True)
    print("\n=== Test 7: Binary cross entropy ===")
    criterion = CrossEntropyLoss(use_sigmoid=True, reduction='mean')
    pred = torch.randn(10, 3)  # Logits
    label = torch.randint(0, 2, (10, 3))  # Binary labels

    loss = criterion(pred, label)
    print(f"Binary cross entropy loss: {loss.item():.6f}")
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify with PyTorch
    expected_loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='mean')
    assert torch.allclose(loss, expected_loss), "Binary cross entropy should match PyTorch"
    print("✓ Test 7 passed")

    # Test 8: Binary cross entropy with pos_weight
    print("\n=== Test 8: Binary cross entropy with pos_weight ===")
    pos_weight = [2.0, 1.0, 3.0]  # Positive class weights for imbalanced data
    criterion = CrossEntropyLoss(use_sigmoid=True, reduction='mean', pos_weight=pos_weight)
    pred = torch.randn(10, 3)
    label = torch.randint(0, 2, (10, 3))

    loss_with_pos_weight = criterion(pred, label)
    criterion_no_pos = CrossEntropyLoss(use_sigmoid=True, reduction='mean')
    loss_no_pos = criterion_no_pos(pred, label)

    print(f"Loss with pos_weight: {loss_with_pos_weight.item():.6f}")
    print(f"Loss without pos_weight: {loss_no_pos.item():.6f}")

    # Verify with PyTorch
    expected_loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=pred.new_tensor(pos_weight), reduction='mean'
    )
    assert torch.allclose(loss_with_pos_weight, expected_loss, rtol=1e-5), "BCE with pos_weight should match PyTorch"
    print("✓ Test 8 passed")

    # Test 9: Cross entropy with avg_factor
    print("\n=== Test 9: Cross entropy with avg_factor ===")
    pred = torch.randn(10, 5)
    label = torch.randint(0, 5, (10,))
    avg_factor = 15.0

    criterion = CrossEntropyLoss(reduction='mean')
    loss = criterion(pred, label, avg_factor=avg_factor)

    # With avg_factor, loss = sum / avg_factor
    loss_none = F.cross_entropy(pred, label, reduction='none')
    expected_loss = loss_none.sum() / avg_factor
    assert torch.allclose(loss, expected_loss), "Loss with avg_factor should match manual calculation"
    print(f"Loss with avg_factor={avg_factor}: {loss.item():.6f}")
    print("✓ Test 9 passed")

    # Test 10: Loss weight parameter
    print("\n=== Test 10: Loss weight parameter ===")
    pred = torch.randn(8, 4)
    label = torch.randint(0, 4, (8,))

    loss_weight_1 = CrossEntropyLoss(loss_weight=1.0, reduction='mean')(pred, label)
    loss_weight_2 = CrossEntropyLoss(loss_weight=2.0, reduction='mean')(pred, label)
    loss_weight_05 = CrossEntropyLoss(loss_weight=0.5, reduction='mean')(pred, label)

    assert torch.allclose(loss_weight_2, loss_weight_1 * 2.0), "Loss weight should scale the loss"
    assert torch.allclose(loss_weight_05, loss_weight_1 * 0.5), "Loss weight should scale the loss"
    print(f"Loss weight 1.0: {loss_weight_1.item():.6f}")
    print(f"Loss weight 2.0: {loss_weight_2.item():.6f}")
    print(f"Loss weight 0.5: {loss_weight_05.item():.6f}")
    print("✓ Test 10 passed")

    # Test 11: Multi-dimensional predictions (e.g., segmentation)
    print("\n=== Test 11: Multi-dimensional predictions ===")
    criterion = CrossEntropyLoss(reduction='mean')
    pred = torch.randn(2, 3, 4, 4)  # (batch, classes, H, W)
    label = torch.randint(0, 3, (2, 4, 4))  # (batch, H, W)

    # Reshape for cross entropy
    pred_2d = pred.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, C)
    label_1d = label.reshape(-1)  # (N,)

    loss = criterion(pred_2d, label_1d)
    print(f"Multi-dimensional CE loss: {loss.item():.6f}")
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    print("✓ Test 11 passed")

    # Test 12: Edge case - all correct predictions
    print("\n=== Test 12: Edge case - all correct predictions ===")
    criterion = CrossEntropyLoss(reduction='mean')
    pred = torch.zeros(5, 3)
    label = torch.tensor([0, 1, 2, 0, 1])

    # Set very high confidence for correct class
    for i, l in enumerate(label):
        pred[i, l] = 10.0

    loss = criterion(pred, label)
    print(f"Loss with high confidence correct predictions: {loss.item():.6f}")
    assert loss.item() < 0.1, "Loss should be very small for confident correct predictions"
    print("✓ Test 12 passed")

    print("\n" + "="*50)
    print("All CrossEntropyLoss tests passed!")
    print("="*50)