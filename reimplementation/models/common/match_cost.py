import torch
from ..losses.l1_loss import smooth_l1_loss
from abc import abstractmethod
from typing import Union, Optional
from torch import Tensor
'''
assigner=dict(
    
    type='HungarianLinesAssigner',
    cost=dict(
        type='MapQueriesCost',
        cls_cost=dict(type='FocalLossCost', weight=1.0),
        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
    ),
),
'''

class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            Subclasses define their own arguments. Typically:
            - For detection: pred_instances, gt_instances containing bboxes, labels, scores
            - For lines/maps: pred lines/scores, gt lines/labels

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass

class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight
    

class LinesL1Cost(object):
    """LinesL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0, beta=0.0, permute=False):
        self.weight = weight
        self.permute = permute
        self.beta = beta

    def __call__(self, lines_pred, gt_lines, **kwargs):
        """
        Args:
            lines_pred (Tensor): predicted normalized lines:
                [num_query, 2*num_points]
            gt_lines (Tensor): Ground truth lines
                [num_gt, 2*num_points] or [num_gt, num_permute, 2*num_points]
        Returns:
            torch.Tensor: reg_cost value with weight
                shape [num_pred, num_gt]
        """        
        if self.permute:
            assert len(gt_lines.shape) == 3
        else:
            assert len(gt_lines.shape) == 2

        num_pred, num_gt = len(lines_pred), len(gt_lines)
        if self.permute:
            # permute-invarint labels
            gt_lines = gt_lines.flatten(0, 1) # (num_gt*num_permute, 2*num_pts)

        num_pts = lines_pred.shape[-1]//2

        if self.beta > 0:
            lines_pred = lines_pred.unsqueeze(1).repeat(1, len(gt_lines), 1)
            gt_lines = gt_lines.unsqueeze(0).repeat(num_pred, 1, 1)
            dist_mat = smooth_l1_loss(lines_pred, gt_lines, reduction='none', beta=self.beta).sum(-1)
        
        else:
            dist_mat = torch.cdist(lines_pred, gt_lines, p=1)

        dist_mat = dist_mat / num_pts

        if self.permute:
            # dist_mat: (num_pred, num_gt*num_permute)
            dist_mat = dist_mat.view(num_pred, num_gt, -1) # (num_pred, num_gt, num_permute)
            dist_mat, gt_permute_index = torch.min(dist_mat, 2)
            return dist_mat * self.weight, gt_permute_index
        
        return dist_mat * self.weight

class MapQueriesCost(object):

    def __init__(self, cls_cost, reg_cost, iou_cost=None):

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

        self.iou_cost = None
        if iou_cost is not None:
            self.iou_cost = build_match_cost(iou_cost)

    def __call__(self, preds: dict, gts: dict, ignore_cls_cost: bool):

        # classification and bboxcost.
        cls_cost = self.cls_cost(preds['scores'], gts['labels'])

        # regression cost
        regkwargs = {}
        if 'masks' in preds and 'masks' in gts:
            assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!'
            regkwargs = {
                'masks_pred': preds['masks'],
                'masks_gt': gts['masks'],
            }

        reg_cost = self.reg_cost(preds['lines'], gts['lines'], **regkwargs)
        if self.reg_cost.permute:
            reg_cost, gt_permute_idx = reg_cost

        # weighted sum of above three costs
        if ignore_cls_cost:
            cost = reg_cost
        else:
            cost = cls_cost + reg_cost

        # Iou
        if self.iou_cost is not None:
            iou_cost = self.iou_cost(preds['lines'],gts['lines'])
            cost += iou_cost
        
        if self.reg_cost.permute:
            return cost, gt_permute_idx
        return cost


def build_match_cost(cfg):
    """Build match cost module from config dict.

    Args:
        cfg (dict or object): Config dict with 'type' key, or an already instantiated object

    Returns:
        Match cost module instance
    """
    if cfg is None:
        return None

    # If already an instance, return it directly
    if not isinstance(cfg, dict):
        return cfg

    # Build from dict config
    cost_type = cfg.get('type')
    cost_kwargs = {k: v for k, v in cfg.items() if k != 'type'}

    if cost_type == 'FocalLossCost':
        return FocalLossCost(**cost_kwargs)
    elif cost_type == 'LinesL1Cost':
        return LinesL1Cost(**cost_kwargs)
    elif cost_type == 'MapQueriesCost':
        return MapQueriesCost(**cost_kwargs)
    else:
        raise ValueError(f"Unknown match cost type: {cost_type}")


if __name__ == '__main__':
    print("Testing match_cost module...")

    # Test 1: FocalLossCost basic functionality
    print("\n=== Test 1: FocalLossCost basic ===")
    focal_cost = FocalLossCost(alpha=0.25, gamma=2.0, weight=1.0)

    # 5 queries, 3 classes
    cls_pred = torch.randn(5, 3)
    # 3 ground truth objects with labels [0, 1, 2]
    gt_labels = torch.tensor([0, 1, 2])

    cost = focal_cost(cls_pred, gt_labels)
    print(f"FocalLossCost output shape: {cost.shape}")
    print(f"Expected shape: torch.Size([5, 3]) (num_queries, num_gt)")
    assert cost.shape == torch.Size([5, 3]), "Cost matrix should be (num_queries, num_gt)"
    assert not torch.isnan(cost).any(), "Cost should not contain NaN"
    print("✓ Test 1 passed")

    # Test 2: FocalLossCost with weight
    print("\n=== Test 2: FocalLossCost with weight ===")
    focal_cost_w2 = FocalLossCost(alpha=0.25, gamma=2.0, weight=2.0)
    cost_w1 = focal_cost(cls_pred, gt_labels)
    cost_w2 = focal_cost_w2(cls_pred, gt_labels)

    print(f"Cost with weight=1.0: {cost_w1[0, 0]:.6f}")
    print(f"Cost with weight=2.0: {cost_w2[0, 0]:.6f}")
    assert torch.allclose(cost_w2, cost_w1 * 2.0, rtol=1e-5), "Weight should scale the cost"
    print("✓ Test 2 passed")

    # Test 3: LinesL1Cost without permutation
    print("\n=== Test 3: LinesL1Cost without permutation ===")
    lines_cost = LinesL1Cost(weight=10.0, beta=0.0, permute=False)

    # 8 queries, each with 20 points (40 coords)
    lines_pred = torch.randn(8, 40)
    # 5 ground truth lines
    gt_lines = torch.randn(5, 40)

    cost = lines_cost(lines_pred, gt_lines)
    print(f"LinesL1Cost output shape: {cost.shape}")
    assert cost.shape == torch.Size([8, 5]), "Cost should be (num_queries, num_gt)"
    assert not torch.isnan(cost).any(), "Cost should not contain NaN"
    print("✓ Test 3 passed")

    # Test 4: LinesL1Cost with permutation
    print("\n=== Test 4: LinesL1Cost with permutation ===")
    lines_cost_perm = LinesL1Cost(weight=10.0, beta=0.01, permute=True)

    # 8 queries
    lines_pred = torch.randn(8, 40)
    # 5 ground truth lines, each with 2 permutations
    gt_lines = torch.randn(5, 2, 40)

    result = lines_cost_perm(lines_pred, gt_lines)
    cost, permute_idx = result

    print(f"LinesL1Cost (permute) output shapes: cost={cost.shape}, permute_idx={permute_idx.shape}")
    assert cost.shape == torch.Size([8, 5]), "Cost should be (num_queries, num_gt)"
    assert permute_idx.shape == torch.Size([8, 5]), "Permute index should be (num_queries, num_gt)"
    assert not torch.isnan(cost).any(), "Cost should not contain NaN"
    print("✓ Test 4 passed")

    # Test 5: MapQueriesCost
    print("\n=== Test 5: MapQueriesCost ===")
    map_cost = MapQueriesCost(
        cls_cost=dict(type='FocalLossCost', weight=1.0),
        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True)
    )

    # Prepare predictions
    preds = {
        'scores': torch.randn(8, 3),  # 8 queries, 3 classes
        'lines': torch.randn(8, 40)    # 8 queries, 20 points
    }

    # Prepare ground truth
    gts = {
        'labels': torch.tensor([0, 1, 2, 0, 1]),  # 5 ground truth
        'lines': torch.randn(5, 2, 40)             # 5 GT, 2 permutations
    }

    result = map_cost(preds, gts, ignore_cls_cost=False)
    cost, permute_idx = result

    print(f"MapQueriesCost output shapes: cost={cost.shape}, permute_idx={permute_idx.shape}")
    assert cost.shape == torch.Size([8, 5]), "Total cost should be (num_queries, num_gt)"
    assert permute_idx.shape == torch.Size([8, 5]), "Permute index should be (num_queries, num_gt)"
    print("✓ Test 5 passed")

    # Test 6: MapQueriesCost with ignore_cls_cost
    print("\n=== Test 6: MapQueriesCost ignore_cls_cost ===")
    result_no_cls = map_cost(preds, gts, ignore_cls_cost=True)
    cost_no_cls, _ = result_no_cls

    result_with_cls = map_cost(preds, gts, ignore_cls_cost=False)
    cost_with_cls, _ = result_with_cls

    print(f"Cost without cls: {cost_no_cls[0, 0]:.6f}")
    print(f"Cost with cls: {cost_with_cls[0, 0]:.6f}")
    # They should be different
    assert not torch.allclose(cost_no_cls, cost_with_cls), "Costs should differ when cls is included/excluded"
    print("✓ Test 6 passed")

    # Test 7: build_match_cost with dict
    print("\n=== Test 7: build_match_cost with dict config ===")
    focal_cfg = dict(type='FocalLossCost', alpha=0.25, gamma=2.0, weight=1.0)
    focal_inst = build_match_cost(focal_cfg)

    assert isinstance(focal_inst, FocalLossCost), "Should build FocalLossCost instance"
    assert focal_inst.alpha == 0.25, "Alpha should be set correctly"
    assert focal_inst.gamma == 2.0, "Gamma should be set correctly"
    print("✓ Test 7 passed")

    # Test 8: build_match_cost with instance
    print("\n=== Test 8: build_match_cost with existing instance ===")
    existing_cost = FocalLossCost(alpha=0.5, gamma=3.0)
    result = build_match_cost(existing_cost)

    assert result is existing_cost, "Should return the same instance"
    print("✓ Test 8 passed")

    # Test 9: build_match_cost with None
    print("\n=== Test 9: build_match_cost with None ===")
    result = build_match_cost(None)
    assert result is None, "Should return None"
    print("✓ Test 9 passed")

    # Test 10: LinesL1Cost with smooth_l1
    print("\n=== Test 10: LinesL1Cost with smooth L1 (beta > 0) ===")
    lines_cost_smooth = LinesL1Cost(weight=5.0, beta=0.1, permute=False)

    lines_pred = torch.randn(6, 40)
    gt_lines = torch.randn(4, 40)

    cost = lines_cost_smooth(lines_pred, gt_lines)
    print(f"Smooth L1 cost shape: {cost.shape}")
    assert cost.shape == torch.Size([6, 4]), "Cost should be (num_queries, num_gt)"
    assert not torch.isnan(cost).any(), "Cost should not contain NaN"
    print("✓ Test 10 passed")

    # Test 11: Edge case - single query, single GT
    print("\n=== Test 11: Edge case - single query, single GT ===")
    focal_cost = FocalLossCost(weight=1.0)
    cls_pred = torch.randn(1, 3)
    gt_labels = torch.tensor([1])

    cost = focal_cost(cls_pred, gt_labels)
    print(f"Single query/GT cost shape: {cost.shape}")
    assert cost.shape == torch.Size([1, 1]), "Cost should be (1, 1)"
    print("✓ Test 11 passed")

    # Test 12: Gradient flow through costs
    print("\n=== Test 12: Gradient flow ===")
    cls_pred = torch.randn(5, 3, requires_grad=True)
    gt_labels = torch.tensor([0, 1, 2])

    focal_cost = FocalLossCost(weight=1.0)
    cost = focal_cost(cls_pred, gt_labels)

    # Take mean and backprop
    cost.mean().backward()

    print(f"Gradient shape: {cls_pred.grad.shape}")
    assert cls_pred.grad is not None, "Gradients should be computed"
    assert not torch.isnan(cls_pred.grad).any(), "Gradients should not contain NaN"
    print("✓ Test 12 passed")

    print("\n" + "="*50)
    print("All match_cost tests passed!")
    print("="*50)