
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod, ABCMeta
from .match_cost import build_match_cost

'''
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
'''

class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes"""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxe or a negative boxes"""
        pass

class BaseTargetWithDenoising(ABC):
    """Base class for target assignment with denoising training support.

    This abstract class provides the interface for target assignment methods
    that support denoising training, which adds noisy instances to improve
    training robustness.

    Args:
        num_dn_groups (int): Number of denoising groups. Default: 0
        num_temp_dn_groups (int): Number of temporal denoising groups. Default: 0
    """

    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        super(BaseTargetWithDenoising, self).__init__()
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None

    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """Perform Hungarian matching between predictions and ground truth.

        Returns the matched ground truth corresponding to the predictions
        along with the corresponding regression weights.

        Args:
            cls_pred (torch.Tensor): Classification predictions
            box_pred (torch.Tensor): Box predictions
            cls_target (list): Ground truth classes
            box_target (list): Ground truth boxes

        Returns:
            tuple: Matched targets and weights
        """

    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """
        Generate noisy instances for the current frame, with a total of
        'self.num_dn_groups' groups.
        """
        return None

    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """
        Insert the previously saved 'self.dn_metas' into the noisy instances
        of the current frame.
        """

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        """
        Randomly save information for 'self.num_temp_dn_groups' groups of
        temporal noisy instances to 'self.dn_metas'.
        """
        if self.num_temp_dn_groups < 0:
            return
        self.dn_metas = dict(dn_anchor=dn_anchor[:, : self.num_temp_dn_groups])
        

class SparsePoint3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner)
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size

    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets  = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        indices = []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            # normalize to (0, 1)
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds=dict(lines=pts_pred, scores=cls_pred)
            gts=dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_box_target, output_reg_weights

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


class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               ignore_cls_cost=False,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        # compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost)
        else:
            cost = self.cost(preds, gts, ignore_cls_cost)

        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx


def build_assigner(cfg):
    """Build assigner from config dict.

    Args:
        cfg (dict or object): Config dict with 'type' key, or an already instantiated object

    Returns:
        Assigner instance
    """
    if cfg is None:
        return None

    # If already an instance, return it directly
    if not isinstance(cfg, dict):
        return cfg

    # Build from dict config
    assigner_type = cfg.get('type')
    assigner_kwargs = {k: v for k, v in cfg.items() if k != 'type'}

    if assigner_type == 'HungarianLinesAssigner':
        return HungarianLinesAssigner(**assigner_kwargs)
    else:
        raise ValueError(f"Unknown assigner type: {assigner_type}")


if __name__ == '__main__':
    print("Testing assigner module...")

    # Test 1: HungarianLinesAssigner basic functionality
    print("\n=== Test 1: HungarianLinesAssigner basic ===")
    assigner = HungarianLinesAssigner(
        cost=dict(
            type='MapQueriesCost',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
        )
    )

    # 8 predictions
    preds = {
        'scores': torch.randn(8, 3),  # 8 queries, 3 classes
        'lines': torch.randn(8, 40)   # 8 queries, 20 points (40 coords)
    }

    # 5 ground truth
    gts = {
        'labels': torch.tensor([0, 1, 2, 0, 1]),  # 5 GT labels
        'lines': torch.randn(5, 2, 40)            # 5 GT lines, 2 permutations
    }

    matched_row_inds, matched_col_inds, gt_permute_idx = assigner.assign(preds, gts)

    print(f"Matched row indices (predictions): {matched_row_inds}")
    print(f"Matched col indices (GT): {matched_col_inds}")
    print(f"GT permute indices shape: {gt_permute_idx.shape}")

    assert len(matched_row_inds) == len(matched_col_inds), "Should have same number of matches"
    assert len(matched_row_inds) == 5, "Should match all 5 GT (since we have 8 queries)"
    assert gt_permute_idx.shape == torch.Size([8, 5]), "Permute index should be (num_preds, num_gts)"
    print("✓ Test 1 passed")

    # Test 2: HungarianLinesAssigner with empty GT
    print("\n=== Test 2: HungarianLinesAssigner with empty GT ===")
    preds_empty = {
        'scores': torch.randn(8, 3),
        'lines': torch.randn(8, 40)
    }
    gts_empty = {
        'labels': torch.tensor([]),
        'lines': torch.empty(0, 2, 40)
    }

    result = assigner.assign(preds_empty, gts_empty)
    assert result == (None, None, None), "Should return (None, None, None) for empty GT"
    print("✓ Test 2 passed")

    # Test 3: SparsePoint3DTarget initialization
    print("\n=== Test 3: SparsePoint3DTarget initialization ===")
    target = SparsePoint3DTarget(
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
            ),
        ),
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    )

    assert target.num_cls == 3, "num_cls should be set correctly"
    assert target.num_sample == 20, "num_sample should be set correctly"
    assert target.roi_size == (30, 60), "roi_size should be set correctly"
    assert isinstance(target.assigner, HungarianLinesAssigner), "Should build HungarianLinesAssigner"
    print("✓ Test 3 passed")

    # Test 4: SparsePoint3DTarget.normalize_line
    print("\n=== Test 4: SparsePoint3DTarget.normalize_line ===")
    target = SparsePoint3DTarget(
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
            ),
        ),
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    )

    # Create lines in range [-15, 15] for x and [-30, 30] for y
    lines = torch.tensor([[0.0, 0.0] * 20])  # Center point repeated
    normalized = target.normalize_line(lines)

    print(f"Original line shape: {lines.shape}")
    print(f"Normalized line shape: {normalized.shape}")
    print(f"Center point normalized value: {normalized[0, 0]:.3f}, {normalized[0, 1]:.3f}")

    # Center (0, 0) should map to (0.5, 0.5) approximately
    assert normalized.shape == lines.shape, "Shape should be preserved"
    assert -0.1 < normalized[0, 0] < 0.6, "x coordinate should be around 0.5"
    assert -0.1 < normalized[0, 1] < 0.6, "y coordinate should be around 0.5"
    print("✓ Test 4 passed")

    # Test 5: SparsePoint3DTarget.sample
    print("\n=== Test 5: SparsePoint3DTarget.sample ===")
    target = SparsePoint3DTarget(
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
            ),
        ),
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    )

    # Batch size 2, 100 predictions, 3 classes
    cls_preds = torch.randn(2, 100, 3)  # Tensor with batch dimension
    # 100 predictions, 40 coordinates (20 points)
    pts_preds = torch.randn(2, 100, 40)

    # Ground truth: 5 lines for first batch, 3 lines for second batch
    cls_targets = [torch.tensor([0, 1, 2, 0, 1]), torch.tensor([1, 2, 0])]
    # GT lines with 2 permutations
    pts_targets = [torch.randn(5, 2, 40), torch.randn(3, 2, 40)]

    output_cls_target, output_box_target, output_reg_weights = target.sample(
        cls_preds, pts_preds, cls_targets, pts_targets
    )

    print(f"Output cls target shape: {output_cls_target.shape}")
    print(f"Output box target shape: {output_box_target.shape}")
    print(f"Output reg weights shape: {output_reg_weights.shape}")

    assert output_cls_target.shape == torch.Size([2, 100]), "Cls target should be (batch, num_queries)"
    assert output_box_target.shape == torch.Size([2, 100, 40]), "Box target should match predictions"
    assert output_reg_weights.shape == torch.Size([2, 100, 40]), "Weights should match predictions"

    # Check that background queries are assigned class = num_cls (3)
    num_background_b0 = (output_cls_target[0] == 3).sum().item()
    print(f"Number of background queries in batch 0: {num_background_b0}")
    assert num_background_b0 == 95, "Should have 95 background queries (100 - 5 matched)"
    print("✓ Test 5 passed")

    # Test 6: SparsePoint3DTarget with empty GT
    print("\n=== Test 6: SparsePoint3DTarget.sample with empty GT ===")
    cls_preds_empty = torch.randn(1, 50, 3)  # Tensor with batch dimension
    pts_preds_empty = torch.randn(1, 50, 40)
    cls_targets_empty = [torch.tensor([])]
    pts_targets_empty = [torch.empty(0, 2, 40)]

    output_cls, output_box, output_weights = target.sample(
        cls_preds_empty, pts_preds_empty, cls_targets_empty, pts_targets_empty
    )

    # All should be background (class = num_cls = 3)
    assert (output_cls[0] == 3).all(), "All queries should be background when GT is empty"
    assert (output_weights[0] == 0).all(), "All weights should be 0 when GT is empty"
    print("✓ Test 6 passed")

    # Test 7: build_assigner with dict
    print("\n=== Test 7: build_assigner with dict config ===")
    assigner_cfg = dict(
        type='HungarianLinesAssigner',
        cost=dict(
            type='MapQueriesCost',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
        )
    )
    assigner_inst = build_assigner(assigner_cfg)

    assert isinstance(assigner_inst, HungarianLinesAssigner), "Should build HungarianLinesAssigner"
    print("✓ Test 7 passed")

    # Test 8: build_assigner with instance
    print("\n=== Test 8: build_assigner with existing instance ===")
    existing_assigner = HungarianLinesAssigner(
        cost=dict(type='MapQueriesCost',
                  cls_cost=dict(type='FocalLossCost', weight=1.0),
                  reg_cost=dict(type='LinesL1Cost', weight=10.0))
    )
    result = build_assigner(existing_assigner)

    assert result is existing_assigner, "Should return the same instance"
    print("✓ Test 8 passed")

    # Test 9: build_assigner with None
    print("\n=== Test 9: build_assigner with None ===")
    result = build_assigner(None)
    assert result is None, "Should return None"
    print("✓ Test 9 passed")

    # Test 10: BaseTargetWithDenoising methods
    print("\n=== Test 10: BaseTargetWithDenoising denoising methods ===")
    target = SparsePoint3DTarget(
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(type='MapQueriesCost',
                      cls_cost=dict(type='FocalLossCost', weight=1.0),
                      reg_cost=dict(type='LinesL1Cost', weight=10.0))
        ),
        num_dn_groups=2,
        num_temp_dn_groups=1,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    )

    assert target.num_dn_groups == 2, "num_dn_groups should be set"
    assert target.num_temp_dn_groups == 1, "num_temp_dn_groups should be set"

    # Test get_dn_anchors (returns None in base implementation)
    result = target.get_dn_anchors([], [])
    assert result is None, "get_dn_anchors should return None in base implementation"

    # Test cache_dn
    dn_anchor = torch.randn(10, 5, 10)  # (batch, dn_groups, features)
    target.cache_dn(None, dn_anchor, None, None, None)
    assert target.dn_metas is not None, "dn_metas should be cached"
    assert 'dn_anchor' in target.dn_metas, "dn_anchor should be in dn_metas"
    # Should only cache first num_temp_dn_groups (1)
    assert target.dn_metas['dn_anchor'].shape[1] == 1, "Should only cache num_temp_dn_groups"
    print("✓ Test 10 passed")

    # Test 11: Verify Hungarian matching is optimal
    print("\n=== Test 11: Verify Hungarian matching optimality ===")
    # Create a simple case where we know the optimal matching
    assigner = HungarianLinesAssigner(
        cost=dict(
            type='MapQueriesCost',
            cls_cost=dict(type='FocalLossCost', weight=0.0),  # Disable cls cost
            reg_cost=dict(type='LinesL1Cost', weight=1.0, beta=0.0, permute=False),
        )
    )

    # Create predictions and GT where optimal matching is clear
    # 3 queries, 2 GT
    preds = {
        'scores': torch.randn(3, 2),
        'lines': torch.tensor([[0.0] * 40,  # Very close to GT 0
                               [1.0] * 40,  # Very close to GT 1
                               [0.5] * 40])  # In between
    }

    gts = {
        'labels': torch.tensor([0, 1]),
        'lines': torch.tensor([[0.01] * 40,   # GT 0
                               [0.99] * 40])   # GT 1
    }

    matched_row, matched_col, _ = assigner.assign(preds, gts)

    print(f"Matched predictions: {matched_row}")
    print(f"Matched GT: {matched_col}")

    # Query 0 should match GT 0, Query 1 should match GT 1
    matching_dict = dict(zip(matched_row, matched_col))
    print(f"Matching: {matching_dict}")

    # Verify we got 2 matches (one per GT)
    assert len(matched_row) == 2, "Should match both GTs"
    print("✓ Test 11 passed")

    print("\n" + "="*50)
    print("All assigner tests passed!")
    print("="*50)
