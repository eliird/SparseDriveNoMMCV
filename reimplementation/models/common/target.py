import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod
from .box3d import *


__all__ = ["SparseBox3DTarget", "BaseTargetWithDenoising"]





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



class SparseBox3DTarget(BaseTargetWithDenoising):
    """Target assignment for 3D object detection with denoising training.

    This class implements Hungarian matching-based target assignment for 3D
    object detection. It computes matching costs using focal loss for classification
    and L1 loss for box regression, then performs optimal assignment using the
    Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Supports denoising training by generating noisy instances from ground truth
    boxes to improve model robustness.

    Args:
        cls_weight (float): Weight for classification cost. Default: 2.0
        alpha (float): Focal loss alpha parameter. Default: 0.25
        gamma (int): Focal loss gamma parameter. Default: 2
        eps (float): Small epsilon for numerical stability. Default: 1e-12
        box_weight (float): Weight for box regression cost. Default: 0.25
        reg_weights (list): Per-dimension regression weights. Default: [1]*8 + [0]*2
        cls_wise_reg_weights (dict): Class-specific regression weights. Default: None
        num_dn_groups (int): Number of denoising groups. Default: 0
        dn_noise_scale (float): Scale of noise for denoising. Default: 0.5
        max_dn_gt (int): Maximum number of GT boxes for denoising. Default: 32
        add_neg_dn (bool): Whether to add negative denoising samples. Default: True
        num_temp_dn_groups (int): Number of temporal denoising groups. Default: 0
    """

    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            # Default weights for [x, y, z, log(w), log(l), log(h), sin_yaw, cos_yaw, vx, vy, vz]
            self.reg_weights = [1.0] * 8 + [0.0] * 3
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        """Encode box targets into regression format.

        Converts box format from [x, y, z, w, l, h, yaw, vx, vy, vz] to
        [x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz].

        Args:
            box_target (list): List of box tensors per batch
            device (torch.device): Device to place outputs. Default: None

        Returns:
            list: Encoded box targets
        """
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
    ):
        """Perform Hungarian matching between predictions and targets.

        Computes classification and box costs, then uses the Hungarian algorithm
        to find the optimal assignment between predictions and ground truth boxes.

        Args:
            cls_pred (torch.Tensor): Classification predictions of shape (B, N, C)
            box_pred (torch.Tensor): Box predictions of shape (B, N, D)
            cls_target (list): List of ground truth class labels per batch
            box_target (list): List of ground truth boxes per batch

        Returns:
            tuple: A tuple containing:
                - output_cls_target (torch.Tensor): Matched class targets (B, N)
                - output_box_target (torch.Tensor): Matched box targets (B, N, D)
                - output_reg_weights (torch.Tensor): Regression weights (B, N, D)
        """
        bs, num_pred, num_cls = cls_pred.shape

        # Compute classification cost using focal loss
        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)

        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])

        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
        self.indices = indices
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )


def test_sparse_box_3d_target():
    """Test SparseBox3DTarget implementation."""
    print("Testing SparseBox3DTarget...")

    # Test parameters
    batch_size = 2
    num_pred = 300
    num_cls = 10
    box_dims_raw = 10  # Raw boxes: [x, y, z, w, l, h, yaw, vx, vy, vz]
    box_dims_encoded = 11  # Encoded: [x, y, z, log(w), log(l), log(h), sin_yaw, cos_yaw, vx, vy, vz]

    # Test 1: Basic Hungarian matching
    print("\n1. Testing basic Hungarian matching...")
    target_sampler = SparseBox3DTarget(
        cls_weight=2.0,
        box_weight=0.25,
        num_dn_groups=0
    )

    # Create dummy predictions (in encoded format)
    cls_pred = torch.randn(batch_size, num_pred, num_cls)
    box_pred = torch.randn(batch_size, num_pred, box_dims_encoded)

    # Create dummy ground truth (different number per batch, in raw format)
    cls_target = [
        torch.tensor([0, 1, 2, 3, 4]),  # 5 GT boxes in batch 0
        torch.tensor([0, 1, 2])          # 3 GT boxes in batch 1
    ]
    box_target = [
        torch.randn(5, box_dims_raw).abs() + 0.1,  # Ensure positive dimensions
        torch.randn(3, box_dims_raw).abs() + 0.1
    ]

    output_cls, output_box, output_weights = target_sampler.sample(
        cls_pred, box_pred, cls_target, box_target
    )

    print(f"   Prediction shape: {cls_pred.shape}")
    print(f"   Output cls target shape: {output_cls.shape}")
    print(f"   Output box target shape: {output_box.shape}")
    print(f"   Output weights shape: {output_weights.shape}")

    assert output_cls.shape == (batch_size, num_pred)
    assert output_box.shape == (batch_size, num_pred, box_dims_encoded)
    assert output_weights.shape == (batch_size, num_pred, box_dims_encoded)
    print("   ✓ Basic Hungarian matching works")

    # Test 2: Denoising anchor generation
    print("\n2. Testing denoising anchor generation...")
    target_sampler_dn = SparseBox3DTarget(
        num_dn_groups=5,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True
    )

    dn_output = target_sampler_dn.get_dn_anchors(cls_target, box_target)

    if dn_output is not None:
        dn_anchor, dn_box_target, dn_cls_target, attn_mask, valid_mask, dn_id_target = dn_output
        print(f"   DN anchor shape: {dn_anchor.shape}")
        print(f"   DN box target shape: {dn_box_target.shape}")
        print(f"   DN cls target shape: {dn_cls_target.shape}")
        print(f"   Attention mask shape: {attn_mask.shape}")
        print(f"   Valid mask shape: {valid_mask.shape}")

        # Check that DN anchors are noisy versions of GT
        assert dn_anchor.shape[0] == batch_size
        assert dn_cls_target.shape[0] == batch_size
        print("   ✓ Denoising anchor generation works")
    else:
        print("   ⚠ No denoising anchors generated (no GT boxes)")

    # Test 3: Encoding regression targets
    print("\n3. Testing box encoding...")
    # Create boxes with proper format [x, y, z, w, l, h, yaw, vx, vy, vz]
    raw_boxes = [
        torch.tensor([
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.5, 0.1, 0.2, 0.0],
            [1.0, 1.0, 1.0, 3.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        ])
    ]

    encoded = target_sampler.encode_reg_target(raw_boxes)

    print(f"   Raw box shape: {raw_boxes[0].shape}")
    print(f"   Encoded box shape: {encoded[0].shape}")

    # Check that width/length/height are log-encoded
    # encoded should be [x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]
    # Raw has 10 dims, encoded has 11 dims (yaw split into sin/cos)
    assert encoded[0].shape[0] == raw_boxes[0].shape[0]  # Same number of boxes
    assert encoded[0].shape[1] == 11  # Encoded has 11 dimensions
    assert torch.allclose(encoded[0][:, :3], raw_boxes[0][:, :3])  # x,y,z unchanged
    assert torch.allclose(encoded[0][:, 3:6], raw_boxes[0][:, 3:6].log())  # w,l,h log-encoded
    # Check yaw encoding: sin and cos
    assert torch.allclose(encoded[0][:, 6], torch.sin(raw_boxes[0][:, 6]))  # sin(yaw)
    assert torch.allclose(encoded[0][:, 7], torch.cos(raw_boxes[0][:, 6]))  # cos(yaw)
    print("   ✓ Box encoding works correctly")

    # Test 4: Class-wise regression weights
    print("\n4. Testing class-wise regression weights...")
    target_sampler_cls_weight = SparseBox3DTarget(
        cls_wise_reg_weights={0: [2.0] * 11, 1: [0.5] * 11}
    )

    output_cls, output_box, output_weights = target_sampler_cls_weight.sample(
        cls_pred, box_pred, cls_target, box_target
    )

    print(f"   Output weights with class-wise scaling: {output_weights.shape}")
    assert output_weights.shape == (batch_size, num_pred, box_dims_encoded)
    print("   ✓ Class-wise regression weights work")

    # Test 5: Empty ground truth handling
    print("\n5. Testing empty ground truth...")
    empty_cls_target = [torch.tensor([]), torch.tensor([])]
    empty_box_target = [torch.empty(0, box_dims_raw), torch.empty(0, box_dims_raw)]

    output_cls, output_box, output_weights = target_sampler.sample(
        cls_pred, box_pred, empty_cls_target, empty_box_target
    )

    print(f"   Output with empty GT: cls={output_cls.shape}, box={output_box.shape}")
    # All predictions should be assigned to background (num_cls)
    assert (output_cls == num_cls).all()
    print("   ✓ Empty ground truth handling works")

    # Test 6: Denoising with instance IDs
    print("\n6. Testing denoising with instance IDs...")
    gt_instance_id = [
        torch.tensor([100, 101, 102, 103, 104]),
        torch.tensor([200, 201, 202])
    ]

    target_sampler_temp_dn = SparseBox3DTarget(
        num_dn_groups=3,
        num_temp_dn_groups=1,
        dn_noise_scale=0.3
    )

    dn_output = target_sampler_temp_dn.get_dn_anchors(
        cls_target, box_target, gt_instance_id
    )

    if dn_output is not None:
        _, _, _, _, _, dn_id = dn_output
        print(f"   DN instance IDs shape: {dn_id.shape if dn_id is not None else None}")
        print("   ✓ Denoising with instance IDs works")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.target
    test_sparse_box_3d_target()
