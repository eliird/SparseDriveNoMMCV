from inspect import signature

import torch
import torch.nn as nn

from .utils.builders import build_from_cfg
from .utils.model_utils import force_fp32
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


class SparseDrive(nn.Module):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        train_cfg=None,
        test_cfg=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__()

        # Build backbone
        self.img_backbone = build_from_cfg(img_backbone)

        # Build neck if provided
        if img_neck is not None:
            self.img_neck = build_from_cfg(img_neck)
        else:
            self.img_neck = None

        # Build head
        self.head = build_from_cfg(head)

        # Grid mask for data augmentation
        self.use_grid_mask = use_grid_mask
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        # Deformable aggregation
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        # Depth branch (optional)
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch)
        else:
            self.depth_branch = None 

    def extract_feat(self, img, return_depth=False, metas=None):
        """Extract features from images.

        Args:
            img: Input images [B, C, H, W] or [B, N, C, H, W] for multi-view
            return_depth: Whether to return depth predictions
            metas: Metadata dictionary

        Returns:
            feature_maps: Extracted multi-scale features
            depths: (optional) Depth predictions if return_depth=True
        """
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def forward(self, img, **data):
        """Forward pass.

        Routes to forward_train or forward_test based on training mode.

        Args:
            img: Input images
            **data: Additional data (ground truth, metadata, etc.)

        Returns:
            During training: Dictionary of losses
            During testing: List of prediction results
        """
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
