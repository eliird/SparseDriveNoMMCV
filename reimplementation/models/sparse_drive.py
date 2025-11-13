"""
SparseDrive: End-to-end autonomous driving model.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
import torch.nn as nn

from .backbones.resnet import ResNet
from .necks.fpn import FPN
from .depth.dense_depth import DenseDepthNet


class SparseDrive(nn.Module):
    """SparseDrive model for end-to-end autonomous driving.

    Args:
        use_grid_mask (bool): Whether to use grid mask augmentation. Default: True
        use_deformable_func (bool): Whether to use deformable functions. Default: True
        img_backbone (dict): Config for image backbone (ResNet)
        img_neck (dict): Config for image neck (FPN)
        depth_branch (dict): Config for depth estimation branch
        head (dict): Config for task heads (detection, map, motion/planning)
    """

    def __init__(
        self,
        use_grid_mask=True,
        use_deformable_func=True,
        img_backbone=None,
        img_neck=None,
        depth_branch=None,
        head=None,
        **kwargs
    ):
        super().__init__()

        self.use_grid_mask = use_grid_mask
        self.use_deformable_func = use_deformable_func

        # Build image backbone
        if img_backbone is not None:
            self.img_backbone = self._build_backbone(img_backbone)
        else:
            self.img_backbone = None

        # Build image neck (FPN)
        if img_neck is not None:
            self.img_neck = self._build_neck(img_neck)
        else:
            self.img_neck = None

        # Build depth branch
        self.depth_branch = None
        if depth_branch is not None:
            self.depth_branch = self._build_depth_branch(depth_branch)

        # Build task head (will implement later)
        self.head = None
        if head is not None:
            # TODO: implement SparseDriveHead
            pass

    def _build_backbone(self, cfg):
        """Build backbone from config dict.

        Args:
            cfg (dict): Backbone config with keys:
                - type: 'ResNet'
                - depth: 50, 101, etc.
                - num_stages: 4
                - frozen_stages: -1
                - norm_eval: False
                - style: 'pytorch'
                - with_cp: True
                - out_indices: (0, 1, 2, 3)
                - norm_cfg: dict(type='BN', requires_grad=True)
                - pretrained: path to pretrained weights
        """
        cfg = cfg.copy()
        backbone_type = cfg.pop('type')

        if backbone_type == 'ResNet':
            # Map config keys to ResNet constructor
            depth = cfg.pop('depth', 50)
            num_stages = cfg.pop('num_stages', 4)
            frozen_stages = cfg.pop('frozen_stages', -1)
            norm_eval = cfg.pop('norm_eval', False)
            style = cfg.pop('style', 'pytorch')
            with_cp = cfg.pop('with_cp', False)
            out_indices = cfg.pop('out_indices', (0, 1, 2, 3))
            norm_cfg = cfg.pop('norm_cfg', None)
            pretrained = cfg.pop('pretrained', None)

            # Create ResNet
            backbone = ResNet(
                depth=depth,
                num_stages=num_stages,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=out_indices,
                style=style,
                frozen_stages=frozen_stages,
                bn_eval=norm_eval,
                bn_frozen=False,
                with_cp=with_cp
            )

            # Load pretrained weights if specified
            if pretrained:
                backbone.init_weights(pretrained=pretrained)
            else:
                backbone.init_weights(pretrained=None)

            return backbone
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def _build_neck(self, cfg):
        """Build neck from config dict.

        Args:
            cfg (dict): Neck config with keys:
                - type: 'FPN'
                - in_channels: [256, 512, 1024, 2048]
                - out_channels: 256
                - num_outs: 4
                - start_level: 0
                - add_extra_convs: 'on_output'
                - relu_before_extra_convs: True
        """
        cfg = cfg.copy()
        neck_type = cfg.pop('type')

        if neck_type == 'FPN':
            in_channels = cfg.pop('in_channels')
            out_channels = cfg.pop('out_channels')
            num_outs = cfg.pop('num_outs')
            start_level = cfg.pop('start_level', 0)
            end_level = cfg.pop('end_level', -1)
            add_extra_convs = cfg.pop('add_extra_convs', False)
            relu_before_extra_convs = cfg.pop('relu_before_extra_convs', False)
            no_norm_on_lateral = cfg.pop('no_norm_on_lateral', False)

            neck = FPN(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                start_level=start_level,
                end_level=end_level,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs,
                no_norm_on_lateral=no_norm_on_lateral
            )

            neck.init_weights()
            return neck
        else:
            raise ValueError(f"Unsupported neck type: {neck_type}")

    def _build_depth_branch(self, cfg):
        """Build depth branch from config dict.

        Args:
            cfg (dict): Depth branch config with keys:
                - type: 'DenseDepthNet'
                - embed_dims: 256
                - num_depth_layers: 3
                - loss_weight: 0.2
        """
        cfg = cfg.copy()
        depth_type = cfg.pop('type')

        if depth_type == 'DenseDepthNet':
            embed_dims = cfg.pop('embed_dims', 256)
            num_depth_layers = cfg.pop('num_depth_layers', 1)
            equal_focal = cfg.pop('equal_focal', 100)
            max_depth = cfg.pop('max_depth', 60)
            loss_weight = cfg.pop('loss_weight', 1.0)

            depth_branch = DenseDepthNet(
                embed_dims=embed_dims,
                num_depth_layers=num_depth_layers,
                equal_focal=equal_focal,
                max_depth=max_depth,
                loss_weight=loss_weight
            )

            return depth_branch
        else:
            raise ValueError(f"Unsupported depth branch type: {depth_type}")

    def extract_img_feat(self, img):
        """Extract features from images.

        Args:
            img (Tensor): Input images of shape (B, N, C, H, W)
                where N is number of camera views

        Returns:
            list[Tensor]: Multi-scale features from FPN
        """
        B, N, C, H, W = img.shape

        # Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        img = img.reshape(B * N, C, H, W)

        # TODO: Apply grid mask if enabled
        # if self.use_grid_mask:
        #     img = self.grid_mask(img)

        # Extract features with backbone
        if self.img_backbone is not None:
            x = self.img_backbone(x=img)
            if isinstance(x, torch.Tensor):
                x = [x]
        else:
            x = [img]

        # Apply neck (FPN)
        if self.img_neck is not None:
            x = self.img_neck(x)

        return x

    def forward(
        self,
        img=None,
        img_metas=None,
        **kwargs
    ):
        """Forward pass.

        Args:
            img (Tensor): Input images of shape (B, N, C, H, W)
            img_metas (list[dict]): Meta information for each image
            **kwargs: Additional arguments

        Returns:
            dict: Model outputs including losses or predictions
        """
        # Extract image features
        img_feats = self.extract_img_feat(img)

        # Forward through depth branch (auxiliary supervision)
        depth_out = None
        if self.depth_branch is not None:
            gt_depths = kwargs.get('gt_depths', None)
            focal = kwargs.get('focal', None)
            depth_out = self.depth_branch(img_feats, focal=focal, gt_depths=gt_depths)

        # TODO: Forward through task head
        # if self.head is not None:
        #     outputs = self.head(img_feats, img_metas, **kwargs)

        # Return features and depth
        outputs = {
            'img_feats': img_feats,
        }
        if depth_out is not None:
            if self.training and isinstance(depth_out, torch.Tensor):
                # During training, depth_out is loss
                outputs['depth_loss'] = depth_out
            else:
                # During inference, depth_out is predictions
                outputs['depths'] = depth_out

        return outputs


def test_sparsedrive():
    """Test SparseDrive model with backbone and neck."""
    import torch

    print("Testing SparseDrive model...")

    # Define config matching sparsedrive_small_stage1.py
    print("\n1. Creating model config...")
    embed_dims = 256
    num_levels = 4

    img_backbone_cfg = dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,  # Set to False for testing (True requires gradient checkpointing)
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained=None,  # No pretrained for testing
    )

    img_neck_cfg = dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    )

    depth_branch_cfg = dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=3,
        loss_weight=0.2,
    )

    # Create model
    print("\n2. Creating SparseDrive model...")
    model = SparseDrive(
        use_grid_mask=False,  # Disable for testing
        use_deformable_func=True,
        img_backbone=img_backbone_cfg,
        img_neck=img_neck_cfg,
        depth_branch=depth_branch_cfg,
    )
    print("   Model created successfully")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    num_cams = 6
    H, W = 256, 704

    # Create dummy input
    img = torch.randn(batch_size, num_cams, 3, H, W)
    print(f"   Input shape: {img.shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(img=img)

    print(f"   Output keys: {outputs.keys()}")
    img_feats = outputs['img_feats']
    print(f"   Number of feature levels: {len(img_feats)}")
    for i, feat in enumerate(img_feats):
        print(f"   Feature[{i}] shape: {feat.shape}")

    if 'depths' in outputs:
        depths = outputs['depths']
        print(f"   Number of depth predictions: {len(depths)}")
        for i, depth in enumerate(depths):
            print(f"   Depth[{i}] shape: {depth.shape}")

    # Test training mode with depth loss
    print("\n4. Testing training mode with depth supervision...")
    model.train()

    # Create dummy ground truth depths (downsampled from input size)
    gt_depths = [
        torch.rand(batch_size * num_cams, H // 4, W // 4) * 60,   # 1/4 scale
        torch.rand(batch_size * num_cams, H // 8, W // 8) * 60,   # 1/8 scale
        torch.rand(batch_size * num_cams, H // 16, W // 16) * 60, # 1/16 scale
    ]

    outputs_train = model(img=img, gt_depths=gt_depths)
    print(f"   Training output keys: {outputs_train.keys()}")
    if 'depth_loss' in outputs_train:
        print(f"   Depth loss: {outputs_train['depth_loss'].item():.4f}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.sparse_drive
    test_sparsedrive()
