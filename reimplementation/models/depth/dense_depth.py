"""
DenseDepthNet: Dense depth estimation network.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class DenseDepthNet(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            # feat shape: (B*N, C, H, W)
            depth = self.depth_layers[i](feat.float()).exp()  # (B*N, 1, H, W)
            # Adjust by focal length
            depth = depth.permute(1, 2, 3, 0)  # (1, H, W, B*N)
            depth = depth * focal / self.equal_focal
            depth = depth.permute(3, 0, 1, 2)  # (B*N, 1, H, W)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


def test_dense_depth():
    """Test DenseDepthNet implementation."""
    import torch

    print("Testing DenseDepthNet...")

    # Create model
    print("\n1. Creating DenseDepthNet...")
    embed_dims = 256
    num_depth_layers = 3
    model = DenseDepthNet(
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2
    )
    print(f"   Model created with {num_depth_layers} depth layers")

    # Test forward pass (inference mode)
    print("\n2. Testing forward pass (inference)...")
    batch_size = 2
    num_cams = 6

    # Simulate FPN outputs at different scales
    feature_maps = [
        torch.randn(batch_size * num_cams, embed_dims, 64, 176),  # 1/4 scale
        torch.randn(batch_size * num_cams, embed_dims, 32, 88),   # 1/8 scale
        torch.randn(batch_size * num_cams, embed_dims, 16, 44),   # 1/16 scale
        torch.randn(batch_size * num_cams, embed_dims, 8, 22),    # 1/32 scale
    ]

    model.eval()
    with torch.no_grad():
        depths = model(feature_maps)

    print(f"   Number of depth predictions: {len(depths)}")
    for i, depth in enumerate(depths):
        print(f"   Depth[{i}] shape: {depth.shape}")

    # Test forward pass with loss (training mode)
    print("\n3. Testing forward pass (training with loss)...")
    model.train()

    # Create dummy ground truth depths
    gt_depths = [
        torch.rand(batch_size * num_cams, 64, 176) * 60,   # Random depths [0, 60m]
        torch.rand(batch_size * num_cams, 32, 88) * 60,
        torch.rand(batch_size * num_cams, 16, 44) * 60,
    ]

    loss = model(feature_maps, gt_depths=gt_depths)
    print(f"   Loss value: {loss.item():.4f}")

    # Test with custom focal lengths
    print("\n4. Testing with custom focal lengths...")
    model.eval()
    focal = torch.tensor([1000.0, 1100.0] * (batch_size * num_cams // 2))
    with torch.no_grad():
        depths_focal = model(feature_maps, focal=focal)
    print(f"   Depth predictions with custom focal: {len(depths_focal)} outputs")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.depth.dense_depth
    test_dense_depth()
