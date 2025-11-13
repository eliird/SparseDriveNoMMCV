"""
AsymmetricFFN: Feedforward network with optional pre-normalization and identity connection.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
import torch.nn as nn

__all__ = ['AsymmetricFFN']


class AsymmetricFFN(nn.Module):
    """Asymmetric Feedforward Network.

    This module implements a feedforward network with multiple fully connected layers,
    activation functions, dropout, and optional pre-normalization and identity connection.

    Args:
        in_channels (int, optional): Input channels. If None, uses embed_dims. Default: None
        pre_norm (dict, optional): Config for pre-normalization layer. Example:
            dict(type='LN', eps=1e-6). If None, no pre-norm is applied. Default: None
        embed_dims (int): Output embedding dimension. Default: 256
        feedforward_channels (int): Hidden dimension in feedforward layers. Default: 1024
        num_fcs (int): Number of fully connected layers (must be >= 2). Default: 2
        act_cfg (dict): Config for activation layer. Example:
            dict(type='ReLU', inplace=True) or dict(type='GELU'). Default: dict(type='ReLU', inplace=True)
        ffn_drop (float): Dropout rate for feedforward layers. Default: 0.0
        dropout_layer (dict, optional): Config for dropout layer. Example:
            dict(type='Dropout', drop_prob=0.1). If None, no dropout is applied. Default: None
        add_identity (bool): Whether to add identity/residual connection. Default: True
        init_cfg (dict, optional): Initialization config (ignored in pure PyTorch). Default: None
        **kwargs: Additional arguments (ignored)

    Example:
        >>> ffn = AsymmetricFFN(embed_dims=256, feedforward_channels=1024, num_fcs=2)
        >>> x = torch.randn(2, 100, 256)
        >>> out = ffn(x)
        >>> print(out.shape)  # torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__()
        assert num_fcs >= 2, (
            f"num_fcs should be no less than 2. got {num_fcs}."
        )
        self.in_channels = in_channels if in_channels is not None else embed_dims
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg

        # Build activation layer from config
        act_type = act_cfg.get('type', 'ReLU')
        if act_type == 'ReLU':
            self.activate = nn.ReLU(inplace=act_cfg.get('inplace', True))
        elif act_type == 'GELU':
            self.activate = nn.GELU()
        elif act_type == 'LeakyReLU':
            self.activate = nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.01),
                                         inplace=act_cfg.get('inplace', True))
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

        # Build pre-normalization layer from config
        if pre_norm is not None:
            norm_type = pre_norm.get('type', 'LN')
            if norm_type in ['LN', 'LayerNorm']:
                self.pre_norm = nn.LayerNorm(self.in_channels, eps=pre_norm.get('eps', 1e-5))
            elif norm_type in ['BN', 'BatchNorm1d']:
                self.pre_norm = nn.BatchNorm1d(self.in_channels, eps=pre_norm.get('eps', 1e-5))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
        else:
            self.pre_norm = None

        # Build feedforward layers
        layers = []
        in_ch = self.in_channels
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_ch, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_ch = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        # Build dropout layer from config
        if dropout_layer and isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get('drop_prob', 0.)
            self.dropout_layer = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        else:
            self.dropout_layer = nn.Identity()

        # Build identity connection
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                nn.Identity()
                if self.in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) or (B, C, H, W)
            identity (torch.Tensor, optional): Identity tensor for residual connection.
                If None, uses x. Default: None

        Returns:
            torch.Tensor: Output tensor of shape (B, N, embed_dims)
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


def test_asymmetric_ffn():
    """Test AsymmetricFFN implementation."""
    print("Testing AsymmetricFFN...")

    # Test 1: Basic usage with default config
    print("\n1. Testing basic usage...")
    ffn = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.1
    )
    x = torch.randn(2, 100, 256)
    out = ffn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (2, 100, 256), f"Expected (2, 100, 256), got {out.shape}"

    # Test 2: With different in_channels (requires projection)
    print("\n2. Testing with different in_channels...")
    ffn_proj = AsymmetricFFN(
        in_channels=128,
        embed_dims=256,
        feedforward_channels=512,
        num_fcs=2
    )
    x2 = torch.randn(4, 50, 128)
    out2 = ffn_proj(x2)
    print(f"   Input shape: {x2.shape}")
    print(f"   Output shape: {out2.shape}")
    assert out2.shape == (4, 50, 256), f"Expected (4, 50, 256), got {out2.shape}"

    # Test 3: With pre-normalization
    print("\n3. Testing with pre-normalization...")
    ffn_norm = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        pre_norm=dict(type='LN', eps=1e-6)
    )
    x3 = torch.randn(2, 100, 256)
    out3 = ffn_norm(x3)
    print(f"   Output shape: {out3.shape}")
    assert out3.shape == (2, 100, 256)

    # Test 4: Without identity connection
    print("\n4. Testing without identity connection...")
    ffn_no_id = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        add_identity=False
    )
    x4 = torch.randn(2, 100, 256)
    out4 = ffn_no_id(x4)
    print(f"   Output shape: {out4.shape}")
    assert out4.shape == (2, 100, 256)

    # Test 5: With different number of FC layers
    print("\n5. Testing with num_fcs=3...")
    ffn_3fc = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=512,
        num_fcs=3
    )
    x5 = torch.randn(2, 100, 256)
    out5 = ffn_3fc(x5)
    print(f"   Output shape: {out5.shape}")
    assert out5.shape == (2, 100, 256)

    # Test 6: With dropout layer config
    print("\n6. Testing with dropout_layer config...")
    ffn_drop = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        dropout_layer=dict(type='Dropout', drop_prob=0.2)
    )
    x6 = torch.randn(2, 100, 256)
    out6 = ffn_drop(x6)
    print(f"   Output shape: {out6.shape}")
    assert out6.shape == (2, 100, 256)

    # Test 7: With GELU activation
    print("\n7. Testing with GELU activation...")
    ffn_gelu = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type='GELU')
    )
    x7 = torch.randn(2, 100, 256)
    out7 = ffn_gelu(x7)
    print(f"   Output shape: {out7.shape}")
    assert out7.shape == (2, 100, 256)

    # Test 8: With custom identity tensor
    print("\n8. Testing with custom identity tensor...")
    ffn_custom_id = AsymmetricFFN(
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2
    )
    x8 = torch.randn(2, 100, 256)
    identity8 = torch.randn(2, 100, 256)
    out8 = ffn_custom_id(x8, identity=identity8)
    print(f"   Output shape: {out8.shape}")
    assert out8.shape == (2, 100, 256)

    # Test 9: Gradient flow
    print("\n9. Testing gradient flow...")
    ffn_grad = AsymmetricFFN(embed_dims=256, feedforward_channels=1024, num_fcs=2)
    x_grad = torch.randn(2, 10, 256, requires_grad=True)
    out_grad = ffn_grad(x_grad)
    loss = out_grad.sum()
    loss.backward()
    print(f"   Gradients computed: {x_grad.grad is not None}")
    assert x_grad.grad is not None

    # Test 10: Check assertion for num_fcs < 2
    print("\n10. Testing num_fcs assertion...")
    try:
        ffn_invalid = AsymmetricFFN(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=1  # Should fail
        )
        print("   ERROR: Should have raised assertion error!")
        assert False
    except AssertionError as e:
        print(f"   ✓ Correctly raised assertion: {str(e)}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.asym_ffn
    test_asymmetric_ffn()
