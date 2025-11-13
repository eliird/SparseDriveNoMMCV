"""
Simplified ConvModule - replaces mmcv.cnn.ConvModule
Pure PyTorch implementation without mmcv dependencies.
"""
import torch.nn as nn


class ConvModule(nn.Module):
    """Simplified conv block: Conv2d + (optional) BatchNorm + (optional) Activation.

    This is a simplified version of mmcv's ConvModule that covers the common use cases
    for FPN and other SparseDrive components.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride. Default: 1
        padding (int): Padding added to input. Default: 0
        dilation (int): Dilation rate. Default: 1
        groups (int): Number of groups for grouped convolution. Default: 1
        bias (bool | str): Whether to use bias. If 'auto', bias is False when
            norm_cfg is specified, True otherwise. Default: 'auto'
        conv_cfg (dict): Config for convolution layer. Default: None (uses Conv2d)
        norm_cfg (dict): Config for normalization. Default: None (no norm)
            Example: dict(type='BN') for BatchNorm2d
        act_cfg (dict): Config for activation. Default: None (no activation)
            Example: dict(type='ReLU') for ReLU
        inplace (bool): Whether to use inplace activation. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=False):
        super().__init__()

        # Determine bias
        if bias == 'auto':
            bias = norm_cfg is None

        # Build conv layer
        # For now, we only support standard Conv2d (conv_cfg=None)
        assert conv_cfg is None, "Only standard Conv2d supported for now"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Build norm layer
        self.norm = None
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN')
            if norm_type in ['BN', 'BatchNorm', 'BatchNorm2d']:
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type in ['GN', 'GroupNorm']:
                num_groups = norm_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups, out_channels)
            elif norm_type in ['LN', 'LayerNorm']:
                self.norm = nn.LayerNorm(out_channels)
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

        # Build activation layer
        self.activation = None
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activation = nn.ReLU(inplace=inplace)
            elif act_type == 'LeakyReLU':
                negative_slope = act_cfg.get('negative_slope', 0.01)
                self.activation = nn.LeakyReLU(negative_slope, inplace=inplace)
            elif act_type == 'GELU':
                self.activation = nn.GELU()
            elif act_type == 'SiLU':
                self.activation = nn.SiLU(inplace=inplace)
            else:
                raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x, activate=True, norm=True):
        """Forward pass.

        Args:
            x: Input tensor
            activate (bool): Whether to apply activation. Default: True
            norm (bool): Whether to apply normalization. Default: True

        Returns:
            Output tensor
        """
        x = self.conv(x)

        if self.norm is not None and norm:
            x = self.norm(x)

        if self.activation is not None and activate:
            x = self.activation(x)

        return x
