"""
Utility functions for model initialization and checkpoint loading.
Replaces mmcv/mmengine utilities with pure PyTorch.
"""
import logging
import torch
import torch.nn as nn


def constant_init(module, val, bias=0.0):
    """Initialize module with constant value.

    Args:
        module: PyTorch module (Conv2d, Linear, BatchNorm, etc.)
        val: Value to initialize weight
        bias: Value to initialize bias
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0):
    """Kaiming/He initialization for Conv2d/Linear layers.

    Args:
        module: PyTorch module
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'
        bias: Value to initialize bias
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1.0, bias=0.0, distribution='normal'):
    """Xavier initialization.

    Args:
        module: PyTorch module
        gain: Scaling factor
        bias: Value to initialize bias
        distribution: 'normal' or 'uniform'
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Normal initialization.

    Args:
        module: PyTorch module
        mean: Mean of normal distribution
        std: Standard deviation
        bias: Value to initialize bias
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def load_checkpoint(model, filename, strict=False, logger=None, map_location='cpu'):
    """Load checkpoint from file.

    Args:
        model: PyTorch model
        filename: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        logger: Logger instance
        map_location: Device to load checkpoint

    Returns:
        Loaded checkpoint dict (or None if file not found)
    """
    if logger:
        logger.info(f'Loading checkpoint from {filename}')

    try:
        checkpoint = torch.load(filename, map_location=map_location)
    except FileNotFoundError:
        if logger:
            logger.warning(f'Checkpoint file not found: {filename}')
        return None

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

    if logger:
        if missing_keys:
            logger.warning(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            logger.warning(f'Unexpected keys: {unexpected_keys}')
        logger.info(f'Checkpoint loaded successfully from {filename}')

    return checkpoint


__all__ = [
    'constant_init',
    'kaiming_init',
    'xavier_init',
    'normal_init',
    'load_checkpoint',
]
