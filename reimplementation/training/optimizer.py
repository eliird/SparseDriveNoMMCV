"""
Optimizer builder for SparseDrive training.
Pure PyTorch implementation without mmcv dependencies.
"""

import torch
from typing import Dict, Any


def build_optimizer(model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from config dict.

    Supports parameter-wise learning rate configuration similar to mmcv.

    Args:
        model: PyTorch model
        optimizer_config: Config dict with format:
            {
                'type': 'AdamW',
                'lr': 4e-4,
                'weight_decay': 0.001,
                'paramwise_cfg': {
                    'custom_keys': {
                        'img_backbone': {'lr_mult': 0.5},
                        'some_other_module': {'lr_mult': 0.1, 'decay_mult': 0.5}
                    }
                }
            }

    Returns:
        PyTorch optimizer

    Example:
        >>> optimizer_config = dict(
        ...     type='AdamW',
        ...     lr=4e-4,
        ...     weight_decay=0.001,
        ...     paramwise_cfg=dict(
        ...         custom_keys={'img_backbone': dict(lr_mult=0.5)}
        ...     )
        ... )
        >>> optimizer = build_optimizer(model, optimizer_config)
    """
    config = optimizer_config.copy()
    optimizer_type = config.pop('type')
    base_lr = config.get('lr', 1e-4)
    base_weight_decay = config.get('weight_decay', 0.0)

    # Get parameter-wise config
    paramwise_cfg = config.pop('paramwise_cfg', None)

    # Organize parameters into groups
    if paramwise_cfg is not None and 'custom_keys' in paramwise_cfg:
        custom_keys = paramwise_cfg['custom_keys']
        param_groups = []

        # Track which parameters have been added to custom groups
        custom_param_ids = set()

        # Create parameter groups for custom keys
        for module_name, cfg in custom_keys.items():
            lr_mult = cfg.get('lr_mult', 1.0)
            decay_mult = cfg.get('decay_mult', 1.0)

            # Find parameters belonging to this module
            module_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and module_name in name:
                    module_params.append(param)
                    custom_param_ids.add(id(param))

            if module_params:
                param_group = {
                    'params': module_params,
                    'lr': base_lr * lr_mult,
                    'weight_decay': base_weight_decay * decay_mult,
                }
                param_groups.append(param_group)
                print(f"  - {module_name}: {len(module_params)} params, "
                      f"lr={base_lr * lr_mult:.6f}, "
                      f"weight_decay={base_weight_decay * decay_mult:.6f}")

        # Add remaining parameters with default settings
        default_params = []
        for param in model.parameters():
            if param.requires_grad and id(param) not in custom_param_ids:
                default_params.append(param)

        if default_params:
            param_groups.append({
                'params': default_params,
                'lr': base_lr,
                'weight_decay': base_weight_decay,
            })
            print(f"  - default: {len(default_params)} params, "
                  f"lr={base_lr:.6f}, "
                  f"weight_decay={base_weight_decay:.6f}")

    else:
        # No parameter-wise config, use all parameters with same settings
        param_groups = [{'params': model.parameters()}]
        print(f"  - Using single parameter group with lr={base_lr:.6f}, "
              f"weight_decay={base_weight_decay:.6f}")

    # Build optimizer
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, **config)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(param_groups, **config)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(param_groups, **config)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer
