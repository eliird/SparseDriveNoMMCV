"""
Dataset and DataLoader builder functions in mmdet style.

Provides build_dataset and build_dataloader functions that accept config dictionaries.
"""

from torch.utils.data import DataLoader, DistributedSampler
from .nuscenes_3d_dataset import NuScenes3DDataset
from .collate import sparse_drive_collate


DATASETS = {
    'NuScenes3DDataset': NuScenes3DDataset,
}


def build_dataset(cfg):
    """Build dataset from config dict.

    Args:
        cfg (dict): Config dict. It should contain:
            - type (str): Dataset type name
            - Other dataset-specific parameters

    Returns:
        Dataset: Built dataset instance

    Example:
        >>> dataset_cfg = dict(
        ...     type='NuScenes3DDataset',
        ...     data_root='data/nuscenes/',
        ...     ann_file='data/infos/nuscenes_infos_train.pkl',
        ...     pipeline=[...],
        ...     classes=['car', 'truck', ...],
        ...     test_mode=False,
        ... )
        >>> dataset = build_dataset(dataset_cfg)
    """
    cfg = cfg.copy()
    dataset_type = cfg.pop('type')

    if dataset_type not in DATASETS:
        raise ValueError(f'Unknown dataset type: {dataset_type}. '
                        f'Available types: {list(DATASETS.keys())}')

    dataset_cls = DATASETS[dataset_type]
    dataset = dataset_cls(**cfg)

    return dataset


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=False,
    shuffle=True,
    seed=None,
    drop_last=False,
    pin_memory=True,
    persistent_workers=False,
    **kwargs
):
    """Build PyTorch DataLoader.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of samples per GPU (batch size per GPU).
        workers_per_gpu (int): Number of workers per GPU.
        num_gpus (int): Number of GPUs. Only used in distributed training.
        dist (bool): Whether to use distributed training. Default: False.
        shuffle (bool): Whether to shuffle the data. Default: True.
        seed (int, optional): Random seed. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch.
            Default: False.
        pin_memory (bool): Whether to use pinned memory. Default: True.
        persistent_workers (bool): Whether to keep workers alive between
            data loading epochs. Default: False.
        **kwargs: Other keyword arguments for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader.

    Example:
        >>> dataset = build_dataset(dataset_cfg)
        >>> dataloader = build_dataloader(
        ...     dataset,
        ...     samples_per_gpu=4,
        ...     workers_per_gpu=2,
        ...     shuffle=True,
        ...     dist=False
        ... )
    """
    # Determine rank and world_size for distributed training
    rank = 0
    world_size = 1

    if dist:
        try:
            import torch.distributed as dist_module
            if dist_module.is_available() and dist_module.is_initialized():
                rank = dist_module.get_rank()
                world_size = dist_module.get_world_size()
        except ImportError:
            pass

    # Create sampler
    sampler = None
    if dist:
        # Use DistributedSampler for distributed training
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed if seed is not None else 0,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling in distributed mode

    # Build init_kwargs for DataLoader
    init_kwargs = {
        'dataset': dataset,
        'batch_size': samples_per_gpu,
        'num_workers': workers_per_gpu,
        'collate_fn': sparse_drive_collate,  # Custom collate for variable-length data
        'pin_memory': pin_memory,
        'shuffle': shuffle if sampler is None else False,
        'sampler': sampler,
        'drop_last': drop_last,
    }

    # Add persistent_workers if workers > 0
    if workers_per_gpu > 0:
        init_kwargs['persistent_workers'] = persistent_workers

    # Add any additional kwargs (can override collate_fn if needed)
    init_kwargs.update(kwargs)

    # Create DataLoader
    data_loader = DataLoader(**init_kwargs)

    return data_loader
