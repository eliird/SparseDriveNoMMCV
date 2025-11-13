"""
InstanceBank: Manages instance features and anchors with temporal caching.
Pure PyTorch implementation without mmcv dependencies.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


class InstanceBank(nn.Module):
    """Instance bank for managing detection/map instance features and anchors.

    This module maintains a bank of instance features and anchors, with support for
    temporal caching across frames. It's used in both detection and map heads.

    Args:
        num_anchor (int): Number of anchor instances
        embed_dims (int): Embedding dimension for instance features
        anchor (str | np.ndarray | list): Initial anchor values or path to .npy file
        anchor_handler (dict): Config for anchor handler (e.g., SparseBox3DKeyPointsGenerator)
        num_temp_instances (int): Number of temporal instances to cache. Default: 0
        default_time_interval (float): Default time interval between frames. Default: 0.5
        confidence_decay (float): Confidence decay factor for temporal instances. Default: 0.6
        anchor_grad (bool): Whether anchor requires gradient. Default: True
        feat_grad (bool): Whether features require gradient. Default: True
        max_time_interval (float): Maximum time interval for temporal matching. Default: 2
    """
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # anchor_handler will be built externally and passed as an instance
        # We'll handle this in the builder method later
        self.anchor_handler_cfg = anchor_handler
        self.anchor_handler = None  # Will be set externally after instantiation
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )


def test_instance_bank():
    """Test InstanceBank implementation."""
    import torch

    print("Testing InstanceBank...")

    # Create model
    print("\n1. Creating InstanceBank...")
    num_anchor = 900
    embed_dims = 256
    num_temp_instances = 600

    # Create dummy anchor data
    anchor = np.random.randn(num_anchor, 10)  # 10D anchor (e.g., 3D box params)

    instance_bank = InstanceBank(
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor=anchor,
        anchor_handler=None,  # No handler for now
        num_temp_instances=num_temp_instances,
        confidence_decay=0.6,
        anchor_grad=False,
        feat_grad=True,
    )
    print(f"   Instance bank created with {num_anchor} anchors, {embed_dims}D features")
    print(f"   Temporal instances: {num_temp_instances}")

    # Test get (initial frame)
    print("\n2. Testing get() for initial frame...")
    batch_size = 2
    metas = {
        'timestamp': torch.tensor([0.0, 0.0]),
        'img_metas': [{'T_global': np.eye(4)} for _ in range(batch_size)]
    }

    instance_feature, anchor_out, cached_feature, cached_anchor, time_interval = instance_bank.get(
        batch_size, metas=metas
    )

    print(f"   instance_feature shape: {instance_feature.shape}")
    print(f"   anchor shape: {anchor_out.shape}")
    print(f"   cached_feature: {cached_feature}")
    print(f"   time_interval: {time_interval}")

    # Test update
    print("\n3. Testing update()...")
    # Simulate predictions
    confidence = torch.rand(batch_size, num_anchor, 10)  # 10 classes
    updated_feature, updated_anchor = instance_bank.update(
        instance_feature, anchor_out, confidence
    )
    print(f"   updated_feature shape: {updated_feature.shape}")
    print(f"   updated_anchor shape: {updated_anchor.shape}")

    # Test cache
    print("\n4. Testing cache()...")
    instance_bank.cache(
        updated_feature, updated_anchor, confidence, metas=metas
    )
    print(f"   Cached {instance_bank.cached_feature.shape[1]} instances")
    print(f"   Confidence shape: {instance_bank.confidence.shape}")

    # Test get with cached data (next frame)
    print("\n5. Testing get() with temporal instances...")
    metas_t1 = {
        'timestamp': torch.tensor([0.5, 0.5]),  # 0.5s later
        'img_metas': [{'T_global': np.eye(4), 'T_global_inv': np.eye(4)}
                      for _ in range(batch_size)]
    }

    instance_feature_t1, anchor_t1, cached_feature_t1, cached_anchor_t1, time_interval_t1 = instance_bank.get(
        batch_size, metas=metas_t1
    )

    print(f"   instance_feature shape: {instance_feature_t1.shape}")
    print(f"   cached_feature shape: {cached_feature_t1.shape}")
    print(f"   time_interval: {time_interval_t1}")
    print(f"   mask shape: {instance_bank.mask.shape}, valid: {instance_bank.mask.sum().item()}")

    # Test instance ID tracking
    print("\n6. Testing instance ID tracking...")
    instance_id = instance_bank.get_instance_id(confidence, threshold=0.5)
    print(f"   instance_id shape: {instance_id.shape}")
    print(f"   Number of unique IDs: {len(instance_id.unique())}")
    print(f"   Next ID will be: {instance_bank.prev_id}")

    # Test reset
    print("\n7. Testing reset()...")
    instance_bank.reset()
    print(f"   cached_feature after reset: {instance_bank.cached_feature}")
    print(f"   confidence after reset: {instance_bank.confidence}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.instance_bank
    test_instance_bank()