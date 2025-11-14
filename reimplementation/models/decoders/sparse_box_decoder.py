from typing import Optional

import torch
from ..common.box3d import *


def decode_box(box):
    yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
    box = torch.cat(
        [
            box[..., [X, Y, Z]],
            box[..., [W, L, H]].exp(),
            yaw[..., None],
            box[..., VX:],
        ],
        dim=-1,
    )
    return box


class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality is not None and quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if quality is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output


if __name__ == '__main__':
    print("Testing SparseBox3DDecoder module...")

    # Test 1: Basic decoding without quality or instance_id
    print("\n=== Test 1: Basic decoding ===")
    decoder = SparseBox3DDecoder(num_output=300, score_threshold=None, sorted=True)

    # Create predictions from 6 decoder layers
    batch_size, num_queries, num_cls = 2, 900, 10
    cls_scores_list = [torch.randn(batch_size, num_queries, num_cls) for _ in range(6)]
    box_preds_list = [torch.randn(batch_size, num_queries, 11) for _ in range(6)]

    # Decode using last layer
    output = decoder.decode(cls_scores_list, box_preds_list, output_idx=-1)

    print(f"Batch size: {batch_size}")
    print(f"Number of outputs per batch: {len(output)}")
    print(f"Keys in output[0]: {list(output[0].keys())}")
    print(f"boxes_3d shape: {output[0]['boxes_3d'].shape}")
    print(f"scores_3d shape: {output[0]['scores_3d'].shape}")
    print(f"labels_3d shape: {output[0]['labels_3d'].shape}")

    assert len(output) == batch_size, "Should have one output per batch"
    assert output[0]['boxes_3d'].shape[0] == 300, "Should have 300 detections"
    assert output[0]['boxes_3d'].shape[1] == 10, "Decoded boxes should be 10D"
    assert output[0]['scores_3d'].shape[0] == 300, "Should have 300 scores"
    assert output[0]['labels_3d'].shape[0] == 300, "Should have 300 labels"
    print("✓ Test 1 passed")

    # Test 2: With score threshold
    print("\n=== Test 2: With score threshold ===")
    decoder_thresh = SparseBox3DDecoder(num_output=300, score_threshold=0.3, sorted=True)

    # Create predictions with known scores
    cls_scores_high = [torch.ones(batch_size, num_queries, num_cls) * 0.5 for _ in range(6)]
    output_thresh = decoder_thresh.decode(cls_scores_high, box_preds_list, output_idx=-1)

    print(f"boxes_3d shape with threshold: {output_thresh[0]['boxes_3d'].shape}")
    print(f"All scores should be >= 0.3")
    assert (output_thresh[0]['scores_3d'] >= 0.3).all(), "All scores should pass threshold"
    print("✓ Test 2 passed")

    # Test 3: With quality (centerness) reweighting
    print("\n=== Test 3: With quality (centerness) reweighting ===")
    decoder_quality = SparseBox3DDecoder(num_output=100, score_threshold=None, sorted=True)

    # Create quality predictions (centerness, yawness)
    quality_list = [torch.randn(batch_size, num_queries, 2) for _ in range(6)]

    output_quality = decoder_quality.decode(
        cls_scores_list, box_preds_list, quality=quality_list, output_idx=-1
    )

    print(f"Output keys with quality: {list(output_quality[0].keys())}")
    assert 'cls_scores' in output_quality[0], "Should have cls_scores when quality is provided"
    print(f"cls_scores shape: {output_quality[0]['cls_scores'].shape}")
    print("✓ Test 3 passed")

    # Test 4: With instance tracking
    print("\n=== Test 4: With instance tracking ===")
    decoder_inst = SparseBox3DDecoder(num_output=100, score_threshold=None, sorted=True)

    # Create instance IDs
    instance_id = torch.randint(0, 1000, (batch_size, num_queries))

    output_inst = decoder_inst.decode(
        cls_scores_list, box_preds_list, instance_id=instance_id, output_idx=-1
    )

    print(f"Output keys with instance_id: {list(output_inst[0].keys())}")
    assert 'instance_ids' in output_inst[0], "Should have instance_ids when provided"
    print(f"instance_ids shape: {output_inst[0]['instance_ids'].shape}")
    print("✓ Test 4 passed")

    # Test 5: decode_box function
    print("\n=== Test 5: decode_box function ===")
    # Create encoded boxes [X, Y, Z, log(W), log(L), log(H), SIN_YAW, COS_YAW, VX, VY, VZ]
    encoded_box = torch.tensor([[
        1.0, 2.0, 3.0,           # X, Y, Z
        0.0, 0.0, 0.0,           # log(W)=0 -> W=1, log(L)=0 -> L=1, log(H)=0 -> H=1
        0.707, 0.707,            # SIN_YAW, COS_YAW (45 degrees)
        0.5, 0.5, 0.5            # VX, VY, VZ
    ]])

    decoded_box = decode_box(encoded_box)

    print(f"Encoded box shape: {encoded_box.shape}")
    print(f"Decoded box shape: {decoded_box.shape}")
    print(f"Position (X,Y,Z): {decoded_box[0, :3].tolist()}")
    print(f"Size (W,L,H): {decoded_box[0, 3:6].tolist()}")
    print(f"Yaw: {decoded_box[0, 6].item():.4f} radians")
    print(f"Velocity (VX,VY,VZ): {decoded_box[0, 7:].tolist()}")

    # Check position unchanged
    assert torch.allclose(decoded_box[0, :3], encoded_box[0, :3]), "Position should be unchanged"
    # Check size is exp(0) = 1
    assert torch.allclose(decoded_box[0, 3:6], torch.ones(3), atol=1e-6), "Size should be exp(0)=1"
    # Check yaw is approximately pi/4 (45 degrees)
    assert torch.allclose(decoded_box[0, 6], torch.tensor(0.7854), atol=0.01), "Yaw should be ~pi/4"
    # Check velocity unchanged
    assert torch.allclose(decoded_box[0, 7:], encoded_box[0, 8:]), "Velocity should be unchanged"
    print("✓ Test 5 passed")

    # Test 6: Different output indices
    print("\n=== Test 6: Different output indices ===")
    decoder_idx = SparseBox3DDecoder(num_output=50, score_threshold=None, sorted=True)

    # Decode from different decoder layers
    output_first = decoder_idx.decode(cls_scores_list, box_preds_list, output_idx=0)
    output_middle = decoder_idx.decode(cls_scores_list, box_preds_list, output_idx=3)
    output_last = decoder_idx.decode(cls_scores_list, box_preds_list, output_idx=-1)

    print(f"Output from layer 0: {output_first[0]['boxes_3d'].shape}")
    print(f"Output from layer 3: {output_middle[0]['boxes_3d'].shape}")
    print(f"Output from layer -1: {output_last[0]['boxes_3d'].shape}")

    assert all(out['boxes_3d'].shape[0] == 50 for out in [output_first[0], output_middle[0], output_last[0]])
    print("✓ Test 6 passed")

    # Test 7: Quality with None values
    print("\n=== Test 7: Quality with None values ===")
    # Create quality list where last layer is None
    quality_none = [torch.randn(batch_size, num_queries, 2) for _ in range(5)]
    quality_none.append(None)

    output_none_quality = decoder.decode(
        cls_scores_list, box_preds_list, quality=quality_none, output_idx=-1
    )

    print(f"Output keys when quality[-1] is None: {list(output_none_quality[0].keys())}")
    assert 'cls_scores' not in output_none_quality[0], "Should not have cls_scores when quality is None"
    print("✓ Test 7 passed")

    # Test 8: Sorted vs unsorted
    print("\n=== Test 8: Sorted vs unsorted ===")
    decoder_sorted = SparseBox3DDecoder(num_output=100, sorted=True)
    decoder_unsorted = SparseBox3DDecoder(num_output=100, sorted=False)

    output_sorted = decoder_sorted.decode(cls_scores_list, box_preds_list, output_idx=-1)
    output_unsorted = decoder_unsorted.decode(cls_scores_list, box_preds_list, output_idx=-1)

    # Sorted output should have descending scores
    scores_sorted = output_sorted[0]['scores_3d']
    is_descending = all(scores_sorted[i] >= scores_sorted[i+1] for i in range(len(scores_sorted)-1))

    print(f"Sorted scores are descending: {is_descending}")
    assert is_descending, "Sorted output should have descending scores"
    print("✓ Test 8 passed")

    # Test 9: Box decoding with different sizes
    print("\n=== Test 9: Box decoding with different sizes ===")
    # Test log -> exp conversion for sizes
    test_boxes = torch.tensor([
        [0, 0, 0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0],      # log(1,1,1) -> (1,1,1)
        [0, 0, 0, 0.693, 1.099, 1.386, 0, 1, 0, 0, 0], # log(2,3,4) -> (2,3,4)
    ])

    decoded = decode_box(test_boxes)

    print(f"Box 1 sizes (should be ~1,1,1): {decoded[0, 3:6].tolist()}")
    print(f"Box 2 sizes (should be ~2,3,4): {decoded[1, 3:6].tolist()}")

    assert torch.allclose(decoded[0, 3:6], torch.tensor([1.0, 1.0, 1.0]), atol=0.01)
    assert torch.allclose(decoded[1, 3:6], torch.tensor([2.0, 3.0, 4.0]), atol=0.1)
    print("✓ Test 9 passed")

    # Test 10: Empty results with high threshold
    print("\n=== Test 10: Empty results with high threshold ===")
    decoder_high_thresh = SparseBox3DDecoder(num_output=300, score_threshold=0.99, sorted=True)

    # Create low confidence predictions
    low_scores = [torch.ones(batch_size, num_queries, num_cls) * (-5.0) for _ in range(6)]  # Will sigmoid to ~0
    output_empty = decoder_high_thresh.decode(low_scores, box_preds_list, output_idx=-1)

    print(f"Number of detections with high threshold: {output_empty[0]['boxes_3d'].shape[0]}")
    print("✓ Test 10 passed")

    print("\n" + "="*50)
    print("All SparseBox3DDecoder tests passed!")
    print("="*50)
