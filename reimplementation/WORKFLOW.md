# SparseDrive Implementation Workflow

**Based on:** `projects/configs/sparsedrive_small_stage1.py`

**Strategy:** Implement models one-by-one in the order they appear in config

---

## Models to Implement (in config order)

### 1. Main Model (line 89)
- [ ] **SparseDrive**

### 2. Backbone (line 93)
- [ ] **ResNet** - Use torchvision.models.resnet50

### 3. Neck (line 105)
- [ ] **FPN** - Feature Pyramid Network

### 4. Depth Branch (line 114)
- [ ] **DenseDepthNet**

### 5. Unified Head (line 120)
- [ ] **SparseDriveHead**

---

## Detection Head Components (lines 122-266)

### 5.1 Main Head (line 123)
- [ ] **Sparse4DHead**

### 5.2 Instance Bank (line 127)
- [ ] **InstanceBank**
- [ ] **SparseBox3DKeyPointsGenerator** (line 131) - anchor_handler

### 5.3 Encoders (line 137)
- [ ] **SparseBox3DEncoder** - anchor_encoder

### 5.4 Graph Models (lines 168, 177)
- [ ] **MultiheadFlashAttention** (temp_graph_model, graph_model)

### 5.5 FFN (line 185)
- [ ] **AsymmetricFFN**

### 5.6 Deformable Aggregation (line 195)
- [ ] **DeformableFeatureAggregation**
- [ ] **SparseBox3DKeyPointsGenerator** (line 205) - kps_generator

### 5.7 Refinement (line 219)
- [ ] **SparseBox3DRefinementModule**

### 5.8 Sampler (line 226)
- [ ] **SparseBox3DTarget**

### 5.9 Losses (lines 251, 258)
- [ ] **FocalLoss** (line 251)
- [ ] **SparseBox3DLoss** (line 258)
  - [ ] **L1Loss** (line 259)
  - [ ] **CrossEntropyLoss** (line 260)
  - [ ] **GaussianFocalLoss** (line 261)

### 5.10 Decoder (line 264)
- [ ] **SparseBox3DDecoder**

---

## Map Head Components (lines 267-398)

### 6.1 Main Head (line 268)
- [ ] **Sparse4DHead** (same as detection, different config)

### 6.2 Instance Bank (line 272)
- [ ] **InstanceBank** (same as detection)
- [ ] **SparsePoint3DKeyPointsGenerator** (line 276)

### 6.3 Encoders (line 282)
- [ ] **SparsePoint3DEncoder**

### 6.4 Graph Models (lines 309, 318)
- [ ] **MultiheadFlashAttention** (same as detection)

### 6.5 FFN (line 326)
- [ ] **AsymmetricFFN** (same as detection)

### 6.6 Deformable Aggregation (line 336)
- [ ] **DeformableFeatureAggregation** (same as detection)
- [ ] **SparsePoint3DKeyPointsGenerator** (line 346)

### 6.7 Refinement (line 355)
- [ ] **SparsePoint3DRefinementModule**

### 6.8 Sampler (line 361)
- [ ] **SparsePoint3DTarget**
  - [ ] **HungarianLinesAssigner** (line 363)
    - [ ] **MapQueriesCost** (line 365)
      - [ ] **FocalLossCost** (line 366)
      - [ ] **LinesL1Cost** (line 367)

### 6.9 Losses (lines 375, 382)
- [ ] **FocalLoss** (line 375)
- [ ] **SparseLineLoss** (line 382)
  - [ ] **LinesL1Loss** (line 384)

### 6.10 Decoder (line 391)
- [ ] **SparsePoint3DDecoder**

---

## Motion & Planning Head Components (lines 399-504)

### 7.1 Main Head (line 400)
- [ ] **MotionPlanningHead**

### 7.2 Instance Queue (line 410)
- [ ] **InstanceQueue**

### 7.3 Graph Models (lines 431, 438, 445)
- [ ] **MultiheadAttention** (line 431) - NOT Flash!
- [ ] **MultiheadFlashAttention** (lines 438, 445)

### 7.4 FFN (line 453)
- [ ] **AsymmetricFFN** (same as detection)

### 7.5 Refinement (line 463)
- [ ] **MotionPlanningRefinementModule**

### 7.6 Samplers (lines 471, 482)
- [ ] **MotionTarget** (line 471)
- [ ] **PlanningTarget** (line 482)

### 7.7 Losses (lines 474, 480, 487, 493, 494)
- [ ] **FocalLoss** (lines 474, 487)
- [ ] **L1Loss** (lines 480, 493, 494)

### 7.8 Decoders (lines 495, 497)
- [ ] **SparseBox3DMotionDecoder** (line 495)
- [ ] **HierarchicalPlanningDecoder** (line 497)

---

## Summary

### 📦 Use External (2)
1. ResNet - torchvision
2. FPN - torchvision or custom

### 🔨 To Implement (38 classes total)

**Unique classes needed:**
1. SparseDrive
2. SparseDriveHead
3. Sparse4DHead
4. SparseBox3DKeyPointsGenerator
5. SparseBox3DEncoder
6. SparseBox3DRefinementModule
7. SparseBox3DTarget
8. SparseBox3DDecoder
9. SparseBox3DLoss
10. SparsePoint3DKeyPointsGenerator
11. SparsePoint3DEncoder
12. SparsePoint3DRefinementModule
13. SparsePoint3DTarget
14. HungarianLinesAssigner
15. MapQueriesCost
16. SparseLineLoss
17. LinesL1Loss
18. LinesL1Cost
19. SparsePoint3DDecoder
20. MotionPlanningHead
21. InstanceQueue
22. MultiheadAttention
23. MotionPlanningRefinementModule
24. MotionTarget
25. PlanningTarget
26. SparseBox3DMotionDecoder
27. HierarchicalPlanningDecoder
28. FocalLoss
29. L1Loss
30. CrossEntropyLoss
31. GaussianFocalLoss
32. FocalLossCost
33. GridMask (used in model but not in config)

---

## Implementation Order (Bottom-Up)

We'll implement in dependency order, starting with the simplest:

1. **Losses** (no dependencies)
2. **Encoders & Generators** (low dependency)
3. **Refinement Modules** (depends on encoders)
4. **Decoders** (low dependency)
5. **Targets/Samplers** (depends on costs/assigners)
6. **Instance Queue** (depends on generators)
7. **Sparse4DHead** (depends on everything above)
8. **MotionPlanningHead** (depends on queue)
9. **SparseDriveHead** (depends on all heads)
10. **FPN** (independent)
11. **SparseDrive** (top level)

---

## Next Step

Tell me which model to start with and I'll:
1. Find the original implementation
2. Copy it with minimal changes
3. Remove mmcv/mmdet dependencies
4. Add tests
5. Mark as complete

**Recommended start:** `FocalLoss` or `SparseBox3DEncoder` (both simple and independent)
