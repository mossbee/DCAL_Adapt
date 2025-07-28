# PromptCAM Adaptation Plan: Twin Face Verification

## Overview
Transform PromptCAM from fine-grained classification to twin face verification using triplet loss with hard negative mining. The goal is to distinguish between same-person pairs and twin-person pairs using class-specific attention maps to identify discriminative facial features.

## Key Changes Summary
- **Task**: Classification → Verification (binary: same person vs twin)
- **Model**: PromptCAMD with 2 class-specific prompts
- **Loss**: Cross-entropy → Triplet loss with hard negative mining
- **Data**: Single images → Image pairs with twin pair structure
- **Backbone**: DINOv2 (frozen)
- **Input**: 224x224 face images (no augmentation)

## Training and testing dataset details:

id_to_images.json

```json
{
    "id1": [
        "/path/to/image_1/of/id1.jpg",
        "/path/to/image_2/of/id1.jpg",
        "/path/to/image_3/of/id1.jpg",
        "/path/to/image_4/of/id1.jpg"
    ],
    "id2": [
        "/path/to/image_1/of/id2.jpg",
        "/path/to/image_2/of/id2.jpg",
        "/path/to/image_3/of/id2.jpg",
        "/path/to/image_4/of/id2.jpg",
        "/path/to/image_5/of/id2.jpg"
    ],
    "id3": [
        "/path/to/image_1/of/id3.jpg",
        "/path/to/image_2/of/id3.jpg",
        "/path/to/image_3/of/id3.jpg",
        "/path/to/image_4/of/id3.jpg",
        "/path/to/image_5/of/id3.jpg",
        "/path/to/image_6/of/id3.jpg"
    ],
    "id4": [
        "/path/to/image_1/of/id4.jpg",
        "/path/to/image_2/of/id4.jpg",
        "/path/to/image_3/of/id4.jpg",
        "/path/to/image_4/of/id4.jpg"
    ]
}
```

train/test_twin_id_pairs.json (2 files, train and test)

```json
[
    [
        "train/test_id1",
        "id_of_train/test_id1_twin"
    ],
    [
        "traintest_id2",
        "id_of_train/test_id2_twin"
    ]
]
```

## Implementation Strategy

### Phase 1: Dataset and Data Pipeline [x] ✅ COMPLETED
**Goal**: Create twin face dataset with pair-based sampling

#### 1.1 Create Twin Face Dataset Class [x]
**File**: `data/dataset/twin_faces.py`
**Strategy**: 
- Load twin pair information from JSON files
- Generate positive pairs (same person, different images)
- Generate negative pairs (twin pairs - hard negatives)
- Implement triplet sampling strategy

#### 1.2 Modify Data Loader [x]
**File**: `experiment/build_loader.py`
**Changes**:
- Add twin face dataset support
- Modify `get_dataset()` to handle twin face data
- Update `gen_loader()` for triplet-based batches

#### 1.3 Update Configuration [x]
**File**: `experiment/config/prompt_cam/dinov2/twin_faces/args.yaml`
**Changes**:
- Set `data: twin_faces`
- Set `data_path: ./data/images/twin_faces`
- Set `vpt_num: 2` (same person vs twin)
- Set `crop_size: 224`
- Disable augmentation

### Phase 2: Model Architecture Modifications [x] ✅ COMPLETED
**Goal**: Adapt PromptCAM for verification task

#### 2.1 Create Twin Face Model [x]
**File**: `model/twin_face_vpt.py`
**Strategy**:
- Extend existing VPT class for twin face verification
- Implement 2 class-specific prompts (same person, twin person)
- Add triplet loss computation
- Add attention map extraction for visualization

#### 2.2 Modify Model Builder [x]
**File**: `experiment/build_model.py`
**Changes**:
- Add twin face model support
- Configure for DINOv2 backbone
- Set up triplet loss parameters

### Phase 3: Training (Week 3) [x] ✅ COMPLETED
1. [x] Modify trainer for triplet loss
2. [x] Implement hard negative mining
3. [x] Add verification metrics
4. [x] Test training loop

### Phase 4: Evaluation (Week 4) [x] ✅ COMPLETED
1. [x] Implement verification evaluation
2. [x] Add attention visualization
3. [x] Create analysis scripts
4. [x] Full end-to-end testing

### Phase 5: Visualization and Analysis [x] ✅ COMPLETED
1. [x] Create Attention Visualizer
2. [x] Add Visualization Pipeline

## File Modification Checklist

### New Files to Create
- [x] `data/dataset/twin_faces.py` - Twin face dataset class
- [x] `model/twin_face_vpt.py` - Twin face VPT model
- [x] `engine/triplet_loss.py` - Triplet loss implementation
- [x] `utils/verification_metrics.py` - Verification metrics
- [x] `utils/attention_visualizer.py` - Attention visualization
- [x] `experiment/config/prompt_cam/dinov2/twin_faces/args.yaml` - Configuration

### Files to Modify
- [x] `experiment/build_loader.py` - Add twin face dataset support
- [x] `experiment/build_model.py` - Add twin face model support
- [x] `engine/trainer.py` - Add triplet loss training
- [x] `experiment/run.py` - Update evaluation pipeline
- [x] `experiment/visualize_run.py` - Add attention visualization

### Files to Reuse (No Changes)
- [x] `model/vision_transformer.py` - DINOv2 backbone
- [x] `model/block.py` - Transformer blocks
- [x] `model/patch_embed.py` - Patch embedding
- [x] `model/mlp.py` - MLP layers
- [x] `utils/setup_logging.py` - Logging utilities
- [x] `utils/misc.py` - Utility functions
- [x] `utils/file_io.py` - File I/O utilities

## Implementation Phases

### Phase 1: Foundation (Week 1) [x] ✅ COMPLETED
1. [x] Create twin face dataset class
2. [x] Modify data loader
3. [x] Create configuration file
4. [x] Test data loading pipeline

### Phase 2: Model (Week 2) [x] ✅ COMPLETED
1. [x] Create twin face VPT model
2. [x] Implement triplet loss
3. [x] Modify model builder
4. [x] Test model forward pass

### Phase 3: Training (Week 3) [x] ✅ COMPLETED
1. [x] Modify trainer for triplet loss
2. [x] Implement hard negative mining
3. [x] Add verification metrics
4. [x] Test training loop

### Phase 4: Evaluation (Week 4) [x] ✅ COMPLETED
1. [x] Implement verification evaluation
2. [x] Add attention visualization
3. [x] Create analysis scripts
4. [x] Full end-to-end testing

### Phase 5: Visualization and Analysis [x] ✅ COMPLETED
1. [x] Create Attention Visualizer
2. [x] Add Visualization Pipeline

## Key Implementation Details

### Triplet Sampling Strategy
- **Positive pairs**: Same person, different images (random sampling)
- **Negative pairs**: Twin pairs (hard negatives from twin pair structure)
- **Batch structure**: (anchor, positive, negative) triplets

### Hard Negative Mining
- Use twin pair information from JSON files
- For each anchor, sample negative from corresponding twin
- Ensure balanced positive/negative pairs

### Attention Visualization
- Extract attention maps from class-specific prompts
- Visualize "same person" vs "twin person" attention patterns
- Overlay attention on original face images
- Create comparison visualizations

### Evaluation Metrics
- **EER**: Equal Error Rate (primary metric)
- **AUC**: Area Under ROC Curve
- **TAR/FAR**: True Accept Rate / False Accept Rate
- **Verification Accuracy**: Binary classification accuracy

## Expected Outcomes
1. **Model**: PromptCAMD with 2 class-specific prompts for twin face verification
2. **Training**: Triplet loss with hard negative mining using twin pair information
3. **Visualization**: Attention maps showing discriminative facial features
4. **Evaluation**: Comprehensive verification metrics (EER, AUC, TAR/FAR)
5. **Analysis**: Understanding of which facial features distinguish twins vs same person

## Success Criteria
- EER < 10% on test set
- AUC > 0.9
- Clear attention maps showing discriminative features
- Successful visualization of twin vs same person attention patterns 