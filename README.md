# Dual Cross-Attention Learning (DCAL) Implementation

This repository contains a PyTorch implementation of the Dual Cross-Attention Learning (DCAL) method for fine-grained visual categorization and object re-identification, as described in the paper "Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification".

## Overview

DCAL introduces two novel cross-attention mechanisms:

1. **Global-Local Cross-Attention (GLCA)**: Enhances interactions between global images and local high-response regions
2. **Pair-Wise Cross-Attention (PWCA)**: Regularizes attention learning by treating another image as a distractor

## Components

### 1. Attention Rollout Mechanism (`attention_rollout.py`)

Implements the attention rollout mechanism following the paper's formulation:
- `S̄ = 0.5S + 0.5E` (considering residual connections)
- `Ŝᵢ = S̄ᵢ ⊗ S̄ᵢ₋₁ ⊗ ... ⊗ S̄₁` (accumulated attention)

**Key Classes:**
- `AttentionRollout`: Core rollout computation
- `AttentionRolloutHook`: Automatic attention weight collection during forward pass

**Usage:**
```python
from attention_rollout import AttentionRolloutHook

# Create hook for a Vision Transformer model
rollout_hook = AttentionRolloutHook(model)

# Forward pass to collect attention weights
_ = model(input_tensor)

# Get accumulated attention rollout
rollout = rollout_hook.get_rollout()

# Get CLS attention to patches
cls_attention = rollout_hook.get_cls_attention()

# Select top tokens based on attention
selection_mask = rollout_hook.select_top_tokens(top_ratio=0.1)
```

### 2. Global-Local Cross-Attention (GLCA) (`glca_module.py`)

Implements cross-attention between selected local queries and global key-value pairs:
- `f_GLCA(Q^l, K^g, V^g) = softmax(Q^l K^g^T / √d) V^g`

**Key Classes:**
- `GlobalLocalCrossAttention`: Core GLCA module
- `GLCAWithBackbone`: GLCA integrated with Vision Transformer backbone

**Usage:**
```python
from glca_module import GlobalLocalCrossAttention

# Create GLCA module
glca = GlobalLocalCrossAttention(
    embed_dim=192,
    num_heads=3,
    top_ratio=0.1  # 10% for FGVC, 30% for Re-ID
)

# Set rollout hook for token selection
glca.set_rollout_hook(rollout_hook)

# Forward pass
output = glca(input_embeddings)
```

### 3. Pair-Wise Cross-Attention (PWCA) (`pwca_module.py`)

Implements cross-attention between query of one image and combined key-value from both images:
- `f_PWCA(Q₁, K_c, V_c) = softmax(Q₁ K_c^T / √d) V_c`
- where `K_c = [K₁; K₂]` and `V_c = [V₁; V₂]`

**Key Classes:**
- `PairWiseCrossAttention`: Core PWCA module
- `PWCAWithBackbone`: PWCA integrated with Vision Transformer backbone
- `ImagePairSampler`: Utility for sampling image pairs
- `PWCAWrapper`: Wrapper with automatic pair sampling

**Usage:**
```python
from pwca_module import PairWiseCrossAttention

# Create PWCA module
pwca = PairWiseCrossAttention(
    embed_dim=192,
    num_heads=3,
    training_only=True  # Only used during training
)

# Forward pass with two images
output = pwca(embeddings1, embeddings2, training=True)
```

### 4. Complete DCAL Model (`dcal_example.py`)

Integrates all components into a complete model with:
- Multi-task learning architecture
- Uncertainty-weighted loss
- Training and inference modes

**Key Classes:**
- `DCALModel`: Complete DCAL implementation
- `create_dcal_model()`: Factory function for creating models

**Usage:**
```python
from dcal_example import create_dcal_model

# Create model for FGVC
model = create_dcal_model(
    backbone_name='deit_tiny_patch16_224',
    num_classes=200,  # CUB-200-2011
    top_ratio=0.1     # 10% for FGVC
)

# Training forward pass
sa_logits, glca_logits, pwca_features = model(x, paired_x, training=True)

# Compute uncertainty-weighted loss
loss_fn = nn.CrossEntropyLoss()
total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)

# Inference
model.eval()
combined_logits = model.inference(x)
```

### 5. Training Infrastructure (`training_infrastructure.py`)

Complete training pipeline with data loading, optimization, and loss functions:
- **Data Loaders**: FGVC and Re-ID dataset support
- **Optimizers**: Adam for FGVC, SGD for Re-ID
- **Loss Functions**: Cross-entropy + Triplet loss for Re-ID
- **Training Loops**: Complete training and validation

**Key Classes:**
- `FGVCDataLoader`: Data loader for fine-grained classification
- `ReIDDataLoader`: Data loader for re-identification
- `DCALTrainer`: Complete trainer with checkpointing
- `TripletLoss`: Triplet loss for Re-ID tasks

**Usage:**
```python
from training_infrastructure import create_fgvc_trainer

# Create trainer
trainer, data_loader = create_fgvc_trainer(
    model=model,
    dataset_name='cub',
    data_root='/path/to/cub/dataset',
    batch_size=16,
    epochs=100
)

# Get dataloaders and train
train_loader, val_loader = data_loader.get_dataloaders()
trainer.train(train_loader, val_loader)
```

### 6. Implementation Details (`implementation_details.py`)

Advanced features and implementation details:
- **Stochastic Depth**: Random layer dropping during training
- **Advanced Pair Sampling**: Difficulty-based pair selection
- **Configuration Management**: Task and backbone-specific configs
- **Batch Strategies**: Optimized batch creation for different tasks

**Key Classes:**
- `DCALModelWithStochasticDepth`: Advanced model with regularization
- `AdvancedImagePairSampler`: Sophisticated pair sampling strategies
- `ConfigManager`: Centralized configuration management
- `BatchStrategyManager`: Optimized batch creation

**Usage:**
```python
from implementation_details import create_advanced_dcal_model

# Create advanced model with stochastic depth
model = create_advanced_dcal_model(
    task_type='fgvc',
    backbone_name='deit_tiny_patch16_224',
    num_classes=200
)
```

## Testing

Run the comprehensive test suite to verify all components:

```bash
python test_components.py
```

This will test:
- Attention rollout mechanism
- GLCA module
- PWCA module
- Complete DCAL model
- Gradient flow through all components

## Complete Example

Run the complete example with all components:

```bash
# Quick test (no arguments needed)
python complete_example.py

# Demo mode (no actual training)
python complete_example.py --demo

# FGVC training
python complete_example.py --task fgvc --dataset cub --data_root /path/to/cub --advanced

# Re-ID training
python complete_example.py --task reid --dataset market1501 --data_root /path/to/market1501 --advanced
```

## Key Features

### 1. Attention Rollout
- Follows paper's formulation with residual connections
- Automatic attention weight collection via hooks
- Token selection based on CLS attention to patches

### 2. GLCA Module
- Cross-attention between local queries and global key-value pairs
- Integration with attention rollout for token selection
- Configurable token selection ratio (10% for FGVC, 30% for Re-ID)

### 3. PWCA Module
- Cross-attention with combined key-value from image pairs
- Training-only implementation (no inference cost)
- Automatic image pair sampling strategies

### 4. Multi-Task Learning
- Uncertainty-weighted loss balancing
- Learnable parameters for dynamic loss weighting
- Combined inference strategy

## Configuration

### FGVC Settings
- Input size: 448×448
- Token selection ratio: 10%
- Backbone: DeiT/ViT variants
- Loss: Cross-entropy

### Re-ID Settings
- Input size: 256×128 (pedestrian) / 256×256 (vehicle)
- Token selection ratio: 30%
- Backbone: DeiT/ViT variants
- Loss: Cross-entropy + triplet loss

## Dependencies

- PyTorch >= 1.8.0
- timm >= 0.4.0
- torchvision
- numpy

## Implementation Notes

1. **Library Re-use**: The implementation leverages existing Vision Transformer components from `timm` library
2. **Efficiency**: PWCA is training-only and adds no inference cost
3. **Flexibility**: Supports various backbone architectures (DeiT, ViT, etc.)
4. **Extensibility**: Easy to integrate with different datasets and tasks

## Paper Results Reproduction

The implementation follows the paper's architecture:
- L=12 SA blocks + M=1 GLCA blocks + T=12 PWCA blocks
- PWCA shares weights with SA, GLCA has separate weights
- Dynamic loss weighting with uncertainty method
- Combined inference strategy (SA + GLCA probabilities)

## License

This implementation is provided for research purposes. Please cite the original paper if you use this code in your research. 