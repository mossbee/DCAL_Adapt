## Core Components to Implement

### 1. **Attention Rollout Mechanism** (Re-use/modify from `vit_rollout.py`)
- **What to re-use**: The `rollout()` function and `VITAttentionRollout` class
- **What to modify**: 
  - Update the rollout formula to match the paper: `S̄ = 0.5S + 0.5E` (considering residual connections)
  - Modify to track accumulated attention from input to layer i: `Ŝᵢ = S̄ᵢ ⊗ S̄ᵢ₋₁ ⊗ ... ⊗ S̄₁`
  - Extract the first row of accumulated attention (CLS token attention to patches)
  - Select top R tokens based on highest responses

### 2. **Global-Local Cross-Attention (GLCA) Module** (New implementation)
- **Core functionality**: Cross-attention between selected local queries and global key-value pairs
- **Key components**:
  - Token selection based on attention rollout results
  - Cross-attention computation: `f_GLCA(Q^l, K^g, V^g) = softmax(Q^l K^g^T / √d) V^g`
  - Integration with existing Vision Transformer architecture

### 3. **Pair-Wise Cross-Attention (PWCA) Module** (New implementation)
- **Core functionality**: Cross-attention between query of one image and combined key-value from both images
- **Key components**:
  - Image pair sampling strategy
  - Key-value concatenation: `K_c = [K₁; K₂]`, `V_c = [V₁; V₂]`
  - Cross-attention computation: `f_PWCA(Q₁, K_c, V_c) = softmax(Q₁ K_c^T / √d) V_c`
  - Training-only implementation (removed during inference)

### 4. **Multi-Task Learning Architecture** (New implementation)
- **Architecture**: L=12 SA blocks + M=1 GLCA blocks + T=12 PWCA blocks
- **Weight sharing**: PWCA shares weights with SA, GLCA has separate weights
- **Dynamic loss weighting**: Uncertainty-based loss balancing with learnable parameters

### 5. **Library Components to Re-use**
- **Vision Transformer backbone**: Use `timm` models (DeiT, ViT) as base
- **Attention mechanisms**: Re-use existing `MultiHeadAttention` implementations
- **Patch embedding**: Use existing `PatchEmbed` from timm
- **MLP blocks**: Use existing feed-forward network implementations
- **Transformer blocks**: Extend existing transformer block implementations

### 6. **Training Infrastructure** (New implementation)
- **Loss functions**: Cross-entropy for FGVC, cross-entropy + triplet loss for Re-ID
- **Optimization**: Adam optimizer with cosine learning rate decay
- **Data loading**: Support for FGVC datasets (CUB, Cars, Aircraft) and Re-ID datasets
- **Inference strategy**: Remove PWCA modules, combine SA and GLCA outputs

### 7. **Key Implementation Details**
- **Token selection ratio**: R=10% for FGVC, R=30% for Re-ID
- **Input sizes**: 448×448 for FGVC, 256×128/256×256 for Re-ID
- **Batch strategies**: Image pairs for PWCA training
- **Stochastic depth**: Random layer dropping during training

The main challenge will be integrating these components into a cohesive architecture while maintaining the efficiency and effectiveness described in the paper. The attention rollout mechanism from `vit_rollout.py` provides a good foundation, but needs significant modification to match the paper's specific formulation.