## **Recommended Adaptation Plan: DCAL for Twin Face Verification**

Twin Face Verification: Determine whether two given face images are from same person or not - based on a threshold of similarity or probability. Twin faces are having fine-grained diferences.

### **Why This Task is Perfect for DCAL:**
- Twin face verification is essentially a **Re-ID problem** (not FGVC) - you're comparing identities, not classifying categories
- DCAL's pair-wise attention is naturally suited for verification tasks
- The fine-grained nature aligns perfectly with detecting subtle differences between twins

### **Core Architecture (Siamese Network)**

```
Input: Twin Face Pair (I₁, I₂)
├── Shared DCAL Backbone (ViT/DeiT)
│   ├── Self-Attention (SA) blocks: L=12
│   ├── Global-Local Cross-Attention (GLCA): M=1  
│   └── Pair-Wise Cross-Attention (PWCA): T=12 (training only)
├── Feature Extraction: Concatenate SA + GLCA class tokens
├── Embedding Projection: 768 → 512 dimensions
└── Similarity Computation: Cosine similarity + threshold
```

### **Key Adaptations:**

#### 1. **GLCA Configuration for Faces**
- **R = 35-40%** (higher than Re-ID's 30% due to twin subtlety)

#### 2. **Enhanced PWCA Strategy** 
```python
# Strategic pair sampling during training
pair_composition = {
    "twin_pairs": 30%            # Hard negatives (twins)
    "same_person": 50%,          # Positive pairs  
    "random_negatives": 20%,     # Easy negatives (non-twins)
}
# There should be three numbers in config control this ratio. Read the following for more information

# Training pairs generation
training_pairs = [
    (id1_img1, id1_img2, 1),  # Same person, different photos
    (id1_img1, id1_twin_img1, 0),  # Twins  
    (id1_img1, id1_non_twin_img1, 0)  # Non-twin
]
```

#### 3. **Multi-Component Loss Function**

```python
total_loss = (
    λ₁ * verification_loss +      # For training a classifier to predict “same” or “different”.
    λ₂ * triplet_loss          # Metric learning loss, directly optimize embedding distance  
)

# Dynamic loss weighting (following DCAL's uncertainty method)
weights = [w₁, w₂] # Learnable parameters
```

```python
def verification_loss(embeddings1, embeddings2, labels):
    """Binary cross-entropy for same/different classification"""
    similarity = cosine_similarity(embeddings1, embeddings2)
    return BCE(similarity, labels)

def twin_triplet_loss(anchor, positive, negative, margin=0.3):
    """Smaller margin for twin faces due to high similarity"""
    return max(0, margin + d(anchor, positive) - d(anchor, negative))
```

#### 4. **Training Strategy**
- **Input Resolution**: 224×224 (sufficient for facial details)
- **Batch Composition**: 8 identities × 4 images each = 32 batch size
- **Progressive Training**:
  1. Phase 1: General face recognition pretraining
  2. Phase 2: Twin dataset fine-tuning with easier pairs
  3. Phase 3: Hard negative mining with most challenging twins

#### 5. **Inference Pipeline**
```python
def verify_twins(face1, face2):
    # Extract embeddings (PWCA removed)
    emb1 = model.extract_features(face1)  # SA + GLCA
    emb2 = model.extract_features(face2)
    
    # Compute similarity
    similarity = cosine_similarity(emb1, emb2)
    prediction = similarity > threshold
    confidence = softmax(similarity)
    
    return prediction, confidence, similarity
```

#### 6. Evaluation Metrics

```python
metrics = {
    "accuracy": binary_accuracy,
    "AUC": roc_auc_score, 
    "EER": equal_error_rate,
    "TAR@FAR": true_accept_rate_at_false_accept_rate,
    "verification_rate": verification_accuracy_at_different_thresholds
}
```

### **Implementation Roadmap:**

#### **Phase 1: Foundation**
1. Implement Siamese ViT/DeiT with basic triplet loss
2. Establish baseline performance on twin dataset
3. Add Re-ID style embedding extraction

#### **Phase 2: GLCA Integration**
1. Implement attention rollout mechanism
2. Add GLCA with R=35% local queries
3. Focus on facial landmark regions

#### **Phase 3: PWCA Enhancement**
1. Implement strategic twin-aware PWCA sampling
2. Add multi-component loss with dynamic weighting
3. Progressive training pipeline

#### **Phase 4: Optimization**
1. Hyperparameter tuning and ablation studies
2. Hard negative mining strategies
3. Performance analysis and refinement

### **Requirements**

- There should be an option of training device in config (cpu with 16GB RAM or cuda on P100 with 16GB RAM).
- Dataset should be loaded from paths specified in id_to_images.json. So during training, it always require two json file: id_to_images.json and twin_pairs_infor.json
- Pretrained model weights should be downloaded with timm, instead of loading a local pretrained weights.
- We will eval the model on hard pairs of image only. Hard pairs of images are images from twin person or same person.
- There should be a config option to save trained model after how many epoch in the config file.
- There should be three config option, one to set the ratio of image pairs sampled from same person (for example, someone has 10 images, we can sample total of 10 * (10 - 1) / 2 = 45 same person image pairs from him. If the ratio set to 0.2, then we sample 9 image pairs from him). The second is ratio of image pairs sampled from twin pairs (A has 10 image, A_twin has 12 images, we can sampled total of 10 * 12 = 120 image pairs, if the ratio set to 0.2, then we sample 24 images). The third is the ratio of non-twin pairs id, each non-twin id pairs, we sample all image pairs from them (suppose we have 20 pairs of ids that is non-twin, a ratio of 0.2 means we randomly take out 4 non-twin pairs, then from these 4 non twin pairs we sample all possible image pairs.)