# Twin Face Verification - Complete Usage Guide

This guide provides comprehensive instructions for using the DCAL-based twin face verification system for determining whether two face images are from the same person or highly similar different persons (twins).

## 🎯 Overview

The twin verification system adapts the Dual Cross-Attention Learning (DCAL) architecture for binary face verification with a focus on distinguishing between:
- **Same person** (different photos of the same individual)
- **Twin pairs** (hard negatives - genetically identical twins)
- **Non-twin pairs** (easy negatives - different individuals)

## 📁 Dataset Structure

The system uses the ND_TWIN dataset with the following structure:

```
ND_TWIN/
├── id_to_images.json          # Maps ID to image paths
├── twin_pairs_infor.json      # Defines twin relationships
└── images/                    # Face images (224x224)
    ├── id_1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── id_2/
        ├── image3.jpg
        └── image4.jpg
```

### Dataset Files

1. **`id_to_images.json`**: Maps each person ID to a list of image paths
   ```json
   {
     "id_1": ["path/to/image1.jpg", "path/to/image2.jpg"],
     "id_2": ["path/to/image3.jpg", "path/to/image4.jpg"]
   }
   ```

2. **`twin_pairs_infor.json`**: Defines twin relationships and pair information
   ```json
   {
     "twin_pairs": [
       {"id1": "id_1", "id2": "id_2", "relationship": "twin"},
       {"id1": "id_3", "id2": "id_4", "relationship": "same_person"}
     ]
   }
   ```

## 🚀 Quick Start

### 1. Test the System

First, test that all components are working:

```bash
python test_twin_complete.py
```

This will run comprehensive tests on all components and verify the implementation.

### 2. Run Training

```bash
python twin_main.py --mode train \
    --config configs/twin_config.yaml \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

### 3. Evaluate Model

```bash
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --config configs/twin_config.yaml \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

### 4. Run Inference

```bash
python twin_main.py --mode infer \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg \
    --save_results results.json
```

### 5. Generate Visualizations

```bash
python twin_main.py --mode visualize \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg \
    --output_dir visualization_results
```

## 📋 Detailed Usage Instructions

### Training Mode

The training mode supports both standard and progressive training strategies.

#### Basic Training

```bash
python twin_main.py --mode train \
    --preset gpu \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

#### Advanced Training with Custom Config

```bash
python twin_main.py --mode train \
    --config configs/twin_config.yaml \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

#### CPU Training (for limited resources)

```bash
python twin_main.py --mode train \
    --preset cpu \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

#### Fast Training (for quick experiments)

```bash
python twin_main.py --mode train \
    --preset fast \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

### Evaluation Mode

The evaluation mode provides comprehensive metrics for model performance.

#### Standard Evaluation

```bash
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

#### Hard Pairs Evaluation

```bash
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json \
    --evaluate_hard_pairs
```

#### Custom Output Directory

```bash
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json \
    --save_dir my_evaluation_results
```

### Inference Mode

The inference mode allows you to verify individual face pairs.

#### Basic Inference

```bash
python twin_main.py --mode infer \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg
```

#### Inference with Results Saving

```bash
python twin_main.py --mode infer \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg \
    --save_results inference_result.json
```

### Visualization Mode

The visualization mode generates attention maps and similarity visualizations.

#### Basic Visualization

```bash
python twin_main.py --mode visualize \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg
```

#### Custom Output Directory

```bash
python twin_main.py --mode visualize \
    --checkpoint checkpoints/best_model.pth \
    --image1 path/to/face1.jpg \
    --image2 path/to/face2.jpg \
    --output_dir my_visualizations
```

## ⚙️ Configuration

### Configuration Files

The system supports multiple configuration presets:

1. **`configs/twin_config.yaml`**: Default configuration
2. **`configs/twin_config_cpu.yaml`**: CPU-optimized configuration
3. **`configs/twin_config_fast.yaml`**: Fast training configuration

### Key Configuration Parameters

```yaml
# Training device
device: "cuda"  # or "cpu"

# Pair sampling ratios
same_person_ratio: 0.5      # Same person, different photos
twin_pairs_ratio: 0.3       # Hard negatives (twins)
non_twin_ratio: 0.2         # Easy negatives (non-twins)

# Model configuration
backbone: "deit_tiny_patch16_224"
embedding_dim: 512
top_ratio: 0.35             # Token selection ratio for GLCA
learnable_threshold: true

# Training configuration
batch_size: 32
learning_rate: 1e-4
weight_decay: 1e-4
num_epochs: 100
save_frequency: 10

# Loss weights
verification_loss_weight: 1.0
triplet_loss_weight: 0.1
use_dynamic_weighting: true

# Progressive training
progressive_training: true
phase1_epochs: 30           # General face recognition
phase2_epochs: 40           # Twin dataset fine-tuning
phase3_epochs: 30           # Hard negative mining
```

### Configuration Presets

#### Default Preset
- Optimized for GPU training
- Full progressive training
- Standard hyperparameters

#### CPU Preset
- Smaller batch size (16)
- No multiprocessing
- Disabled progressive training
- Optimized for CPU resources

#### Fast Preset
- Reduced epochs (20)
- Smaller batch size (16)
- Faster learning rate
- For quick experiments

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Standard Metrics
- **Accuracy**: Overall classification accuracy
- **AUC**: Area Under ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Verification-Specific Metrics
- **EER**: Equal Error Rate (where FAR = FRR)
- **FAR**: False Accept Rate
- **FRR**: False Reject Rate
- **TAR**: True Accept Rate
- **TRR**: True Reject Rate
- **TAR@FAR**: True Accept Rate at specific False Accept Rate

### Hard Pairs Metrics
- Separate evaluation on twin pairs only
- Focus on the most challenging cases
- Twin-specific performance analysis

## 🎨 Visualization Features

### Attention Maps
- **Attention Rollout**: Shows which facial regions the model focuses on
- **Cross-Attention Visualization**: Displays GLCA attention patterns
- **Token Selection**: Highlights selected high-response tokens

### Similarity Visualization
- **Image Comparison**: Side-by-side image display
- **Similarity Score**: Numerical similarity with confidence
- **Decision Visualization**: Clear same/different indication

### Training History
- **Loss Curves**: Training and validation loss over time
- **Accuracy Curves**: Training and validation accuracy
- **AUC Curves**: Training and validation AUC scores

## 🔧 Advanced Features

### Progressive Training Strategy

The system implements a three-phase progressive training strategy:

1. **Phase 1 (General Face Recognition)**: 
   - Focus on general face recognition
   - Standard triplet loss
   - All pair types

2. **Phase 2 (Twin Dataset Fine-tuning)**:
   - Focus on twin dataset specifics
   - Twin-aware triplet loss
   - Balanced pair sampling

3. **Phase 3 (Hard Negative Mining)**:
   - Focus on challenging twin pairs
   - Smaller margin for twins
   - Hard negative mining

### Dynamic Loss Weighting

The system uses uncertainty-based loss weighting to automatically balance:
- Verification loss (binary cross-entropy)
- Triplet loss (metric learning)

### Learnable Threshold

The model can learn the optimal verification threshold during training, adapting to the specific dataset characteristics.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU preset or reduce batch size
   python twin_main.py --mode train --preset cpu
   ```

2. **Dataset Loading Errors**
   - Verify JSON file formats
   - Check image paths exist
   - Ensure proper file permissions

3. **Model Loading Errors**
   - Verify checkpoint file exists
   - Check model architecture matches
   - Ensure compatible PyTorch version

### Performance Optimization

1. **For CPU Training**:
   - Use `--preset cpu`
   - Reduce batch size
   - Disable progressive training

2. **For GPU Training**:
   - Use `--preset gpu`
   - Increase batch size if memory allows
   - Enable progressive training

3. **For Quick Experiments**:
   - Use `--preset fast`
   - Reduce number of epochs
   - Use smaller model backbone

## 📈 Expected Results

### Performance Benchmarks

Based on the DCAL paper and twin verification adaptation:

- **Overall Accuracy**: 85-95%
- **AUC**: 0.90-0.98
- **EER**: 5-15%
- **TAR@FAR=0.01**: 70-90%

### Training Time Estimates

- **CPU (16GB RAM)**: 8-12 hours for 100 epochs
- **GPU (P100, 16GB RAM)**: 2-4 hours for 100 epochs
- **Fast Preset**: 1-2 hours for 20 epochs

## 🔄 Workflow Examples

### Complete Training and Evaluation Workflow

```bash
# 1. Test the system
python test_twin_complete.py

# 2. Train the model
python twin_main.py --mode train \
    --config configs/twin_config.yaml \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json

# 3. Evaluate the model
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json \
    --evaluate_hard_pairs

# 4. Test inference on sample pairs
python twin_main.py --mode infer \
    --checkpoint checkpoints/best_model.pth \
    --image1 sample_faces/face1.jpg \
    --image2 sample_faces/face2.jpg

# 5. Generate visualizations
python twin_main.py --mode visualize \
    --checkpoint checkpoints/best_model.pth \
    --image1 sample_faces/face1.jpg \
    --image2 sample_faces/face2.jpg
```

### Quick Experiment Workflow

```bash
# 1. Quick training
python twin_main.py --mode train \
    --preset fast \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json

# 2. Quick evaluation
python twin_main.py --mode eval \
    --checkpoint checkpoints/best_model.pth \
    --id_to_images id_to_images_local.json \
    --twin_pairs twin_pairs_infor.json
```

## 📚 Additional Resources

- **Paper**: "Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification"
- **Dataset**: ND_TWIN dataset documentation
- **Implementation**: `twin_adaptation_plan.md` for technical details
- **Testing**: `test_twin_complete.py` for comprehensive testing

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test outputs
3. Verify configuration parameters
4. Check dataset format and paths

The system is designed to be robust and user-friendly, with comprehensive error handling and informative output messages. 