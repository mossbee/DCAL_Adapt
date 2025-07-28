# PromptCAM Twin Face Verification

This repository adapts PromptCAM for identical twin face verification using triplet loss with hard negative mining. The system distinguishes between same-person pairs and twin-person pairs using class-specific attention maps.

## Overview

- **Task**: Binary verification (same person vs twin)
- **Model**: PromptCAMD with DINOv2 backbone
- **Loss**: Triplet loss with hard negative mining
- **Dataset**: ND TWIN 2009-2010 (pre-processed face images)
- **Input**: 224x224 face images (no augmentation)

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch torchvision timm wandb iopath dotwiz tabulate

# Set up WandB (for experiment tracking)
export WANDB_API_KEY="your_api_key_here"
```

### 2. Dataset Preparation

Ensure your dataset follows this structure:

```
data/
├── images/
│   ├── person1_img1.jpg
│   ├── person1_img2.jpg
│   └── ...
├── id_to_images.json
├── train_twin_id_pairs.json
└── test_twin_id_pairs.json
```

**JSON Format:**
- `id_to_images.json`: Maps person IDs to image paths
- `train_twin_id_pairs.json` / `test_twin_id_pairs.json`: Twin pair relationships

### 3. Training

#### Fresh Training
```bash
python experiment/run.py \
    --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml \
    --data twin_faces \
    --output_dir ./outputs/twin_faces_experiment
```

#### Fast Training (20% of triplets)
```bash
python experiment/run.py \
    --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml \
    --data twin_faces \
    --output_dir ./outputs/twin_faces_fast \
    --triplet_portion 0.2
```

#### Resume from Checkpoint
```bash
python experiment/run.py \
    --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml \
    --data twin_faces \
    --output_dir ./outputs/twin_faces_experiment \
    --resume
```

### 4. Evaluation

```bash
python experiment/run.py \
    --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml \
    --data twin_faces \
    --output_dir ./outputs/twin_faces_experiment \
    --eval
```

**Output Metrics:**
- EER (Equal Error Rate)
- AUC (Area Under Curve)
- TAR/FAR (True/False Accept Rate)
- Verification Accuracy

### 5. Visualization

#### Generate Attention Maps
```bash
python experiment/visualize_run.py \
    --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml \
    --data twin_faces \
    --model_path ./outputs/twin_faces_experiment/model.pt \
    --output_dir ./visualizations
```

#### Visualization Types
- **Individual Attention**: Single image attention heatmaps
- **Verification Pairs**: Side-by-side attention comparison
- **Attention Summary**: Overall attention pattern analysis

## Configuration

Key parameters in `experiment/config/prompt_cam/dinov2/twin_faces/args.yaml`:

```yaml
# Model
model: dinov2_vitb14
vpt_num: 2  # Same person vs twin
crop_size: 224

# Training
batch_size: 16
epochs: 100
lr: 0.001
margin: 0.3  # Triplet loss margin

# Data
data: twin_faces
data_path: ./data/images

# Training speed optimization
triplet_portion: 1.0  # Fraction of triplets to use (0.0-1.0). Use 0.2 for faster training.
```

## Model Architecture

- **Backbone**: DINOv2 ViT-B/14 (frozen)
- **Prompts**: 2 class-specific prompts for verification
- **Loss**: Triplet loss with hard negative mining
- **Parameters**: ~85.7M total (all trainable)

## Training Details

### Triplet Sampling Strategy
- **Positive pairs**: Same person, different images
- **Negative pairs**: Twin pairs (hard negatives)
- **Batch structure**: (anchor, positive, negative) triplets
- **Speed optimization**: Use `triplet_portion` parameter to reduce training time
  - `triplet_portion=1.0`: Use all possible triplets (default)
  - `triplet_portion=0.2`: Use 20% of triplets (5x faster training)
  - Ensures all twin pairs are represented regardless of portion

### Hard Negative Mining
- Uses twin pair information from JSON files
- For each anchor, samples negative from corresponding twin
- Ensures balanced positive/negative pairs

### Checkpointing
- Model saved every 2 epochs
- Supports resume training
- Best model based on validation EER

## Evaluation Metrics

- **EER**: Equal Error Rate (primary metric)
- **AUC**: Area Under ROC Curve
- **TAR/FAR**: True/False Accept Rate curves
- **Verification Accuracy**: Binary classification accuracy

## Visualization Features

### Attention Maps
- Extract attention from class-specific prompts
- Visualize "same person" vs "twin person" attention
- Overlay attention on original face images

### Output Files
- `attention_heatmaps/`: Individual attention visualizations
- `verification_pairs/`: Side-by-side comparisons
- `attention_summary.png`: Overall attention analysis

## Kaggle Integration

For Kaggle training:

```python
import os
from kaggle_secrets import UserSecretsClient

# Setup
user_secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")

# Clone repo
!git clone "https://github.com/your_username/this_repo.git"
os.chdir('/kaggle/working/this_repo')

# Training
!python experiment/run.py --config experiment/config/prompt_cam/dinov2/twin_faces/args.yaml --data twin_faces
```

## File Structure

```
├── data/dataset/
│   ├── twin_faces.py              # Twin face dataset
│   └── verification_dataset.py    # Verification pairs
├── model/
│   └── twin_face_vpt.py          # Twin face model
├── engine/
│   ├── triplet_loss.py           # Triplet loss
│   └── twin_face_trainer.py      # Training loop
├── utils/
│   ├── verification_metrics.py   # Evaluation metrics
│   └── attention_visualizer.py   # Visualization
├── experiment/
│   ├── run.py                    # Main training script
│   ├── visualize_run.py          # Visualization script
│   └── config/prompt_cam/dinov2/twin_faces/args.yaml
└── test_phase*.py                # Unit tests
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use gradient accumulation

2. **Missing Dependencies**
   ```bash
   pip install iopath dotwiz tabulate
   ```

3. **Dataset Loading Errors**
   - Check JSON file format
   - Verify image paths exist
   - Ensure proper file permissions

### Testing

Run unit tests to verify components:

```bash
python test_phase1.py    # Dataset and metrics
python test_phase2.py    # Model architecture
python test_phase3.py    # Training pipeline
python test_phase4.py    # Visualization
```

## Performance Expectations

- **Target EER**: < 10%
- **Target AUC**: > 0.9
- **Training Time**: 
  - Full training: ~2-4 hours on Tesla P100
  - Fast training (20% triplets): ~30-60 minutes on Tesla P100
- **Memory Usage**: ~12GB VRAM

## Citation

If you use this code, please cite the original PromptCAM paper and mention this adaptation for twin face verification.

## License

[Add your license information here]
