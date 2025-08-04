#!/usr/bin/env python3
"""
Twin Face Verification - Main Script

This script provides a comprehensive interface for twin face verification using DCAL:
- Training: Progressive training with configurable strategies
- Evaluation: Comprehensive metrics and analysis
- Inference: Single pair and batch verification
- Visualization: Attention maps and results visualization

Usage:
    python twin_main.py --mode train --config configs/twin_config.yaml
    python twin_main.py --mode eval --checkpoint checkpoints/best_model.pth
    python twin_main.py --mode infer --image1 path/to/face1.jpg --image2 path/to/face2.jpg
    python twin_main.py --mode visualize --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys
import os

# Import twin-specific modules
from twin_config import load_twin_config, ConfigManager
from twin_model import create_twin_model
from twin_dataset import TwinDataLoader
from twin_trainer import create_twin_trainer, ProgressiveTwinTrainer
from twin_evaluation import evaluate_twin_model, TwinVerificationEvaluator
from twin_losses import create_twin_loss


def setup_device(config):
    """Setup device based on configuration"""
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def train_mode(args):
    """Training mode"""
    print("=" * 60)
    print("TWIN FACE VERIFICATION - TRAINING MODE")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_twin_config(config_path=args.config)
    else:
        config = ConfigManager.get_preset_config(args.preset)
    
    # Validate configuration
    if not config.validate():
        print("Configuration validation failed!")
        return
    
    # Setup device
    device = setup_device(config)
    
    # Create model
    print(f"\nCreating twin verification model...")
    model = create_twin_model(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        top_ratio=config.top_ratio,
        learnable_threshold=config.learnable_threshold
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create data loader
    print(f"\nSetting up data loader...")
    data_loader = TwinDataLoader(
        id_to_images_path=args.id_to_images,
        twin_pairs_path=args.twin_pairs,
        same_person_ratio=config.same_person_ratio,
        twin_pairs_ratio=config.twin_pairs_ratio,
        non_twin_ratio=config.non_twin_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create trainer
    print(f"\nSetting up trainer...")
    if config.progressive_training:
        trainer = ProgressiveTwinTrainer(
            model=model,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_epochs=config.num_epochs,
            save_dir=config.save_dir,
            save_frequency=config.save_frequency,
            verification_weight=config.verification_loss_weight,
            triplet_weight=config.triplet_loss_weight,
            use_dynamic_weighting=config.use_dynamic_weighting,
            progressive_training=config.progressive_training
        )
        
        # Progressive training
        print("Starting progressive training...")
        trainer.train_progressive(
            train_loader=train_loader,
            val_loader=val_loader,
            phase1_epochs=config.phase1_epochs,
            phase2_epochs=config.phase2_epochs,
            phase3_epochs=config.phase3_epochs
        )
    else:
        trainer = create_twin_trainer(
            model=model,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_epochs=config.num_epochs,
            save_dir=config.save_dir,
            save_frequency=config.save_frequency,
            progressive_training=False,
            verification_weight=config.verification_loss_weight,
            triplet_weight=config.triplet_loss_weight,
            use_dynamic_weighting=config.use_dynamic_weighting
        )
        
        # Standard training
        print("Starting standard training...")
        trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history(save_path=f"{config.save_dir}/training_history.png")
    
    print(f"\n✓ Training completed!")
    print(f"Checkpoints saved in: {config.save_dir}")
    print(f"Training history plot saved as: {config.save_dir}/training_history.png")


def eval_mode(args):
    """Evaluation mode"""
    print("=" * 60)
    print("TWIN FACE VERIFICATION - EVALUATION MODE")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_twin_config(config_path=args.config)
    else:
        config = ConfigManager.get_preset_config(args.preset)
    
    # Setup device
    device = setup_device(config)
    
    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model = create_twin_model(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        top_ratio=config.top_ratio,
        learnable_threshold=config.learnable_threshold
    )
    
    # Load checkpoint with compatibility for older PyTorch versions
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with weights_only=False: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Create data loader for evaluation
    print(f"\nSetting up evaluation data loader...")
    data_loader = TwinDataLoader(
        id_to_images_path=args.id_to_images,
        twin_pairs_path=args.twin_pairs,
        same_person_ratio=config.same_person_ratio,
        twin_pairs_ratio=config.twin_pairs_ratio,
        non_twin_ratio=config.non_twin_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    _, _, test_loader = data_loader.get_dataloaders()
    
    # Evaluate model
    print(f"\nStarting evaluation...")
    results = evaluate_twin_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir or "./evaluation_results",
        evaluate_hard_pairs=args.evaluate_hard_pairs
    )
    
    # Print results
    print(f"\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    if 'hard_pairs_metrics' in results:
        print(f"\nHard Pairs Results:")
        for metric, value in results['hard_pairs_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print(f"\n✓ Evaluation completed!")
    print(f"Results saved in: {args.save_dir or './evaluation_results'}")


def infer_mode(args):
    """Inference mode"""
    print("=" * 60)
    print("TWIN FACE VERIFICATION - INFERENCE MODE")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_twin_config(config_path=args.config)
    else:
        config = ConfigManager.get_preset_config(args.preset)
    
    # Setup device
    device = setup_device(config)
    
    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model = create_twin_model(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        top_ratio=config.top_ratio,
        learnable_threshold=config.learnable_threshold
    )
    
    # Load checkpoint with compatibility for older PyTorch versions
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with weights_only=False: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess images
    print(f"\nLoading images...")
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images
    image1 = Image.open(args.image1).convert('RGB')
    image2 = Image.open(args.image2).convert('RGB')
    
    # Preprocess
    img1_tensor = transform(image1).unsqueeze(0).to(device)
    img2_tensor = transform(image2).unsqueeze(0).to(device)
    
    # Run inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        # Extract embeddings
        emb1 = model.extract_features(img1_tensor)
        emb2 = model.extract_features(img2_tensor)
        
        # Compute similarity
        similarity = model.compute_similarity(emb1, emb2)
        
        # Get prediction
        threshold = model.threshold if hasattr(model, 'threshold') else 0.5
        prediction = similarity > threshold
        confidence = torch.sigmoid(similarity).item()
    
    # Print results
    print(f"\n" + "=" * 40)
    print("INFERENCE RESULTS")
    print("=" * 40)
    print(f"Similarity Score: {similarity.item():.4f}")
    print(f"Prediction: {'Same Person' if prediction.item() else 'Different Persons'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # Save results if requested
    if args.save_results:
        results = {
            'image1': args.image1,
            'image2': args.image2,
            'similarity_score': similarity.item(),
            'prediction': prediction.item(),
            'confidence': confidence,
            'threshold': threshold
        }
        
        output_path = Path(args.save_results)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print(f"\n✓ Inference completed!")


def visualize_mode(args):
    """Visualization mode"""
    print("=" * 60)
    print("TWIN FACE VERIFICATION - VISUALIZATION MODE")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_twin_config(config_path=args.config)
    else:
        config = ConfigManager.get_preset_config(args.preset)
    
    # Setup device
    device = setup_device(config)
    
    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model = create_twin_model(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        top_ratio=config.top_ratio,
        learnable_threshold=config.learnable_threshold
    )
    
    # Load checkpoint with compatibility for older PyTorch versions
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with weights_only=False: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess images
    print(f"\nLoading images for visualization...")
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images
    image1 = Image.open(args.image1).convert('RGB')
    image2 = Image.open(args.image2).convert('RGB')
    
    # Preprocess
    img1_tensor = transform(image1).unsqueeze(0).to(device)
    img2_tensor = transform(image2).unsqueeze(0).to(device)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # Create output directory
    output_dir = Path(args.output_dir or "./visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate attention maps
    from twin_evaluation import plot_attention_maps
    plot_attention_maps(
        model=model,
        image1=img1_tensor,
        image2=img2_tensor,
        save_path=str(output_dir / "attention_maps.png")
    )
    
    # Generate similarity visualization
    with torch.no_grad():
        emb1 = model.extract_features(img1_tensor)
        emb2 = model.extract_features(img2_tensor)
        similarity = model.compute_similarity(emb1, emb2)
    
    # Create similarity visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original images
    ax1.imshow(image1)
    ax1.set_title("Image 1")
    ax1.axis('off')
    
    ax2.imshow(image2)
    ax2.set_title("Image 2")
    ax2.axis('off')
    
    # Similarity score
    ax3.text(0.5, 0.5, f"Similarity: {similarity.item():.4f}", 
             ha='center', va='center', fontsize=16, transform=ax3.transAxes)
    ax3.set_title("Similarity Score")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualizations completed!")
    print(f"Results saved in: {output_dir}")
    print(f"  - attention_maps.png: Attention rollout visualization")
    print(f"  - similarity_comparison.png: Image comparison with similarity score")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Twin Face Verification - Main Script')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'eval', 'infer', 'visualize'],
                       help='Mode: train, eval, infer, or visualize')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'cpu', 'gpu', 'fast'],
                       help='Preset configuration')
    
    # Data paths
    parser.add_argument('--id_to_images', type=str,
                       help='Path to id_to_images.json file (required for train and eval modes)')
    parser.add_argument('--twin_pairs', type=str,
                       help='Path to twin_pairs_infor.json file (required for train and eval modes)')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (required for eval, infer, visualize)')
    
    # Images for inference/visualization
    parser.add_argument('--image1', type=str,
                       help='Path to first image (required for infer, visualize)')
    parser.add_argument('--image2', type=str,
                       help='Path to second image (required for infer, visualize)')
    
    # Output
    parser.add_argument('--save_dir', type=str,
                       help='Directory to save results')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save visualizations')
    parser.add_argument('--save_results', type=str,
                       help='Path to save inference results as JSON')
    
    # Evaluation options
    parser.add_argument('--evaluate_hard_pairs', action='store_true',
                       help='Evaluate on hard pairs only')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ['train', 'eval']:
        if not args.id_to_images:
            print("Error: --id_to_images is required for train and eval modes!")
            return
        if not args.twin_pairs:
            print("Error: --twin_pairs is required for train and eval modes!")
            return
    
    if args.mode in ['eval', 'infer', 'visualize'] and not args.checkpoint:
        print("Error: --checkpoint is required for eval, infer, and visualize modes!")
        return
    
    if args.mode in ['infer', 'visualize']:
        if not args.image1 or not args.image2:
            print("Error: --image1 and --image2 are required for infer and visualize modes!")
            return
    
    # Run appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)
    elif args.mode == 'visualize':
        visualize_mode(args)


if __name__ == "__main__":
    main() 