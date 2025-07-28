#!/usr/bin/env python3
"""
Visualization pipeline for twin face verification model.

This script generates attention visualizations to understand which facial features
the model focuses on to distinguish between same person and twin pairs.
"""

import os
import sys
import torch
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.build_model import get_model
from experiment.build_loader import get_loader
from utils.attention_visualizer import AttentionVisualizer, visualize_model_attention
from utils.setup_logging import get_logger
from utils.misc import set_seed


logger = get_logger("Prompt_CAM_Visualization")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace-like object
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return Config(**config)


def visualize_attention_maps(config, model_path, save_dir, num_samples=20):
    """
    Generate attention visualizations for trained model.
    
    Args:
        config: Configuration object
        model_path: Path to trained model checkpoint
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    logger.info(f"Loading model from {model_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, _, _ = get_model(config)
    model = model.to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights")
    else:
        logger.warning(f"Model checkpoint not found at {model_path}")
        return
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_loader(config, logger)
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device)
    
    # Generate visualizations for test set
    logger.info(f"Generating attention visualizations for {num_samples} samples...")
    visualize_model_attention(model, test_loader, save_dir, num_samples=num_samples)
    
    # Generate additional verification pair visualizations
    logger.info("Generating verification pair visualizations...")
    generate_verification_visualizations(model, test_loader, visualizer, save_dir, num_samples=10)


def generate_verification_visualizations(model, data_loader, visualizer, save_dir, num_samples=10):
    """
    Generate visualizations for verification pairs.
    
    Args:
        model: Trained model
        data_loader: DataLoader with verification pairs
        visualizer: AttentionVisualizer instance
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    verification_dir = os.path.join(save_dir, 'verification_pairs')
    os.makedirs(verification_dir, exist_ok=True)
    
    sample_count = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            # Extract verification pair data
            img1 = batch_data['img1'].to(model.device)
            img2 = batch_data['img2'].to(model.device)
            labels = batch_data['label']
            
            # Compute verification scores
            scores = model.compute_verification_score(img1, img2)
            
            # Get attention maps
            attention1_same = model.get_attention_maps(img1, target_class=0)
            attention1_twin = model.get_attention_maps(img1, target_class=1)
            attention2_same = model.get_attention_maps(img2, target_class=0)
            attention2_twin = model.get_attention_maps(img2, target_class=1)
            
            for i in range(min(img1.shape[0], num_samples - sample_count)):
                # Save verification pair visualization
                save_path = os.path.join(verification_dir, f'verification_pair_{sample_count + i}.png')
                visualizer.visualize_verification_pair(
                    img1[i:i+1], img2[i:i+1], 
                    scores[i].item(), labels[i].item(),
                    save_path=save_path
                )
                
                # Save attention comparison for first image
                if attention1_same is not None and attention1_twin is not None:
                    comp_path = os.path.join(verification_dir, f'attention_comparison_{sample_count + i}.png')
                    visualizer.visualize_twin_comparison(
                        img1[i:i+1], img1[i:i+1],
                        attention1_same[i:i+1], attention1_twin[i:i+1],
                        save_path=comp_path,
                        title1="Same Person Attention", title2="Twin Attention"
                    )
                
                sample_count += 1
    
    logger.info(f"Saved verification pair visualizations to {verification_dir}")


def analyze_attention_patterns(model, data_loader, save_dir, num_samples=50):
    """
    Analyze attention patterns across multiple samples.
    
    Args:
        model: Trained model
        data_loader: DataLoader with images
        save_dir: Directory to save analysis
        num_samples: Number of samples to analyze
    """
    analysis_dir = os.path.join(save_dir, 'attention_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    same_attention_maps = []
    twin_attention_maps = []
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            # Get images from batch
            if 'anchor' in batch_data:
                images = batch_data['anchor']
            elif 'img1' in batch_data:
                images = batch_data['img1']
            else:
                continue
            
            images = images.to(model.device)
            
            for i in range(min(images.shape[0], num_samples - sample_count)):
                img = images[i:i+1]
                
                # Get attention maps for both classes
                same_attention = model.get_attention_maps(img, target_class=0)
                twin_attention = model.get_attention_maps(img, target_class=1)
                
                if same_attention is not None and twin_attention is not None:
                    same_attention_maps.append(same_attention.squeeze(0))
                    twin_attention_maps.append(twin_attention.squeeze(0))
                
                sample_count += 1
    
    # Create analysis visualizations
    if same_attention_maps and twin_attention_maps:
        visualizer = AttentionVisualizer(model)
        
        # Same person attention summary
        same_summary_path = os.path.join(analysis_dir, 'same_person_attention_summary.png')
        visualizer.create_attention_summary(same_attention_maps, save_path=same_summary_path)
        
        # Twin attention summary
        twin_summary_path = os.path.join(analysis_dir, 'twin_attention_summary.png')
        visualizer.create_attention_summary(twin_attention_maps, save_path=twin_summary_path)
        
        logger.info(f"Saved attention analysis to {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate attention visualizations for twin face model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate visualizations
    logger.info("Starting attention visualization pipeline...")
    visualize_attention_maps(config, args.model_path, args.save_dir, args.num_samples)
    
    # Load data for additional analysis
    logger.info("Loading data for attention pattern analysis...")
    train_loader, val_loader, test_loader = get_loader(config, logger)
    
    # Load model for analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = get_model(config)
    model = model.to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Analyze attention patterns
    logger.info("Analyzing attention patterns...")
    analyze_attention_patterns(model, test_loader, args.save_dir, num_samples=50)
    
    logger.info(f"Visualization pipeline completed. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()




