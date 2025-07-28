#!/usr/bin/env python3
"""
Standalone training script for twin face verification.
Can be run from any directory.
"""

import os
import sys
import argparse
import yaml

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from experiment.run import basic_run
from utils.misc import set_seed
from utils.setup_logging import get_logger

def setup_parser():
    parser = argparse.ArgumentParser(description='Twin Face Verification Training')
    
    # Config file
    parser.add_argument('--config', type=str, 
                       default='experiment/config/prompt_cam/dinov2/twin_faces/args.yaml',
                       help='Path to YAML config file')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='twin_faces',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='.',
                       help='Data path (use . for absolute paths in JSON)')
    
    # Training arguments
    parser.add_argument('--triplet_portion', type=float, default=1.0,
                       help='Fraction of triplets to use (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epoch', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/twin_faces_experiment',
                       help='Output directory')
    
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Load config file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {}
    
    # Override config with command line arguments
    config.update({
        'data': args.data,
        'data_path': args.data_path,
        'triplet_portion': args.triplet_portion,
        'batch_size': args.batch_size,
        'epoch': args.epoch,
        'lr': args.lr,
        'output_dir': args.output_dir,
    })
    
    # Set default values for required parameters
    defaults = {
        'crop_size': 224,
        'model': 'dinov2',
        'pretrained_weights': 'vit_base_patch14_dinov2',
        'train_type': 'prompt_cam',
        'vpt_num': 2,
        'optimizer': 'sgd',
        'wd': 0.001,
        'momentum': 0.9,
        'warmup_epoch': 20,
        'lr_min': 1e-6,
        'warmup_lr_init': 0,
        'drop_path_rate': 0.1,
        'vpt_dropout': 0.1,
        'vpt_layer': None,
        'vpt_mode': None,
        'final_run': True,
        'debug': True,
        'gpu_num': 1,
        'random_seed': 42,
        'eval_freq': 5,
        'early_patience': 101,
        'store_ckp': True,
        'test_batch_size': 32,
        'normalized': True,
        'full': False,
        'final_acc_hp': True,
        'triplet_margin': 0.3,
        'hard_negative_mining': True,
    }
    
    # Apply defaults for missing values
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Convert to object with attributes
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    params = Config(**config)
    
    # Set random seed
    set_seed(params.random_seed)
    
    # Setup logging
    logger = get_logger("Twin_Face_Training")
    logger.info(f"Starting training with triplet_portion={params.triplet_portion}")
    logger.info(f"Data path: {params.data_path}")
    logger.info(f"Output directory: {params.output_dir}")
    
    # Check if required files exist
    required_files = ['id_to_images.json', 'train_twin_id_pairs.json', 'test_twin_id_pairs.json']
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Required file not found: {file}")
            logger.error("Please ensure you're in the correct directory with the dataset files.")
            return
    
    # Run training
    try:
        basic_run(params)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 