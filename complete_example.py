#!/usr/bin/env python3
"""
Complete DCAL Implementation Example

This script demonstrates the complete Dual Cross-Attention Learning (DCAL) implementation
with all components: Attention Rollout, GLCA, PWCA, training infrastructure, and implementation details.
"""

import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from dcal_example import create_dcal_model
from training_infrastructure import create_fgvc_trainer, create_reid_trainer
from implementation_details import create_advanced_dcal_model, get_implementation_details_summary


def main():
    parser = argparse.ArgumentParser(description='DCAL Complete Implementation Example')
    parser.add_argument('--task', type=str, default='fgvc', choices=['fgvc', 'reid'],
                       help='Task type: fgvc or reid')
    parser.add_argument('--dataset', type=str, default='cub',
                       help='Dataset name (cub, cars, aircraft for FGVC; market1501, dukemtmc, msmt17, veri776 for Re-ID)')
    parser.add_argument('--data_root', type=str, required=False,
                       help='Path to dataset root directory (not needed for demo mode)')
    parser.add_argument('--backbone', type=str, default='deit_tiny_patch16_224',
                       help='Backbone model name')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (will use default if not specified)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (will use default if not specified)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (will use default if not specified)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--advanced', action='store_true',
                       help='Use advanced features (stochastic depth, etc.)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode without actual training')
    
    args = parser.parse_args()
    
    # Print implementation details summary
    print("=" * 60)
    print("DCAL Implementation Details Summary")
    print("=" * 60)
    summary = get_implementation_details_summary()
    
    for category, details in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    
    # Check if dataset directory exists (only for non-demo mode)
    if not args.demo:
        if args.data_root is None:
            print("Error: --data_root is required for training mode!")
            print("Use --demo flag for demo mode without dataset.")
            return
        if not os.path.exists(args.data_root):
            print(f"Error: Dataset directory {args.data_root} does not exist!")
            print("Please provide the correct path to your dataset.")
            return
    
    # Create model
    print(f"\nCreating DCAL model for {args.task} task...")
    
    if args.advanced:
        # Use advanced model with stochastic depth
        model = create_advanced_dcal_model(
            task_type=args.task,
            backbone_name=args.backbone,
            num_classes=200 if args.task == 'fgvc' else 751,  # Default class counts
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        print("✓ Advanced DCAL model created with stochastic depth")
    else:
        # Use basic model
        model = create_dcal_model(
            backbone_name=args.backbone,
            num_classes=200 if args.task == 'fgvc' else 751,
            top_ratio=0.1 if args.task == 'fgvc' else 0.3
        )
        print("✓ Basic DCAL model created")
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.demo:
        print("\n" + "=" * 60)
        print("DEMO MODE - No actual training will be performed")
        print("=" * 60)
        
        # Demo forward pass
        print("\nRunning demo forward pass...")
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy data
        if args.task == 'fgvc':
            batch_size = 4
            input_size = 224  # Use 224 for demo to match model expectations
            x = torch.randn(batch_size, 3, input_size, input_size).to(device)
            paired_x = torch.randn(batch_size, 3, input_size, input_size).to(device)
            targets = torch.randint(0, 200, (batch_size,)).to(device)
        else:  # reid
            batch_size = 8
            input_size = (224, 224)  # Use 224x224 for demo to match model expectations
            x = torch.randn(batch_size, 3, *input_size).to(device)
            paired_x = torch.randn(batch_size, 3, *input_size).to(device)
            targets = torch.randint(0, 751, (batch_size,)).to(device)
        
        # Forward pass
        model.train()
        with torch.no_grad():
            sa_logits, glca_logits, pwca_features = model(x, paired_x, training=True)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
        
        # Inference
        model.eval()
        with torch.no_grad():
            combined_logits = model.inference(x)
        
        print(f"✓ Demo forward pass completed successfully!")
        print(f"  Input shape: {x.shape}")
        print(f"  SA logits shape: {sa_logits.shape}")
        print(f"  GLCA logits shape: {glca_logits.shape}")
        print(f"  PWCA features shape: {pwca_features.shape}")
        print(f"  Combined logits shape: {combined_logits.shape}")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Clean up
        model.remove_hooks()
        print("\n✓ Demo completed successfully!")
        
    else:
        # Actual training
        print(f"\nStarting training for {args.task} task...")
        
        # Create trainer and data loader
        if args.task == 'fgvc':
            trainer, data_loader = create_fgvc_trainer(
                model=model,
                dataset_name=args.dataset,
                data_root=args.data_root,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
                save_dir=args.save_dir
            )
            
            # Get dataloaders
            train_loader, val_loader = data_loader.get_dataloaders()
            
            # Start training
            trainer.train(train_loader, val_loader)
            
        else:  # reid
            trainer, data_loader = create_reid_trainer(
                model=model,
                dataset_name=args.dataset,
                data_root=args.data_root,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
                save_dir=args.save_dir
            )
            
            # Get dataloaders
            train_loader, query_loader, gallery_loader = data_loader.get_dataloaders()
            
            # Start training (using query_loader as validation for simplicity)
            trainer.train(train_loader, query_loader)
        
        print(f"\n✓ Training completed for {args.task} task!")
        print(f"Checkpoints saved in: {args.save_dir}")
    
    print("\n" + "=" * 60)
    print("DCAL Implementation Complete!")
    print("=" * 60)


def run_quick_test():
    """Run a quick test to verify all components work"""
    print("Running quick test of all DCAL components...")
    
    # Test basic model
    model = create_dcal_model(
        backbone_name='deit_tiny_patch16_224',
        num_classes=200,
        top_ratio=0.1
    )
    
    # Test advanced model
    advanced_model = create_advanced_dcal_model(
        task_type='fgvc',
        backbone_name='deit_tiny_patch16_224',
        num_classes=200
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    paired_x = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 200, (batch_size,))
    
    # Basic model
    model.train()
    sa_logits, glca_logits, pwca_features = model(x, paired_x, training=True)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
    
    # Advanced model
    advanced_model.train()
    sa_logits_adv, glca_logits_adv, pwca_features_adv = advanced_model(x, paired_x, training=True)
    total_loss_adv = advanced_model.compute_loss(sa_logits_adv, glca_logits_adv, targets, loss_fn)
    
    print("✓ All components working correctly!")
    print(f"  Basic model loss: {total_loss.item():.4f}")
    print(f"  Advanced model loss: {total_loss_adv.item():.4f}")
    
    # Clean up
    model.remove_hooks()
    advanced_model.remove_hooks()
    
    print("✓ Quick test completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run quick test
        run_quick_test()
    else:
        # Run main function with arguments
        main() 