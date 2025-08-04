#!/usr/bin/env python3
"""
Test script for twin verification components
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json

from twin_dataset import TwinDataset, TwinDataLoader
from twin_model import create_twin_model, TwinVerificationModel
from twin_losses import create_twin_loss, CombinedTwinLoss
from twin_trainer import create_twin_trainer


def create_dummy_dataset_files():
    """Create dummy dataset files for testing"""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create dummy id_to_images.json
    id_to_images = {
        "id1": [
            str(temp_dir / "id1_img1.jpg"),
            str(temp_dir / "id1_img2.jpg"),
            str(temp_dir / "id1_img3.jpg")
        ],
        "id2": [
            str(temp_dir / "id2_img1.jpg"),
            str(temp_dir / "id2_img2.jpg"),
            str(temp_dir / "id2_img3.jpg")
        ],
        "id3": [
            str(temp_dir / "id3_img1.jpg"),
            str(temp_dir / "id3_img2.jpg")
        ],
        "id4": [
            str(temp_dir / "id4_img1.jpg"),
            str(temp_dir / "id4_img2.jpg")
        ]
    }
    
    # Create dummy twin_pairs_infor.json
    twin_pairs_info = {
        "train": [
            ["id1", "id2"]  # id1 and id2 are twins
        ],
        "val": [
            ["id3", "id4"]  # id3 and id4 are twins
        ],
        "test": [
            ["id1", "id2"]  # Same twins in test
        ]
    }
    
    # Save files
    with open(temp_dir / "id_to_images.json", 'w') as f:
        json.dump(id_to_images, f, indent=2)
        
    with open(temp_dir / "twin_pairs_infor.json", 'w') as f:
        json.dump(twin_pairs_info, f, indent=2)
        
    return temp_dir, str(temp_dir / "id_to_images.json"), str(temp_dir / "twin_pairs_infor.json")


def test_twin_dataset():
    """Test twin dataset creation and pair generation"""
    print("Testing Twin Dataset...")
    
    # Create dummy dataset files
    temp_dir, id_to_images_path, twin_pairs_path = create_dummy_dataset_files()
    
    try:
        # Create dataset
        dataset = TwinDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            split='train',
            same_person_ratio=0.5,
            twin_pairs_ratio=0.3,
            non_twin_ratio=0.2
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total pairs: {len(dataset)}")
        
        # Test getting an item
        if len(dataset) > 0:
            img1, img2, label = dataset[0]
            print(f"✓ Sample item retrieved")
            print(f"  Label: {label}")
            
        # Test statistics
        stats = dataset.get_pair_statistics()
        print(f"✓ Pair statistics computed")
        print(f"  Same-person pairs: {stats['same_person_pairs']}")
        print(f"  Different-person pairs: {stats['different_person_pairs']}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False
        
    return True


def test_twin_model():
    """Test twin model creation and forward pass"""
    print("\nTesting Twin Model...")
    
    try:
        # Create model
        model = create_twin_model(
            backbone_name='deit_tiny_patch16_224',
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        x1 = torch.randn(batch_size, 3, 224, 224)
        x2 = torch.randn(batch_size, 3, 224, 224)
        
        embeddings1, embeddings2, similarity_scores = model(x1, x2, training=True)
        
        print(f"✓ Forward pass successful")
        print(f"  Embeddings1 shape: {embeddings1.shape}")
        print(f"  Embeddings2 shape: {embeddings2.shape}")
        print(f"  Similarity scores shape: {similarity_scores.shape}")
        
        # Test verification
        predictions, confidence, similarities = model.verify_twins(x1, x2)
        print(f"✓ Verification successful")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Confidence shape: {confidence.shape}")
        
        # Test threshold
        threshold = model.get_threshold()
        print(f"  Current threshold: {threshold:.4f}")
        
        model.set_threshold(0.6)
        new_threshold = model.get_threshold()
        print(f"  New threshold: {new_threshold:.4f}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False
        
    return True


def test_twin_losses():
    """Test twin loss functions"""
    print("\nTesting Twin Losses...")
    
    try:
        # Create loss function
        loss_fn = create_twin_loss(
            verification_weight=1.0,
            triplet_weight=0.1,
            use_dynamic_weighting=True
        )
        
        print(f"✓ Loss function created successfully")
        
        # Test verification loss
        batch_size = 8
        similarity_scores = torch.randn(batch_size)
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        loss = loss_fn(similarity_scores, labels)
        print(f"✓ Verification loss computed: {loss.item():.4f}")
        
        # Test with triplet loss
        anchor = torch.randn(batch_size, 512)
        positive = torch.randn(batch_size, 512)
        negative = torch.randn(batch_size, 512)
        
        loss_with_triplet = loss_fn(similarity_scores, labels, anchor, positive, negative)
        print(f"✓ Combined loss computed: {loss_with_triplet.item():.4f}")
        
        # Test loss weights
        verif_weight, triplet_weight = loss_fn.get_loss_weights()
        print(f"✓ Loss weights retrieved: {verif_weight:.4f}, {triplet_weight:.4f}")
        
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        return False
        
    return True


def test_twin_trainer():
    """Test twin trainer creation"""
    print("\nTesting Twin Trainer...")
    
    try:
        # Create model
        model = create_twin_model(
            backbone_name='deit_tiny_patch16_224',
            embedding_dim=512,
            top_ratio=0.35
        )
        
        # Create trainer
        trainer = create_twin_trainer(
            model=model,
            device='cpu',  # Use CPU for testing
            learning_rate=1e-4,
            weight_decay=1e-4,
            num_epochs=10,
            save_dir='./test_checkpoints',
            save_frequency=5,
            progressive_training=False  # Disable for testing
        )
        
        print(f"✓ Trainer created successfully")
        print(f"  Device: {trainer.device}")
        print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
        print(f"  Save frequency: {trainer.save_frequency}")
        
        # Test checkpoint saving
        trainer.save_checkpoint("test_checkpoint.pth", 0, 0.5)
        print(f"✓ Checkpoint saving works")
        
        # Clean up
        import shutil
        if Path('./test_checkpoints').exists():
            shutil.rmtree('./test_checkpoints')
            
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        return False
        
    return True


def test_integration():
    """Test integration of all components"""
    print("\nTesting Integration...")
    
    try:
        # Create dummy dataset files
        temp_dir, id_to_images_path, twin_pairs_path = create_dummy_dataset_files()
        
        # Create data loader
        data_loader = TwinDataLoader(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            input_size=224,
            same_person_ratio=0.5,
            twin_pairs_ratio=0.3,
            non_twin_ratio=0.2
        )
        
        # Get dataloaders
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        print(f"✓ Data loaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Create model
        model = create_twin_model(
            backbone_name='deit_tiny_patch16_224',
            embedding_dim=512,
            top_ratio=0.35
        )
        
        # Create loss function
        loss_fn = create_twin_loss(
            verification_weight=1.0,
            triplet_weight=0.1,
            use_dynamic_weighting=True
        )
        
        # Test one batch
        for img1, img2, labels in train_loader:
            # Forward pass
            embeddings1, embeddings2, similarity_scores = model(img1, img2, training=True)
            
            # Compute loss
            loss = loss_fn(similarity_scores, labels)
            
            print(f"✓ Integration test successful")
            print(f"  Input shapes: {img1.shape}, {img2.shape}")
            print(f"  Embedding shapes: {embeddings1.shape}, {embeddings2.shape}")
            print(f"  Similarity scores: {similarity_scores.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Loss: {loss.item():.4f}")
            break
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
        
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Twin Verification Components")
    print("=" * 60)
    
    tests = [
        test_twin_dataset,
        test_twin_model,
        test_twin_losses,
        test_twin_trainer,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Twin verification components are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 