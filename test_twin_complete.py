#!/usr/bin/env python3
"""
Comprehensive test script for twin verification components
Tests Priority 1 and Priority 2 components together
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml

# Import all twin verification components
from twin_dataset import TwinDataset, TwinDataLoader
from twin_model import create_twin_model, TwinVerificationModel
from twin_losses import create_twin_loss, CombinedTwinLoss
from twin_trainer import create_twin_trainer
from twin_config import TwinConfig, ConfigManager, load_twin_config
from twin_evaluation import TwinVerificationMetrics, TwinVerificationEvaluator
from training_infrastructure import create_twin_trainer_with_loader


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


def test_priority_1_components():
    """Test Priority 1 components (core implementation)"""
    print("=" * 60)
    print("Testing Priority 1 Components (Core Implementation)")
    print("=" * 60)
    
    # Test 1: Twin Dataset
    print("\n1. Testing Twin Dataset...")
    temp_dir, id_to_images_path, twin_pairs_path = create_dummy_dataset_files()
    
    try:
        dataset = TwinDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            split='train',
            same_person_ratio=0.5,
            twin_pairs_ratio=0.3,
            non_twin_ratio=0.2
        )
        print("✓ Twin Dataset created successfully")
        
        # Test data loading
        img1, img2, label = dataset[0]
        print(f"✓ Data loading works - shapes: {img1.shape}, {img2.shape}, label: {label}")
        
    except Exception as e:
        print(f"✗ Twin Dataset test failed: {e}")
        return False
    
    # Test 2: Twin Model
    print("\n2. Testing Twin Model...")
    try:
        model = create_twin_model(
            backbone_name='deit_tiny_patch16_224',
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
        print("✓ Twin Model created successfully")
        
        # Test forward pass
        batch_size = 4
        x1 = torch.randn(batch_size, 3, 224, 224)
        x2 = torch.randn(batch_size, 3, 224, 224)
        
        embeddings1, embeddings2, similarity_scores = model(x1, x2, training=True)
        print(f"✓ Forward pass works - embeddings: {embeddings1.shape}, similarity: {similarity_scores.shape}")
        
        # Test verification
        predictions, confidence, similarities = model.verify_twins(x1, x2)
        print(f"✓ Verification works - predictions: {predictions.shape}")
        
    except Exception as e:
        print(f"✗ Twin Model test failed: {e}")
        return False
    
    # Test 3: Twin Losses
    print("\n3. Testing Twin Losses...")
    try:
        loss_fn = create_twin_loss(
            verification_weight=1.0,
            triplet_weight=0.1,
            use_dynamic_weighting=True
        )
        print("✓ Twin Loss function created successfully")
        
        # Test loss computation
        batch_size = 8
        similarity_scores = torch.randn(batch_size)
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        loss = loss_fn(similarity_scores, labels)
        print(f"✓ Loss computation works - loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Twin Losses test failed: {e}")
        return False
    
    # Test 4: Twin Trainer
    print("\n4. Testing Twin Trainer...")
    try:
        trainer = create_twin_trainer(
            model=model,
            device='cpu',  # Use CPU for testing
            learning_rate=1e-4,
            weight_decay=1e-4,
            num_epochs=10,
            save_dir='./test_checkpoints',
            save_frequency=5,
            progressive_training=False
        )
        print("✓ Twin Trainer created successfully")
        
        # Test checkpoint saving
        trainer.save_checkpoint("test_checkpoint.pth", 0, 0.5)
        print("✓ Checkpoint saving works")
        
    except Exception as e:
        print(f"✗ Twin Trainer test failed: {e}")
        return False
    
    print("\n✓ All Priority 1 components working correctly!")
    return True


def test_priority_2_components():
    """Test Priority 2 components (configuration and evaluation)"""
    print("\n" + "=" * 60)
    print("Testing Priority 2 Components (Configuration & Evaluation)")
    print("=" * 60)
    
    # Test 1: Twin Configuration
    print("\n1. Testing Twin Configuration...")
    try:
        # Test default config
        config = TwinConfig()
        print("✓ Default TwinConfig created successfully")
        
        # Test validation
        if config.validate():
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
            return False
        
        # Test preset configs
        gpu_config = ConfigManager.get_preset_config('gpu')
        cpu_config = ConfigManager.get_preset_config('cpu')
        fast_config = ConfigManager.get_preset_config('fast')
        print("✓ Preset configurations loaded successfully")
        
        # Test config creation
        custom_config = ConfigManager.create_config(
            device='cpu',
            batch_size=16,
            num_epochs=50
        )
        print("✓ Custom configuration created successfully")
        
    except Exception as e:
        print(f"✗ Twin Configuration test failed: {e}")
        return False
    
    # Test 2: Twin Evaluation
    print("\n2. Testing Twin Evaluation...")
    try:
        # Create metrics calculator
        metrics_calc = TwinVerificationMetrics()
        print("✓ TwinVerificationMetrics created successfully")
        
        # Test metrics computation
        similarity_scores = np.random.randn(100)
        labels = np.random.randint(0, 2, 100)
        
        metrics = metrics_calc.compute_metrics(similarity_scores, labels)
        print("✓ Metrics computation works")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  EER: {metrics['eer']:.4f}")
        
        # Test optimal threshold finding
        optimal_threshold = metrics_calc.find_optimal_threshold(similarity_scores, labels)
        print(f"✓ Optimal threshold found: {optimal_threshold:.4f}")
        
        # Test evaluator
        evaluator = TwinVerificationEvaluator(save_dir='./test_evaluation')
        print("✓ TwinVerificationEvaluator created successfully")
        
    except Exception as e:
        print(f"✗ Twin Evaluation test failed: {e}")
        return False
    
    print("\n✓ All Priority 2 components working correctly!")
    return True


def test_integration():
    """Test integration of all components"""
    print("\n" + "=" * 60)
    print("Testing Integration of All Components")
    print("=" * 60)
    
    try:
        # Create dummy dataset files
        temp_dir, id_to_images_path, twin_pairs_path = create_dummy_dataset_files()
        
        # Test 1: Data Loader Integration
        print("\n1. Testing Data Loader Integration...")
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
        
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        print("✓ Data loaders created successfully")
        
        # Test 2: Model and Loss Integration
        print("\n2. Testing Model and Loss Integration...")
        model = create_twin_model(
            backbone_name='deit_tiny_patch16_224',
            embedding_dim=512,
            top_ratio=0.35
        )
        
        loss_fn = create_twin_loss(
            verification_weight=1.0,
            triplet_weight=0.1,
            use_dynamic_weighting=True
        )
        print("✓ Model and loss function created successfully")
        
        # Test 3: Training Integration
        print("\n3. Testing Training Integration...")
        trainer = create_twin_trainer(
            model=model,
            device='cpu',
            learning_rate=1e-4,
            weight_decay=1e-4,
            num_epochs=2,  # Very short for testing
            save_dir='./test_checkpoints',
            save_frequency=1,
            progressive_training=False
        )
        print("✓ Trainer created successfully")
        
        # Test 4: End-to-End Forward Pass
        print("\n4. Testing End-to-End Forward Pass...")
        for img1, img2, labels in train_loader:
            # Forward pass
            embeddings1, embeddings2, similarity_scores = model(img1, img2, training=True)
            
            # Compute loss
            loss = loss_fn(similarity_scores, labels)
            
            print("✓ End-to-end forward pass successful")
            print(f"  Input shapes: {img1.shape}, {img2.shape}")
            print(f"  Embedding shapes: {embeddings1.shape}, {embeddings2.shape}")
            print(f"  Similarity scores: {similarity_scores.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Loss: {loss.item():.4f}")
            break
        
        # Test 5: Configuration Integration
        print("\n5. Testing Configuration Integration...")
        config = load_twin_config(preset='fast')
        print("✓ Configuration loaded successfully")
        print(f"  Device: {config.device}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        
        # Test 6: Evaluation Integration
        print("\n6. Testing Evaluation Integration...")
        metrics_calc = TwinVerificationMetrics()
        
        # Simulate evaluation
        all_similarities = []
        all_labels = []
        
        for img1, img2, labels in test_loader:
            with torch.no_grad():
                embeddings1, embeddings2, similarity_scores = model(img1, img2, training=False)
                all_similarities.extend(similarity_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            break  # Just test one batch
        
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        metrics = metrics_calc.compute_metrics(all_similarities, all_labels)
        print("✓ Evaluation integration successful")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
    
    print("\n✓ All integration tests passed!")
    return True


def test_training_infrastructure_integration():
    """Test integration with training infrastructure"""
    print("\n" + "=" * 60)
    print("Testing Training Infrastructure Integration")
    print("=" * 60)
    
    try:
        # Create dummy dataset files
        temp_dir, id_to_images_path, twin_pairs_path = create_dummy_dataset_files()
        
        # Test the convenience function
        trainer, data_loader = create_twin_trainer_with_loader(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            device='cpu',
            batch_size=4,
            num_epochs=2,
            progressive_training=False
        )
        
        print("✓ Training infrastructure integration successful")
        print(f"  Trainer device: {trainer.device}")
        print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # Test data loader
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        print(f"  Data loaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
    except Exception as e:
        print(f"✗ Training infrastructure integration failed: {e}")
        return False
    
    print("\n✓ Training infrastructure integration tests passed!")
    return True


def cleanup():
    """Clean up test files"""
    import shutil
    
    # Remove test directories
    test_dirs = ['./test_checkpoints', './test_evaluation']
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")


def main():
    """Run all tests"""
    print("=" * 80)
    print("COMPREHENSIVE TWIN VERIFICATION COMPONENTS TEST")
    print("=" * 80)
    
    tests = [
        ("Priority 1 Components", test_priority_1_components),
        ("Priority 2 Components", test_priority_2_components),
        ("Component Integration", test_integration),
        ("Training Infrastructure Integration", test_training_infrastructure_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: FAILED with exception: {e}")
    
    # Cleanup
    cleanup()
    
    print("\n" + "=" * 80)
    print(f"FINAL TEST RESULTS: {passed}/{total} test suites passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Twin verification implementation is complete and working.")
        print("\n✅ Priority 1 Components (Core Implementation):")
        print("   - Twin Dataset: Custom dataset for ND_TWIN with pair generation")
        print("   - Twin Model: Siamese DCAL model with embedding extraction")
        print("   - Twin Losses: Verification and triplet losses with dynamic weighting")
        print("   - Twin Trainer: Training pipeline with progressive strategy")
        
        print("\n✅ Priority 2 Components (Configuration & Evaluation):")
        print("   - Twin Configuration: YAML/JSON config management with validation")
        print("   - Twin Evaluation: Comprehensive metrics and evaluation tools")
        print("   - Training Infrastructure Integration: Seamless integration with existing code")
        
        print("\n🚀 Ready for twin face verification training and evaluation!")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 