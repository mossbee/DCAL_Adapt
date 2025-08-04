#!/usr/bin/env python3
"""
Twin Face Verification - Simple Example

This script demonstrates the key features of the twin verification system
with minimal setup and configuration.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import tempfile
import os

# Import twin-specific modules
from twin_config import ConfigManager
from twin_model import create_twin_model
from twin_dataset import TwinDataLoader
from twin_trainer import create_twin_trainer
from twin_evaluation import evaluate_twin_model
from twin_losses import create_twin_loss


def create_sample_dataset():
    """Create a small sample dataset for demonstration"""
    print("Creating sample dataset...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    images_dir = temp_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create sample images (random noise for demonstration)
    sample_images = {}
    for i in range(1, 6):  # 5 IDs
        id_dir = images_dir / f"id_{i}"
        id_dir.mkdir(exist_ok=True)
        
        images = []
        for j in range(3):  # 3 images per ID
            # Create random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = id_dir / f"image_{j}.jpg"
            img.save(img_path)
            images.append(str(img_path))
        
        sample_images[f"id_{i}"] = images
    
    # Create id_to_images.json
    id_to_images_path = temp_dir / "id_to_images.json"
    with open(id_to_images_path, 'w') as f:
        json.dump(sample_images, f, indent=2)
    
    # Create twin_pairs_infor.json
    twin_pairs = {
        "twin_pairs": [
            {"id1": "id_1", "id2": "id_2", "relationship": "twin"},
            {"id1": "id_3", "id2": "id_4", "relationship": "same_person"},
            {"id1": "id_1", "id2": "id_3", "relationship": "non_twin"},
            {"id1": "id_2", "id2": "id_5", "relationship": "non_twin"}
        ]
    }
    
    twin_pairs_path = temp_dir / "twin_pairs_infor.json"
    with open(twin_pairs_path, 'w') as f:
        json.dump(twin_pairs, f, indent=2)
    
    print(f"Sample dataset created in: {temp_dir}")
    return temp_dir, str(id_to_images_path), str(twin_pairs_path)


def demonstrate_model_creation():
    """Demonstrate model creation and basic functionality"""
    print("\n" + "=" * 50)
    print("MODEL CREATION DEMONSTRATION")
    print("=" * 50)
    
    # Create model
    model = create_twin_model(
        backbone_name="deit_tiny_patch16_224",
        embedding_dim=512,
        top_ratio=0.35,
        learnable_threshold=True
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        embeddings = model.extract_features(x)
        print(f"Input shape: {x.shape}")
        print(f"Output embedding shape: {embeddings.shape}")
    
    # Test similarity computation
    emb1 = torch.randn(1, 512)
    emb2 = torch.randn(1, 512)
    similarity = model.compute_similarity(emb1, emb2)
    print(f"Similarity score: {similarity.item():.4f}")
    
    return model


def demonstrate_training_setup():
    """Demonstrate training setup with sample data"""
    print("\n" + "=" * 50)
    print("TRAINING SETUP DEMONSTRATION")
    print("=" * 50)
    
    # Create sample dataset
    temp_dir, id_to_images_path, twin_pairs_path = create_sample_dataset()
    
    # Create configuration
    config = ConfigManager.get_preset_config('fast')
    config.batch_size = 4
    config.num_epochs = 2
    config.progressive_training = False
    
    # Setup device
    device = torch.device('cpu')  # Use CPU for demonstration
    print(f"Using device: {device}")
    
    # Create model
    model = create_twin_model(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        top_ratio=config.top_ratio,
        learnable_threshold=config.learnable_threshold
    )
    
    # Create data loader
    data_loader = TwinDataLoader(
        id_to_images_path=id_to_images_path,
        twin_pairs_path=twin_pairs_path,
        same_person_ratio=config.same_person_ratio,
        twin_pairs_ratio=config.twin_pairs_ratio,
        non_twin_ratio=config.non_twin_ratio,
        batch_size=config.batch_size,
        num_workers=0  # No multiprocessing for CPU
    )
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create trainer
    trainer = create_twin_trainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        save_dir="./demo_checkpoints",
        save_frequency=1,
        progressive_training=False,
        verification_weight=config.verification_loss_weight,
        triplet_weight=config.triplet_loss_weight,
        use_dynamic_weighting=config.use_dynamic_weighting
    )
    
    print("Trainer created successfully!")
    print(f"Training will save checkpoints to: {trainer.save_dir}")
    
    return trainer, train_loader, val_loader, temp_dir


def demonstrate_training():
    """Demonstrate training process"""
    print("\n" + "=" * 50)
    print("TRAINING DEMONSTRATION")
    print("=" * 50)
    
    # Setup training
    trainer, train_loader, val_loader, temp_dir = demonstrate_training_setup()
    
    # Run training
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history(save_path="./demo_checkpoints/training_history.png")
    print("Training history plot saved to: ./demo_checkpoints/training_history.png")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary dataset")
    
    return trainer


def demonstrate_evaluation():
    """Demonstrate evaluation process"""
    print("\n" + "=" * 50)
    print("EVALUATION DEMONSTRATION")
    print("=" * 50)
    
    # Create sample dataset
    temp_dir, id_to_images_path, twin_pairs_path = create_sample_dataset()
    
    # Load trained model (if available)
    checkpoint_path = "./demo_checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Creating a new model for demonstration...")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
    else:
        print(f"Loading trained model from: {checkpoint_path}")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    data_loader = TwinDataLoader(
        id_to_images_path=id_to_images_path,
        twin_pairs_path=twin_pairs_path,
        same_person_ratio=0.5,
        twin_pairs_ratio=0.3,
        non_twin_ratio=0.2,
        batch_size=4,
        num_workers=0
    )
    
    _, _, test_loader = data_loader.get_dataloaders()
    
    # Evaluate model
    print("Running evaluation...")
    results = evaluate_twin_model(
        model=model,
        test_loader=test_loader,
        device='cpu',
        save_dir="./demo_evaluation",
        evaluate_hard_pairs=True
    )
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary dataset")
    
    return results


def demonstrate_inference():
    """Demonstrate inference on sample images"""
    print("\n" + "=" * 50)
    print("INFERENCE DEMONSTRATION")
    print("=" * 50)
    
    # Create sample images
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create two sample face images
    img1_path = temp_dir / "face1.jpg"
    img2_path = temp_dir / "face2.jpg"
    
    # Create random images (in real usage, these would be actual face images)
    img1_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img2_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    Image.fromarray(img1_array).save(img1_path)
    Image.fromarray(img2_array).save(img2_path)
    
    # Load model
    checkpoint_path = "./demo_checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Creating a new model for demonstration...")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
    else:
        print(f"Loading trained model from: {checkpoint_path}")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load and preprocess images
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(img2_path).convert('RGB')
    
    img1_tensor = transform(image1).unsqueeze(0)
    img2_tensor = transform(image2).unsqueeze(0)
    
    # Run inference
    print("Running inference...")
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
    print(f"\nInference Results:")
    print(f"  Similarity Score: {similarity.item():.4f}")
    print(f"  Prediction: {'Same Person' if prediction.item() else 'Different Persons'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary images")
    
    return similarity.item(), prediction.item(), confidence


def demonstrate_visualization():
    """Demonstrate visualization features"""
    print("\n" + "=" * 50)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Create sample images
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create two sample face images
    img1_path = temp_dir / "face1.jpg"
    img2_path = temp_dir / "face2.jpg"
    
    # Create random images with different patterns
    img1_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img2_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    Image.fromarray(img1_array).save(img1_path)
    Image.fromarray(img2_array).save(img2_path)
    
    # Load model
    checkpoint_path = "./demo_checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Creating a new model for demonstration...")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
    else:
        print(f"Loading trained model from: {checkpoint_path}")
        model = create_twin_model(
            backbone_name="deit_tiny_patch16_224",
            embedding_dim=512,
            top_ratio=0.35,
            learnable_threshold=True
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load and preprocess images
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(img2_path).convert('RGB')
    
    img1_tensor = transform(image1).unsqueeze(0)
    img2_tensor = transform(image2).unsqueeze(0)
    
    # Create output directory
    output_dir = Path("./demo_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # Generate similarity visualization
    print("Generating visualizations...")
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
    
    print(f"Visualization saved to: {output_dir / 'similarity_comparison.png'}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary images")


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("TWIN FACE VERIFICATION - COMPLETE DEMONSTRATION")
    print("=" * 60)
    
    print("\nThis demonstration will show you all the key features of the twin verification system.")
    print("It will create sample data, train a model, evaluate it, and demonstrate inference and visualization.")
    
    try:
        # 1. Model creation demonstration
        model = demonstrate_model_creation()
        
        # 2. Training setup demonstration
        trainer, train_loader, val_loader, temp_dir = demonstrate_training_setup()
        
        # 3. Training demonstration
        trainer = demonstrate_training()
        
        # 4. Evaluation demonstration
        results = demonstrate_evaluation()
        
        # 5. Inference demonstration
        similarity, prediction, confidence = demonstrate_inference()
        
        # 6. Visualization demonstration
        demonstrate_visualization()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nWhat was demonstrated:")
        print("✅ Model creation and architecture")
        print("✅ Dataset loading and pair generation")
        print("✅ Training setup and execution")
        print("✅ Model evaluation with comprehensive metrics")
        print("✅ Inference on individual face pairs")
        print("✅ Visualization of results")
        
        print("\nNext steps:")
        print("1. Use your own dataset by replacing the sample data")
        print("2. Run full training with: python twin_main.py --mode train")
        print("3. Evaluate your model with: python twin_main.py --mode eval")
        print("4. Test inference with: python twin_main.py --mode infer")
        print("5. Generate visualizations with: python twin_main.py --mode visualize")
        
        print("\nCheck the generated files:")
        print("- ./demo_checkpoints/: Training checkpoints and history")
        print("- ./demo_evaluation/: Evaluation results and plots")
        print("- ./demo_visualization/: Visualization outputs")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        print("Please check the error message and ensure all dependencies are installed.")
        return False
    
    return True


if __name__ == "__main__":
    main() 