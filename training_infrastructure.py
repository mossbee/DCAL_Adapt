import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from pathlib import Path

from dcal_example import DCALModel, create_dcal_model

# Import twin verification components
try:
    from twin_dataset import TwinDataLoader
    from twin_model import create_twin_model
    from twin_losses import create_twin_loss
    from twin_trainer import create_twin_trainer
    TWIN_AVAILABLE = True
except ImportError:
    TWIN_AVAILABLE = False


class TripletLoss(nn.Module):
    """
    Triplet loss for Re-ID tasks
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class from anchor)
            
        Returns:
            Triplet loss
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class FGVCDataLoader:
    """
    Data loader for Fine-Grained Visual Categorization datasets
    """
    
    def __init__(self, 
                 dataset_name: str,
                 data_root: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 input_size: int = 448,
                 top_ratio: float = 0.1):
        """
        Args:
            dataset_name: Dataset name ('cub', 'cars', 'aircraft')
            data_root: Path to dataset root
            batch_size: Batch size
            num_workers: Number of workers for data loading
            input_size: Input image size
            top_ratio: Token selection ratio
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.top_ratio = top_ratio
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the specified dataset"""
        if self.dataset_name == 'cub':
            self._load_cub_dataset()
        elif self.dataset_name == 'cars':
            self._load_cars_dataset()
        elif self.dataset_name == 'aircraft':
            self._load_aircraft_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
    def _load_cub_dataset(self):
        """Load CUB-200-2011 dataset"""
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'test')
        
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"CUB-200-2011: {self.num_classes} classes")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def _load_cars_dataset(self):
        """Load Stanford Cars dataset"""
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'test')
        
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"Stanford Cars: {self.num_classes} classes")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def _load_aircraft_dataset(self):
        """Load FGVC-Aircraft dataset"""
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'test')
        
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"FGVC-Aircraft: {self.num_classes} classes")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader


class ReIDDataLoader:
    """
    Data loader for Re-Identification datasets
    """
    
    def __init__(self, 
                 dataset_name: str,
                 data_root: str,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 input_size: Tuple[int, int] = (256, 128),
                 top_ratio: float = 0.3,
                 num_instances: int = 4):
        """
        Args:
            dataset_name: Dataset name ('market1501', 'dukemtmc', 'msmt17', 'veri776')
            data_root: Path to dataset root
            batch_size: Batch size
            num_workers: Number of workers for data loading
            input_size: Input image size (height, width)
            top_ratio: Token selection ratio
            num_instances: Number of instances per ID in a batch
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.top_ratio = top_ratio
        self.num_instances = num_instances
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the specified Re-ID dataset"""
        # This is a simplified implementation
        # In practice, you would need to implement specific dataset classes for each Re-ID dataset
        
        if self.dataset_name == 'market1501':
            self._load_market1501_dataset()
        elif self.dataset_name == 'dukemtmc':
            self._load_dukemtmc_dataset()
        elif self.dataset_name == 'msmt17':
            self._load_msmt17_dataset()
        elif self.dataset_name == 'veri776':
            self._load_veri776_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
    def _load_market1501_dataset(self):
        """Load Market-1501 dataset (simplified)"""
        train_dir = os.path.join(self.data_root, 'train')
        query_dir = os.path.join(self.data_root, 'query')
        gallery_dir = os.path.join(self.data_root, 'gallery')
        
        # Simplified implementation - in practice, you'd need proper Re-ID dataset classes
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.query_dataset = datasets.ImageFolder(query_dir, transform=self.val_transform)
        self.gallery_dataset = datasets.ImageFolder(gallery_dir, transform=self.val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"Market-1501: {self.num_classes} classes")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Query samples: {len(self.query_dataset)}")
        print(f"Gallery samples: {len(self.gallery_dataset)}")
        
    def _load_dukemtmc_dataset(self):
        """Load DukeMTMC-ReID dataset (simplified)"""
        # Similar to Market-1501
        self._load_market1501_dataset()
        
    def _load_msmt17_dataset(self):
        """Load MSMT17 dataset (simplified)"""
        # Similar to Market-1501
        self._load_market1501_dataset()
        
    def _load_veri776_dataset(self):
        """Load VeRi-776 dataset (simplified)"""
        # Similar to Market-1501
        self._load_market1501_dataset()
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, query, and gallery dataloaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        query_loader = DataLoader(
            self.query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        gallery_loader = DataLoader(
            self.gallery_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, query_loader, gallery_loader


class DCALTrainer:
    """
    Trainer for DCAL model
    """
    
    def __init__(self,
                 model: DCALModel,
                 task_type: str = 'fgvc',
                 device: str = 'cuda',
                 learning_rate: float = 5e-4,
                 weight_decay: float = 0.05,
                 momentum: float = 0.9,
                 num_epochs: int = 100,
                 save_dir: str = './checkpoints'):
        """
        Args:
            model: DCAL model
            task_type: Task type ('fgvc' or 'reid')
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            momentum: Momentum for SGD
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup optimizer
        if task_type == 'fgvc':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:  # reid
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
            
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Setup loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        if task_type == 'reid':
            self.triplet_loss = TripletLoss(margin=0.3)
            
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch_fgvc(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch for FGVC task"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Create paired images for PWCA (random sampling)
            paired_images = images[torch.randperm(images.size(0))]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            sa_logits, glca_logits, _ = self.model(images, paired_images, training=True)
            
            # Compute loss
            loss = self.model.compute_loss(sa_logits, glca_logits, targets, self.ce_loss)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Compute accuracy (using combined predictions)
            combined_logits = self.model.inference(images)
            _, predicted = combined_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
        
    def train_epoch_reid(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch for Re-ID task"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Create paired images for PWCA (random sampling)
            paired_images = images[torch.randperm(images.size(0))]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            sa_logits, glca_logits, pwca_features = self.model(images, paired_images, training=True)
            
            # Compute classification loss
            ce_loss = self.ce_loss(sa_logits, targets) + self.ce_loss(glca_logits, targets)
            
            # Compute triplet loss (simplified - in practice, you'd need proper triplet mining)
            if pwca_features.size(0) >= 3:
                anchor = pwca_features[0:1]
                positive = pwca_features[1:2]
                negative = pwca_features[2:3]
                triplet_loss = self.triplet_loss(anchor, positive, negative)
            else:
                triplet_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = ce_loss + 0.1 * triplet_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Compute accuracy
            combined_logits = self.model.inference(images)
            _, predicted = combined_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
        
    def validate_fgvc(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for FGVC task"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass (no PWCA during inference)
                sa_logits, glca_logits, _ = self.model(images, training=False)
                
                # Compute loss
                loss = self.model.compute_loss(sa_logits, glca_logits, targets, self.ce_loss)
                
                # Statistics
                total_loss += loss.item()
                
                # Compute accuracy
                combined_logits = self.model.inference(images)
                _, predicted = combined_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model"""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            # Training
            if self.task_type == 'fgvc':
                train_loss, train_acc = self.train_epoch_fgvc(train_loader)
            else:
                train_loss, train_acc = self.train_epoch_reid(train_loader)
                
            # Validation
            val_loss, val_acc = self.validate_fgvc(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f"best_model.pth", epoch, val_acc)
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, val_acc)
                
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Checkpoint saved: {filename}")
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        
        print(f"Checkpoint loaded: {filename}")
        print(f"Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")


def create_fgvc_trainer(model: DCALModel, 
                       dataset_name: str,
                       data_root: str,
                       **kwargs) -> DCALTrainer:
    """Create trainer for FGVC task"""
    # Create data loader
    data_loader = FGVCDataLoader(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=kwargs.get('batch_size', 16),
        input_size=kwargs.get('input_size', 448),
        top_ratio=kwargs.get('top_ratio', 0.1)
    )
    
    # Create trainer
    trainer = DCALTrainer(
        model=model,
        task_type='fgvc',
        learning_rate=kwargs.get('learning_rate', 5e-4),
        weight_decay=kwargs.get('weight_decay', 0.05),
        num_epochs=kwargs.get('num_epochs', 100),
        **kwargs
    )
    
    return trainer, data_loader


def create_reid_trainer(model: DCALModel,
                       dataset_name: str,
                       data_root: str,
                       **kwargs) -> DCALTrainer:
    """Create trainer for Re-ID task"""
    # Create data loader
    data_loader = ReIDDataLoader(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=kwargs.get('batch_size', 64),
        input_size=kwargs.get('input_size', (256, 128)),
        top_ratio=kwargs.get('top_ratio', 0.3)
    )
    
    # Create trainer
    trainer = DCALTrainer(
        model=model,
        task_type='reid',
        learning_rate=kwargs.get('learning_rate', 0.008),
        weight_decay=kwargs.get('weight_decay', 1e-4),
        momentum=kwargs.get('momentum', 0.9),
        num_epochs=kwargs.get('num_epochs', 120),
        **kwargs
    )
    
    return trainer, data_loader 


def create_twin_trainer_with_loader(id_to_images_path: str,
                                   twin_pairs_path: str,
                                   **kwargs):
    """
    Create twin verification trainer with data loader
    
    Args:
        id_to_images_path: Path to id_to_images.json
        twin_pairs_path: Path to twin_pairs_infor.json
        **kwargs: Additional trainer parameters
        
    Returns:
        Tuple of (trainer, data_loader)
    """
    if not TWIN_AVAILABLE:
        raise ImportError("Twin verification components not available. Please install required dependencies.")
        
    # Create twin model
    model = create_twin_model(
        backbone_name=kwargs.get('backbone', 'deit_tiny_patch16_224'),
        embedding_dim=kwargs.get('embedding_dim', 512),
        top_ratio=kwargs.get('top_ratio', 0.35),
        learnable_threshold=kwargs.get('learnable_threshold', True)
    )
    
    # Create data loader
    data_loader = TwinDataLoader(
        id_to_images_path=id_to_images_path,
        twin_pairs_path=twin_pairs_path,
        batch_size=kwargs.get('batch_size', 32),
        num_workers=kwargs.get('num_workers', 4),
        input_size=kwargs.get('input_size', 224),
        same_person_ratio=kwargs.get('same_person_ratio', 0.5),
        twin_pairs_ratio=kwargs.get('twin_pairs_ratio', 0.3),
        non_twin_ratio=kwargs.get('non_twin_ratio', 0.2),
        max_pairs_per_epoch=kwargs.get('max_pairs_per_epoch', None),
        shuffle=kwargs.get('shuffle', True)
    )
    
    # Create trainer
    trainer = create_twin_trainer(
        model=model,
        device=kwargs.get('device', 'cuda'),
        learning_rate=kwargs.get('learning_rate', 1e-4),
        weight_decay=kwargs.get('weight_decay', 1e-4),
        num_epochs=kwargs.get('num_epochs', 100),
        save_dir=kwargs.get('save_dir', './checkpoints'),
        save_frequency=kwargs.get('save_frequency', 10),
        verification_weight=kwargs.get('verification_loss_weight', 1.0),
        triplet_weight=kwargs.get('triplet_loss_weight', 0.1),
        use_dynamic_weighting=kwargs.get('use_dynamic_weighting', True),
        progressive_training=kwargs.get('progressive_training', True)
    )
    
    return trainer, data_loader 