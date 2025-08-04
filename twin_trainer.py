import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from twin_model import TwinVerificationModel, create_twin_model
from twin_losses import CombinedTwinLoss, create_twin_loss
from twin_dataset import TwinDataLoader


class TwinTrainer:
    """
    Trainer for twin verification model
    
    Extends existing trainer with twin-specific logic, progressive training phases,
    and verification metrics.
    """
    
    def __init__(self,
                 model: TwinVerificationModel,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 num_epochs: int = 100,
                 save_dir: str = './checkpoints',
                 save_frequency: int = 10,
                 verification_weight: float = 1.0,
                 triplet_weight: float = 0.1,
                 use_dynamic_weighting: bool = True,
                 progressive_training: bool = True):
        """
        Args:
            model: Twin verification model
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
            verification_weight: Weight for verification loss
            triplet_weight: Weight for triplet loss
            use_dynamic_weighting: Whether to use dynamic weighting
            progressive_training: Whether to use progressive training strategy
        """
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_frequency = save_frequency
        self.progressive_training = progressive_training
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Setup loss function
        self.loss_fn = create_twin_loss(
            verification_weight=verification_weight,
            triplet_weight=triplet_weight,
            use_dynamic_weighting=use_dynamic_weighting
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_aucs = []
        self.val_aucs = []
        self.best_val_auc = 0.0
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (avg_loss, accuracy, auc)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings1, embeddings2, similarity_scores = self.model(img1, img2, training=True)
            
            # Compute loss
            loss = self.loss_fn(similarity_scores, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            predictions = (similarity_scores > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Compute AUC (convert similarity scores to probabilities)
        similarity_probs = (torch.tensor(all_predictions).float() + 1) / 2
        auc = roc_auc_score(all_labels, similarity_probs)
        
        return avg_loss, accuracy, auc
        
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, accuracy, auc)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_similarities = []
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings1, embeddings2, similarity_scores = self.model(img1, img2, training=False)
                
                # Compute loss
                loss = self.loss_fn(similarity_scores, labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                predictions = (similarity_scores > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_similarities.extend(similarity_scores.cpu().numpy())
                
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Compute AUC
        auc = roc_auc_score(all_labels, all_similarities)
        
        return avg_loss, accuracy, auc
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc, train_auc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_auc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Get current loss weights
            verif_weight, triplet_weight = self.loss_fn.get_loss_weights()
            print(f"  Loss Weights - Verification: {verif_weight:.4f}, Triplet: {triplet_weight:.4f}")
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.save_checkpoint(f"best_model.pth", epoch, val_auc)
                print(f"  ✓ New best model saved (AUC: {val_auc:.4f})")
                
            # Save checkpoint periodically
            if (epoch + 1) % self.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, val_auc)
                
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        
    def save_checkpoint(self, filename: str, epoch: int, val_auc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_auc': val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_aucs': self.train_aucs,
            'val_aucs': self.val_aucs,
            'best_val_auc': self.best_val_auc
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
        self.train_aucs = checkpoint['train_aucs']
        self.val_aucs = checkpoint['val_aucs']
        self.best_val_auc = checkpoint['best_val_auc']
        
        print(f"Checkpoint loaded: {filename}")
        print(f"Epoch: {checkpoint['epoch']}, Val AUC: {checkpoint['val_auc']:.4f}")
        
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.train_accs, label='Train')
        axes[0, 1].plot(self.val_accs, label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(self.train_aucs, label='Train')
        axes[1, 0].plot(self.val_aucs, label='Validation')
        axes[1, 0].set_title('AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Learning rate
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        axes[1, 1].plot(lrs)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history saved to {save_path}")
        else:
            plt.show()


class ProgressiveTwinTrainer(TwinTrainer):
    """
    Progressive trainer with multi-phase training strategy
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_phase = 1
        
    def train_progressive(self, 
                         train_loader: DataLoader, 
                         val_loader: DataLoader,
                         phase1_epochs: int = 30,
                         phase2_epochs: int = 40,
                         phase3_epochs: int = 30):
        """
        Progressive training with three phases
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            phase1_epochs: Number of epochs for phase 1 (general pretraining)
            phase2_epochs: Number of epochs for phase 2 (twin fine-tuning)
            phase3_epochs: Number of epochs for phase 3 (hard negative mining)
        """
        print("Starting progressive training...")
        
        # Phase 1: General face recognition pretraining
        print(f"\n=== Phase 1: General Pretraining ({phase1_epochs} epochs) ===")
        self.current_phase = 1
        self._train_phase(train_loader, val_loader, phase1_epochs, "general")
        
        # Phase 2: Twin dataset fine-tuning
        print(f"\n=== Phase 2: Twin Fine-tuning ({phase2_epochs} epochs) ===")
        self.current_phase = 2
        self._train_phase(train_loader, val_loader, phase2_epochs, "twin")
        
        # Phase 3: Hard negative mining
        print(f"\n=== Phase 3: Hard Negative Mining ({phase3_epochs} epochs) ===")
        self.current_phase = 3
        self._train_phase(train_loader, val_loader, phase3_epochs, "hard")
        
        print("\nProgressive training completed!")
        
    def _train_phase(self, 
                    train_loader: DataLoader, 
                    val_loader: DataLoader,
                    num_epochs: int,
                    phase_name: str):
        """Train a specific phase"""
        start_epoch = len(self.train_losses)
        
        for epoch in range(num_epochs):
            global_epoch = start_epoch + epoch
            
            # Adjust learning rate for current phase
            if phase_name == "general":
                # Higher learning rate for general pretraining
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 2e-4
            elif phase_name == "twin":
                # Medium learning rate for twin fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-4
            else:  # hard
                # Lower learning rate for hard negative mining
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 5e-5
                    
            # Training
            train_loss, train_acc, train_auc = self.train_epoch(train_loader, global_epoch)
            
            # Validation
            val_loss, val_acc, val_auc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            
            # Print progress
            print(f"Phase {self.current_phase} - Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.save_checkpoint(f"best_model_phase{self.current_phase}.pth", global_epoch, val_auc)
                print(f"  ✓ New best model saved (AUC: {val_auc:.4f})")


def create_twin_trainer(model: TwinVerificationModel,
                       device: str = 'cuda',
                       learning_rate: float = 1e-4,
                       weight_decay: float = 1e-4,
                       num_epochs: int = 100,
                       save_dir: str = './checkpoints',
                       save_frequency: int = 10,
                       progressive_training: bool = True,
                       **kwargs) -> TwinTrainer:
    """
    Create a twin trainer
    
    Args:
        model: Twin verification model
        device: Device to use
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N epochs
        progressive_training: Whether to use progressive training
        **kwargs: Additional arguments for trainer
        
    Returns:
        Twin trainer
    """
    if progressive_training:
        return ProgressiveTwinTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            save_dir=save_dir,
            save_frequency=save_frequency,
            progressive_training=progressive_training,
            **kwargs
        )
    else:
        return TwinTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            save_dir=save_dir,
            save_frequency=save_frequency,
            progressive_training=progressive_training,
            **kwargs
        ) 