import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Union
import random
import numpy as np


class StochasticDepth(nn.Module):
    """
    Stochastic Depth for regularization during training
    """
    
    def __init__(self, drop_prob: float = 0.1):
        """
        Args:
            drop_prob: Probability of dropping a layer during training
        """
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with stochastic depth
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (possibly dropped during training)
        """
        if not self.training or self.drop_prob == 0.0:
            return x
            
        # Randomly drop the layer
        if random.random() < self.drop_prob:
            return x
            
        # Scale the output to maintain expected value
        scale = 1.0 / (1.0 - self.drop_prob)
        return x * scale


class DCALModelWithStochasticDepth(nn.Module):
    """
    DCAL model with stochastic depth regularization
    """
    
    def __init__(self,
                 backbone_name: str = 'deit_tiny_patch16_224',
                 num_classes: int = 1000,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 dropout: float = 0.0,
                 top_ratio: float = 0.1,
                 head_fusion: str = "mean",
                 training_only_pwca: bool = True,
                 stochastic_depth_prob: float = 0.1):
        """
        Args:
            backbone_name: Name of the backbone model from timm
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_ratio: Ratio of top tokens to select for GLCA
            head_fusion: How to fuse attention heads for rollout
            training_only_pwca: Whether PWCA is training-only
            stochastic_depth_prob: Probability for stochastic depth
        """
        super().__init__()
        
        from dcal_example import DCALModel
        
        # Create base DCAL model
        self.base_model = DCALModel(
            backbone_name=backbone_name,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_ratio=top_ratio,
            head_fusion=head_fusion,
            training_only_pwca=training_only_pwca
        )
        
        # Add stochastic depth to GLCA and PWCA modules
        self.glca_stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.pwca_stochastic_depth = StochasticDepth(stochastic_depth_prob)
        
        # Override the forward methods to include stochastic depth
        self._original_glca_forward = self.base_model.glca.forward
        self._original_pwca_forward = self.base_model.pwca.forward
        
        # Patch the forward methods
        self.base_model.glca.forward = self._glca_forward_with_stochastic_depth
        self.base_model.pwca.forward = self._pwca_forward_with_stochastic_depth
        
    def _glca_forward_with_stochastic_depth(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """GLCA forward pass with stochastic depth"""
        output = self._original_glca_forward(x, layer_idx)
        return self.glca_stochastic_depth(output)
        
    def _pwca_forward_with_stochastic_depth(self, x1: torch.Tensor, x2: torch.Tensor, training: bool = True) -> torch.Tensor:
        """PWCA forward pass with stochastic depth"""
        output = self._original_pwca_forward(x1, x2, training)
        return self.pwca_stochastic_depth(output)
        
    def forward(self, 
                x: torch.Tensor,
                paired_x: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through DCAL model with stochastic depth"""
        return self.base_model(x, paired_x, training)
        
    def compute_loss(self, 
                     sa_logits: torch.Tensor,
                     glca_logits: torch.Tensor,
                     targets: torch.Tensor,
                     loss_fn: nn.Module) -> torch.Tensor:
        """Compute uncertainty-weighted loss"""
        return self.base_model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
        
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode (no PWCA)"""
        return self.base_model.inference(x)
        
    def remove_hooks(self):
        """Remove attention rollout hooks"""
        self.base_model.remove_hooks()


class AdvancedImagePairSampler:
    """
    Advanced image pair sampling strategies for PWCA
    """
    
    def __init__(self, 
                 dataset_size: int,
                 same_class_prob: float = 0.5,
                 intra_class_prob: float = 0.3,
                 inter_class_prob: float = 0.7):
        """
        Args:
            dataset_size: Size of the dataset
            same_class_prob: Probability of sampling from same class
            intra_class_prob: Probability of sampling intra-class pairs
            inter_class_prob: Probability of sampling inter-class pairs
        """
        self.dataset_size = dataset_size
        self.same_class_prob = same_class_prob
        self.intra_class_prob = intra_class_prob
        self.inter_class_prob = inter_class_prob
        self.indices = list(range(dataset_size))
        
    def sample_pair_advanced(self, 
                           current_idx: int, 
                           labels: Optional[List[int]] = None,
                           difficulty_level: str = 'medium') -> int:
        """
        Advanced pair sampling with difficulty levels
        
        Args:
            current_idx: Current image index
            labels: List of labels for all images
            difficulty_level: Difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            Paired image index
        """
        if labels is None:
            return self._random_sampling(current_idx)
            
        current_label = labels[current_idx]
        
        # Adjust probabilities based on difficulty level
        if difficulty_level == 'easy':
            # Prefer same class
            adjusted_same_class_prob = 0.8
        elif difficulty_level == 'hard':
            # Prefer different class
            adjusted_same_class_prob = 0.2
        else:  # medium
            adjusted_same_class_prob = self.same_class_prob
            
        # Sample based on adjusted probability
        if random.random() < adjusted_same_class_prob:
            return self._sample_same_class(current_idx, current_label, labels)
        else:
            return self._sample_different_class(current_idx, current_label, labels)
            
    def _random_sampling(self, current_idx: int) -> int:
        """Random sampling without class consideration"""
        available_indices = [i for i in self.indices if i != current_idx]
        return random.choice(available_indices)
        
    def _sample_same_class(self, current_idx: int, current_label: int, labels: List[int]) -> int:
        """Sample from same class"""
        same_class_indices = [i for i, label in enumerate(labels) 
                            if label == current_label and i != current_idx]
        if same_class_indices:
            return random.choice(same_class_indices)
        return self._random_sampling(current_idx)
        
    def _sample_different_class(self, current_idx: int, current_label: int, labels: List[int]) -> int:
        """Sample from different class"""
        different_class_indices = [i for i, label in enumerate(labels) 
                                 if label != current_label and i != current_idx]
        if different_class_indices:
            return random.choice(different_class_indices)
        return self._random_sampling(current_idx)


class ConfigManager:
    """
    Configuration manager for DCAL implementation details
    """
    
    # Default configurations for different tasks
    FGVC_CONFIG = {
        'input_size': 448,
        'top_ratio': 0.1,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'weight_decay': 0.05,
        'num_epochs': 100,
        'stochastic_depth_prob': 0.1,
        'head_fusion': 'mean'
    }
    
    REID_CONFIG = {
        'input_size': (256, 128),  # (height, width)
        'top_ratio': 0.3,
        'batch_size': 64,
        'learning_rate': 0.008,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'num_epochs': 120,
        'stochastic_depth_prob': 0.1,
        'head_fusion': 'mean'
    }
    
    TWIN_CONFIG = {
        'input_size': 224,
        'top_ratio': 0.35,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'stochastic_depth_prob': 0.1,
        'head_fusion': 'mean',
        'embedding_dim': 512,
        'same_person_ratio': 0.5,
        'twin_pairs_ratio': 0.3,
        'non_twin_ratio': 0.2,
        'verification_loss_weight': 1.0,
        'triplet_loss_weight': 0.1,
        'use_dynamic_weighting': True,
        'progressive_training': True
    }
    
    # Backbone-specific configurations
    BACKBONE_CONFIGS = {
        'deit_tiny_patch16_224': {
            'embed_dim': 192,
            'num_heads': 3,
            'num_layers': 12
        },
        'deit_small_patch16_224': {
            'embed_dim': 384,
            'num_heads': 6,
            'num_layers': 12
        },
        'deit_base_patch16_224': {
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12
        },
        'vit_base_patch16_224': {
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12
        }
    }
    
    @classmethod
    def get_config(cls, task_type: str, backbone_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get configuration for specific task and backbone
        
        Args:
            task_type: Task type ('fgvc', 'reid', or 'twin')
            backbone_name: Backbone model name
            **kwargs: Additional configuration parameters
            
        Returns:
            Configuration dictionary
        """
        # Get base configuration
        if task_type == 'fgvc':
            config = cls.FGVC_CONFIG.copy()
        elif task_type == 'reid':
            config = cls.REID_CONFIG.copy()
        elif task_type == 'twin':
            config = cls.TWIN_CONFIG.copy()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        # Add backbone-specific configuration
        if backbone_name in cls.BACKBONE_CONFIGS:
            config.update(cls.BACKBONE_CONFIGS[backbone_name])
            
        # Override with provided kwargs
        config.update(kwargs)
        
        return config


class BatchStrategyManager:
    """
    Manager for different batch strategies
    """
    
    @staticmethod
    def create_fgvc_batch(images: torch.Tensor, 
                         targets: torch.Tensor,
                         batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batch for FGVC training
        
        Args:
            images: Input images
            targets: Target labels
            batch_size: Batch size
            
        Returns:
            Tuple of (images, targets, paired_images)
        """
        # For FGVC, we use random pairing
        paired_images = images[torch.randperm(images.size(0))]
        return images, targets, paired_images
        
    @staticmethod
    def create_reid_batch(images: torch.Tensor,
                         targets: torch.Tensor,
                         batch_size: int = 64,
                         num_instances: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batch for Re-ID training with proper instance sampling
        
        Args:
            images: Input images
            targets: Target labels
            batch_size: Batch size
            num_instances: Number of instances per ID
            
        Returns:
            Tuple of (images, targets, paired_images)
        """
        # For Re-ID, we ensure proper instance sampling
        # This is a simplified implementation
        # In practice, you'd need more sophisticated sampling strategies
        
        # Random pairing for now
        paired_images = images[torch.randperm(images.size(0))]
        return images, targets, paired_images


class InputSizeManager:
    """
    Manager for handling different input sizes
    """
    
    @staticmethod
    def get_transform_for_size(input_size: Union[int, Tuple[int, int]], 
                              task_type: str = 'fgvc') -> Dict[str, Any]:
        """
        Get transforms for specific input size and task
        
        Args:
            input_size: Input size (int for square, tuple for rectangular)
            task_type: Task type ('fgvc' or 'reid')
            
        Returns:
            Dictionary with train and val transforms
        """
        if isinstance(input_size, int):
            # Square input (FGVC)
            resize_size = (input_size, input_size)
        else:
            # Rectangular input (Re-ID)
            resize_size = input_size
            
        train_transform = {
            'resize': resize_size,
            'random_horizontal_flip': True,
            'random_rotation': 10,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        val_transform = {
            'resize': resize_size,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        return {
            'train': train_transform,
            'val': val_transform
        }


def create_advanced_dcal_model(task_type: str = 'fgvc',
                              backbone_name: str = 'deit_tiny_patch16_224',
                              num_classes: int = 200,
                              **kwargs) -> DCALModelWithStochasticDepth:
    """
    Create advanced DCAL model with all implementation details
    
    Args:
        task_type: Task type ('fgvc' or 'reid')
        backbone_name: Backbone model name
        num_classes: Number of classes
        **kwargs: Additional configuration parameters
        
    Returns:
        Advanced DCAL model with stochastic depth
    """
    # Get configuration
    config = ConfigManager.get_config(task_type, backbone_name, **kwargs)
    
    # Create model
    model = DCALModelWithStochasticDepth(
        backbone_name=backbone_name,
        num_classes=num_classes,
        embed_dim=config.get('embed_dim', 192),
        num_heads=config.get('num_heads', 3),
        dropout=config.get('dropout', 0.0),
        top_ratio=config.get('top_ratio', 0.1),
        head_fusion=config.get('head_fusion', 'mean'),
        training_only_pwca=True,
        stochastic_depth_prob=config.get('stochastic_depth_prob', 0.1)
    )
    
    return model


def get_implementation_details_summary() -> Dict[str, Any]:
    """
    Get summary of all implementation details
    
    Returns:
        Dictionary with implementation details
    """
    return {
        'fgvc_settings': {
            'input_size': 448,
            'token_selection_ratio': '10%',
            'batch_size': 16,
            'optimizer': 'Adam',
            'learning_rate': '5e-4',
            'weight_decay': 0.05,
            'epochs': 100,
            'loss': 'Cross-entropy'
        },
        'reid_settings': {
            'input_size': '256×128 (pedestrian) / 256×256 (vehicle)',
            'token_selection_ratio': '30%',
            'batch_size': 64,
            'optimizer': 'SGD',
            'learning_rate': 0.008,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'epochs': 120,
            'loss': 'Cross-entropy + Triplet loss'
        },
        'advanced_features': {
            'stochastic_depth': 'Random layer dropping during training',
            'attention_rollout': 'S̄ = 0.5S + 0.5E with residual connections',
            'uncertainty_weighting': 'Dynamic loss balancing with learnable parameters',
            'pair_sampling': 'Advanced strategies for intra/inter-class pairs',
            'inference_strategy': 'Remove PWCA, combine SA and GLCA outputs'
        }
    } 