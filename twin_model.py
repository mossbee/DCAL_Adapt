import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
import numpy as np

from dcal_example import DCALModel
from attention_rollout import AttentionRolloutHook


class SiameseDCALModel(nn.Module):
    """
    Siamese DCAL model for twin face verification
    
    Modifies the existing DCAL model to output embeddings instead of logits
    and adds cosine similarity computation for verification.
    """
    
    def __init__(self,
                 backbone_name: str = 'deit_tiny_patch16_224',
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 dropout: float = 0.0,
                 top_ratio: float = 0.35,  # Higher for twin verification
                 head_fusion: str = "mean",
                 training_only_pwca: bool = True,
                 embedding_dim: int = 512,  # Output embedding dimension
                 learnable_threshold: bool = True):
        """
        Args:
            backbone_name: Name of the backbone model from timm
            embed_dim: Embedding dimension from backbone
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_ratio: Ratio of top tokens to select for GLCA (35% for twins)
            head_fusion: How to fuse attention heads for rollout
            training_only_pwca: Whether PWCA is training-only
            embedding_dim: Output embedding dimension (512 for face verification)
            learnable_threshold: Whether to use learnable threshold
        """
        super().__init__()
        
        # Create base DCAL model
        self.base_model = DCALModel(
            backbone_name=backbone_name,
            num_classes=embedding_dim,  # Use embedding_dim as num_classes for projection
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_ratio=top_ratio,
            head_fusion=head_fusion,
            training_only_pwca=training_only_pwca
        )
        
        # Get actual embed_dim from backbone
        if hasattr(self.base_model, 'embed_dim'):
            actual_embed_dim = self.base_model.embed_dim
        else:
            actual_embed_dim = embed_dim
            
        # Embedding projection layer (backbone_dim -> embedding_dim)
        # The base model outputs logits of size embedding_dim, so we need to project from there
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # Already the right size from base model
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Learnable threshold for verification
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(0.5))
        else:
            self.threshold = 0.5
            
        self.embedding_dim = embedding_dim
        self.learnable_threshold = learnable_threshold
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from a single image
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Reset attention rollout hook
        self.base_model.rollout_hook.reset()
        
        # Forward pass through base model (no PWCA during inference)
        sa_logits, glca_logits, _ = self.base_model(x, training=False)
        
        # Combine SA and GLCA features (following DCAL paper)
        # Use the logits as features since they're already projected
        combined_features = sa_logits + glca_logits
        
        # Project to final embedding dimension
        embeddings = self.embedding_projection(combined_features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
        
    def forward(self, 
                x1: torch.Tensor,
                x2: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for twin verification
        
        Args:
            x1: First image tensor
            x2: Second image tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (embeddings1, embeddings2, similarity_scores)
        """
        # Extract embeddings for both images
        embeddings1 = self.extract_features(x1)
        embeddings2 = self.extract_features(x2)
        
        # Compute cosine similarity
        similarity_scores = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        return embeddings1, embeddings2, similarity_scores
        
    def verify_twins(self, 
                    x1: torch.Tensor, 
                    x2: torch.Tensor,
                    threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Verify if two faces are from the same person
        
        Args:
            x1: First face image
            x2: Second face image
            threshold: Similarity threshold (uses learnable threshold if None)
            
        Returns:
            Tuple of (predictions, confidence_scores, similarity_scores)
        """
        # Extract embeddings and compute similarity
        embeddings1, embeddings2, similarity_scores = self.forward(x1, x2, training=False)
        
        # Use provided threshold or learnable threshold
        if threshold is None:
            threshold = self.threshold
            
        # Make predictions
        predictions = (similarity_scores > threshold).float()
        
        # Compute confidence scores (distance from threshold)
        confidence_scores = torch.abs(similarity_scores - threshold)
        
        return predictions, confidence_scores, similarity_scores
        
    def get_threshold(self) -> float:
        """Get current threshold value"""
        if self.learnable_threshold:
            return self.threshold.item()
        else:
            return self.threshold
            
    def set_threshold(self, threshold: float):
        """Set threshold value"""
        if self.learnable_threshold:
            with torch.no_grad():
                self.threshold.fill_(threshold)
        else:
            self.threshold = threshold
            
    def remove_hooks(self):
        """Remove attention rollout hooks"""
        self.base_model.remove_hooks()


class TwinVerificationModel(nn.Module):
    """
    Complete twin verification model with all components
    """
    
    def __init__(self,
                 backbone_name: str = 'deit_tiny_patch16_224',
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 dropout: float = 0.0,
                 top_ratio: float = 0.35,
                 head_fusion: str = "mean",
                 training_only_pwca: bool = True,
                 embedding_dim: int = 512,
                 learnable_threshold: bool = True):
        """
        Args:
            backbone_name: Name of the backbone model from timm
            embed_dim: Embedding dimension from backbone
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_ratio: Ratio of top tokens to select for GLCA
            head_fusion: How to fuse attention heads for rollout
            training_only_pwca: Whether PWCA is training-only
            embedding_dim: Output embedding dimension
            learnable_threshold: Whether to use learnable threshold
        """
        super().__init__()
        
        # Create Siamese DCAL model
        self.siamese_model = SiameseDCALModel(
            backbone_name=backbone_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_ratio=top_ratio,
            head_fusion=head_fusion,
            training_only_pwca=training_only_pwca,
            embedding_dim=embedding_dim,
            learnable_threshold=learnable_threshold
        )
        
        # Uncertainty weights for loss balancing
        self.w1 = nn.Parameter(torch.tensor(0.0))  # Verification loss weight
        self.w2 = nn.Parameter(torch.tensor(0.0))  # Triplet loss weight
        
    def forward(self, 
                x1: torch.Tensor,
                x2: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x1: First image tensor
            x2: Second image tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (embeddings1, embeddings2, similarity_scores)
        """
        return self.siamese_model(x1, x2, training)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a single image"""
        return self.siamese_model.extract_features(x)
        
    def verify_twins(self, 
                    x1: torch.Tensor, 
                    x2: torch.Tensor,
                    threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Verify if two faces are from the same person"""
        return self.siamese_model.verify_twins(x1, x2, threshold)
        
    def get_threshold(self) -> float:
        """Get current threshold value"""
        return self.siamese_model.get_threshold()
        
    def set_threshold(self, threshold: float):
        """Set threshold value"""
        self.siamese_model.set_threshold(threshold)
        
    def remove_hooks(self):
        """Remove attention rollout hooks"""
        self.siamese_model.remove_hooks()


def create_twin_model(backbone_name: str = 'deit_tiny_patch16_224',
                     embedding_dim: int = 512,
                     top_ratio: float = 0.35,
                     learnable_threshold: bool = True) -> TwinVerificationModel:
    """
    Create a twin verification model
    
    Args:
        backbone_name: Backbone model name
        embedding_dim: Output embedding dimension
        top_ratio: Token selection ratio for GLCA
        learnable_threshold: Whether to use learnable threshold
        
    Returns:
        Twin verification model
    """
    return TwinVerificationModel(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        top_ratio=top_ratio,
        learnable_threshold=learnable_threshold
    )


def compute_cosine_similarity(embeddings1: torch.Tensor, 
                            embeddings2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Cosine similarity scores
    """
    # Ensure embeddings are L2 normalized
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)
    
    return similarity


def compute_euclidean_distance(embeddings1: torch.Tensor, 
                              embeddings2: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between two sets of embeddings
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Euclidean distances
    """
    # Compute Euclidean distance
    distance = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
    
    return distance 