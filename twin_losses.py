import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class VerificationLoss(nn.Module):
    """
    Binary cross-entropy loss for twin verification
    
    Computes loss for same/different classification based on similarity scores.
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: Margin for similarity scores (0.0 for standard BCE)
        """
        super().__init__()
        self.margin = margin
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                similarity_scores: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute verification loss
        
        Args:
            similarity_scores: Cosine similarity scores between embeddings
            labels: Binary labels (1 for same person, 0 for different)
            
        Returns:
            Verification loss
        """
        # Apply margin if specified
        if self.margin > 0.0:
            # Shift similarity scores by margin
            # For same person (label=1): similarity should be > margin
            # For different person (label=0): similarity should be < -margin
            adjusted_scores = similarity_scores - self.margin * (2 * labels - 1)
        else:
            adjusted_scores = similarity_scores
            
        # Convert similarity scores to logits (sigmoid input)
        # Map similarity from [-1, 1] to logits space
        logits = torch.tanh(adjusted_scores) * 10  # Scale for better gradients
        
        # Compute binary cross-entropy loss
        loss = self.bce_loss(logits, labels)
        
        return loss


class TwinTripletLoss(nn.Module):
    """
    Twin-aware triplet loss for metric learning
    
    Uses smaller margin for twin faces due to their high similarity.
    """
    
    def __init__(self, 
                 margin: float = 0.3,
                 twin_margin: float = 0.1,
                 distance_metric: str = 'cosine'):
        """
        Args:
            margin: Standard margin for triplet loss
            twin_margin: Smaller margin for twin pairs
            distance_metric: Distance metric ('cosine' or 'euclidean')
        """
        super().__init__()
        self.margin = margin
        self.twin_margin = twin_margin
        self.distance_metric = distance_metric
        
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor,
                is_twin_negative: bool = False) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same person as anchor)
            negative: Negative embeddings (different person from anchor)
            is_twin_negative: Whether negative is a twin (use smaller margin)
            
        Returns:
            Triplet loss
        """
        # Choose margin based on whether negative is a twin
        margin = self.twin_margin if is_twin_negative else self.margin
        
        # Compute distances
        if self.distance_metric == 'cosine':
            # Convert cosine similarity to distance
            pos_sim = F.cosine_similarity(anchor, positive, dim=1)
            neg_sim = F.cosine_similarity(anchor, negative, dim=1)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        else:  # euclidean
            pos_dist = torch.norm(anchor - positive, p=2, dim=1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=1)
            
        # Compute triplet loss
        losses = torch.relu(pos_dist - neg_dist + margin)
        
        return losses.mean()


class DynamicLossWeighting(nn.Module):
    """
    Dynamic loss weighting using uncertainty method (following DCAL paper)
    """
    
    def __init__(self, num_losses: int = 2):
        """
        Args:
            num_losses: Number of losses to weight
        """
        super().__init__()
        self.num_losses = num_losses
        self.weights = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted loss
        
        Args:
            losses: List of individual losses
            
        Returns:
            Weighted total loss
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")
            
        # Compute weighted loss using uncertainty method
        weighted_losses = []
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.weights[i])
            weighted_losses.append(weight * loss)
            
        # Add regularization terms
        regularization = torch.sum(self.weights)
        
        # Total loss - convert list to tensor for sum
        total_loss = sum(weighted_losses) + regularization
        
        return total_loss
        
    def get_weights(self) -> List[float]:
        """Get current loss weights"""
        return [torch.exp(-w).item() for w in self.weights]


class CombinedTwinLoss(nn.Module):
    """
    Combined loss function for twin verification
    
    Combines verification loss and triplet loss with dynamic weighting.
    """
    
    def __init__(self,
                 verification_weight: float = 1.0,
                 triplet_weight: float = 0.1,
                 use_dynamic_weighting: bool = True,
                 verification_margin: float = 0.0,
                 triplet_margin: float = 0.3,
                 twin_triplet_margin: float = 0.1,
                 distance_metric: str = 'cosine'):
        """
        Args:
            verification_weight: Weight for verification loss
            triplet_weight: Weight for triplet loss
            use_dynamic_weighting: Whether to use dynamic weighting
            verification_margin: Margin for verification loss
            triplet_margin: Standard margin for triplet loss
            twin_triplet_margin: Margin for twin triplet loss
            distance_metric: Distance metric for triplet loss
        """
        super().__init__()
        
        self.verification_weight = verification_weight
        self.triplet_weight = triplet_weight
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Loss functions
        self.verification_loss = VerificationLoss(margin=verification_margin)
        self.triplet_loss = TwinTripletLoss(
            margin=triplet_margin,
            twin_margin=twin_triplet_margin,
            distance_metric=distance_metric
        )
        
        # Dynamic weighting
        if use_dynamic_weighting:
            self.dynamic_weighting = DynamicLossWeighting(num_losses=2)
        else:
            self.dynamic_weighting = None
            
    def forward(self,
                similarity_scores: torch.Tensor,
                labels: torch.Tensor,
                anchor: Optional[torch.Tensor] = None,
                positive: Optional[torch.Tensor] = None,
                negative: Optional[torch.Tensor] = None,
                is_twin_negative: bool = False) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            similarity_scores: Cosine similarity scores
            labels: Binary labels (1 for same person, 0 for different)
            anchor: Anchor embeddings for triplet loss
            positive: Positive embeddings for triplet loss
            negative: Negative embeddings for triplet loss
            is_twin_negative: Whether negative is a twin
            
        Returns:
            Combined loss
        """
        # Verification loss
        verif_loss = self.verification_loss(similarity_scores, labels)
        
        # Triplet loss (if embeddings provided)
        if anchor is not None and positive is not None and negative is not None:
            triplet_loss = self.triplet_loss(anchor, positive, negative, is_twin_negative)
        else:
            triplet_loss = torch.tensor(0.0, device=verif_loss.device)
            
        # Combine losses
        if self.use_dynamic_weighting and self.dynamic_weighting is not None:
            # Use dynamic weighting
            total_loss = self.dynamic_weighting([verif_loss, triplet_loss])
        else:
            # Use fixed weights
            total_loss = (self.verification_weight * verif_loss + 
                         self.triplet_weight * triplet_loss)
            
        return total_loss
            
        return total_loss
        
    def get_loss_weights(self) -> Tuple[float, float]:
        """Get current loss weights"""
        if self.use_dynamic_weighting and self.dynamic_weighting is not None:
            weights = self.dynamic_weighting.get_weights()
            return weights[0], weights[1]
        else:
            return self.verification_weight, self.triplet_weight


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for twin verification
    
    Alternative to triplet loss that directly optimizes embedding distances.
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        """
        Args:
            margin: Margin for contrastive loss
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self,
                embeddings1: torch.Tensor,
                embeddings2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Binary labels (1 for same person, 0 for different)
            
        Returns:
            Contrastive loss
        """
        # Compute distances
        if self.distance_metric == 'cosine':
            # Convert cosine similarity to distance
            similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)
            distances = 1 - similarity
        else:  # euclidean
            distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
            
        # Contrastive loss
        # For same person (label=1): minimize distance
        # For different person (label=0): maximize distance (with margin)
        same_person_loss = labels * distances.pow(2)
        different_person_loss = (1 - labels) * torch.relu(self.margin - distances).pow(2)
        
        total_loss = same_person_loss + different_person_loss
        
        return total_loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for twin verification
    
    Uses angular margin to improve discriminative power of embeddings.
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 num_classes: int,
                 margin: float = 0.5,
                 scale: float = 64.0):
        """
        Args:
            embedding_dim: Embedding dimension
            num_classes: Number of classes (persons)
            margin: Angular margin
            scale: Scaling factor
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss
        
        Args:
            embeddings: Input embeddings
            labels: Class labels
            
        Returns:
            ArcFace loss
        """
        # Normalize embeddings and classifier weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        classifier_weights = F.normalize(self.classifier.weight, p=2, dim=1)
        
        # Compute cosine similarities
        cos_theta = F.linear(embeddings, classifier_weights)
        
        # Apply angular margin
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * torch.cos(self.margin) - sin_theta * torch.sin(self.margin)
        
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply margin only to correct class
        cos_theta_masked = torch.where(one_hot.bool(), cos_theta_m, cos_theta)
        
        # Scale and compute cross-entropy
        logits = cos_theta_masked * self.scale
        loss = F.cross_entropy(logits, labels)
        
        return loss


def create_twin_loss(verification_weight: float = 1.0,
                    triplet_weight: float = 0.1,
                    use_dynamic_weighting: bool = True,
                    **kwargs) -> CombinedTwinLoss:
    """
    Create a combined twin loss function
    
    Args:
        verification_weight: Weight for verification loss
        triplet_weight: Weight for triplet loss
        use_dynamic_weighting: Whether to use dynamic weighting
        **kwargs: Additional arguments for loss functions
        
    Returns:
        Combined twin loss function
    """
    return CombinedTwinLoss(
        verification_weight=verification_weight,
        triplet_weight=triplet_weight,
        use_dynamic_weighting=use_dynamic_weighting,
        **kwargs
    ) 