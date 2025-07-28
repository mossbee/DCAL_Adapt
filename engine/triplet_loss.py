import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining for twin face verification.
    
    Uses twin pairs as hard negatives to force the model to learn
    subtle differences between identical twins.
    """
    
    def __init__(self, margin=0.3, distance_metric='euclidean'):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, anchor_emb, positive_emb, negative_emb, labels=None):
        """
        Compute triplet loss.
        
        Args:
            anchor_emb: Anchor embeddings [batch_size, embed_dim]
            positive_emb: Positive embeddings [batch_size, embed_dim]
            negative_emb: Negative embeddings [batch_size, embed_dim]
            labels: Optional labels for additional supervision
            
        Returns:
            loss: Triplet loss value
            stats: Dictionary with loss statistics
        """
        if self.distance_metric == 'euclidean':
            # Compute Euclidean distances
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
        elif self.distance_metric == 'cosine':
            # Compute cosine distances (1 - cosine_similarity)
            pos_dist = 1 - F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            neg_dist = 1 - F.cosine_similarity(anchor_emb, negative_emb, dim=1)
        else:
            raise ValueError(f"Distance metric {self.distance_metric} not supported")
        
        # Compute triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Compute statistics
        num_valid_triplets = (triplet_loss > 0).float().sum()
        avg_pos_dist = pos_dist.mean()
        avg_neg_dist = neg_dist.mean()
        
        # Compute accuracy (percentage of valid triplets)
        accuracy = (num_valid_triplets / anchor_emb.size(0)).item()
        
        # Average loss over valid triplets
        if num_valid_triplets > 0:
            loss = triplet_loss.sum() / num_valid_triplets
        else:
            loss = triplet_loss.mean()
        
        stats = {
            'triplet_loss': loss.item(),
            'pos_dist': avg_pos_dist.item(),
            'neg_dist': avg_neg_dist.item(),
            'accuracy': accuracy,
            'num_valid_triplets': num_valid_triplets.item(),
            'total_triplets': anchor_emb.size(0)
        }
        
        return loss, stats


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss with semi-hard negative mining.
    
    Finds the hardest positive and semi-hard negative for each anchor
    in the batch.
    """
    
    def __init__(self, margin=0.3, distance_metric='euclidean'):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, embeddings, labels):
        """
        Compute online triplet loss.
        
        Args:
            embeddings: All embeddings in batch [batch_size, embed_dim]
            labels: Labels for each embedding [batch_size]
            
        Returns:
            loss: Triplet loss value
            stats: Dictionary with loss statistics
        """
        if self.distance_metric == 'euclidean':
            # Compute pairwise distances
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        elif self.distance_metric == 'cosine':
            # Compute cosine distances
            similarity_matrix = torch.mm(embeddings, embeddings.t())
            dist_matrix = 1 - similarity_matrix
        else:
            raise ValueError(f"Distance metric {self.distance_metric} not supported")
        
        # Create mask for positive pairs (same label)
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Create mask for negative pairs (different label)
        negative_mask = ~labels_matrix
        
        # Find hardest positive for each anchor
        hardest_positives = torch.max(dist_matrix * labels_matrix.float(), dim=1)[0]
        
        # Find semi-hard negatives (negative with distance > positive distance but < positive + margin)
        semi_hard_mask = (dist_matrix > hardest_positives.unsqueeze(1)) & \
                        (dist_matrix < hardest_positives.unsqueeze(1) + self.margin) & \
                        negative_mask
        
        # If no semi-hard negatives, use hardest negative
        if semi_hard_mask.sum() == 0:
            hardest_negatives = torch.max(dist_matrix * negative_mask.float(), dim=1)[0]
        else:
            hardest_negatives = torch.min(dist_matrix + (1 - semi_hard_mask.float()) * 1e6, dim=1)[0]
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positives - hardest_negatives + self.margin)
        
        # Compute statistics
        num_valid_triplets = (triplet_loss > 0).float().sum()
        avg_pos_dist = hardest_positives.mean()
        avg_neg_dist = hardest_negatives.mean()
        
        # Compute accuracy
        accuracy = (num_valid_triplets / embeddings.size(0)).item()
        
        # Average loss over valid triplets
        if num_valid_triplets > 0:
            loss = triplet_loss.sum() / num_valid_triplets
        else:
            loss = triplet_loss.mean()
        
        stats = {
            'triplet_loss': loss.item(),
            'pos_dist': avg_pos_dist.item(),
            'neg_dist': avg_neg_dist.item(),
            'accuracy': accuracy,
            'num_valid_triplets': num_valid_triplets.item(),
            'total_triplets': embeddings.size(0)
        }
        
        return loss, stats


def compute_similarity_scores(emb1, emb2, distance_metric='cosine'):
    """
    Compute similarity scores between two sets of embeddings.
    
    Args:
        emb1: First set of embeddings [batch_size, embed_dim]
        emb2: Second set of embeddings [batch_size, embed_dim]
        distance_metric: 'euclidean' or 'cosine'
        
    Returns:
        scores: Similarity scores [batch_size]
    """
    if distance_metric == 'euclidean':
        # Convert distance to similarity (higher distance = lower similarity)
        distances = F.pairwise_distance(emb1, emb2, p=2)
        scores = -distances  # Negative distance as similarity
    elif distance_metric == 'cosine':
        # Cosine similarity (higher = more similar)
        scores = F.cosine_similarity(emb1, emb2, dim=1)
    else:
        raise ValueError(f"Distance metric {distance_metric} not supported")
    
    return scores 