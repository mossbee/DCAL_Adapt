import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import random


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention (PWCA) module
    
    Implements cross-attention between query of one image and combined key-value from both images:
    f_PWCA(Q₁, K_c, V_c) = softmax(Q₁ K_c^T / √d) V_c
    where K_c = [K₁; K₂] and V_c = [V₁; V₂]
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 training_only: bool = True):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            training_only: Whether to only use during training
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.training_only = training_only
        
        # Multi-head attention for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def cross_attention_with_combined_kv(self, 
                                       q: torch.Tensor, 
                                       k1: torch.Tensor, 
                                       v1: torch.Tensor,
                                       k2: torch.Tensor, 
                                       v2: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention with combined key-value pairs from two images
        
        Args:
            q: Query tensor from image 1 of shape (batch, seq_len, embed_dim)
            k1: Key tensor from image 1 of shape (batch, seq_len, embed_dim)
            v1: Value tensor from image 1 of shape (batch, seq_len, embed_dim)
            k2: Key tensor from image 2 of shape (batch, seq_len, embed_dim)
            v2: Value tensor from image 2 of shape (batch, seq_len, embed_dim)
            
        Returns:
            Cross-attention output of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = q.shape
        
        # Concatenate key-value pairs: K_c = [K₁; K₂], V_c = [V₁; V₂]
        k_combined = torch.cat([k1, k2], dim=1)  # (batch, 2*seq_len, embed_dim)
        v_combined = torch.cat([v1, v2], dim=1)  # (batch, 2*seq_len, embed_dim)
        
        # Project to Q, K, V
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_combined).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_combined).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Project output
        output = self.out_proj(attn_output)
        
        return output
        
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        """
        Forward pass of PWCA module
        
        Args:
            x1: Input embeddings from image 1 of shape (batch, seq_len, embed_dim)
            x2: Input embeddings from image 2 of shape (batch, seq_len, embed_dim)
            training: Whether in training mode
            
        Returns:
            Output embeddings for image 1 of shape (batch, seq_len, embed_dim)
        """
        # During inference, if training_only is True, return input unchanged
        if not training and self.training_only:
            return x1
            
        # Apply cross-attention with combined key-value pairs
        cross_attn_output = self.cross_attention_with_combined_kv(x1, x1, x1, x2, x2)
        
        # Apply residual connection and layer norm
        output = self.norm1(cross_attn_output + x1)
        
        # Apply feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output


class PWCAWithBackbone(nn.Module):
    """
    PWCA module integrated with a Vision Transformer backbone
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 training_only: bool = True):
        """
        Args:
            backbone: Vision Transformer backbone (e.g., DeiT, ViT)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            training_only: Whether to only use during training
        """
        super().__init__()
        
        self.backbone = backbone
        self.pwca = PairWiseCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            training_only=training_only
        )
        
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        """
        Forward pass through backbone and PWCA
        
        Args:
            x1: Input tensor for image 1
            x2: Input tensor for image 2
            training: Whether in training mode
            
        Returns:
            Output tensor for image 1
        """
        # Apply PWCA
        output = self.pwca(x1, x2, training)
        
        return output


class ImagePairSampler:
    """
    Utility class for sampling image pairs for PWCA training
    """
    
    def __init__(self, dataset_size: int, same_class_prob: float = 0.5):
        """
        Args:
            dataset_size: Size of the dataset
            same_class_prob: Probability of sampling from same class
        """
        self.dataset_size = dataset_size
        self.same_class_prob = same_class_prob
        self.indices = list(range(dataset_size))
        
    def sample_pair(self, current_idx: int, labels: Optional[List[int]] = None) -> int:
        """
        Sample a pair index for the current image
        
        Args:
            current_idx: Current image index
            labels: List of labels for all images (optional)
            
        Returns:
            Paired image index
        """
        if labels is None:
            # Random sampling without class consideration
            available_indices = [i for i in self.indices if i != current_idx]
            return random.choice(available_indices)
        
        current_label = labels[current_idx]
        
        # Decide whether to sample from same class or different class
        if random.random() < self.same_class_prob:
            # Sample from same class
            same_class_indices = [i for i, label in enumerate(labels) 
                                if label == current_label and i != current_idx]
            if same_class_indices:
                return random.choice(same_class_indices)
        
        # Sample from different class
        different_class_indices = [i for i, label in enumerate(labels) 
                                 if label != current_label and i != current_idx]
        if different_class_indices:
            return random.choice(different_class_indices)
        
        # Fallback to random sampling
        available_indices = [i for i in self.indices if i != current_idx]
        return random.choice(available_indices)


class PWCAWrapper(nn.Module):
    """
    Wrapper for PWCA that handles image pair sampling and training/inference modes
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 training_only: bool = True,
                 dataset_size: int = 1000):
        """
        Args:
            backbone: Vision Transformer backbone
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            training_only: Whether to only use during training
            dataset_size: Size of the dataset for pair sampling
        """
        super().__init__()
        
        self.pwca_module = PWCAWithBackbone(
            backbone=backbone,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            training_only=training_only
        )
        
        self.sampler = ImagePairSampler(dataset_size)
        self.training_only = training_only
        
    def forward(self, 
                x: torch.Tensor,
                labels: Optional[List[int]] = None,
                current_idx: Optional[int] = None,
                paired_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic pair sampling
        
        Args:
            x: Input tensor
            labels: List of labels for pair sampling
            current_idx: Current image index for pair sampling
            paired_x: Pre-sampled paired image tensor (optional)
            
        Returns:
            Output tensor
        """
        if self.training and self.training_only:
            if paired_x is None:
                # Sample a paired image
                if current_idx is not None:
                    paired_idx = self.sampler.sample_pair(current_idx, labels)
                    # Note: In practice, you would need to get the paired image from your dataset
                    # This is a placeholder - you'll need to implement the actual sampling
                    paired_x = x  # Placeholder
                else:
                    # Use the same image as pair (fallback)
                    paired_x = x
                    
            return self.pwca_module(x, paired_x, training=True)
        else:
            # During inference, return input unchanged
            return x 