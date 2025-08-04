import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from attention_rollout import AttentionRolloutHook


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention (GLCA) module
    
    Implements cross-attention between selected local queries and global key-value pairs:
    f_GLCA(Q^l, K^g, V^g) = softmax(Q^l K^g^T / √d) V^g
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 top_ratio: float = 0.1,
                 head_fusion: str = "mean"):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_ratio: Ratio of top tokens to select (e.g., 0.1 for 10%)
            head_fusion: How to fuse attention heads for rollout
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_ratio = top_ratio
        self.head_fusion = head_fusion
        
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
        
        # Attention rollout hook
        self.rollout_hook = None
        
    def set_rollout_hook(self, rollout_hook: AttentionRolloutHook):
        """Set the attention rollout hook for token selection"""
        self.rollout_hook = rollout_hook
        
    def select_local_queries(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select local queries based on attention rollout
        
        Args:
            x: Input embeddings of shape (batch, seq_len, embed_dim)
            layer_idx: Layer index for rollout computation
            
        Returns:
            Tuple of (local_queries, selection_mask)
            - local_queries: Selected query embeddings (batch, num_selected, embed_dim)
            - selection_mask: Boolean mask of selected tokens (batch, seq_len)
        """
        if self.rollout_hook is None:
            raise ValueError("Rollout hook not set. Call set_rollout_hook() first.")
            
        # Get token selection mask based on attention rollout
        selection_mask = self.rollout_hook.select_top_tokens(
            top_ratio=self.top_ratio, 
            layer_idx=layer_idx
        )  # (batch, num_patches)
        
        # Expand mask to include CLS token (always selected)
        batch_size, num_patches = selection_mask.shape
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=selection_mask.device)
        full_mask = torch.cat([cls_mask, selection_mask], dim=1)  # (batch, seq_len)
        
        # Select local queries
        local_queries = x[full_mask]  # (batch * num_selected, embed_dim)
        local_queries = local_queries.view(batch_size, -1, self.embed_dim)
        
        return local_queries, full_mask
        
    def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention: softmax(Q K^T / √d) V
        
        Args:
            q: Query tensor of shape (batch, num_queries, embed_dim)
            k: Key tensor of shape (batch, seq_len, embed_dim)
            v: Value tensor of shape (batch, seq_len, embed_dim)
            
        Returns:
            Cross-attention output of shape (batch, num_queries, embed_dim)
        """
        batch_size, num_queries, _ = q.shape
        _, seq_len, _ = k.shape
        
        # Project to Q, K, V
        q = self.q_proj(q).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.embed_dim
        )
        
        # Project output
        output = self.out_proj(attn_output)
        
        return output
        
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass of GLCA module
        
        Args:
            x: Input embeddings of shape (batch, seq_len, embed_dim)
            layer_idx: Layer index for rollout computation
            
        Returns:
            Output embeddings of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Select local queries based on attention rollout
        local_queries, selection_mask = self.select_local_queries(x, layer_idx)
        
        # Apply cross-attention between local queries and global key-value pairs
        cross_attn_output = self.cross_attention(local_queries, x, x)
        
        # Create output tensor with same shape as input
        output = torch.zeros_like(x)
        
        # Place cross-attention outputs at selected positions
        output[selection_mask] = cross_attn_output.view(-1, embed_dim)
        
        # Apply residual connection and layer norm
        output = self.norm1(output + x)
        
        # Apply feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output


class GLCAWithBackbone(nn.Module):
    """
    GLCA module integrated with a Vision Transformer backbone
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 top_ratio: float = 0.1,
                 head_fusion: str = "mean"):
        """
        Args:
            backbone: Vision Transformer backbone (e.g., DeiT, ViT)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_ratio: Ratio of top tokens to select
            head_fusion: How to fuse attention heads for rollout
        """
        super().__init__()
        
        self.backbone = backbone
        self.glca = GlobalLocalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_ratio=top_ratio,
            head_fusion=head_fusion
        )
        
        # Set up attention rollout hook
        self.rollout_hook = AttentionRolloutHook(backbone)
        self.glca.set_rollout_hook(self.rollout_hook)
        
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through backbone and GLCA
        
        Args:
            x: Input tensor
            layer_idx: Layer index for rollout computation
            
        Returns:
            Output tensor
        """
        # Reset rollout hook
        self.rollout_hook.reset()
        
        # Forward through backbone to collect attention weights
        with torch.no_grad():
            _ = self.backbone(x)
            
        # Apply GLCA
        output = self.glca(x, layer_idx)
        
        return output
        
    def remove_hooks(self):
        """Remove attention rollout hooks"""
        self.rollout_hook.remove_hooks() 