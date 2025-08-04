import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AttentionRollout:
    """
    Attention rollout mechanism following the paper's formulation:
    S̄ = 0.5S + 0.5E (considering residual connections)
    Ŝᵢ = S̄ᵢ ⊗ S̄ᵢ₋₁ ⊗ ... ⊗ S̄₁ (accumulated attention)
    """
    
    def __init__(self, head_fusion: str = "mean"):
        """
        Args:
            head_fusion: How to fuse attention heads ('mean', 'max', 'min')
        """
        self.head_fusion = head_fusion
        self.attentions = []
        
    def reset(self):
        """Reset stored attentions"""
        self.attentions = []
        
    def add_attention(self, attention: torch.Tensor):
        """
        Add attention weights from a layer
        
        Args:
            attention: Attention weights of shape (batch, heads, seq_len, seq_len)
        """
        self.attentions.append(attention.detach())
        
    def fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Fuse attention heads according to specified method
        
        Args:
            attention: Attention weights of shape (batch, heads, seq_len, seq_len)
            
        Returns:
            Fused attention weights of shape (batch, seq_len, seq_len)
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unsupported head fusion: {self.head_fusion}")
    
    def compute_rollout(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute attention rollout up to specified layer
        
        Args:
            layer_idx: Layer index to compute rollout up to (None for all layers)
            
        Returns:
            Accumulated attention weights of shape (batch, seq_len, seq_len)
        """
        if not self.attentions:
            raise ValueError("No attention weights stored")
            
        attentions = self.attentions[:layer_idx] if layer_idx is not None else self.attentions
        
        # Start with identity matrix
        batch_size, seq_len = attentions[0].shape[0], attentions[0].shape[-1]
        result = torch.eye(seq_len, device=attentions[0].device).unsqueeze(0).expand(batch_size, -1, -1)
        
        for attention in attentions:
            # Fuse attention heads
            fused_attention = self.fuse_heads(attention)  # (batch, seq_len, seq_len)
            
            # Apply paper's formulation: S̄ = 0.5S + 0.5E
            identity = torch.eye(seq_len, device=attention.device).unsqueeze(0).expand(batch_size, -1, -1)
            normalized_attention = 0.5 * fused_attention + 0.5 * identity
            
            # Normalize
            normalized_attention = normalized_attention / normalized_attention.sum(dim=-1, keepdim=True)
            
            # Accumulate: result = normalized_attention @ result
            result = torch.matmul(normalized_attention, result)
            
        return result
    
    def get_cls_attention_to_patches(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get CLS token attention to patches from accumulated attention
        
        Args:
            layer_idx: Layer index to compute rollout up to
            
        Returns:
            CLS attention to patches of shape (batch, num_patches)
        """
        accumulated_attention = self.compute_rollout(layer_idx)
        
        # Extract CLS token attention to patches (first row, excluding CLS token itself)
        # accumulated_attention shape: (batch, seq_len, seq_len)
        # First row: CLS token attention to all tokens
        # We want CLS attention to patches (excluding CLS token)
        cls_attention = accumulated_attention[:, 0, 1:]  # (batch, num_patches)
        
        return cls_attention
    
    def select_top_tokens(self, cls_attention: torch.Tensor, top_ratio: float) -> torch.Tensor:
        """
        Select top tokens based on CLS attention scores
        
        Args:
            cls_attention: CLS attention to patches of shape (batch, num_patches)
            top_ratio: Ratio of top tokens to select (e.g., 0.1 for 10%)
            
        Returns:
            Boolean mask of selected tokens of shape (batch, num_patches)
        """
        batch_size, num_patches = cls_attention.shape
        num_select = int(num_patches * top_ratio)
        
        # Get top-k indices for each batch
        _, top_indices = torch.topk(cls_attention, k=num_select, dim=-1)
        
        # Create boolean mask
        mask = torch.zeros_like(cls_attention, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        
        return mask


class AttentionRolloutHook:
    """
    Hook to automatically collect attention weights during forward pass
    """
    
    def __init__(self, model: nn.Module, attention_layer_name: str = 'attn'):
        """
        Args:
            model: Vision Transformer model
            attention_layer_name: Name pattern to identify attention layers
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.rollout = AttentionRollout()
        self.hooks = []
        
        # Register hooks to all attention modules
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to attention modules"""
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name and hasattr(module, 'num_heads'):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)
                
    def _attention_hook(self, module, input, output):
        """Hook function to extract attention weights"""
        # Extract attention weights from the attention module
        x = input[0]  # Input to attention module
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply normalization if available
        if hasattr(module, 'q_norm') and hasattr(module, 'k_norm'):
            q, k = module.q_norm(q), module.k_norm(k)
        
        # Compute attention weights
        if hasattr(module, 'scale'):
            q = q * module.scale
        else:
            q = q * (module.head_dim ** -0.5)
            
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        self.rollout.add_attention(attn)
        
    def reset(self):
        """Reset stored attentions"""
        self.rollout.reset()
        
    def get_rollout(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get attention rollout"""
        return self.rollout.compute_rollout(layer_idx)
        
    def get_cls_attention(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get CLS attention to patches"""
        return self.rollout.get_cls_attention_to_patches(layer_idx)
        
    def select_top_tokens(self, top_ratio: float, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Select top tokens based on CLS attention"""
        cls_attention = self.get_cls_attention(layer_idx)
        return self.rollout.select_top_tokens(cls_attention, top_ratio)
        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 