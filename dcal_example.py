import torch
import torch.nn as nn
import timm
from typing import Optional, List, Tuple

from attention_rollout import AttentionRolloutHook
from glca_module import GlobalLocalCrossAttention, GLCAWithBackbone
from pwca_module import PairWiseCrossAttention, PWCAWithBackbone, ImagePairSampler


class DCALModel(nn.Module):
    """
    Dual Cross-Attention Learning (DCAL) model
    
    Integrates Attention Rollout, GLCA, and PWCA modules with a Vision Transformer backbone
    """
    
    def __init__(self,
                 backbone_name: str = 'deit_tiny_patch16_224',
                 num_classes: int = 1000,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 dropout: float = 0.0,
                 top_ratio: float = 0.1,
                 head_fusion: str = "mean",
                 training_only_pwca: bool = True):
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
        """
        super().__init__()
        
        # Load backbone model
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
        
        # Get actual embed_dim from backbone
        if hasattr(self.backbone, 'embed_dim'):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'proj'):
            embed_dim = self.backbone.patch_embed.proj.out_channels
            
        # Get actual num_heads from backbone
        if hasattr(self.backbone, 'num_heads'):
            num_heads = self.backbone.num_heads
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Set up attention rollout hook
        self.rollout_hook = AttentionRolloutHook(self.backbone)
        
        # GLCA module
        self.glca = GlobalLocalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_ratio=top_ratio,
            head_fusion=head_fusion
        )
        self.glca.set_rollout_hook(self.rollout_hook)
        
        # PWCA module
        self.pwca = PairWiseCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            training_only=training_only_pwca
        )
        
        # Classifiers for different branches
        self.sa_classifier = nn.Linear(embed_dim, num_classes)
        self.glca_classifier = nn.Linear(embed_dim, num_classes)
        
        # Uncertainty loss weights (learnable parameters)
        self.w1 = nn.Parameter(torch.tensor(0.0))  # SA branch weight
        self.w2 = nn.Parameter(torch.tensor(0.0))  # GLCA branch weight
        
    def forward(self, 
                x: torch.Tensor,
                paired_x: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through DCAL model
        
        Args:
            x: Input tensor
            paired_x: Paired image tensor for PWCA (optional)
            training: Whether in training mode
            
        Returns:
            Tuple of (sa_output, glca_output, pwca_output)
        """
        batch_size = x.shape[0]
        
        # Reset attention rollout hook
        self.rollout_hook.reset()
        
        # Self-attention branch (backbone)
        # First pass to collect attention weights for rollout (no gradients needed)
        with torch.no_grad():
            _ = self.backbone(x)
        
        # Second pass to get features with gradients
        sa_features = self.backbone.forward_features(x)
        sa_logits = self.sa_classifier(sa_features[:, 0])  # Use CLS token
        
        # GLCA branch
        glca_features = self.glca(sa_features)
        glca_logits = self.glca_classifier(glca_features[:, 0])  # Use CLS token
        
        # PWCA branch (training only)
        if training and paired_x is not None:
            # Extract features from paired image
            paired_features = self.backbone.forward_features(paired_x)
            pwca_features = self.pwca(sa_features, paired_features, training=True)
        else:
            pwca_features = sa_features  # No PWCA during inference
            
        return sa_logits, glca_logits, pwca_features
    
    def compute_loss(self, 
                     sa_logits: torch.Tensor,
                     glca_logits: torch.Tensor,
                     targets: torch.Tensor,
                     loss_fn: nn.Module) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss
        
        Args:
            sa_logits: Self-attention branch logits
            glca_logits: GLCA branch logits
            targets: Ground truth labels
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            
        Returns:
            Total weighted loss
        """
        # Individual losses
        sa_loss = loss_fn(sa_logits, targets)
        glca_loss = loss_fn(glca_logits, targets)
        
        # Uncertainty-weighted loss
        total_loss = 0.5 * (
            torch.exp(-self.w1) * sa_loss + 
            torch.exp(-self.w2) * glca_loss + 
            self.w1 + self.w2
        )
        
        return total_loss
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode (no PWCA)
        
        Args:
            x: Input tensor
            
        Returns:
            Combined logits
        """
        sa_logits, glca_logits, _ = self.forward(x, training=False)
        
        # Combine predictions (add probabilities as mentioned in paper)
        sa_probs = torch.softmax(sa_logits, dim=1)
        glca_probs = torch.softmax(glca_logits, dim=1)
        
        combined_probs = sa_probs + glca_probs
        combined_logits = torch.log(combined_probs + 1e-8)
        
        return combined_logits
    
    def remove_hooks(self):
        """Remove attention rollout hooks"""
        self.rollout_hook.remove_hooks()


def create_dcal_model(backbone_name: str = 'deit_tiny_patch16_224',
                     num_classes: int = 200,  # CUB-200-2011
                     top_ratio: float = 0.1,  # 10% for FGVC
                     training_only_pwca: bool = True) -> DCALModel:
    """
    Create a DCAL model with specified configuration
    
    Args:
        backbone_name: Backbone model name
        num_classes: Number of classes
        top_ratio: Token selection ratio
        training_only_pwca: Whether PWCA is training-only
        
    Returns:
        DCAL model
    """
    return DCALModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        top_ratio=top_ratio,
        training_only_pwca=training_only_pwca
    )


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_dcal_model(
        backbone_name='deit_tiny_patch16_224',
        num_classes=200,  # CUB-200-2011
        top_ratio=0.1     # 10% for FGVC
    )
    
    # Example input
    batch_size = 4
    input_size = 224
    x = torch.randn(batch_size, 3, input_size, input_size)
    paired_x = torch.randn(batch_size, 3, input_size, input_size)
    targets = torch.randint(0, 200, (batch_size,))
    
    # Training forward pass
    model.train()
    sa_logits, glca_logits, pwca_features = model(x, paired_x, training=True)
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
    
    print(f"SA logits shape: {sa_logits.shape}")
    print(f"GLCA logits shape: {glca_logits.shape}")
    print(f"PWCA features shape: {pwca_features.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Inference forward pass
    model.eval()
    with torch.no_grad():
        combined_logits = model.inference(x)
    
    print(f"Combined logits shape: {combined_logits.shape}")
    
    # Clean up hooks
    model.remove_hooks() 