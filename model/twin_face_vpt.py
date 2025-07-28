import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
import math
from torch.nn.modules.utils import _pair

from model.vpt import VPT
from model.vision_transformer import VisionTransformerPETL


class TwinFaceVPT(VPT):
    """
    Twin Face VPT model for verification task.
    
    Extends VPT to handle twin face verification with:
    - 2 class-specific prompts (same person vs twin)
    - Triplet loss computation
    - Attention map extraction for visualization
    """
    
    def __init__(self, params, depth, patch_size, embed_dim):
        # Set required attributes for VPT
        if not hasattr(params, 'vpt_mode'):
            params.vpt_mode = 'deep'
        if not hasattr(params, 'vpt_num'):
            params.vpt_num = 2  # same person vs twin
        if not hasattr(params, 'vpt_dropout'):
            params.vpt_dropout = 0.1
        if not hasattr(params, 'vpt_layer'):
            params.vpt_layer = None
        
        super().__init__(params, depth, patch_size, embed_dim)
        
        # Override to use 2 prompts for twin face verification
        self.params.vpt_num = 2  # same person vs twin
        
        # Reinitialize prompt embeddings for 2 classes
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + embed_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            depth, 2, embed_dim))  # 2 class-specific prompts
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        # Shared scoring vector for verification
        self.shared_scorer = nn.Parameter(torch.randn(embed_dim))
        nn.init.normal_(self.shared_scorer, std=0.02)
        
        # Distance metric for triplet loss
        self.distance_metric = getattr(params, 'distance_metric', 'cosine')
        
    def forward_triplet(self, anchor, positive, negative, return_attention=False):
        """
        Forward pass for triplet training.
        
        Args:
            anchor: Anchor images [batch_size, 3, H, W]
            positive: Positive images [batch_size, 3, H, W]
            negative: Negative images [batch_size, 3, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            triplet_output: Dictionary with embeddings and scores
        """
        # Get embeddings for each image in triplet
        anchor_emb = self.forward_single(anchor, return_attention=return_attention)
        positive_emb = self.forward_single(positive, return_attention=return_attention)
        negative_emb = self.forward_single(negative, return_attention=return_attention)
        
        # Extract embeddings and attention maps
        anchor_features = anchor_emb['features']
        positive_features = positive_emb['features']
        negative_features = negative_emb['features']
        
        # Compute similarity scores
        anchor_scores = anchor_emb['scores']
        positive_scores = positive_emb['scores']
        negative_scores = negative_emb['scores']
        
        output = {
            'anchor_features': anchor_features,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'anchor_scores': anchor_scores,
            'positive_scores': positive_scores,
            'negative_scores': negative_scores,
        }
        
        if return_attention:
            output.update({
                'anchor_attention': anchor_emb.get('attention', None),
                'positive_attention': positive_emb.get('attention', None),
                'negative_attention': negative_emb.get('attention', None),
            })
        
        return output
    
    def forward_single(self, x, return_attention=False):
        """
        Forward pass for single image.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            output: Dictionary with features, scores, and optional attention
        """
        # Get features from the base model
        features, attention_maps = self.get_features(x, return_attention=return_attention)
        
        # Compute class-specific scores
        scores = self.compute_scores(features)
        
        output = {
            'features': features,
            'scores': scores
        }
        
        if return_attention:
            output['attention'] = attention_maps
        
        return output
    
    def get_features(self, x, return_attention=False):
        """
        Extract features from input images.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            features: Class-specific features [batch_size, embed_dim]
            attention_maps: Optional attention maps
        """
        # This method should be implemented by the specific model architecture
        # For now, we'll use a placeholder that will be overridden
        raise NotImplementedError("get_features should be implemented by specific model")
    
    def compute_scores(self, features):
        """
        Compute class-specific scores from features.
        
        Args:
            features: Class-specific features [batch_size, 2, embed_dim]
            
        Returns:
            scores: Similarity scores [batch_size, 2]
        """
        # Compute scores using shared scorer
        scores = torch.matmul(features, self.shared_scorer.unsqueeze(1)).squeeze(-1)
        return scores
    
    def get_attention_maps(self, x, target_class=0):
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            target_class: Which class to visualize (0: same person, 1: twin)
            
        Returns:
            attention_maps: Attention maps for visualization
        """
        # This method should be implemented by the specific model architecture
        raise NotImplementedError("get_attention_maps should be implemented by specific model")
    
    def compute_similarity(self, emb1, emb2):
        """
        Compute similarity between two embeddings.
        
        Args:
            emb1: First embeddings [batch_size, embed_dim]
            emb2: Second embeddings [batch_size, embed_dim]
            
        Returns:
            similarity: Similarity scores [batch_size]
        """
        if self.distance_metric == 'cosine':
            similarity = F.cosine_similarity(emb1, emb2, dim=1)
        elif self.distance_metric == 'euclidean':
            distance = F.pairwise_distance(emb1, emb2, p=2)
            similarity = -distance  # Convert distance to similarity
        else:
            raise ValueError(f"Distance metric {self.distance_metric} not supported")
        
        return similarity


class TwinFaceVisionTransformer(VisionTransformerPETL):
    """
    Vision Transformer with Twin Face VPT for verification task.
    """
    
    def __init__(self, *args, **kwargs):
        # Set required attributes for VisionTransformerPETL
        params = kwargs.get('params')
        if params:
            if not hasattr(params, 'train_type'):
                params.train_type = 'prompt_cam'
            if not hasattr(params, 'vpt_mode'):
                params.vpt_mode = 'deep'
            if not hasattr(params, 'vpt_layer'):
                params.vpt_layer = None
            if not hasattr(params, 'vpt_dropout'):
                params.vpt_dropout = 0.1
        
        super().__init__(*args, **kwargs)
        
        # Add twin face VPT
        self.twin_face_vpt = TwinFaceVPT(
            params=params,
            depth=kwargs.get('depth', 12),
            patch_size=kwargs.get('patch_size', 14),
            embed_dim=kwargs.get('embed_dim', 768)
        )
        
        # Override class number for verification task
        self.num_classes = 2  # same person vs twin
    
    def forward_features(self, x, blur_head_lst=[], target_cls=-1):
        """
        Forward pass through feature extraction layers.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            blur_head_lst: List of attention heads to blur
            target_cls: Target class for attention visualization
            
        Returns:
            features: Extracted features
            attention_maps: Attention maps if requested
        """
        # Get base features from parent class
        features, attention_maps = super().forward_features(x, blur_head_lst, target_cls)
        
        # Apply twin face VPT
        vpt_features = self.twin_face_vpt.get_features_from_attention(
            features, attention_maps, target_cls
        )
        
        return vpt_features, attention_maps
    
    def forward_triplet(self, anchor, positive, negative, return_attention=False):
        """
        Forward pass for triplet training.
        
        Args:
            anchor: Anchor images [batch_size, 3, H, W]
            positive: Positive images [batch_size, 3, H, W]
            negative: Negative images [batch_size, 3, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            triplet_output: Dictionary with embeddings and scores
        """
        return self.twin_face_vpt.forward_triplet(
            anchor, positive, negative, return_attention=return_attention
        )
    
    def get_attention_maps(self, x, target_class=0):
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            target_class: Which class to visualize (0: same person, 1: twin)
            
        Returns:
            attention_maps: Attention maps for visualization
        """
        # Get features and attention maps
        features, attention_maps = self.forward_features(x, target_cls=target_class)
        
        # Extract class-specific attention maps
        if attention_maps is not None:
            # Get attention maps for the target class prompt
            class_attention = attention_maps[:, target_class, :, :]  # [batch_size, num_heads, seq_len]
            return class_attention
        
        return None
    
    def compute_verification_score(self, img1, img2):
        """
        Compute verification score between two images.
        
        Args:
            img1: First image [batch_size, 3, H, W]
            img2: Second image [batch_size, 3, H, W]
            
        Returns:
            score: Verification score [batch_size]
        """
        # Get embeddings for both images
        emb1 = self.forward_single(img1)['features']
        emb2 = self.forward_single(img2)['features']
        
        # Compute similarity
        similarity = self.twin_face_vpt.compute_similarity(emb1, emb2)
        
        return similarity


def create_twin_face_model(params):
    """
    Create twin face model with DINOv2 backbone.
    
    Args:
        params: Configuration parameters
        
    Returns:
        model: Twin face model
    """
    # Create base DINOv2 model
    model = TwinFaceVisionTransformer(
        img_size=params.crop_size,
        patch_size=14,  # DINOv2 uses 14x14 patches
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=params.drop_path_rate,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        params=params
    )
    
    return model 