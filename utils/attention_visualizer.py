import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import List, Tuple, Optional
import seaborn as sns


class AttentionVisualizer:
    """
    Attention visualizer for twin face verification.
    
    Generates heatmaps showing which facial features the model focuses on
    to distinguish between same person and twin pairs.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_attention_heatmap(self, image, attention_maps, target_class=0, 
                                 save_path=None, alpha=0.6):
        """
        Generate attention heatmap for a single image.
        
        Args:
            image: Input image tensor [1, 3, H, W] or PIL Image
            attention_maps: Attention maps from model [num_heads, seq_len]
            target_class: Which class to visualize (0: same person, 1: twin)
            save_path: Path to save the heatmap
            alpha: Transparency for overlay
            
        Returns:
            heatmap: Attention heatmap
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_np = (image.squeeze(0).cpu() * std + mean).permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = np.array(image) / 255.0
        
        # Get attention maps for target class
        if attention_maps.dim() == 3:  # [batch, num_heads, seq_len]
            attention_maps = attention_maps[0]  # Remove batch dimension
        
        # Average attention across heads
        attention_avg = attention_maps.mean(dim=0)  # [seq_len]
        
        # Remove CLS token attention (first token)
        patch_attention = attention_avg[1:]  # [num_patches]
        
        # Convert to numpy if it's a tensor
        if isinstance(patch_attention, torch.Tensor):
            patch_attention = patch_attention.numpy()
        
        # Reshape to spatial dimensions (assuming square patches)
        num_patches = patch_attention.shape[0]
        patch_size = int(np.sqrt(num_patches))
        attention_2d = patch_attention.reshape(patch_size, patch_size)
        
        # Convert to tensor for interpolation
        attention_2d_tensor = torch.from_numpy(attention_2d).unsqueeze(0).unsqueeze(0)
        attention_2d_tensor = F.interpolate(attention_2d_tensor, size=image_np.shape[:2], 
                                          mode='bilinear', align_corners=False)
        attention_2d = attention_2d_tensor.squeeze().numpy()
        
        # Normalize attention
        attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_2d * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = heatmap / 255.0
        
        # Overlay on image
        overlay = alpha * heatmap + (1 - alpha) * image_np
        overlay = np.clip(overlay, 0, 1)
        
        if save_path:
            plt.figure(figsize=(12, 4))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            
            # Attention heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(attention_2d, cmap='jet')
            plt.title(f'Attention Map (Class {target_class})')
            plt.axis('off')
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Attention Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return attention_2d, overlay
    
    def visualize_twin_comparison(self, img1, img2, attention1, attention2, 
                                save_path=None, title1="Same Person", title2="Twin"):
        """
        Visualize attention comparison between two images.
        
        Args:
            img1: First image tensor or PIL Image
            img2: Second image tensor or PIL Image
            attention1: Attention maps for first image
            attention2: Attention maps for second image
            save_path: Path to save the comparison
            title1: Title for first image
            title2: Title for second image
        """
        # Generate heatmaps
        heatmap1, overlay1 = self.generate_attention_heatmap(img1, attention1, target_class=0)
        heatmap2, overlay2 = self.generate_attention_heatmap(img2, attention2, target_class=1)
        
        if save_path:
            plt.figure(figsize=(15, 6))
            
            # First image
            plt.subplot(2, 3, 1)
            if isinstance(img1, torch.Tensor):
                img1_np = (img1.squeeze(0).cpu().permute(1, 2, 0) * 0.229 + 0.485).numpy()
                img1_np = np.clip(img1_np, 0, 1)
            else:
                img1_np = np.array(img1) / 255.0
            plt.imshow(img1_np)
            plt.title(f'{title1} - Original')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(heatmap1, cmap='jet')
            plt.title(f'{title1} - Attention')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(overlay1)
            plt.title(f'{title1} - Overlay')
            plt.axis('off')
            
            # Second image
            plt.subplot(2, 3, 4)
            if isinstance(img2, torch.Tensor):
                img2_np = (img2.squeeze(0).cpu().permute(1, 2, 0) * 0.229 + 0.485).numpy()
                img2_np = np.clip(img2_np, 0, 1)
            else:
                img2_np = np.array(img2) / 255.0
            plt.imshow(img2_np)
            plt.title(f'{title2} - Original')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            plt.imshow(heatmap2, cmap='jet')
            plt.title(f'{title2} - Attention')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            plt.imshow(overlay2)
            plt.title(f'{title2} - Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_verification_pair(self, img1, img2, score, label, attention1=None, attention2=None,
                                  save_path=None):
        """
        Visualize a verification pair with attention maps.
        
        Args:
            img1: First image
            img2: Second image
            score: Verification score
            label: Ground truth label (1: same person, 0: different)
            attention1: Attention maps for first image (optional)
            attention2: Attention maps for second image (optional)
            save_path: Path to save visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Convert images to numpy
        if isinstance(img1, torch.Tensor):
            img1_np = (img1.squeeze(0).cpu().permute(1, 2, 0) * 0.229 + 0.485).numpy()
            img1_np = np.clip(img1_np, 0, 1)
        else:
            img1_np = np.array(img1) / 255.0
            
        if isinstance(img2, torch.Tensor):
            img2_np = (img2.squeeze(0).cpu().permute(1, 2, 0) * 0.229 + 0.485).numpy()
            img2_np = np.clip(img2_np, 0, 1)
        else:
            img2_np = np.array(img2) / 255.0
        
        # Plot images
        plt.subplot(1, 2, 1)
        plt.imshow(img1_np)
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2_np)
        plt.title('Image 2')
        plt.axis('off')
        
        # Add text with verification info
        plt.figtext(0.5, 0.02, 
                   f'Score: {score:.4f}, Label: {label} ({("Same Person" if label == 1 else "Different Person")})',
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_attention_summary(self, attention_maps_list, save_path=None):
        """
        Create a summary of attention patterns across multiple samples.
        
        Args:
            attention_maps_list: List of attention maps
            save_path: Path to save summary
        """
        if not attention_maps_list:
            return
        
        # Average attention across samples
        avg_attention = torch.stack(attention_maps_list).mean(dim=0)
        
        # Create summary visualization
        plt.figure(figsize=(10, 8))
        
        # Average attention heatmap
        plt.subplot(2, 2, 1)
        attention_2d = avg_attention.mean(dim=0)[1:].reshape(14, 14).numpy()  # Assuming 14x14 patches
        plt.imshow(attention_2d, cmap='jet')
        plt.title('Average Attention Pattern')
        plt.colorbar()
        plt.axis('off')
        
        # Attention distribution
        plt.subplot(2, 2, 2)
        attention_flat = avg_attention.mean(dim=0)[1:].numpy()
        plt.hist(attention_flat, bins=50, alpha=0.7)
        plt.title('Attention Distribution')
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        
        # Top attention regions
        plt.subplot(2, 2, 3)
        top_indices = torch.topk(avg_attention.mean(dim=0)[1:], k=10).indices
        top_attention = avg_attention.mean(dim=0)[1:][top_indices]
        plt.bar(range(10), top_attention.numpy())
        plt.title('Top 10 Attention Regions')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Weight')
        
        # Attention by head
        plt.subplot(2, 2, 4)
        head_attention = avg_attention.mean(dim=1).numpy()
        plt.imshow(head_attention, cmap='viridis', aspect='auto')
        plt.title('Attention by Head')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Head')
        plt.colorbar()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


def visualize_model_attention(model, data_loader, save_dir, num_samples=10):
    """
    Visualize model attention for a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader with images
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    visualizer = AttentionVisualizer(model)
    
    sample_count = 0
    attention_maps_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            # Get images from batch
            if 'anchor' in batch_data:
                images = batch_data['anchor']
            elif 'img1' in batch_data:
                images = batch_data['img1']
            else:
                continue
            
            images = images.to(model.device)
            
            # Extract attention maps for each image
            for i in range(min(images.shape[0], num_samples - sample_count)):
                img = images[i:i+1]
                
                # Get attention maps for both classes
                same_attention = model.get_attention_maps(img, target_class=0)
                twin_attention = model.get_attention_maps(img, target_class=1)
                
                if same_attention is not None and twin_attention is not None:
                    # Save individual attention maps
                    save_path = os.path.join(save_dir, f'attention_sample_{sample_count + i}.png')
                    visualizer.generate_attention_heatmap(
                        img, same_attention, target_class=0, save_path=save_path
                    )
                    
                    # Save comparison
                    comp_path = os.path.join(save_dir, f'comparison_sample_{sample_count + i}.png')
                    visualizer.visualize_twin_comparison(
                        img, img, same_attention, twin_attention,
                        save_path=comp_path, title1="Same Person", title2="Twin"
                    )
                    
                    # Collect attention maps for summary
                    attention_maps_list.append(same_attention)
                
                sample_count += 1
    
    # Create attention summary
    if attention_maps_list:
        summary_path = os.path.join(save_dir, 'attention_summary.png')
        visualizer.create_attention_summary(attention_maps_list, save_path=summary_path)
    
    print(f"Saved attention visualizations for {sample_count} samples to {save_dir}") 