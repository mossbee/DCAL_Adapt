import torch
import torch.nn as nn
import timm
import numpy as np

from attention_rollout import AttentionRollout, AttentionRolloutHook
from glca_module import GlobalLocalCrossAttention
from pwca_module import PairWiseCrossAttention
from dcal_example import DCALModel, create_dcal_model


def test_attention_rollout():
    """Test attention rollout mechanism"""
    print("Testing Attention Rollout...")
    
    # Create a simple model for testing
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=1000)
    
    # Create rollout hook
    rollout_hook = AttentionRolloutHook(model)
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass to collect attention weights
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Test rollout computation
    rollout = rollout_hook.get_rollout()
    print(f"Rollout shape: {rollout.shape}")
    
    # Test CLS attention
    cls_attention = rollout_hook.get_cls_attention()
    print(f"CLS attention shape: {cls_attention.shape}")
    
    # Test token selection
    selection_mask = rollout_hook.select_top_tokens(top_ratio=0.1)
    print(f"Selection mask shape: {selection_mask.shape}")
    print(f"Number of selected tokens: {selection_mask.sum().item()}")
    
    rollout_hook.remove_hooks()
    print("✓ Attention Rollout test passed\n")


def test_glca():
    """Test Global-Local Cross-Attention module"""
    print("Testing GLCA Module...")
    
    # Create model and rollout hook
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=1000)
    rollout_hook = AttentionRolloutHook(model)
    
    # Create GLCA module
    embed_dim = 192  # DeiT-Tiny embed_dim
    num_heads = 3    # DeiT-Tiny num_heads
    glca = GlobalLocalCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        top_ratio=0.1
    )
    glca.set_rollout_hook(rollout_hook)
    
    # Test input
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass through model to collect attention weights
    with torch.no_grad():
        _ = model(torch.randn(batch_size, 3, 224, 224))
    
    # Test GLCA forward pass
    output = glca(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "GLCA output shape should match input shape"
    
    rollout_hook.remove_hooks()
    print("✓ GLCA test passed\n")


def test_pwca():
    """Test Pair-Wise Cross-Attention module"""
    print("Testing PWCA Module...")
    
    # Create PWCA module
    embed_dim = 192
    num_heads = 3
    pwca = PairWiseCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        training_only=True
    )
    
    # Test input
    batch_size = 2
    seq_len = 197
    x1 = torch.randn(batch_size, seq_len, embed_dim)
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test training mode
    output_train = pwca(x1, x2, training=True)
    print(f"Training output shape: {output_train.shape}")
    assert output_train.shape == x1.shape, "PWCA output shape should match input shape"
    
    # Test inference mode
    output_inference = pwca(x1, x2, training=False)
    print(f"Inference output shape: {output_inference.shape}")
    assert torch.allclose(output_inference, x1), "PWCA should return input unchanged during inference"
    
    print("✓ PWCA test passed\n")


def test_dcal_model():
    """Test complete DCAL model"""
    print("Testing DCAL Model...")
    
    # Create DCAL model
    model = create_dcal_model(
        backbone_name='deit_tiny_patch16_224',
        num_classes=200,
        top_ratio=0.1
    )
    
    # Test input
    batch_size = 4
    input_size = 224
    x = torch.randn(batch_size, 3, input_size, input_size)
    paired_x = torch.randn(batch_size, 3, input_size, input_size)
    targets = torch.randint(0, 200, (batch_size,))
    
    # Test training forward pass
    model.train()
    sa_logits, glca_logits, pwca_features = model(x, paired_x, training=True)
    
    print(f"SA logits shape: {sa_logits.shape}")
    print(f"GLCA logits shape: {glca_logits.shape}")
    print(f"PWCA features shape: {pwca_features.shape}")
    
    assert sa_logits.shape == (batch_size, 200), "SA logits shape incorrect"
    assert glca_logits.shape == (batch_size, 200), "GLCA logits shape incorrect"
    assert pwca_features.shape[0] == batch_size, "PWCA features batch size incorrect"
    
    # Test loss computation
    loss_fn = nn.CrossEntropyLoss()
    total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
    print(f"Total loss: {total_loss.item():.4f}")
    assert total_loss.item() > 0, "Loss should be positive"
    
    # Test inference
    model.eval()
    with torch.no_grad():
        combined_logits = model.inference(x)
    
    print(f"Combined logits shape: {combined_logits.shape}")
    assert combined_logits.shape == (batch_size, 200), "Combined logits shape incorrect"
    
    # Clean up
    model.remove_hooks()
    print("✓ DCAL Model test passed\n")


def test_gradient_flow():
    """Test that gradients flow properly through all components"""
    print("Testing Gradient Flow...")
    
    # Create model
    model = create_dcal_model(
        backbone_name='deit_tiny_patch16_224',
        num_classes=200,
        top_ratio=0.1
    )
    
    # Test input
    batch_size = 2
    input_size = 224
    x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)
    paired_x = torch.randn(batch_size, 3, input_size, input_size)
    targets = torch.randint(0, 200, (batch_size,))
    
    # Forward pass
    model.train()
    sa_logits, glca_logits, _ = model(x, paired_x, training=True)
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    total_loss = model.compute_loss(sa_logits, glca_logits, targets, loss_fn)
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Input gradients should exist"
    assert model.w1.grad is not None, "Uncertainty weight gradients should exist"
    assert model.w2.grad is not None, "Uncertainty weight gradients should exist"
    
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    print(f"w1 gradient: {model.w1.grad.item():.4f}")
    print(f"w2 gradient: {model.w2.grad.item():.4f}")
    
    model.remove_hooks()
    print("✓ Gradient Flow test passed\n")


def main():
    """Run all tests"""
    print("Running DCAL Component Tests\n")
    print("=" * 50)
    
    try:
        test_attention_rollout()
        test_glca()
        test_pwca()
        test_dcal_model()
        test_gradient_flow()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 