"""
Test script to verify that token selection now works correctly with proper token extraction.
"""
import torch
import sys
sys.path.insert(0, '/data/workspace/Projects/CLIP_benchmark')

from clip_benchmark.models import load_clip
from clip_benchmark.metrics.zeroshot_classification import get_image_features_with_tokens
from clip_benchmark.metrics.token_selection import apply_token_selection

def test_token_extraction():
    """Test that we can extract unpooled token features from CLIP model."""
    print("=" * 60)
    print("Testing Token Extraction from CLIP Model")
    print("=" * 60)
    
    # Load a CLIP model
    print("\n1. Loading CLIP model (ViT-B-32)...")
    model, transform, tokenizer = load_clip(
        model_type="open_clip",
        model_name="ViT-B-32",
        pretrained="openai",
        cache_dir=None,
        device="cpu"
    )
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Create dummy images
    print("\n2. Creating dummy input images (batch_size=2)...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {dummy_images.shape}")
    
    # Test standard encode_image
    print("\n3. Testing standard encode_image (pooled features)...")
    with torch.no_grad():
        pooled_features = model.encode_image(dummy_images)
    print(f"   Pooled features shape: {pooled_features.shape}")
    print(f"   Expected: (2, D) where D is embedding dimension")
    
    # Test unpooled token extraction
    print("\n4. Testing get_image_features_with_tokens (unpooled)...")
    with torch.no_grad():
        token_features = get_image_features_with_tokens(model, dummy_images)
    print(f"   Token features shape: {token_features.shape}")
    
    if len(token_features.shape) == 3:
        print(f"   ✓ SUCCESS: Got 3D tensor (B, N, D)")
        print(f"   Batch size: {token_features.shape[0]}")
        print(f"   Number of tokens: {token_features.shape[1]}")
        print(f"   Feature dimension: {token_features.shape[2]}")
    else:
        print(f"   ✗ FAILED: Got {len(token_features.shape)}D tensor instead of 3D")
        return False
    
    # Test token selection
    print("\n5. Testing token selection on unpooled features...")
    with torch.no_grad():
        selected_features = apply_token_selection(
            token_features,
            k=10,
            m=20,
            alpha=0.5,
            enabled=True
        )
    print(f"   Selected features shape: {selected_features.shape}")
    
    # Check that some tokens are zero (not selected)
    non_zero_tokens = (selected_features.abs().sum(dim=-1) > 0).sum(dim=1)
    print(f"   Non-zero tokens per sample: {non_zero_tokens.tolist()}")
    print(f"   Expected: around k+m = 30 tokens per sample")
    
    if non_zero_tokens[0] > 0 and non_zero_tokens[0] < token_features.shape[1]:
        print(f"   ✓ SUCCESS: Token selection is working (some tokens are zero)")
    else:
        print(f"   ✗ FAILED: All tokens are non-zero, selection didn't work")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! Token selection should now work correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_token_extraction()
    sys.exit(0 if success else 1)
