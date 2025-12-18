"""
Simple test script to verify token selection functionality.
"""
import torch
import sys
sys.path.insert(0, '/data/workspace/Projects/CLIP_benchmark')

from clip_benchmark.metrics.token_selection import (
    select_anchor_tokens,
    compute_importance_scores,
    compute_diversity_scores,
    select_tokens_with_scoring,
    apply_token_selection
)


def test_token_selection():
    """Test token selection with synthetic data."""
    print("Testing token selection module...")
    
    # Create synthetic image features (batch_size=2, num_tokens=50, feature_dim=512)
    B, N, D = 2, 50, 512
    image_features = torch.randn(B, N, D)
    
    print(f"\nInput shape: {image_features.shape}")
    
    # Test 1: Select anchor tokens
    print("\n1. Testing anchor token selection...")
    k = 10
    anchor_indices = select_anchor_tokens(image_features, k=k)
    print(f"   Anchor indices shape: {anchor_indices.shape}")
    print(f"   Sample anchor indices (batch 0): {anchor_indices[0].tolist()}")
    assert anchor_indices.shape == (B, k), f"Expected shape ({B}, {k}), got {anchor_indices.shape}"
    
    # Test 2: Compute importance scores
    print("\n2. Testing importance score computation...")
    importance_scores = compute_importance_scores(image_features, anchor_indices)
    print(f"   Importance scores shape: {importance_scores.shape}")
    print(f"   Sample importance scores (batch 0, first 5): {importance_scores[0, :5].tolist()}")
    assert importance_scores.shape == (B, N), f"Expected shape ({B}, {N}), got {importance_scores.shape}"
    
    # Test 3: Compute diversity scores
    print("\n3. Testing diversity score computation...")
    diversity_scores = compute_diversity_scores(image_features, anchor_indices)
    print(f"   Diversity scores shape: {diversity_scores.shape}")
    print(f"   Sample diversity scores (batch 0, first 5): {diversity_scores[0, :5].tolist()}")
    assert diversity_scores.shape == (B, N), f"Expected shape ({B}, {N}), got {diversity_scores.shape}"
    
    # Test 4: Select tokens with scoring
    print("\n4. Testing token selection with scoring...")
    m = 30
    alpha = 0.5
    selected_features = select_tokens_with_scoring(
        image_features, anchor_indices, importance_scores, diversity_scores, m=m, alpha=alpha
    )
    print(f"   Selected features shape: {selected_features.shape}")
    
    # Count non-zero tokens
    non_zero_mask = (selected_features.abs().sum(dim=-1) > 0)
    num_selected = non_zero_mask.sum(dim=1)
    print(f"   Number of selected tokens per batch: {num_selected.tolist()}")
    print(f"   Expected: {k + m} tokens per batch")
    assert selected_features.shape == image_features.shape, "Output shape should match input shape"
    
    # Test 5: End-to-end token selection
    print("\n5. Testing end-to-end token selection...")
    selected_features_e2e = apply_token_selection(
        image_features, k=k, m=m, alpha=alpha, enabled=True
    )
    print(f"   End-to-end selected features shape: {selected_features_e2e.shape}")
    non_zero_mask_e2e = (selected_features_e2e.abs().sum(dim=-1) > 0)
    num_selected_e2e = non_zero_mask_e2e.sum(dim=1)
    print(f"   Number of selected tokens per batch: {num_selected_e2e.tolist()}")
    
    # Test 6: Test with disabled token selection
    print("\n6. Testing with disabled token selection...")
    output_disabled = apply_token_selection(
        image_features, k=k, m=m, alpha=alpha, enabled=False
    )
    assert torch.allclose(output_disabled, image_features), "Disabled mode should return original features"
    print("   ✓ Disabled mode returns original features")
    
    # Test 7: Test with 2D input (already pooled)
    print("\n7. Testing with 2D input (already pooled)...")
    pooled_features = torch.randn(B, D)
    output_2d = apply_token_selection(pooled_features, k=k, m=m, alpha=alpha, enabled=True)
    assert torch.allclose(output_2d, pooled_features), "2D input should be returned as-is"
    print("   ✓ 2D input handled correctly")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_token_selection()
