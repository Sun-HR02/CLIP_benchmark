#!/usr/bin/env python3
"""
测试token剪枝功能的调试脚本
"""
import torch
import sys
sys.path.insert(0, '/data/workspace/Projects/CLIP_benchmark')

from clip_benchmark.models import load_clip
from clip_benchmark.metrics.token_pruning import apply_pruning_to_model

def test_pruning():
    print("="*80)
    print("Token Pruning Debug Test")
    print("="*80)
    
    # 加载模型
    print("\n1. Loading CLIP model...")
    model, transform, tokenizer = load_clip(
        model_type='open_clip',
        model_name='ViT-B-32',
        pretrained='openai',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.eval()
    print(f"   Model loaded on device: {next(model.parameters()).device}")
    
    # 创建测试图像
    print("\n2. Creating test images...")
    device = next(model.parameters()).device
    test_images = torch.randn(2, 3, 224, 224).to(device)
    print(f"   Test images shape: {test_images.shape}")
    
    # 测试原始编码
    print("\n3. Testing original encoding...")
    with torch.no_grad():
        original_features = model.encode_image(test_images)
    print(f"   Original features shape: {original_features.shape}")
    print(f"   Original features norm: {original_features.norm(dim=-1)}")
    
    # 应用剪枝
    print("\n4. Applying token pruning...")
    pruned_encoder = apply_pruning_to_model(
        model,
        k_anchors=10,
        top_m=50,
        alpha=0.5,
        verbose=True
    )
    
    # 测试剪枝后的编码
    print("\n5. Testing pruned encoding...")
    with torch.no_grad():
        pruned_features = model.encode_image(test_images)
    print(f"   Pruned features shape: {pruned_features.shape}")
    print(f"   Pruned features norm: {pruned_features.norm(dim=-1)}")
    
    # 比较特征
    print("\n6. Comparing features...")
    feature_diff = (original_features - pruned_features).abs().mean()
    cosine_sim = torch.nn.functional.cosine_similarity(original_features, pruned_features, dim=-1)
    print(f"   Mean absolute difference: {feature_diff.item():.6f}")
    print(f"   Cosine similarity: {cosine_sim.tolist()}")
    
    # 检查剪枝是否真的应用了
    print("\n7. Verification...")
    if pruned_encoder.pruning_applied:
        print("   ✓ Pruning was successfully applied!")
    else:
        print("   ✗ Pruning was NOT applied!")
    
    if feature_diff.item() > 1e-6:
        print("   ✓ Features are different (pruning is working!)")
    else:
        print("   ✗ Features are identical (pruning may not be working)")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

if __name__ == "__main__":
    test_pruning()
