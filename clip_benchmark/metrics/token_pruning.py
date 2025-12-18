"""
Token pruning module for vision transformers.
Implements attention-based token selection to reduce computational cost.
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_attention_from_features(features, num_heads=12):
    """
    从特征直接计算注意力权重（简化版本，用于演示）
    
    Args:
        features: token特征 [batch_size, seq_len, dim]
        num_heads: 注意力头数量
    
    Returns:
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    """
    batch_size, seq_len, dim = features.shape
    head_dim = dim // num_heads
    
    # 简化：使用特征相似度作为注意力的近似
    # 归一化特征
    features_norm = F.normalize(features, dim=-1)
    
    # 计算相似度矩阵
    attention = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, seq_len, seq_len]
    
    # 应用softmax
    attention = F.softmax(attention / np.sqrt(head_dim), dim=-1)
    
    # 扩展到多头（简化：所有头使用相同的注意力）
    attention = attention.unsqueeze(1).expand(-1, num_heads, -1, -1)
    
    return attention


def select_anchor_tokens(attention_weights, k):
    """
    根据CLS token的注意力选择k个锚点token
    
    Args:
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        k: 选择的锚点数量
    
    Returns:
        anchor_indices: 锚点token的索引 [batch_size, k]
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # 提取CLS token (索引0) 对其他token的注意力
    # A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
    cls_attention = attention_weights[:, :, 0, 1:]  # [batch_size, num_heads, seq_len-1]
    
    # 对所有头的注意力求平均
    cls_attention_avg = cls_attention.mean(dim=1)  # [batch_size, seq_len-1]
    
    # 选择top-k个注意力最高的token作为锚点
    _, anchor_indices = torch.topk(cls_attention_avg, k, dim=1)  # [batch_size, k]
    
    # 注意：索引需要+1，因为我们排除了CLS token
    anchor_indices = anchor_indices + 1
    
    return anchor_indices


def compute_token_importance(attention_weights, anchor_indices):
    """
    计算每个token相对于锚点的重要性
    I_i = 1/|S_anchor| * sum_{x_j in S_anchor} Attn(x_i, x_j)
    
    Args:
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        anchor_indices: 锚点索引 [batch_size, k]
    
    Returns:
        importance_scores: 重要性分数 [batch_size, seq_len-1]
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    k = anchor_indices.shape[1]
    
    # 对所有头的注意力求平均
    attention_avg = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    
    # 排除CLS token，只考虑patch tokens
    attention_patches = attention_avg[:, 1:, :]  # [batch_size, seq_len-1, seq_len]
    
    # 收集每个token对锚点的注意力
    importance_scores = torch.zeros(batch_size, seq_len - 1, device=attention_weights.device)
    
    for b in range(batch_size):
        # 获取当前batch的锚点索引
        anchors = anchor_indices[b]  # [k]
        
        # 计算每个token对所有锚点的平均注意力
        # attention_patches[b]: [seq_len-1, seq_len]
        # 选择对锚点的注意力
        attention_to_anchors = attention_patches[b, :, anchors]  # [seq_len-1, k]
        importance_scores[b] = attention_to_anchors.mean(dim=1)  # [seq_len-1]
    
    return importance_scores


def compute_token_diversity(features, anchor_indices):
    """
    计算每个token相对于锚点的多样性（基于余弦相似度）
    Sim(x_i, S_anchor) = max_{x_j in S_anchor}(x_i · x_j / (|x_i|_2 * |x_j|_2))
    D_i = 1 - Sim(x_i, S_anchor)
    
    Args:
        features: token特征 [batch_size, seq_len, dim]
        anchor_indices: 锚点索引 [batch_size, k]
    
    Returns:
        diversity_scores: 多样性分数 [batch_size, seq_len-1]
    """
    batch_size, seq_len, dim = features.shape
    k = anchor_indices.shape[1]
    
    # 排除CLS token
    patch_features = features[:, 1:, :]  # [batch_size, seq_len-1, dim]
    
    # 归一化特征
    patch_features_norm = F.normalize(patch_features, dim=-1)  # [batch_size, seq_len-1, dim]
    
    diversity_scores = torch.zeros(batch_size, seq_len - 1, device=features.device)
    
    for b in range(batch_size):
        # 获取锚点特征
        anchors = anchor_indices[b]  # [k]
        anchor_features = features[b, anchors, :]  # [k, dim]
        anchor_features_norm = F.normalize(anchor_features, dim=-1)  # [k, dim]
        
        # 计算余弦相似度
        # patch_features_norm[b]: [seq_len-1, dim]
        # anchor_features_norm: [k, dim]
        similarity = torch.mm(patch_features_norm[b], anchor_features_norm.t())  # [seq_len-1, k]
        
        # 取最大相似度
        max_similarity, _ = similarity.max(dim=1)  # [seq_len-1]
        
        # 多样性 = 1 - 相似度
        diversity_scores[b] = 1.0 - max_similarity
    
    return diversity_scores


def select_tokens_with_pruning(attention_weights, features, k_anchors, top_m, alpha):
    """
    基于注意力和特征相似度选择token
    
    Args:
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        features: token特征 [batch_size, seq_len, dim]
        k_anchors: 锚点数量
        top_m: 最终保留的非锚点token数量
        alpha: 重要性和多样性的权重平衡参数 (0-1之间)
    
    Returns:
        selected_indices: 选中的token索引 [batch_size, k_anchors + top_m]
        selected_features: 选中的token特征 [batch_size, k_anchors + top_m, dim]
    """
    batch_size, seq_len, dim = features.shape
    
    # 1. 选择锚点
    anchor_indices = select_anchor_tokens(attention_weights, k_anchors)  # [batch_size, k_anchors]
    
    # 2. 计算重要性分数
    importance_scores = compute_token_importance(attention_weights, anchor_indices)  # [batch_size, seq_len-1]
    
    # 3. 计算多样性分数
    diversity_scores = compute_token_diversity(features, anchor_indices)  # [batch_size, seq_len-1]
    
    # 4. 归一化分数到[0, 1]
    importance_scores = (importance_scores - importance_scores.min(dim=1, keepdim=True)[0]) / \
                       (importance_scores.max(dim=1, keepdim=True)[0] - importance_scores.min(dim=1, keepdim=True)[0] + 1e-8)
    diversity_scores = (diversity_scores - diversity_scores.min(dim=1, keepdim=True)[0]) / \
                      (diversity_scores.max(dim=1, keepdim=True)[0] - diversity_scores.min(dim=1, keepdim=True)[0] + 1e-8)
    
    # 5. 计算最终分数: S_i = alpha * I_i + (1 - alpha) * D_i
    final_scores = alpha * importance_scores + (1 - alpha) * diversity_scores  # [batch_size, seq_len-1]
    
    # 6. 创建mask，排除已经是锚点的token
    mask = torch.ones_like(final_scores, dtype=torch.bool)
    for b in range(batch_size):
        # anchor_indices中的索引是包含CLS的，需要-1来对应patch tokens
        anchor_patch_indices = anchor_indices[b] - 1
        mask[b, anchor_patch_indices] = False
    
    # 7. 从非锚点token中选择top-m
    final_scores_masked = final_scores.clone()
    final_scores_masked[~mask] = -float('inf')
    
    _, top_m_indices = torch.topk(final_scores_masked, min(top_m, mask.sum(dim=1).min().item()), dim=1)
    top_m_indices = top_m_indices + 1  # 转换为包含CLS的索引
    
    # 8. 合并CLS、锚点和选中的token
    selected_indices_list = []
    selected_features_list = []
    
    for b in range(batch_size):
        # CLS token (索引0)
        cls_idx = torch.tensor([0], device=features.device)
        
        # 合并所有索引
        all_indices = torch.cat([cls_idx, anchor_indices[b], top_m_indices[b]])
        
        # 去重并排序
        all_indices = torch.unique(all_indices, sorted=True)
        
        selected_indices_list.append(all_indices)
        selected_features_list.append(features[b, all_indices, :])
    
    # 由于每个batch可能有不同数量的token，我们需要padding
    max_len = max([idx.shape[0] for idx in selected_indices_list])
    
    selected_indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=features.device)
    selected_features = torch.zeros(batch_size, max_len, dim, device=features.device)
    
    for b in range(batch_size):
        length = selected_indices_list[b].shape[0]
        selected_indices[b, :length] = selected_indices_list[b]
        selected_features[b, :length, :] = selected_features_list[b]
    
    return selected_indices, selected_features


class PrunedVisionEncoder:
    """
    包装CLIP模型的视觉编码器，添加token剪枝功能
    """
    def __init__(self, model, k_anchors=10, top_m=50, alpha=0.5, verbose=False):
        self.model = model
        self.k_anchors = k_anchors
        self.top_m = top_m
        self.alpha = alpha
        self.verbose = verbose
        self.original_encode_image = model.encode_image
        
        # 用于存储中间结果
        self.intermediate_features = None
        self.pruning_applied = False
        
    def _feature_hook(self, module, input, output):
        """Hook函数用于捕获transformer最后一层的输出特征"""
        if isinstance(output, torch.Tensor):
            # 保存特征 [batch, seq_len, dim]
            self.intermediate_features = output.clone()
        elif isinstance(output, tuple) and len(output) > 0:
            self.intermediate_features = output[0].clone()
    
    def register_hooks(self):
        """注册hooks到模型的最后一个transformer block"""
        hooks = []
        if hasattr(self.model, 'visual'):
            visual = self.model.visual
            
            # 尝试找到transformer blocks
            if hasattr(visual, 'transformer'):
                if hasattr(visual.transformer, 'resblocks'):
                    # OpenCLIP style - 在最后一个block之前捕获特征
                    # 使用倒数第二个block的输出，这样包含了位置信息但还没有最终的pooling
                    target_block = visual.transformer.resblocks[-2] if len(visual.transformer.resblocks) > 1 else visual.transformer.resblocks[-1]
                    hooks.append(target_block.register_forward_hook(self._feature_hook))
                    if self.verbose:
                        print(f"Registered hook on transformer block {len(visual.transformer.resblocks)-2}")
                elif hasattr(visual.transformer, 'layers'):
                    # 另一种可能的命名
                    target_block = visual.transformer.layers[-2] if len(visual.transformer.layers) > 1 else visual.transformer.layers[-1]
                    hooks.append(target_block.register_forward_hook(self._feature_hook))
                    if self.verbose:
                        print(f"Registered hook on transformer layer {len(visual.transformer.layers)-2}")
        
        return hooks
    
    def encode_image_with_pruning(self, images):
        """
        使用token剪枝的图像编码
        """
        # 注册hooks
        hooks = self.register_hooks()
        
        try:
            # 重置中间结果
            self.intermediate_features = None
            self.pruning_applied = False
            
            # 执行原始的encode_image
            original_features = self.original_encode_image(images)
            
            # 如果成功捕获了中间特征，应用剪枝
            if self.intermediate_features is not None:
                try:
                    # 确保维度正确
                    if len(self.intermediate_features.shape) == 3:
                        batch_size, seq_len, dim = self.intermediate_features.shape
                        
                        if self.verbose:
                            print(f"Captured features shape: {self.intermediate_features.shape}")
                            print(f"Original features shape: {original_features.shape}")
                        
                        # 手动计算注意力权重（基于特征相似度）
                        # 这是一个简化版本，实际的attention计算更复杂
                        num_heads = 12  # 大多数ViT使用12个头
                        attention_weights = compute_attention_from_features(
                            self.intermediate_features, 
                            num_heads=num_heads
                        )
                        
                        if self.verbose:
                            print(f"Computed attention weights shape: {attention_weights.shape}")
                        
                        # 应用token剪枝
                        _, pruned_features = select_tokens_with_pruning(
                            attention_weights,
                            self.intermediate_features,
                            k_anchors=self.k_anchors,
                            top_m=self.top_m,
                            alpha=self.alpha
                        )
                        
                        if self.verbose:
                            print(f"Pruned features shape: {pruned_features.shape}")
                        
                        # 对剪枝后的特征进行池化
                        # 只对非零特征求平均（因为有padding）
                        mask = (pruned_features.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
                        num_valid_tokens = mask.sum(dim=1)
                        
                        if self.verbose:
                            print(f"Valid tokens per sample: {num_valid_tokens.squeeze().tolist()}")
                        
                        # 使用mean pooling
                        pruned_pooled = (pruned_features * mask).sum(dim=1) / (num_valid_tokens + 1e-8)
                        
                        # 归一化
                        pruned_pooled = F.normalize(pruned_pooled, dim=-1)
                        
                        self.pruning_applied = True
                        
                        if self.verbose:
                            print(f"✓ Token pruning successfully applied!")
                            print(f"  Original tokens: {seq_len}, Kept tokens: ~{int(num_valid_tokens.mean().item())}")
                        
                        return pruned_pooled
                        
                except Exception as e:
                    if self.verbose:
                        print(f"✗ Token pruning failed: {e}")
                        import traceback
                        traceback.print_exc()
                    print(f"Warning: Token pruning failed: {e}, using original features")
            else:
                if self.verbose:
                    print("✗ Failed to capture intermediate features, using original features")
            
            return original_features
            
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()


def apply_pruning_to_model(model, k_anchors=10, top_m=50, alpha=0.5, verbose=False):
    """
    为模型应用token剪枝
    
    Args:
        model: CLIP模型
        k_anchors: 锚点数量
        top_m: 保留的非锚点token数量
        alpha: 重要性和多样性的权重平衡参数
        verbose: 是否打印详细信息
    
    Returns:
        pruned_encoder: 包装后的编码器
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Applying Token Pruning to Model")
        print(f"{'='*60}")
        print(f"  k_anchors: {k_anchors}")
        print(f"  top_m: {top_m}")
        print(f"  alpha: {alpha}")
        print(f"{'='*60}\n")
    
    pruned_encoder = PrunedVisionEncoder(model, k_anchors, top_m, alpha, verbose=verbose)
    
    # 替换模型的encode_image方法
    model.encode_image = pruned_encoder.encode_image_with_pruning
    
    return pruned_encoder


def apply_token_pruning(model, images, k_anchors=10, top_m=50, alpha=0.5):
    """
    对模型的图像编码过程应用token剪枝
    
    Args:
        model: CLIP模型
        images: 输入图像 [batch_size, 3, H, W]
        k_anchors: 锚点数量
        top_m: 保留的非锚点token数量
        alpha: 重要性和多样性的权重平衡参数
    
    Returns:
        image_features: 剪枝后的图像特征
    """
    # 这个函数需要根据具体的模型实现来调整
    # 这里提供一个通用的框架
    raise NotImplementedError("This function needs to be customized for specific model architecture")