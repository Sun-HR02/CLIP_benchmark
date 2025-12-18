"""
Token pruning module for vision transformers.
Implements attention-based token selection to reduce computational cost.
"""
import torch
import torch.nn.functional as F


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
    def __init__(self, model, k_anchors=10, top_m=50, alpha=0.5):
        self.model = model
        self.k_anchors = k_anchors
        self.top_m = top_m
        self.alpha = alpha
        self.original_encode_image = model.encode_image
        
        # 用于存储中间结果
        self.attention_weights = None
        self.intermediate_features = None
        
    def _attention_hook(self, module, input, output):
        """Hook函数用于捕获注意力权重"""
        # 根据不同的模型架构，注意力权重的位置可能不同
        if isinstance(output, tuple):
            # 有些模型返回 (output, attention_weights)
            if len(output) > 1 and isinstance(output[1], torch.Tensor):
                self.attention_weights = output[1]
        elif hasattr(output, 'attentions'):
            self.attention_weights = output.attentions
            
    def _feature_hook(self, module, input, output):
        """Hook函数用于捕获中间特征"""
        if isinstance(output, torch.Tensor):
            self.intermediate_features = output
        elif isinstance(output, tuple) and len(output) > 0:
            self.intermediate_features = output[0]
    
    def register_hooks(self):
        """注册hooks到模型"""
        hooks = []
        if hasattr(self.model, 'visual'):
            visual = self.model.visual
            
            # 尝试找到transformer blocks
            if hasattr(visual, 'transformer'):
                if hasattr(visual.transformer, 'resblocks'):
                    # OpenCLIP style
                    last_block = visual.transformer.resblocks[-1]
                    if hasattr(last_block, 'attn'):
                        hooks.append(last_block.attn.register_forward_hook(self._attention_hook))
                    hooks.append(last_block.register_forward_hook(self._feature_hook))
                elif hasattr(visual.transformer, 'layers'):
                    # 另一种可能的命名
                    last_block = visual.transformer.layers[-1]
                    if hasattr(last_block, 'attn') or hasattr(last_block, 'self_attn'):
                        attn_module = last_block.attn if hasattr(last_block, 'attn') else last_block.self_attn
                        hooks.append(attn_module.register_forward_hook(self._attention_hook))
                    hooks.append(last_block.register_forward_hook(self._feature_hook))
        
        return hooks
    
    def encode_image_with_pruning(self, images):
        """
        使用token剪枝的图像编码
        """
        # 注册hooks
        hooks = self.register_hooks()
        
        try:
            # 重置中间结果
            self.attention_weights = None
            self.intermediate_features = None
            
            # 执行原始的encode_image
            features = self.original_encode_image(images)
            
            # 如果成功捕获了注意力和特征，应用剪枝
            if self.attention_weights is not None and self.intermediate_features is not None:
                try:
                    # 确保维度正确
                    if len(self.intermediate_features.shape) == 3:
                        # [batch, seq_len, dim]
                        _, pruned_features = select_tokens_with_pruning(
                            self.attention_weights,
                            self.intermediate_features,
                            k_anchors=self.k_anchors,
                            top_m=self.top_m,
                            alpha=self.alpha
                        )
                        # 对剪枝后的特征进行池化
                        # 只对非零特征求平均（因为有padding）
                        mask = (pruned_features.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
                        features = (pruned_features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                except Exception as e:
                    print(f"Token pruning failed: {e}, using original features")
            
            return features
            
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()


def apply_pruning_to_model(model, k_anchors=10, top_m=50, alpha=0.5):
    """
    为模型应用token剪枝
    
    Args:
        model: CLIP模型
        k_anchors: 锚点数量
        top_m: 保留的非锚点token数量
        alpha: 重要性和多样性的权重平衡参数
    
    Returns:
        pruned_encoder: 包装后的编码器
    """
    pruned_encoder = PrunedVisionEncoder(model, k_anchors, top_m, alpha)
    
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