# Token Selection 维度转换修复

## 问题描述

在运行 `chap2.sh` 时，token selection 功能没有生效，输出结果与未启用 token selection 时完全相同。

## 根本原因

CLIP 模型的 `encode_image()` 方法默认返回的是**池化后的 2D 特征** `(B, D)`，其中：
- B = batch size
- D = embedding dimension

但是 `apply_token_selection()` 函数需要的是**未池化的 3D token 特征** `(B, N, D)`，其中：
- B = batch size  
- N = number of tokens (包括 CLS token)
- D = feature dimension

当输入是 2D 时，`apply_token_selection()` 会直接返回原始特征，导致 token selection 没有实际生效。

## 解决方案

### 1. 添加 `get_image_features_with_tokens()` 函数

在 `zeroshot_classification.py` 中添加了新函数，用于从 CLIP 视觉编码器中提取未池化的 token 特征：

```python
def get_image_features_with_tokens(model, images):
    """
    Get image features with all tokens (unpooled) from CLIP model.
    
    Returns:
        image_features: torch.Tensor of shape (B, N, D) for ViT models
    """
```

该函数的工作流程：
1. 通过 `visual.conv1` 将图像转换为 patch embeddings
2. 添加 class token
3. 添加 positional embeddings
4. 通过 transformer blocks
5. **返回所有 tokens（不进行池化）**

### 2. 修改 `run_classification()` 函数

修改了特征提取和处理流程：

**启用 token selection 时：**
```python
# 1. 获取未池化的 token 特征 (B, N, D)
image_features = get_image_features_with_tokens(model, images)

# 2. 应用 token selection（返回稀疏表示）
image_features = apply_token_selection(image_features, k, m, alpha)

# 3. 应用 layer norm
image_features = model.visual.ln_post(image_features)

# 4. 对选中的 tokens 进行平均池化
token_mask = (image_features.abs().sum(dim=-1) > 0).float()
image_features = (image_features * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1)

# 5. 应用投影矩阵
image_features = image_features @ model.visual.proj

# 6. 归一化并计算 logits
image_features = F.normalize(image_features, dim=-1)
logits = 100. * image_features @ classifier
```

**未启用 token selection 时：**
```python
# 使用标准的 encode_image（池化后的特征）
image_features = model.encode_image(images)
```

## 关键改进

1. **正确的维度处理**：现在能够获取 3D token 特征 `(B, N, D)`
2. **稀疏表示的池化**：只对选中的 tokens（非零）进行平均池化
3. **完整的处理流程**：包括 layer norm 和 projection，与标准 CLIP 流程一致
4. **向后兼容**：未启用 token selection 时使用标准路径

## 测试

运行测试脚本验证修改：

```bash
cd /data/workspace/Projects/CLIP_benchmark
python test_token_selection_fix.py
```

预期输出：
- 标准 `encode_image` 返回 2D 特征 `(B, D)`
- `get_image_features_with_tokens` 返回 3D 特征 `(B, N, D)`
- Token selection 后有部分 tokens 为零（稀疏表示）

## 运行 chap2.sh

现在可以正常运行：

```bash
bash chap2.sh
```

Token selection 应该会生效，结果会与未启用时有所不同。

## 技术细节

### ViT-B-32 的 Token 数量

对于 224x224 的输入图像和 patch size 32：
- Grid size = 224 / 32 = 7
- Patch tokens = 7 × 7 = 49
- 加上 CLS token = 50 tokens
- 因此 N = 50

### Token Selection 参数（chap2.sh）

- `k=15`: 选择 15 个 anchor tokens
- `m=60`: 额外选择 60 个 tokens（但实际最多 50-15=35 个）
- `alpha=0.6`: 重要性权重 0.6，多样性权重 0.4

## 文件修改清单

- ✅ `/data/workspace/Projects/CLIP_benchmark/clip_benchmark/metrics/zeroshot_classification.py`
  - 添加 `get_image_features_with_tokens()` 函数
  - 修改 `run_classification()` 函数的特征提取和池化逻辑
  
- ✅ `/data/workspace/Projects/CLIP_benchmark/test_token_selection_fix.py`
  - 新增测试脚本

## 注意事项

1. 该修改仅适用于 **Vision Transformer (ViT)** 模型
2. 对于 ResNet 等 CNN 模型，会自动回退到标准的 `encode_image`
3. Token selection 会增加计算开销（需要处理所有 tokens）
4. 确保有足够的 GPU 内存来处理未池化的 token 特征
