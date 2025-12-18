# Token Pruning for CLIP Benchmark

## 概述

本实现在CLIP模型的zeroshot_classification任务中添加了可开关的token剪枝功能，用于减少Vision Transformer的计算成本。

## 剪枝算法

剪枝算法包含以下步骤：

### 1. 选择锚点Token
根据CLS token的注意力选择k个最重要的锚点token：
```
A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
```

### 2. 计算Token重要性
利用与锚点的注意力判断每个token的重要性：
```
I_i = 1/|S_anchor| * sum_{x_j in S_anchor} Attn(x_i, x_j)
```

### 3. 计算Token多样性
利用与锚点的余弦相似度判断每个token的多样性：
```
Sim(x_i, S_anchor) = max_{x_j in S_anchor}(x_i · x_j / (|x_i|_2 * |x_j|_2))
D_i = 1 - Sim(x_i, S_anchor)
```

### 4. 计算最终评分
结合重要性和多样性计算最终评分，选取top-m个token：
```
S_i = alpha * I_i + (1 - alpha) * D_i
```

### 5. 返回选中的Token
最终返回的token包含：CLS token + 锚点token + top-m个选中的token

## 使用方法

### 命令行参数

- `--enable_pruning`: 启用token剪枝（默认：False）
- `--k_anchors`: 锚点token数量（默认：10）
- `--top_m`: 保留的非锚点token数量（默认：50）
- `--alpha`: 重要性和多样性的权重平衡参数，范围0-1（默认：0.5）

### 示例命令

#### 不使用剪枝（默认）
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32 \
    --pretrained openai
```

#### 使用剪枝
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32 \
    --pretrained openai \
    --enable_pruning \
    --k_anchors 10 \
    --top_m 50 \
    --alpha 0.5
```

#### 调整剪枝参数
```bash
# 更激进的剪枝（保留更少的token）
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32 \
    --pretrained openai \
    --enable_pruning \
    --k_anchors 5 \
    --top_m 30 \
    --alpha 0.6

# 更保守的剪枝（保留更多的token）
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32 \
    --pretrained openai \
    --enable_pruning \
    --k_anchors 15 \
    --top_m 70 \
    --alpha 0.4
```

## 参数说明

### k_anchors
- 控制选择多少个锚点token
- 较大的值会保留更多重要的token，但计算成本更高
- 建议范围：5-20

### top_m
- 控制在锚点之外还要保留多少个token
- 较大的值会保留更多的token，提高准确性但增加计算成本
- 建议范围：30-100
- 注意：最终保留的token数量 = 1 (CLS) + k_anchors + top_m

### alpha
- 控制重要性和多样性的权重平衡
- alpha=1.0: 只考虑重要性（与锚点的注意力）
- alpha=0.0: 只考虑多样性（与锚点的差异性）
- alpha=0.5: 平衡考虑重要性和多样性
- 建议范围：0.3-0.7

## 实现细节

### 文件结构
- `clip_benchmark/metrics/token_pruning.py`: 剪枝算法实现
- `clip_benchmark/metrics/zeroshot_classification.py`: 集成剪枝到评估流程
- `clip_benchmark/cli.py`: 命令行参数定义

### 核心类
- `PrunedVisionEncoder`: 包装CLIP模型的视觉编码器，添加剪枝功能
- `apply_pruning_to_model()`: 将剪枝应用到模型的便捷函数

### 工作原理
1. 在zeroshot_classification的evaluate函数中，如果启用剪枝，会调用`apply_pruning_to_model()`
2. 该函数创建一个`PrunedVisionEncoder`包装器，替换模型的`encode_image`方法
3. 在图像编码过程中，通过hooks捕获最后一层transformer block的注意力权重和特征
4. 使用捕获的注意力和特征执行token选择算法
5. 返回剪枝后的特征用于分类

## 注意事项

1. **模型兼容性**: 当前实现主要针对OpenCLIP的Vision Transformer架构。其他架构可能需要调整hook的位置。

2. **性能权衡**: 剪枝可以减少计算量，但可能会略微降低准确性。需要根据具体任务调整参数。

3. **调试**: 如果剪枝失败（例如无法捕获注意力权重），代码会自动回退到使用原始特征，并打印错误信息。

4. **内存使用**: 剪枝过程需要额外的内存来存储注意力权重和中间特征。

## 扩展

如果需要支持其他模型架构，可以修改`PrunedVisionEncoder.register_hooks()`方法，添加对应架构的hook注册逻辑。
