# Token Pruning 功能说明

## 概述

现在已经实现了在评估过程中应用token剪枝的功能。剪枝逻辑会在模型前向传播时自动生效，因此评估得到的准确率等指标是**剪枝后模型**的性能。

## 工作原理

### 1. 剪枝策略

基于注意力的四步剪枝策略：

1. **选择锚点token**：根据CLS token的注意力选取k个最重要的锚点
   ```
   A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
   ```

2. **计算重要性**：利用与锚点的注意力判断每个token的重要性
   ```
   I_i = (1/|S_anchor|) * sum(Attn(x_i, x_j)) for x_j in S_anchor
   ```

3. **计算多样性**：利用与锚点的相似度判断每个token的多样性
   ```
   Sim(x_i, S_anchor) = max(x_i · x_j / (|x_i|_2 * |x_j|_2)) for x_j in S_anchor
   D_i = 1 - Sim(x_i, S_anchor)
   ```

4. **最终评分**：结合重要性和多样性，选取top-m个token
   ```
   S_i = alpha * I_i + (1 - alpha) * D_i
   ```

最终返回的token包含：**CLS token + k个锚点 + top-m个选中的token**

### 2. 实现方式

- 在评估**之前**，通过 `apply_pruning_to_model()` 函数为模型的视觉编码器注入剪枝逻辑
- 剪枝逻辑会替换每个transformer block的注意力层的forward方法
- 在前向传播时，从第2层开始自动应用剪枝（第1层不剪枝以保留完整信息）
- 评估**之后**，恢复模型的原始forward方法

## 命令行参数

### 剪枝相关参数

- `--enable_pruning`: 启用token剪枝（默认：False）
- `--k_anchors`: 锚点token数量（默认：10）
- `--top_m`: 保留的非锚点token数量（默认：50）
- `--alpha`: 重要性和多样性的权重平衡参数，范围[0,1]（默认：0.5）
  - alpha=1.0: 只考虑重要性
  - alpha=0.0: 只考虑多样性
  - alpha=0.5: 平衡考虑

### 注意力提取参数

- `--extract_attention`: 提取并保存注意力权重（默认：False）
- `--attention_output`: 注意力权重输出文件路径

## 使用示例

### 1. 不使用剪枝（基线）

```bash
python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --dataset "cifar10" \
  --task "zeroshot_classification" \
  --output "results_baseline.json"
```

### 2. 使用剪枝评估

```bash
python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --dataset "cifar10" \
  --task "zeroshot_classification" \
  --enable_pruning \
  --k_anchors 10 \
  --top_m 50 \
  --alpha 0.5 \
  --output "results_pruned.json"
```

### 3. 使用剪枝并提取注意力权重

```bash
python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --dataset "cifar10" \
  --task "zeroshot_classification" \
  --extract_attention \
  --attention_output "attention_weights.pt" \
  --enable_pruning \
  --k_anchors 10 \
  --top_m 50 \
  --alpha 0.5 \
  --output "results_pruned_with_attn.json"
```

### 4. 对比测试

使用提供的对比脚本：

```bash
bash prune_compare.sh
```

这会运行两次评估（无剪枝和有剪枝），并对比结果。

## 快速测试

使用简化的测试脚本：

```bash
bash prune_test.sh
```

## 参数调优建议

### k_anchors（锚点数量）

- **较小值（5-10）**：更激进的剪枝，速度更快，但可能损失精度
- **较大值（15-20）**：保守的剪枝，精度更高，但加速效果减弱
- **推荐**：从10开始尝试

### top_m（保留token数量）

- **较小值（30-50）**：更激进的剪枝，对于ViT-B-32（原始197个token），保留约25-30%
- **较大值（80-100）**：保守的剪枝，保留约40-50%
- **推荐**：从50开始尝试（约25%保留率）

### alpha（权重平衡）

- **alpha=1.0**：只考虑重要性，可能导致选中的token过于相似
- **alpha=0.5**：平衡重要性和多样性，通常效果最好
- **alpha=0.0**：只考虑多样性，可能忽略重要信息
- **推荐**：0.5

## 预期效果

对于ViT-B-32（197个token）：

- **无剪枝**：197个token
- **剪枝后**（k_anchors=10, top_m=50）：1 (CLS) + 10 (anchors) + 50 (top-m) = 61个token
- **压缩率**：约31%的token保留率，理论上可以加速约3倍

## 注意事项

1. **第一层不剪枝**：为了保留完整的图像信息，第一个transformer block不应用剪枝
2. **批次一致性**：当前实现使用第一个batch的token选择应用到所有batch，确保批处理的一致性
3. **attn_mask处理**：剪枝时会同步调整attention mask的维度
4. **评估时机**：剪枝在评估过程中生效，因此评估结果反映的是剪枝后模型的性能

## 代码结构

- `apply_pruning_to_model()`: 为模型注入剪枝逻辑
- `select_tokens_by_pruning()`: 实现四步剪枝策略
- `get_vision_attention()`: 提取注意力权重（可选）

## 故障排除

### 如果遇到维度不匹配错误

确保：
1. `k_anchors + top_m + 1` (CLS) 不超过原始token数量
2. 对于ViT-B-32，原始token数 = 197 (1 CLS + 196 patches)

### 如果准确率下降过多

尝试：
1. 增加 `top_m` 值
2. 增加 `k_anchors` 值
3. 调整 `alpha` 值
4. 只在更深的层应用剪枝（修改代码中的 `layer_idx > 0` 条件）
