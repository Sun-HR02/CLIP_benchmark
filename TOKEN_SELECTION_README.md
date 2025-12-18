# Token Selection Feature

## 概述

本功能在CLIP benchmark中添加了可开关的token选择逻辑，用于在计算metric之前对图像特征进行智能筛选。该功能基于注意力机制，通过以下步骤选择最重要和最具多样性的token：

## 算法流程

### 1. 锚点选择（Anchor Selection）
根据CLS token的注意力选取k个锚点token：

```
A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
```

### 2. 重要性计算（Importance Scoring）
利用与锚点的注意力判断每个token的重要性：

```
I_i = (1 / |S_anchor|) * sum_{x_j in S_anchor} Attn(x_i, x_j)
```

### 3. 多样性计算（Diversity Scoring）
利用与锚点的相似度判断每个token的多样性：

```
Sim(x_i, S_anchor) = max_{x_j in S_anchor} (x_i · x_j / (|x_i|_2 * |x_j|_2))
D_i = 1 - Sim(x_i, S_anchor)
```

### 4. 最终评分与选择
计算综合评分并选取top-m个token：

```
S_i = alpha * I_i + (1 - alpha) * D_i
```

最终返回的token序列长度不变，在对应位置上为锚点和选取的top-m特征，其余位置为0。

## 使用方法

### 命令行参数

- `--enable_token_selection`: 启用token选择功能（默认：False）
- `--token_selection_k`: 锚点token数量（默认：10）
- `--token_selection_m`: 额外选择的token数量（默认：50）
- `--token_selection_alpha`: 重要性与多样性的权重（0到1之间，默认：0.5）

### 示例命令

#### 基本使用（启用token选择）
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --enable_token_selection
```

#### 自定义参数
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --enable_token_selection \
    --token_selection_k 15 \
    --token_selection_m 60 \
    --token_selection_alpha 0.6
```

#### 不使用token选择（默认行为）
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

## 参数说明

### token_selection_k（锚点数量）
- 控制基于CLS注意力选择的锚点token数量
- 较大的k值会选择更多的锚点，可能提高覆盖面但降低选择性
- 推荐范围：5-20

### token_selection_m（额外选择数量）
- 控制基于重要性和多样性评分额外选择的token数量
- 较大的m值会保留更多信息，但可能引入噪声
- 推荐范围：30-100

### token_selection_alpha（权重参数）
- 控制重要性（importance）和多样性（diversity）的平衡
- alpha=1.0：完全基于重要性
- alpha=0.0：完全基于多样性
- alpha=0.5：平衡两者
- 推荐范围：0.3-0.7

## 实现细节

### 文件结构
- `clip_benchmark/metrics/token_selection.py`: token选择核心实现
- `clip_benchmark/metrics/zeroshot_classification.py`: 集成token选择到分类流程
- `clip_benchmark/cli.py`: 命令行参数定义

### 关键函数
- `select_anchor_tokens()`: 基于CLS注意力选择锚点
- `compute_importance_scores()`: 计算token重要性
- `compute_diversity_scores()`: 计算token多样性
- `select_tokens_with_scoring()`: 综合评分并选择token
- `apply_token_selection()`: 主入口函数

## 注意事项

1. **特征维度要求**：token选择仅在图像特征为3D张量（B, N, D）时生效，对于已经池化的2D特征（B, D）会直接返回原特征。

2. **性能影响**：启用token选择会增加一定的计算开销，主要来自注意力计算和相似度计算。

3. **适用任务**：当前实现主要针对`zeroshot_classification`任务，其他任务（如retrieval、linear_probe）暂未集成。

4. **参数调优**：建议根据具体数据集和模型进行参数调优，不同的k、m、alpha组合可能产生不同的效果。

## 扩展建议

如需将token选择应用到其他任务（如zeroshot_retrieval、linear_probe等），可以参考`zeroshot_classification.py`中的实现方式，在相应的`run_*`函数中调用`apply_token_selection()`。
