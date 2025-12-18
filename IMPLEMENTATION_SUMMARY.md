# Token Selection Implementation Summary

## 实现概述

已成功在CLIP benchmark项目中实现了可开关的token选择逻辑，该功能在计算metric之前对图像特征进行智能筛选。

## 修改的文件

### 1. 新增文件

#### `/data/workspace/Projects/CLIP_benchmark/clip_benchmark/metrics/token_selection.py`
- **功能**: Token选择核心实现模块
- **主要函数**:
  - `select_anchor_tokens()`: 基于CLS token注意力选择k个锚点token
  - `compute_importance_scores()`: 计算每个token与锚点的注意力重要性
  - `compute_diversity_scores()`: 计算每个token与锚点的余弦相似度多样性
  - `select_tokens_with_scoring()`: 综合重要性和多样性评分选择top-m个token
  - `apply_token_selection()`: 主入口函数，提供完整的token选择流程

#### `/data/workspace/Projects/CLIP_benchmark/TOKEN_SELECTION_README.md`
- **功能**: 详细的使用说明文档
- **内容**: 算法原理、使用方法、参数说明、示例命令

#### `/data/workspace/Projects/CLIP_benchmark/test_token_selection.py`
- **功能**: 单元测试脚本
- **测试覆盖**: 所有核心函数和边界情况

### 2. 修改的文件

#### `/data/workspace/Projects/CLIP_benchmark/clip_benchmark/cli.py`
- **修改内容**:
  - 添加4个新的命令行参数:
    - `--enable_token_selection`: 启用token选择（布尔开关）
    - `--token_selection_k`: 锚点数量（默认10）
    - `--token_selection_m`: 额外选择数量（默认50）
    - `--token_selection_alpha`: 重要性/多样性权重（默认0.5）
  - 在`run()`函数中将这些参数传递给`zeroshot_classification.evaluate()`

#### `/data/workspace/Projects/CLIP_benchmark/clip_benchmark/metrics/zeroshot_classification.py`
- **修改内容**:
  - 导入`apply_token_selection`函数
  - 修改`run_classification()`函数签名，添加token选择相关参数
  - 在`model.encode_image()`之后、特征归一化之前插入token选择逻辑
  - 修改`evaluate()`函数签名，添加token选择相关参数并传递给`run_classification()`

## 算法实现细节

### 步骤1: 锚点选择
```python
# 计算CLS token对所有token的注意力
attention_scores = (cls_token @ all_tokens.T) / sqrt(d)
attention_weights = Softmax(attention_scores)
# 选择top-k个token作为锚点
anchor_indices = topk(attention_weights, k)
```

### 步骤2: 重要性计算
```python
# 计算每个token对锚点的平均注意力
attention_to_anchors = Softmax((all_tokens @ anchor_tokens.T) / sqrt(d))
importance_scores = mean(attention_to_anchors, dim=-1)
```

### 步骤3: 多样性计算
```python
# 计算每个token与锚点的最大余弦相似度
similarity = max(normalize(all_tokens) @ normalize(anchor_tokens).T)
diversity_scores = 1 - similarity
```

### 步骤4: 综合评分与选择
```python
# 综合重要性和多样性
combined_scores = alpha * importance_scores + (1 - alpha) * diversity_scores
# 选择top-m个token（排除已选的锚点）
selected_indices = topk(combined_scores, m)
# 创建稀疏表示（锚点+选中的token，其余为0）
output[anchor_indices] = features[anchor_indices]
output[selected_indices] = features[selected_indices]
```

## 使用示例

### 启用token选择
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --enable_token_selection \
    --token_selection_k 10 \
    --token_selection_m 50 \
    --token_selection_alpha 0.5
```

### 不启用（默认行为）
```bash
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

## 测试结果

运行`test_token_selection.py`的测试结果：
- ✓ 锚点选择功能正常
- ✓ 重要性计算正确
- ✓ 多样性计算正确
- ✓ Token选择与评分正确
- ✓ 端到端流程正常
- ✓ 禁用模式返回原始特征
- ✓ 2D输入（已池化特征）处理正确

所有测试通过！

## 技术特点

1. **可开关设计**: 通过`--enable_token_selection`参数控制，不影响原有功能
2. **灵活配置**: k、m、alpha三个参数可独立调整
3. **兼容性好**: 
   - 自动处理2D和3D特征
   - 对已池化的特征直接返回
   - 不改变特征维度
4. **稀疏表示**: 返回与输入相同shape的张量，未选中位置为0
5. **高效实现**: 使用批量矩阵乘法和PyTorch原生操作

## 注意事项

1. **仅支持zeroshot_classification任务**: 当前实现仅集成到零样本分类任务中
2. **需要3D特征**: Token选择需要完整的token序列（B, N, D），对于已池化的特征会直接返回
3. **参数调优**: 建议根据具体任务调整k、m、alpha参数
4. **计算开销**: 启用token选择会增加注意力计算和相似度计算的开销

## 扩展方向

如需将token选择应用到其他任务：
1. 在对应的metrics文件中导入`apply_token_selection`
2. 在`run_*`函数中的特征提取后调用
3. 在CLI中传递相应参数

参考`zeroshot_classification.py`的实现即可。

## 文件清单

```
CLIP_benchmark/
├── clip_benchmark/
│   ├── metrics/
│   │   ├── token_selection.py          # 新增：核心实现
│   │   └── zeroshot_classification.py  # 修改：集成token选择
│   └── cli.py                          # 修改：添加命令行参数
├── TOKEN_SELECTION_README.md           # 新增：使用文档
├── IMPLEMENTATION_SUMMARY.md           # 新增：实现总结（本文件）
└── test_token_selection.py             # 新增：测试脚本
```
