# Token Selection Quick Reference

## 快速开始

```bash
# 启用token选择（使用默认参数）
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --enable_token_selection

# 自定义参数
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --enable_token_selection \
    --token_selection_k 15 \
    --token_selection_m 60 \
    --token_selection_alpha 0.6
```

## 参数速查

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--enable_token_selection` | False | - | 启用token选择 |
| `--token_selection_k` | 10 | 5-20 | 锚点token数量 |
| `--token_selection_m` | 50 | 30-100 | 额外选择的token数量 |
| `--token_selection_alpha` | 0.5 | 0.0-1.0 | 重要性权重（1.0=纯重要性，0.0=纯多样性） |

## 算法公式

```
1. 锚点选择:    A = Softmax(x_cls·W_Q·(X·W_K)^T / √d)
2. 重要性评分:  I_i = (1/|S|) Σ Attn(x_i, x_j)  for x_j ∈ S_anchor
3. 多样性评分:  D_i = 1 - max(x_i·x_j / (|x_i||x_j|))  for x_j ∈ S_anchor
4. 综合评分:    S_i = α·I_i + (1-α)·D_i
5. 选择top-m:  返回锚点+top-m，其余位置为0
```

## 测试

```bash
# 运行单元测试
cd /data/workspace/Projects/CLIP_benchmark
python test_token_selection.py
```

## 文档

- 详细文档: [TOKEN_SELECTION_README.md](TOKEN_SELECTION_README.md)
- 实现总结: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 核心代码

```python
from clip_benchmark.metrics.token_selection import apply_token_selection

# 在特征提取后应用
image_features = model.encode_image(images)  # (B, N, D)
selected_features = apply_token_selection(
    image_features, 
    k=10,      # 锚点数量
    m=50,      # 额外选择数量
    alpha=0.5, # 重要性权重
    enabled=True
)
```

## 常见问题

**Q: 为什么返回的特征中有很多0？**  
A: 这是设计行为。Token选择返回稀疏表示，只在选中的token位置保留特征，其余位置为0。

**Q: 可以用于其他任务吗？**  
A: 当前仅集成到zeroshot_classification。要用于其他任务，需要在对应的metrics文件中调用`apply_token_selection()`。

**Q: 对性能有什么影响？**  
A: 会增加注意力计算和相似度计算的开销，但通常可以接受。

**Q: 如何选择最佳参数？**  
A: 建议从默认值开始，根据具体数据集和模型进行调优。一般来说：
- k越大，覆盖面越广
- m越大，保留信息越多
- alpha=0.5平衡重要性和多样性
