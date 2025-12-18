#!/bin/bash

echo "=========================================="
echo "Testing Token Pruning with Verbose Output"
echo "=========================================="

# 测试不启用剪枝
echo ""
echo "1. Running WITHOUT pruning (baseline)..."
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --dataset_root /data/datasets \
    --model ViT-B-32 \
    --pretrained openai \
    --batch_size 128 \
    --output results_no_pruning.json \
    --verbose

# 测试启用剪枝
echo ""
echo "2. Running WITH pruning..."
python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --dataset_root /data/datasets \
    --model ViT-B-32 \
    --pretrained openai \
    --batch_size 128 \
    --enable_pruning \
    --k_anchors 10 \
    --top_m 50 \
    --alpha 0.5 \
    --output results_with_pruning.json \
    --verbose

echo ""
echo "=========================================="
echo "Comparing Results"
echo "=========================================="
echo ""
echo "Without pruning:"
cat results_no_pruning.json | python -m json.tool | grep -A 5 "metrics"
echo ""
echo "With pruning:"
cat results_with_pruning.json | python -m json.tool | grep -A 5 "metrics"
