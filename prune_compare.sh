#!/bin/bash

echo "=========================================="
echo "Testing WITHOUT pruning..."
echo "=========================================="
python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --language "en" \
  --task "zeroshot_classification"  \
  --dataset "cifar10"  \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  --output "results_no_pruning.json"

echo ""
echo "=========================================="
echo "Testing WITH pruning..."
echo "=========================================="
python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --language "en" \
  --task "zeroshot_classification"  \
  --dataset "cifar10"  \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  --extract_attention \
  --attention_output "my_attention_weights.pt" \
  --enable_pruning \
  --k_anchors 10 \
  --top_m 50 \
  --alpha 0.5 \
  --output "results_with_pruning.json"

echo ""
echo "=========================================="
echo "Comparison Results:"
echo "=========================================="
echo "Without pruning:"
cat results_no_pruning.json | grep -E '"acc1"|"acc5"'
echo ""
echo "With pruning (k_anchors=10, top_m=50, alpha=0.5):"
cat results_with_pruning.json | grep -E '"acc1"|"acc5"'
