#!/bin/bash

# 设置输出目录
OUTPUT_DIR="results_comparison"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "第一步：不启用 Token Selection 的评估"
echo "=========================================="

python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-L-14 \
    --pretrained openai \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_baseline.json"

echo ""
echo "=========================================="
echo "第二步：启用 Token Selection 的评估（激进参数）"
echo "=========================================="
echo "使用更激进的参数：k=10, m=30, alpha=0.5"
echo "这将只保留 40 个 token（10 anchor + 30 selected），而不是全部 257 个"
echo ""

python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-L-14 \
    --pretrained openai \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --enable_token_selection \
    --token_selection_k 10 \
    --token_selection_m 30 \
    --token_selection_alpha 0.5 \
    --output "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_token_selection_aggressive.json"

echo ""
echo "=========================================="
echo "第三步：启用 Token Selection 的评估（不同 alpha）"
echo "=========================================="
echo "使用 alpha=1.0（完全基于重要性）"
echo ""

python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-L-14 \
    --pretrained openai \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --enable_token_selection \
    --token_selection_k 10 \
    --token_selection_m 30 \
    --token_selection_alpha 1.0 \
    --output "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_token_selection_alpha1.json"

echo ""
echo "=========================================="
echo "第四步：对比结果"
echo "=========================================="

# 使用 Python 脚本对比三个 JSON 文件
python - <<EOF
import json
import os

baseline_file = "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_baseline.json"
token_selection_aggressive_file = "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_token_selection_aggressive.json"
token_selection_alpha1_file = "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_token_selection_alpha1.json"

print("\n对比结果：")
print("=" * 100)

try:
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(token_selection_aggressive_file, 'r') as f:
        token_selection_aggressive = json.load(f)
    
    with open(token_selection_alpha1_file, 'r') as f:
        token_selection_alpha1 = json.load(f)
    
    print(f"\n{'指标':<30} {'Baseline':<20} {'Token Sel (α=0.5)':<20} {'Token Sel (α=1.0)':<20}")
    print("-" * 100)
    
    baseline_metrics = baseline.get('metrics', {})
    aggressive_metrics = token_selection_aggressive.get('metrics', {})
    alpha1_metrics = token_selection_alpha1.get('metrics', {})
    
    # 对比所有指标
    all_keys = set(baseline_metrics.keys()) | set(aggressive_metrics.keys()) | set(alpha1_metrics.keys())
    
    for key in sorted(all_keys):
        baseline_val = baseline_metrics.get(key, 'N/A')
        aggressive_val = aggressive_metrics.get(key, 'N/A')
        alpha1_val = alpha1_metrics.get(key, 'N/A')
        
        if isinstance(baseline_val, (int, float)) and isinstance(aggressive_val, (int, float)):
            diff_aggressive = aggressive_val - baseline_val
            diff_alpha1 = alpha1_val - baseline_val
            print(f"{key:<30} {baseline_val:<20.4f} {aggressive_val:<20.4f} {alpha1_val:<20.4f}")
            print(f"{'  差异':<30} {'':<20} {diff_aggressive:+.4f} ({diff_aggressive/baseline_val*100:+.2f}%) {diff_alpha1:+.4f} ({diff_alpha1/baseline_val*100:+.2f}%)")
        else:
            print(f"{key:<30} {str(baseline_val):<20} {str(aggressive_val):<20} {str(alpha1_val):<20}")
    
    print("=" * 100)
    print("\n结果文件已保存到:")
    print(f"  - Baseline: {baseline_file}")
    print(f"  - Token Selection (α=0.5, k=10, m=30): {token_selection_aggressive_file}")
    print(f"  - Token Selection (α=1.0, k=10, m=30): {token_selection_alpha1_file}")
    
except FileNotFoundError as e:
    print(f"错误：找不到结果文件 - {e}")
except json.JSONDecodeError as e:
    print(f"错误：JSON 解析失败 - {e}")
except Exception as e:
    print(f"错误：{e}")

EOF

echo ""
echo "=========================================="
echo "评估和对比完成！"
echo "=========================================="
