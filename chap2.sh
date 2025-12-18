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
echo "第二步：启用 Token Selection 的评估"
echo "=========================================="

python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-L-14 \
    --pretrained openai \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --enable_token_selection \
    --token_selection_k 15 \
    --token_selection_m 150 \
    --token_selection_alpha 0.6 \
    --output "${OUTPUT_DIR}/cifar10_openai_ViT-L-14_en_zeroshot_classification_token_selection.json"

echo ""
echo "=========================================="
echo "第三步：对比结果"
echo "=========================================="

# 使用 Python 脚本对比两个 JSON 文件
python - <<EOF
import json
import os

baseline_file = "${OUTPUT_DIR}/cifar10_openai_ViT-B-32_en_zeroshot_classification_baseline.json"
token_selection_file = "${OUTPUT_DIR}/cifar10_openai_ViT-B-32_en_zeroshot_classification_token_selection.json"

print("\n对比结果：")
print("=" * 80)

try:
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(token_selection_file, 'r') as f:
        token_selection = json.load(f)
    
    print(f"\n{'指标':<30} {'Baseline':<20} {'Token Selection':<20} {'差异':<15}")
    print("-" * 80)
    
    baseline_metrics = baseline.get('metrics', {})
    token_selection_metrics = token_selection.get('metrics', {})
    
    # 对比所有指标
    all_keys = set(baseline_metrics.keys()) | set(token_selection_metrics.keys())
    
    for key in sorted(all_keys):
        baseline_val = baseline_metrics.get(key, 'N/A')
        token_val = token_selection_metrics.get(key, 'N/A')
        
        if isinstance(baseline_val, (int, float)) and isinstance(token_val, (int, float)):
            diff = token_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
            print(f"{key:<30} {baseline_val:<20.4f} {token_val:<20.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
        else:
            print(f"{key:<30} {str(baseline_val):<20} {str(token_val):<20} {'N/A':<15}")
    
    print("=" * 80)
    print("\n结果文件已保存到:")
    print(f"  - Baseline: {baseline_file}")
    print(f"  - Token Selection: {token_selection_file}")
    
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