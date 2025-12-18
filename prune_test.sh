python3 clip_benchmark/cli.py eval \
  --model_type "open_clip" \
  --model "ViT-B-32" \
  --pretrained "openai" \
  --language "en" \
  --task "zeroshot_classification"  \
  --dataset "cifar10"  \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  # --extract_attention \
  # --attention_output "my_attention_weights.pt" \
  --enable_pruning \
  --k_anchors 10 \
  --top_m 50 \
  --alpha 0.5