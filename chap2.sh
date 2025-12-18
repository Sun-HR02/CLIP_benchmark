python -m clip_benchmark.cli eval \
    --dataset cifar10 \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --enable_token_selection \
    --token_selection_k 15 \
    --token_selection_m 400 \
    --token_selection_alpha 0.6