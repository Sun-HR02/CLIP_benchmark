python3 clip_benchmark/cli.py eval \
  --model_type "openai" \ 
  --pretrained "ViT-B-32" \  
  --language "en" \
  --task "zeroshot_classification"  \
  --dataset "cifar10"  \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main"
