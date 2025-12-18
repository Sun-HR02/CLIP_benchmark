"""Console script for clip_benchmark."""
import argparse
import csv
import json
import os
import random
import sys
from copy import copy
from itertools import product

import torch

from clip_benchmark.datasets.builder import (build_dataset, dataset_collection,
                                             get_dataset_collate_fn,
                                             get_dataset_collection_from_file,
                                             get_dataset_default_task)
from clip_benchmark.metrics import (captioning, image_caption_selection,
                                    linear_probe, zeroshot_classification,
                                    zeroshot_retrieval)
from clip_benchmark.model_collection import (get_model_collection_from_file,
                                             model_collection)
from clip_benchmark.models import MODEL_TYPES, load_clip


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+", help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str, help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--test_split', dest="split", action='store', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--train_split', type=str, nargs="+", default="train", help="Dataset(s) train split names")
    mutually_exclusive = parser_eval.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+", help="Dataset(s) validation split names. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+", help="what is the share of the train dataset will be used for validation part, if it doesn't predefined. Mutually exclusive with val_split")
    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"], help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, nargs="+", default=["laion400m_e32"], help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+", help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto", choices=["zeroshot_classification", "zeroshot_retrieval", "linear_probe", "captioning", "image_caption_selection", "auto"], help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True, help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int, help="for retrieval, select the k for Recall@K metric. ", nargs="+",)
    parser_eval.add_argument('--fewshot_k', default=-1, type=int, help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float, help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true", help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--normalize', default=True, type=bool, help="features normalization")
    parser_eval.add_argument('--model_cache_dir', default=None, type=str, help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str, help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str, help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str, help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")
    parser_eval.add_argument('--dump_classnames', default=False, action="store_true", help="dump classnames to the results json file.")
    parser_eval.add_argument('--dump_templates', default=False, action="store_true", help="dump templates to the results json file.")

    parser_eval.add_argument('--language', default="en", type=str, nargs="+", help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output', default="{dataset}_{pretrained}_{model}_{language}_{task}.json", type=str, help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str, help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str, help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true", help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str, help="optional cache directory for webdataset only")
    parser_eval.add_argument('--extract_attention', default=False, action="store_true", help="whether to extract and save attention weights from vision encoder")
    parser_eval.add_argument('--attention_output', default="attention_{dataset}_{pretrained}_{model}.pt", type=str, help="output file for attention weights")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str,  nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str,  default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    args = parser.parse_args()
    return parser, args

def main():
    parser, base = get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)

def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()
    def process_file(path: str):
        data = json.load(open(path))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        row['model_fullname'] = row['model'] + ' ' + row['pretrained']
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)
    for path in base.files:
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
            for file in files:
                process_file(file)
        else:
            process_file(path)
    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main_eval(base):
    # Get list of pre-trained models to evaluate
    pretrained_model = _as_list(base.pretrained_model)
    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                # if path, read file, each line is a pre-trained model
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                # if part of `model_collection`, retrieve from it
                models.extend(model_collection[name])
            else:
                # if not, assume it is in the form of `model,pretrained`
                model, pretrained = name.split(',')
                models.append((model, pretrained))
    else:
        models = list(product(base.model, base.pretrained))

    # Ge list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)
    
    train_splits = _as_list(base.train_split)
    train_splits = _single_option_to_multiple_datasets(train_splits, datasets, "train_split")
    proportions, val_splits = None, None
    if base.val_split is not None:
        val_splits = _as_list(base.val_split)
        val_splits = _single_option_to_multiple_datasets(val_splits, datasets, "val_split")
    if base.val_proportion is not None:
        proportions = _as_list(base.val_proportion)
        proportions = _single_option_to_multiple_datasets(proportions, datasets, "val_proportion")

    dataset_info = {}
    for i in range(len(datasets)):
        dataset_info[datasets[i]] = {
            "train_split": train_splits[i],
            "val_split": val_splits[i] if val_splits is not None else None,
            "proportion": proportions[i] if proportions is not None else None
        }
    
    # Get list of languages to evaluate on
    languages = _as_list(base.language)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Languages: {languages}")
    runs = product(models, datasets, languages)
    if base.distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        # randomize runs so that runs are balanced across gpus
        random.seed(base.seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]
    for (model, pretrained), (dataset), (language) in runs:
        # We iterative over all possible model/dataset/languages
        args = copy(base)
        args.model = model
        args.pretrained = pretrained
        args.dataset = dataset
        args.language = language
        args.train_split = dataset_info[dataset]["train_split"]
        args.val_split = dataset_info[dataset]["val_split"]
        args.val_proportion = dataset_info[dataset]["proportion"]
        run(args)

def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l

def _single_option_to_multiple_datasets(cur_option, datasets, name):
    cur_len = len(cur_option)
    ds_len = len(datasets)
    if cur_len != ds_len:
        # If user wants to use same value for all datasets
        if cur_len == 1:
            return [cur_option[0]] * ds_len
        else:
            raise ValueError(f"The incommensurable number of {name}")
    else:
        return cur_option

def run(args):
    """Console script for clip_benchmark."""
    if torch.cuda.is_available():
        if args.distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = 'cuda:%d' % local_rank
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        args.device = device
    else:
        args.device = "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = os.path.basename(args.pretrained) if os.path.isfile(args.pretrained) else args.pretrained
    pretrained_slug_full_path = args.pretrained.replace('/', '_') if os.path.isfile(args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        model=args.model, 
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task, 
        dataset=dataset_slug,
        language=args.language
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"Running '{task}' on '{dataset_name}' with the model '{args.pretrained}' on language '{args.language}'")
    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
        model.eval()
        if args.model.count("nllb-clip") > 0:
            # for NLLB-CLIP models, we need to set the language prior to running the tests
            from clip_benchmark.models.nllb_clip import set_language

            set_language(tokenizer, args.language)
        dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform, 
            split=args.split, 
            annotation_file=args.annotation_file,
            download=True,
            language=args.language,
            task=task,
            custom_template_file=args.custom_template_file,
            custom_classname_file=args.custom_classname_file,
            wds_cache_dir=args.wds_cache_dir,
        )
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if hasattr(dataset, "classes") and dataset.classes:
                try:
                    print(f"Dataset classes: {dataset.classes}")
                    print(f"Dataset number of classes: {len(dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        if args.dataset.startswith("wds/"):
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(args.batch_size), batch_size=None, 
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.num_workers, 
                collate_fn=collate_fn
            )
    if task == "zeroshot_classification":
        zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics = zeroshot_classification.evaluate(
            model, 
            dataloader, 
            tokenizer, 
            classnames, zeroshot_templates, 
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            save_clf=args.save_clf,
            load_clfs=args.load_clfs,
        )
        
        # 提取注意力权重（如果需要）
        if args.extract_attention:
            if args.verbose:
                print("Extracting attention weights from vision encoder...")
            # 获取一个批次的图像
            sample_images, _ = next(iter(dataloader))
            if isinstance(sample_images, (tuple, list)):
                sample_images = sample_images[0]
            
            # 提取注意力权重
            attention_weights = get_vision_attention(model, sample_images, device=args.device)
            
            # 保存注意力权重
            attention_output = args.attention_output.format(
                model=args.model,
                pretrained=pretrained_slug,
                dataset=dataset_slug
            )
            
            attention_data = {
                'attention_weights': attention_weights,
                'model': args.model,
                'pretrained': args.pretrained,
                'dataset': args.dataset,
                'num_layers': len(attention_weights),
                'shape_info': {
                    'last_layer_shape': str(attention_weights[-1].shape) if attention_weights else None,
                    'description': '[batch_size, num_heads, seq_len, seq_len]'
                }
            }
            
            torch.save(attention_data, attention_output)
            if args.verbose:
                print(f"Attention weights saved to: {attention_output}")
                if attention_weights:
                    print(f"Number of layers: {len(attention_weights)}")
                    print(f"Last layer attention shape: {attention_weights[-1].shape}")
    elif task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model, 
            dataloader, 
            tokenizer, 
            recall_k_list=args.recall_k,
            device=args.device, 
            amp=args.amp
        )
    elif task == "image_caption_selection":
        metrics = image_caption_selection.evaluate(
            model,
            dataloader,
            tokenizer,
            device=args.device,
            amp=args.amp,
        )
    elif task == "linear_probe":
        # we also need the train and validation splits for linear probing.
        train_dataset = None
        train_dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform, 
            split=args.train_split, 
            annotation_file=args.annotation_file,
            download=True,
        )
        if args.val_split is not None:
            val_dataset = build_dataset(
                dataset_name=args.dataset, 
                root=dataset_root, 
                transform=transform, 
                split=args.val_split, 
                annotation_file=args.annotation_file,
                download=True,
            )
        elif args.val_proportion is not None:
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [1 - args.val_proportion, args.val_proportion])
        else:
            val_dataset = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, 
            collate_fn=collate_fn, pin_memory=True,
        )
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.num_workers, 
                collate_fn=collate_fn, pin_memory=True,
            )
        else:
            val_dataloader = None
        metrics = linear_probe.evaluate(
            model,
            train_dataloader,
            dataloader, 
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            val_dataloader=val_dataloader,
            device=args.device, 
            normalize=args.normalize,
            amp=args.amp,
            verbose=args.verbose,
        )
    elif task == "captioning":
        metrics = captioning.evaluate(
            model=model, 
            dataloader=dataloader, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            transform=transform
        )
    else:
        raise ValueError("Unsupported task: {}. task should be `zeroshot_classification`, `zeroshot_retrieval`, `linear_probe`, or `captioning`".format(task))
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if hasattr(dataset, "classes") and dataset.classes and args.dump_classnames:
        dump["classnames"] = dataset.classes
    if hasattr(dataset, "templates") and dataset.templates and args.dump_templates:
        dump["templates"] = dataset.templates
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size

def get_vision_attention(model, images, device='cuda', enable_pruning=True, k_anchors=10, top_m=50, alpha=0.5):
    """
    从视觉编码器获取注意力矩阵，支持可选的token剪枝
    
    Args:
        model: OpenCLIP 模型
        images: 输入图像张量 [batch_size, 3, H, W]
        device: 设备
        enable_pruning: 是否启用token剪枝
        k_anchors: 选择的锚点token数量
        top_m: 最终保留的非锚点token数量
        alpha: 重要性和多样性的权重平衡参数
    
    Returns:
        attention_weights: List of attention tensors, 每层一个
                          shape: [batch_size, num_heads, seq_len, seq_len]
    """
    # 访问视觉编码器
    visual = model.visual
    attention_weights = []
    
    def select_tokens_by_pruning(x, attn_weights, k_anchors, top_m, alpha):
        """
        基于注意力的token剪枝策略
        
        Args:
            x: token特征 [seq_len, batch_size, embed_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
            k_anchors: 锚点数量
            top_m: 保留的非锚点token数量
            alpha: 重要性和多样性的权重
            
        Returns:
            selected_indices: 选中的token索引 [batch_size, k_anchors + top_m]
            pruned_x: 剪枝后的token特征
        """
        seq_len, batch_size, embed_dim = x.shape
        
        # 转换x的维度: [seq_len, batch_size, embed_dim] -> [batch_size, seq_len, embed_dim]
        x_batch = x.permute(1, 0, 2)
        
        # 1. 使用cls token (第0个token) 的注意力选择k个锚点
        # A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
        # 这里直接使用已计算的注意力权重中cls token对其他token的注意力
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        
        # 平均所有注意力头的cls token注意力
        cls_attention = attn_weights[:, :, 0, :].mean(dim=1)  # [batch_size, seq_len]
        
        # 排除cls token自身，选择top-k作为锚点
        cls_attention_no_cls = cls_attention[:, 1:]  # [batch_size, seq_len-1]
        _, anchor_indices_relative = torch.topk(cls_attention_no_cls, k=min(k_anchors, seq_len-1), dim=1)
        anchor_indices = anchor_indices_relative + 1  # 加1因为排除了cls token
        
        # 2. 计算每个token与锚点的平均注意力 (重要性)
        # I_i = (1/|S_anchor|) * sum(Attn(x_i, x_j)) for x_j in S_anchor
        
        # 获取所有token对所有token的平均注意力
        avg_attn = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        importance_scores = []
        diversity_scores = []
        
        for b in range(batch_size):
            anchor_idx = anchor_indices[b]  # [k_anchors]
            
            # 计算重要性: 每个token对锚点的平均注意力
            # avg_attn[b]: [seq_len, seq_len]
            # 选择每个token对锚点的注意力
            attn_to_anchors = avg_attn[b, :, anchor_idx]  # [seq_len, k_anchors]
            importance = attn_to_anchors.mean(dim=1)  # [seq_len]
            
            # 3. 计算每个token与锚点的相似度 (多样性)
            # Sim(x_i, S_anchor) = max(x_i · x_j / (|x_i|_2 * |x_j|_2)) for x_j in S_anchor
            # D_i = 1 - Sim(x_i, S_anchor)
            
            # 归一化特征
            x_norm = torch.nn.functional.normalize(x_batch[b], p=2, dim=1)  # [seq_len, embed_dim]
            anchor_features = x_norm[anchor_idx]  # [k_anchors, embed_dim]
            
            # 计算余弦相似度
            similarity = torch.matmul(x_norm, anchor_features.T)  # [seq_len, k_anchors]
            max_similarity = similarity.max(dim=1)[0]  # [seq_len]
            diversity = 1 - max_similarity  # [seq_len]
            
            importance_scores.append(importance)
            diversity_scores.append(diversity)
        
        importance_scores = torch.stack(importance_scores)  # [batch_size, seq_len]
        diversity_scores = torch.stack(diversity_scores)  # [batch_size, seq_len]
        
        # 4. 计算最终评分并选择top-m个token
        # S_i = alpha * I_i + (1 - alpha) * D_i
        final_scores = alpha * importance_scores + (1 - alpha) * diversity_scores
        
        # 排除cls token和已选择的锚点
        selected_indices_list = []
        for b in range(batch_size):
            # 创建mask排除cls和锚点
            mask = torch.ones(seq_len, dtype=torch.bool, device=final_scores.device)
            mask[0] = False  # 排除cls
            mask[anchor_indices[b]] = False  # 排除锚点
            
            # 在剩余token中选择top-m
            remaining_scores = final_scores[b].clone()
            remaining_scores[~mask] = -float('inf')
            
            _, top_m_indices = torch.topk(remaining_scores, k=min(top_m, mask.sum().item()), dim=0)
            
            # 合并: cls + 锚点 + top-m
            selected = torch.cat([
                torch.tensor([0], device=device),  # cls token
                anchor_indices[b],  # 锚点
                top_m_indices  # top-m tokens
            ])
            
            # 排序以保持原始顺序
            selected = torch.sort(selected)[0]
            selected_indices_list.append(selected)
        
        # 由于不同batch可能选择不同数量的token，这里返回索引列表
        return selected_indices_list
    
    def make_attention_hook(layer_idx):
        """创建一个闭包来捕获特定层的注意力权重"""
        def hook(module, input, output):
            # 在 MultiheadAttention 中，我们需要手动计算或捕获注意力权重
            # OpenCLIP 的注意力层通常不直接返回注意力权重
            # 我们需要在 forward 过程中捕获 Q, K, V
            pass
        return hook
    
    # 更好的方法：直接修改前向传播来获取注意力权重
    def attention_forward_hook(module, input, output):
        """捕获注意力模块的输出"""
        # 对于 nn.MultiheadAttention，需要在调用时设置 need_weights=True
        # 但 OpenCLIP 可能没有直接暴露这个选项
        # 我们需要手动计算注意力权重
        pass
    
    # 注册 hook 到每个 transformer block 的注意力层
    hooks = []
    original_forwards = []
    
    # 保存原始的 forward 方法并替换为自定义版本
    for i, block in enumerate(visual.transformer.resblocks):
        # 保存原始 forward
        original_forward = block.attn.forward
        original_forwards.append(original_forward)
        
        def make_custom_forward(orig_forward, layer_idx):
            def custom_forward(x, attn_mask=None):
                # 手动计算注意力权重
                # x shape: [seq_len, batch_size, embed_dim]
                seq_len, batch_size, embed_dim = x.shape
                
                # 获取 Q, K, V
                qkv = torch.nn.functional.linear(x, block.attn.in_proj_weight, block.attn.in_proj_bias)
                qkv = qkv.reshape(seq_len, batch_size, 3, block.attn.num_heads, embed_dim // block.attn.num_heads)
                qkv = qkv.permute(2, 1, 3, 0, 4)  # [3, batch, heads, seq_len, head_dim]
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # 计算注意力权重
                scale = (embed_dim // block.attn.num_heads) ** -0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq_len, seq_len]
                
                if attn_mask is not None:
                    attn = attn + attn_mask
                
                attn_weights = torch.nn.functional.softmax(attn, dim=-1)
                attention_weights.append(attn_weights.detach().cpu())
                
                # 如果启用剪枝，在前向传播中进行token选择
                if enable_pruning and layer_idx > 0:  # 第一层不剪枝
                    with torch.no_grad():
                        selected_indices_list = select_tokens_by_pruning(
                            x, attn_weights.detach(), k_anchors, top_m, alpha
                        )
                        
                        # 对每个batch进行token选择
                        pruned_x_list = []
                        for b in range(batch_size):
                            selected_idx = selected_indices_list[b]
                            pruned_x_list.append(x[selected_idx, b, :])
                        
                        # 由于不同batch可能有不同数量的token，需要padding
                        # 这里简化处理：使用第一个batch的选择应用到所有batch
                        # 实际应用中可能需要更复杂的处理
                        selected_idx = selected_indices_list[0]
                        x_pruned = x[selected_idx, :, :]
                        
                        # 使用剪枝后的x继续前向传播
                        return orig_forward(x_pruned, attn_mask)
                
                # 调用原始 forward
                return orig_forward(x, attn_mask)
            return custom_forward
        
        # 替换 forward 方法
        block.attn.forward = make_custom_forward(original_forward, i)
    
    # 前向传播
    with torch.no_grad():
        images = images.to(device)
        _ = visual(images)
    
    # 恢复原始 forward 方法
    for i, block in enumerate(visual.transformer.resblocks):
        block.attn.forward = original_forwards[i]
    
    return attention_weights


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
