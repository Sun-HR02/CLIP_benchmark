"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score
from .token_selection import apply_token_selection



def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    with torch.no_grad(), torch.autocast(device, enabled=amp):
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def get_image_features_with_tokens(model, images):
    """
    Get image features with all tokens (unpooled) from CLIP model.
    
    Args:
        model: CLIP model
        images: input images tensor
    
    Returns:
        image_features: torch.Tensor of shape (B, N, D) for ViT models or (B, D) for others
    """
    # Try to get unpooled features from vision transformer
    if hasattr(model, 'visual'):
        visual = model.visual
        
        # For open_clip ViT models, we can directly call the visual encoder
        # and extract features before the final pooling
        try:
            # Encode patches
            x = visual.conv1(images)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            
            # Add class token
            x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + visual.positional_embedding.to(x.dtype)
            
            # Apply transformer
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            # Apply ln_post to all tokens (not just CLS)
            x = visual.ln_post(x)  # (B, N, D)
            
            # Apply projection to all tokens if exists
            # This transforms from transformer dim (768) to embedding dim (512)
            if hasattr(visual, 'proj') and visual.proj is not None:
                B, N, D = x.shape
                # Reshape to (B*N, D), apply projection, reshape back
                x_flat = x.reshape(B * N, D)
                x_flat = x_flat @ visual.proj
                x = x_flat.reshape(B, N, -1)
            
            # Return all tokens AFTER ln_post and projection
            # This gives us (B, N, D_embed) where D_embed is the final embedding dim (e.g., 512)
            return x
        except Exception as e:
            # If anything fails, fall back to standard encode_image
            pass
    
    # Fallback: use standard encode_image (returns pooled features)
    return model.encode_image(images)


def run_classification(model, classifier, dataloader, device, amp=True, 
                      enable_token_selection=False, token_selection_k=10, 
                      token_selection_m=50, token_selection_alpha=0.5):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    enable_token_selection: bool
        whether to apply token selection before computing metrics
    
    token_selection_k: int
        number of anchor tokens to select
    
    token_selection_m: int
        number of additional tokens to select
    
    token_selection_alpha: float
        weight for importance vs diversity
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):

            images = images.to(device)
            target = target.to(device)

            with torch.autocast(device, enabled=amp):
                # Get image features (with tokens if possible)
                if enable_token_selection:
                    # # Get unpooled token features (already with ln_post and projection applied)
                    # image_features = get_image_features_with_tokens(model, images)
                    
                    # print(f"[Debug] Image features after encode (with ln_post & proj): {image_features.shape}")
                    
                    # # Apply token selection on the projected features
                    # # Token selection works in the final embedding space (e.g., 512-dim)
                    # image_features = apply_token_selection(
                    #     image_features, 
                    #     k=token_selection_k, 
                    #     m=token_selection_m, 
                    #     alpha=token_selection_alpha,
                    #     enabled=True
                    # )
                    
                    # Get unpooled token features (B, N, D) with ln_post and projection applied
                    image_features = get_image_features_with_tokens(model, images)
                    print(f"[Debug] Unpooled image features shape: {image_features.shape}")
                    
                    # Apply token selection to get sparse representation
                    image_features = apply_token_selection(
                        image_features, 
                        k=token_selection_k, 
                        m=token_selection_m, 
                        alpha=token_selection_alpha,
                        enabled=True
                    )
                    print(f"[Token Selection] After selection (sparse), shape: {image_features.shape}")
                    
                    # Pool the selected tokens: mean pooling over non-zero tokens
                    if len(image_features.shape) == 3:
                        # Create mask for non-zero tokens
                        token_mask = (image_features.abs().sum(dim=-1) > 0).float()  # (B, N)
                        num_selected = token_mask.sum(dim=1).mean().item()
                        print(f"[Token Selection] Average number of selected tokens: {num_selected:.1f}")
                        # Sum features and divide by number of non-zero tokens
                        image_features = (image_features * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        print(f"[Debug] After pooling selected tokens, shape: {image_features.shape}")
                else:
                    # Standard path: use pooled features
                    image_features = model.encode_image(images)
                
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, 
            save_clf=None, load_clfs=[], enable_token_selection=False, token_selection_k=10,
            token_selection_m=50, token_selection_alpha=0.5):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model
    
    enable_token_selection: bool
        whether to apply token selection before computing metrics
    
    token_selection_k: int
        number of anchor tokens to select
    
    token_selection_m: int
        number of additional tokens to select
    
    token_selection_alpha: float
        weight for importance vs diversity

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(model, classifier, dataloader, device, amp=amp,
                                       enable_token_selection=enable_token_selection,
                                       token_selection_k=token_selection_k,
                                       token_selection_m=token_selection_m,
                                       token_selection_alpha=token_selection_alpha)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}
