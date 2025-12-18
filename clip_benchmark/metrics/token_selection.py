"""
Token selection module for CLIP image features.
Implements anchor-based token selection with importance and diversity scoring.
"""
import torch
import torch.nn.functional as F


def select_anchor_tokens(image_features, k=10):
    """
    Select k anchor tokens based on CLS token attention.
    
    Args:
        image_features: torch.Tensor of shape (B, N, D) where
            B is batch size, N is number of tokens (including CLS), D is feature dimension
        k: number of anchor tokens to select
    
    Returns:
        anchor_indices: torch.Tensor of shape (B, k) containing indices of anchor tokens
    """
    # Assume first token is CLS token
    # A_coarse = Softmax(x_cls * W_Q * (X * W_K)^T / sqrt(d))
    # For simplicity, we use dot product attention without explicit W_Q, W_K
    # This can be extended to use learned projection matrices
    
    B, N, D = image_features.shape
    cls_token = image_features[:, 0:1, :]  # (B, 1, D)
    all_tokens = image_features  # (B, N, D)
    
    # Compute attention scores from CLS to all tokens
    # attention_scores = (cls_token @ all_tokens.transpose(-2, -1)) / (D ** 0.5)  # (B, 1, N)
    attention_scores = torch.bmm(cls_token, all_tokens.transpose(-2, -1)) / (D ** 0.5)  # (B, 1, N)
    attention_weights = F.softmax(attention_scores, dim=-1)  # (B, 1, N)
    
    # Select top-k tokens (excluding CLS token itself at index 0)
    # We'll select from tokens 1 to N-1
    attention_weights_no_cls = attention_weights[:, :, 1:]  # (B, 1, N-1)
    topk_values, topk_indices = torch.topk(attention_weights_no_cls, k=min(k, N-1), dim=-1)  # (B, 1, k)
    
    # Adjust indices to account for excluding CLS (add 1 back)
    anchor_indices = topk_indices.squeeze(1) + 1  # (B, k)
    
    return anchor_indices


def compute_importance_scores(image_features, anchor_indices):
    """
    Compute importance score for each token based on attention to anchor tokens.
    I_i = (1 / |S_anchor|) * sum_{x_j in S_anchor} Attn(x_i, x_j)
    
    Args:
        image_features: torch.Tensor of shape (B, N, D)
        anchor_indices: torch.Tensor of shape (B, k)
    
    Returns:
        importance_scores: torch.Tensor of shape (B, N)
    """
    B, N, D = image_features.shape
    k = anchor_indices.shape[1]
    
    # Gather anchor tokens
    # anchor_indices: (B, k) -> expand to (B, k, D)
    anchor_indices_expanded = anchor_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
    anchor_tokens = torch.gather(image_features, 1, anchor_indices_expanded)  # (B, k, D)
    
    # Compute attention from each token to anchor tokens
    # all_tokens: (B, N, D), anchor_tokens: (B, k, D)
    attention_scores = torch.bmm(image_features, anchor_tokens.transpose(-2, -1)) / (D ** 0.5)  # (B, N, k)
    attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, k)
    
    # Average attention to all anchors
    importance_scores = attention_weights.mean(dim=-1)  # (B, N)
    
    return importance_scores


def compute_diversity_scores(image_features, anchor_indices):
    """
    Compute diversity score for each token based on similarity to anchor tokens.
    Sim(x_i, S_anchor) = max_{x_j in S_anchor} (x_i · x_j / (|x_i|_2 * |x_j|_2))
    D_i = 1 - Sim(x_i, S_anchor)
    
    Args:
        image_features: torch.Tensor of shape (B, N, D)
        anchor_indices: torch.Tensor of shape (B, k)
    
    Returns:
        diversity_scores: torch.Tensor of shape (B, N)
    """
    B, N, D = image_features.shape
    k = anchor_indices.shape[1]
    
    # Gather anchor tokens
    anchor_indices_expanded = anchor_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
    anchor_tokens = torch.gather(image_features, 1, anchor_indices_expanded)  # (B, k, D)
    
    # Normalize features
    image_features_norm = F.normalize(image_features, dim=-1)  # (B, N, D)
    anchor_tokens_norm = F.normalize(anchor_tokens, dim=-1)  # (B, k, D)
    
    # Compute cosine similarity
    similarity = torch.bmm(image_features_norm, anchor_tokens_norm.transpose(-2, -1))  # (B, N, k)
    
    # Max similarity to any anchor
    max_similarity, _ = similarity.max(dim=-1)  # (B, N)
    
    # Diversity is 1 - similarity
    diversity_scores = 1.0 - max_similarity  # (B, N)
    
    return diversity_scores


def select_tokens_with_scoring(image_features, anchor_indices, importance_scores, diversity_scores, 
                                m, alpha=0.5):
    """
    Select top-m tokens based on combined importance and diversity scores.
    S_i = alpha * I_i + (1 - alpha) * D_i
    
    Args:
        image_features: torch.Tensor of shape (B, N, D)
        anchor_indices: torch.Tensor of shape (B, k)
        importance_scores: torch.Tensor of shape (B, N)
        diversity_scores: torch.Tensor of shape (B, N)
        m: number of tokens to select (in addition to anchors)
        alpha: weight for importance vs diversity (0 to 1)
    
    Returns:
        selected_features: torch.Tensor of shape (B, N, D) with selected tokens and zeros elsewhere
    """
    B, N, D = image_features.shape
    k = anchor_indices.shape[1]
    
    # Compute combined scores
    combined_scores = alpha * importance_scores + (1 - alpha) * diversity_scores  # (B, N)
    
    # Mask out anchor tokens from selection (they're already included)
    # Create a mask for anchor positions
    anchor_mask = torch.zeros(B, N, device=image_features.device, dtype=torch.bool)
    for b in range(B):
        anchor_mask[b, anchor_indices[b]] = True
    
    # Set anchor positions to very low score so they won't be selected again
    combined_scores_masked = combined_scores.clone()
    combined_scores_masked[anchor_mask] = -float('inf')
    
    # Select top-m tokens (excluding anchors)
    topk_values, topk_indices = torch.topk(combined_scores_masked, k=min(m, N-k), dim=-1)  # (B, m)
    
    # Create output tensor with zeros
    selected_features = torch.zeros_like(image_features)  # (B, N, D)
    
    # Fill in anchor tokens
    anchor_indices_expanded = anchor_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
    anchor_tokens = torch.gather(image_features, 1, anchor_indices_expanded)  # (B, k, D)
    selected_features.scatter_(1, anchor_indices_expanded, anchor_tokens)
    
    # Fill in selected tokens
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, m, D)
    selected_tokens = torch.gather(image_features, 1, topk_indices_expanded)  # (B, m, D)
    selected_features.scatter_(1, topk_indices_expanded, selected_tokens)
    
    return selected_features


def apply_token_selection(image_features, k=10, m=50, alpha=0.5, enabled=True):
    """
    Apply token selection to image features.
    
    Args:
        image_features: torch.Tensor of shape (B, N, D) or (B, D)
            If (B, D), assumes already pooled and returns as-is
        k: number of anchor tokens to select
        m: number of additional tokens to select based on importance/diversity
        alpha: weight for importance vs diversity (0 to 1)
        enabled: whether to apply token selection (if False, returns original features)
    
    Returns:
        selected_features: torch.Tensor of same shape as input
            If token selection is applied, returns (B, N, D) with selected tokens and zeros elsewhere
    """
    if not enabled:
        return image_features
    
    # Check if features are already pooled (2D)
    if len(image_features.shape) == 2:
        # Already pooled, return as-is
        return image_features
    
    # Features should be (B, N, D)
    if len(image_features.shape) != 3:
        raise ValueError(f"Expected image_features to be 3D (B, N, D) or 2D (B, D), got shape {image_features.shape}")
    
    B, N, D = image_features.shape
    
    # Ensure k and m are valid
    k = min(k, N - 1)  # At least leave 1 token for selection
    m = min(m, N - k - 1)  # Ensure we don't exceed total tokens
    
    if k <= 0 or m <= 0:
        # Not enough tokens, return original
        return image_features
    
    # Step 1: Select anchor tokens based on CLS attention
    anchor_indices = select_anchor_tokens(image_features, k=k)
    
    # Step 2: Compute importance scores
    importance_scores = compute_importance_scores(image_features, anchor_indices)
    
    # Step 3: Compute diversity scores
    diversity_scores = compute_diversity_scores(image_features, anchor_indices)
    
    # Step 4: Select top-m tokens and create sparse representation
    selected_features = select_tokens_with_scoring(
        image_features, anchor_indices, importance_scores, diversity_scores, m=m, alpha=alpha
    )
    
    return selected_features
