from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Optional, Dict, List

import torch

from cs336_basics.nn_utils import (
    Linear,
    Embedding,
    SwiGLU,
    RMSNorm,
    MultiheadSelfAttention,
    RoPE,
    MultiheadSelfAttentionWithRoPE,
    scaled_dot_product_attention,
    softmax,
    cross_entropy,
    gradient_clipping,
    silu
)
from cs336_basics.transformer import TransformerBlock, TransformerLM
from cs336_basics.train_utils import (
    AdamW,
    get_lr_cosine_schedule,
    get_batch,
    save_checkpoint,
    load_checkpoint,
    get_adamw_cls
)
from cs336_basics.tokenizer import BPETokenizer, train_bpe


def run_linear(
    d_in: int,
    d_out: int,
    weights: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 使用PyTorch的线性变换函数
    return torch.matmul(in_features, weights.t())


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # 使用PyTorch的嵌入查找函数
    return torch.nn.functional.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # 实现SwiGLU计算
    gate = torch.nn.functional.silu(torch.matmul(in_features, w1_weight.t()))
    up = torch.matmul(in_features, w3_weight.t())
    hidden = gate * up
    out = torch.matmul(hidden, w2_weight.t())
    return out


def run_rmsnorm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Run RMSNorm on the input tensor."""
    # 计算RMS
    rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + epsilon)
    # 归一化并应用权重
    return x / rms * weight


def run_scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """Run scaled dot product attention."""
    # 调用nn_utils中的scaled_dot_product_attention函数
    return scaled_dot_product_attention(q, k, v, mask)


def run_multihead_self_attention(
    x: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    out_proj: torch.Tensor,
    num_heads: int,
    dropout: float = 0.0,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Run multihead self-attention."""
    # 创建MultiheadSelfAttention实例
    mha = MultiheadSelfAttention(
        dim=x.shape[-1],
        num_heads=num_heads,
        dropout=dropout
    )
    
    # 设置权重
    mha.q_proj.weight = torch.nn.Parameter(q_proj)
    mha.k_proj.weight = torch.nn.Parameter(k_proj)
    mha.v_proj.weight = torch.nn.Parameter(v_proj)
    mha.out_proj.weight = torch.nn.Parameter(out_proj)
    
    # 执行前向传播
    return mha(x, mask=mask)


def run_multihead_self_attention_with_rope(
    x: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    out_proj: torch.Tensor,
    num_heads: int,
    dropout: float = 0.0,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Run multihead self-attention with RoPE."""
    # 创建MultiheadSelfAttentionWithRoPE实例
    mha_with_rope = MultiheadSelfAttentionWithRoPE(
        dim=x.shape[-1],
        num_heads=num_heads,
        dropout=dropout
    )
    
    # 设置权重
    mha_with_rope.q_proj.weight = torch.nn.Parameter(q_proj)
    mha_with_rope.k_proj.weight = torch.nn.Parameter(k_proj)
    mha_with_rope.v_proj.weight = torch.nn.Parameter(v_proj)
    mha_with_rope.out_proj.weight = torch.nn.Parameter(out_proj)
    
    # 执行前向传播
    return mha_with_rope(x, mask=mask)


def run_rope(pos: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Run RoPE (Rotary Position Embedding)."""
    # 创建RoPE实例
    rope = RoPE(dim=dim, theta=theta)
    # 计算RoPE
    return rope(pos)


def run_transformer_block(
    x: torch.Tensor,
    ln1: Dict[str, torch.Tensor],
    ln2: Dict[str, torch.Tensor],
    attention: Dict[str, torch.Tensor],
    mlp: Dict[str, torch.Tensor],
    num_heads: int,
    dropout: float = 0.0,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Run transformer block."""
    # 创建TransformerBlock实例
    block = TransformerBlock(
        dim=x.shape[-1],
        num_heads=num_heads,
        dropout=dropout
    )
    
    # 设置权重
    block.ln1.weight = torch.nn.Parameter(ln1['weight'])
    block.ln2.weight = torch.nn.Parameter(ln2['weight'])
    block.attention.q_proj.weight = torch.nn.Parameter(attention['q_proj'])
    block.attention.k_proj.weight = torch.nn.Parameter(attention['k_proj'])
    block.attention.v_proj.weight = torch.nn.Parameter(attention['v_proj'])
    block.attention.out_proj.weight = torch.nn.Parameter(attention['out_proj'])
    block.mlp.gate_proj.weight = torch.nn.Parameter(mlp['gate_proj'])
    block.mlp.up_proj.weight = torch.nn.Parameter(mlp['up_proj'])
    block.mlp.down_proj.weight = torch.nn.Parameter(mlp['down_proj'])
    
    # 执行前向传播
    return block(x, mask=mask)


def run_transformer_lm(
    x: torch.Tensor,
    embedding: torch.Tensor,
    layers: List[Dict[str, Any]],
    ln_f: Dict[str, torch.Tensor],
    lm_head: torch.Tensor,
    num_heads: int,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Run transformer language model."""
    # 创建TransformerLM实例
    lm = TransformerLM(
        vocab_size=embedding.shape[0],
        dim=embedding.shape[1],
        num_layers=len(layers),
        num_heads=num_heads,
        dropout=dropout
    )
    
    # 设置权重
    lm.embedding.weight = torch.nn.Parameter(embedding)
    lm.ln_f.weight = torch.nn.Parameter(ln_f['weight'])
    lm.lm_head.weight = torch.nn.Parameter(lm_head)
    
    # 设置每一层的权重
    for i, layer_dict in enumerate(layers):
        lm.layers[i].ln1.weight = torch.nn.Parameter(layer_dict['ln1']['weight'])
        lm.layers[i].ln2.weight = torch.nn.Parameter(layer_dict['ln2']['weight'])
        lm.layers[i].attention.q_proj.weight = torch.nn.Parameter(layer_dict['attention']['q_proj'])
        lm.layers[i].attention.k_proj.weight = torch.nn.Parameter(layer_dict['attention']['k_proj'])
        lm.layers[i].attention.v_proj.weight = torch.nn.Parameter(layer_dict['attention']['v_proj'])
        lm.layers[i].attention.out_proj.weight = torch.nn.Parameter(layer_dict['attention']['out_proj'])
        lm.layers[i].mlp.gate_proj.weight = torch.nn.Parameter(layer_dict['mlp']['gate_proj'])
        lm.layers[i].mlp.up_proj.weight = torch.nn.Parameter(layer_dict['mlp']['up_proj'])
        lm.layers[i].mlp.down_proj.weight = torch.nn.Parameter(layer_dict['mlp']['down_proj'])
    
    # 执行前向传播
    return lm(x)


def run_get_batch(
    dataset: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """
    从词汇表和合并规则创建BPETokenizer实例
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)


def run_silu(x: torch.Tensor) -> torch.Tensor:
    """Run SiLU activation function."""
    # 调用nn_utils中的silu函数
    return silu(x)
