import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_utils import softmax, silu, rmsnorm

def linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    """
    实现线性层变换
    """
    return torch.matmul(in_features, weights.T)

def embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    """
    实现嵌入层，根据token IDs获取嵌入向量
    """
    return weights[token_ids]

def scaled_dot_product_attention(
    Q: Tensor, 
    K: Tensor, 
    V: Tensor, 
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    实现缩放点积注意力机制
    """
    # 获取维度信息
    d_k = Q.size(-1)
    
    # 计算注意力分数
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 应用掩码（如果有）
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
    
    # 应用softmax获取注意力权重
    attn_weights = softmax(attn_scores, dim=-1)
    
    # 加权求和得到输出
    return torch.matmul(attn_weights, V)

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor
) -> Tensor:
    """
    实现多头自注意力机制
    """
    batch_size = in_features.size(0)
    seq_len = in_features.size(1)
    
    # 计算每个头的维度
    d_k = d_model // num_heads
    
    # 进行投影
    Q = linear(d_model, d_model, q_proj_weight, in_features)
    K = linear(d_model, d_model, k_proj_weight, in_features)
    V = linear(d_model, d_model, v_proj_weight, in_features)
    
    # 重塑为多头格式: [batch_size, seq_len, num_heads, d_k]
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 计算注意力
    attn_output = scaled_dot_product_attention(Q, K, V)
    
    # 重新合并多头
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # 输出投影
    return linear(d_model, d_model, o_proj_weight, attn_output)

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor
) -> Tensor:
    """
    实现旋转位置编码 (RoPE)
    """
    # 确保d_k是偶数
    assert d_k % 2 == 0, "d_k must be even for RoPE"
    
    # 计算旋转角度
    positions = token_positions.unsqueeze(-1)  # [batch_size, seq_len, 1]
    dims = torch.arange(0, d_k, 2, device=in_query_or_key.device)  # [0, 2, 4, ..., d_k-2]
    inv_freq = 1.0 / (theta ** (dims / d_k))  # [1/theta^(0/d_k), 1/theta^(2/d_k), ...]
    
    # 计算旋转角度: [batch_size, seq_len, d_k//2]
    angles = positions * inv_freq
    cos_angles = torch.cos(angles).repeat(1, 1, 2)
    sin_angles = torch.sin(angles).repeat(1, 1, 2)
    
    # 应用旋转
    # 将输入分为偶数和奇数维度
    x_even = in_query_or_key[..., ::2]
    x_odd = in_query_or_key[..., 1::2]
    
    # 旋转操作
    rotated_even = x_even * cos_angles - x_odd * sin_angles
    rotated_odd = x_even * sin_angles + x_odd * cos_angles
    
    # 重新组合
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1).reshape_as(in_query_or_key)
    
    return rotated

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Optional[Tensor] = None
) -> Tensor:
    """
    实现带RoPE的多头自注意力机制
    """
    batch_size = in_features.size(0)
    seq_len = in_features.size(1)
    
    # 如果没有提供位置信息，生成默认的位置序列
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).repeat(batch_size, 1)
    
    # 计算每个头的维度
    d_k = d_model // num_heads
    
    # 进行投影
    Q = linear(d_model, d_model, q_proj_weight, in_features)
    K = linear(d_model, d_model, k_proj_weight, in_features)
    V = linear(d_model, d_model, v_proj_weight, in_features)
    
    # 重塑为多头格式并转置
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 对Q和K应用RoPE
    # 为每个头分别应用RoPE
    Q_roped = []
    K_roped = []
    for i in range(num_heads):
        Q_roped.append(rope(d_k, theta, max_seq_len, Q[:, i], token_positions))
        K_roped.append(rope(d_k, theta, max_seq_len, K[:, i], token_positions))
    
    # 重新组合多头
    Q_roped = torch.stack(Q_roped, dim=1)
    K_roped = torch.stack(K_roped, dim=1)
    
    # 计算注意力
    attn_output = scaled_dot_product_attention(Q_roped, K_roped, V)
    
    # 重新合并多头
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # 输出投影
    return linear(d_model, d_model, o_proj_weight, attn_output)

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor
) -> Tensor:
    """
    实现SwiGLU激活函数的前馈网络
    """
    # 第一层投影
    x1 = linear(d_model, d_ff, w1_weight, in_features)
    # 第三层投影
    x3 = linear(d_model, d_ff, w3_weight, in_features)
    # 应用SiLU激活
    x_silu = silu(x1)
    # 按元素相乘
    x_mul = x_silu * x3
    # 第二层投影
    return linear(d_ff, d_model, w2_weight, x_mul)

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor
) -> Tensor:
    """
    实现Transformer块
    """
    # 层归一化1
    ln1_output = rmsnorm(
        d_model, 
        1e-5, 
        weights["ln1.weight"], 
        in_features
    )
    
    # 多头自注意力
    attn_output = multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        ln1_output
    )
    
    # 残差连接1
    residual1 = in_features + attn_output
    
    # 层归一化2
    ln2_output = rmsnorm(
        d_model, 
        1e-5, 
        weights["ln2.weight"], 
        residual1
    )
    
    # SwiGLU前馈网络
    ffn_output = swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        ln2_output
    )
    
    # 残差连接2
    residual2 = residual1 + ffn_output
    
    return residual2

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor
) -> Tensor:
    """
    实现Transformer语言模型
    """
    # 输入嵌入
    token_embeddings = embedding(
        vocab_size,
        d_model,
        weights["token_embeddings.weight"],
        in_indices
    )
    
    # 通过多层Transformer块
    x = token_embeddings
    for layer_idx in range(num_layers):
        layer_weights = {
            f"attn.{k}": v 
            for k, v in weights.items() 
            if f"layers.{layer_idx}.attn." in k
        }
        layer_weights.update({
            f"ffn.{k}": v 
            for k, v in weights.items() 
            if f"layers.{layer_idx}.ffn." in k
        })
        layer_weights.update({
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"]
        })
        
        x = transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            layer_weights,
            x
        )
    
    # 最终层归一化
    x = rmsnorm(
        d_model,
        1e-5,
        weights["ln_final.weight"],
        x
    )
    
    # 语言模型头部
    logits = linear(
        d_model,
        vocab_size,
        weights["lm_head.weight"],
        x
    )
    
    return logits