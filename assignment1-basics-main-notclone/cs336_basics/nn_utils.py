import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Optional
import math

class Linear(torch.nn.Module):
    """
    线性层实现
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

class Embedding(torch.nn.Module):
    """
    词嵌入层实现
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        
        # 初始化权重
        torch.nn.init.normal_(self.weight, mean=0.0, std=embedding_dim ** -0.5)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight)

class SwiGLU(torch.nn.Module):
    """
    SwiGLU激活函数实现
    """
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim * 4
        self.w_gate = torch.nn.Parameter(torch.randn(embed_dim, self.hidden_dim))
        self.w_proj = torch.nn.Parameter(torch.randn(embed_dim, self.hidden_dim))
        self.b_gate = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_proj = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        
        # 初始化权重
        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_proj, a=math.sqrt(5))
    
    def forward(self, x: Tensor) -> Tensor:
        gate = F.linear(x, self.w_gate, self.b_gate)
        gate = F.silu(gate)
        proj = F.linear(x, self.w_proj, self.b_proj)
        return gate * proj

class RMSNorm(torch.nn.Module):
    """
    RMSNorm实现
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x: Tensor) -> Tensor:
        # 计算均方根
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # 归一化并应用权重
        return (x / rms) * self.weight

def scaled_dot_product_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor, 
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    实现缩放点积注意力
    """
    # 获取注意力维度
    d_k = q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码
    if mask is not None:
        scores = scores + mask
    
    # 应用softmax
    attention = softmax(scores, dim=-1)
    
    # 与值相乘
    return torch.matmul(attention, v)

class MultiheadSelfAttention(torch.nn.Module):
    """
    多头自注意力实现
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV投影权重
        self.w_qkv = torch.nn.Parameter(torch.randn(embed_dim, 3 * embed_dim))
        # 输出投影权重
        self.w_out = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.w_qkv)
        torch.nn.init.xavier_uniform_(self.w_out)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV投影
        qkv = F.linear(x, self.w_qkv)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return F.linear(attn_output, self.w_out)

class RoPE:
    """
    旋转位置编码实现
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta
    
    def apply_rotary_pos_emb(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        
        # 计算位置编码
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        freqs = torch.outer(position, inv_freq)
        
        # 创建旋转矩阵
        cos_freq = freqs.cos()
        sin_freq = freqs.sin()
        
        # 重塑以应用到输入上
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # 应用旋转
        rotated_x = torch.zeros_like(x)
        rotated_x[..., ::2] = x1 * cos_freq.unsqueeze(0) - x2 * sin_freq.unsqueeze(0)
        rotated_x[..., 1::2] = x1 * sin_freq.unsqueeze(0) + x2 * cos_freq.unsqueeze(0)
        
        return rotated_x

class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    """
    带RoPE的多头自注意力实现
    """
    def __init__(self, embed_dim: int, num_heads: int, rope_theta: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = RoPE(self.head_dim, rope_theta)
        
        # QKV投影权重
        self.w_qkv = torch.nn.Parameter(torch.randn(embed_dim, 3 * embed_dim))
        # 输出投影权重
        self.w_out = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.w_qkv)
        torch.nn.init.xavier_uniform_(self.w_out)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV投影
        qkv = F.linear(x, self.w_qkv)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 应用RoPE到查询和键
        q = self.rope.apply_rotary_pos_emb(q)
        k = self.rope.apply_rotary_pos_emb(k)
        
        # 注意力计算
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return F.linear(attn_output, self.w_out)

def softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    实现softmax函数，处理数值稳定性问题
    """
    # 为了数值稳定性，减去每行的最大值
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_vals = torch.exp(in_features - max_vals)
    return exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    实现交叉熵损失函数
    """
    # 为了数值稳定性，减去每行的最大值
    max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
    inputs_stable = inputs - max_vals
    exp_logits = torch.exp(inputs_stable)
    log_sum_exp = torch.log(torch.sum(exp_logits, dim=1))
    log_probs = inputs_stable - log_sum_exp
    # 选择正确类别的log概率
    correct_log_probs = log_probs[range(inputs.size(0)), targets]
    return -torch.mean(correct_log_probs)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    实现梯度裁剪，确保梯度的L2范数不超过max_l2_norm
    """
    # 收集所有需要梯度的参数
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    
    # 计算当前梯度的L2范数
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 如果范数超过阈值，则裁剪
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.data.mul_(clip_coef)

def silu(in_features: Tensor) -> Tensor:
    """
    实现SiLU激活函数 (x * sigmoid(x))
    """
    return in_features * torch.sigmoid(in_features)