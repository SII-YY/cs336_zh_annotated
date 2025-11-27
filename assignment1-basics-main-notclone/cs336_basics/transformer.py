import torch
from torch import Tensor
from typing import Optional
from cs336_basics.nn_utils import Linear, MultiheadSelfAttentionWithRoPE, RMSNorm, SwiGLU

class TransformerBlock(torch.nn.Module):
    """
    Transformer块实现
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 自注意力层
        self.self_attn = MultiheadSelfAttentionWithRoPE(embed_dim, num_heads)
        self.norm1 = RMSNorm(embed_dim)
        
        # MLP层
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim)
        self.norm2 = RMSNorm(embed_dim)
    
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # 残差连接 + 自注意力
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attention_mask)
        x = x + residual
        
        # 残差连接 + MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class TransformerLM(torch.nn.Module):
    """
    Transformer语言模型实现
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 嵌入层
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        
        # Transformer块
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm_final = RMSNorm(embed_dim)
        self.output = Linear(embed_dim, vocab_size, bias=False)
        
        # 权重绑定：输出层权重与嵌入层共享
        self.output.weight = self.token_embedding.weight
    
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # 输入嵌入
        x = self.token_embedding(x)
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # 最终归一化和输出
        x = self.norm_final(x)
        logits = self.output(x)
        
        return logits