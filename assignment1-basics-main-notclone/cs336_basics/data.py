import torch
from torch import Tensor
from typing import List, Tuple
from jaxtyping import Int

def get_batch(
    split: str,
    batch_size: int,
    context_length: int,
    train_data: Tensor,
    val_data: Tensor,
    device: torch.device
) -> Tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
    """
    从训练或验证数据中获取一批样本
    """
    # 选择数据源
    data = train_data if split == 'train' else val_data
    
    # 随机选择batch_size个起始位置
    indices = torch.randint(0, data.size(0) - context_length, (batch_size,))
    
    # 创建输入和目标张量
    x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    
    # 填充数据
    for i, idx in enumerate(indices):
        x[i] = data[idx:idx+context_length]
        y[i] = data[idx+1:idx+context_length+1]
    
    return x, y