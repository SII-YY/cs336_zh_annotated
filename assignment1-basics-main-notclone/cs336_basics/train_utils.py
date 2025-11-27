import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import os
import math

class AdamW(torch.optim.Optimizer):
    """
    AdamW优化器实现
    """
    def __init__(self,
                 params,
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 correct_bias: bool = True):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 权重衰减（L2正则化）
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 计算一阶和二阶矩
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr']
                
                # 参数更新
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

def get_lr_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int
) -> List[float]:
    """
    余弦学习率调度实现
    """
    lrs = []
    
    for step in range(total_steps):
        if step < warmup_steps:
            # 预热阶段：线性增加
            lr = max_lr * (step + 1) / warmup_steps
        else:
            # 余弦衰减阶段
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    
    return lrs

def get_batch(data: List[List[int]], batch_size: int, block_size: int) -> Tuple[Tensor, Tensor]:
    """
    从数据中获取批次数据
    """
    # 随机选择起始位置
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # 获取输入和目标
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    
    return x, y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    """
    保存模型检查点
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(checkpoint, path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    加载模型检查点
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def get_adamw_cls():
    """
    返回AdamW优化器类
    """
    return AdamW