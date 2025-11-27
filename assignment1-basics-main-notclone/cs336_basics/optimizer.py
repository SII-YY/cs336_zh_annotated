import torch
import math
from typing import Dict, Any

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩估计
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # 二阶矩估计
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                state["step"] += 1
                
                # 应用权重衰减
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                
                # 更新一阶矩和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算偏差校正后的估计
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group["lr"]
                
                # 更新参数
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

def get_adamw_cls():
    """
    返回AdamW优化器类
    """
    return AdamW

def get_lr_cosine_schedule(
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
    current_step: int
) -> float:
    """
    实现余弦学习率调度器
    """
    if current_step < warmup_steps:
        # 预热阶段：线性增长
        lr = max_lr * current_step / warmup_steps
    elif current_step >= max_steps:
        # 结束阶段：保持最小学习率
        lr = min_lr
    else:
        # 余弦衰减阶段
        progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    return lr

def save_checkpoint(
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Dict[str, Any],
    epoch: int,
    filepath: str
) -> None:
    """
    保存模型和优化器的状态到检查点文件
    """
    torch.save({
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": epoch
    }, filepath)

def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    从检查点文件加载模型和优化器的状态
    """
    return torch.load(filepath)