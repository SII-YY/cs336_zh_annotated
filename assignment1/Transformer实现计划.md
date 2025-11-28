# Transformer语言模型实现计划

## 项目概述

根据CS336作业1要求，我们需要从零开始实现一个完整的Transformer语言模型。本项目遵循严格的"从零实现"原则，不使用PyTorch的高级模块，仅使用基础张量操作和允许的组件。

## 核心任务

1. **实现 Byte-pair encoding (BPE) 分词器**
2. **实现 Transformer 语言模型**
3. **实现交叉熵损失函数和 AdamW 优化器**
4. **实现训练循环，支持模型和优化器状态的保存与加载**

## 实验任务

1. 在 TinyStories 数据集上训练 BPE 分词器
2. 使用训练好的分词器将数据集转换为整数 ID 序列
3. 在 TinyStories 数据集上训练 Transformer 语言模型
4. 使用训练好的模型生成文本样本并评估困惑度
5. 在 OpenWebText 数据集上训练模型，并将困惑度提交到排行榜

## 编程限制

### 禁止使用
- `torch.nn` 中的绝大部分模块
- `torch.nn.functional`
- `torch.optim`（除了 `Optimizer` 基类）

### 允许使用
- `torch.nn.Parameter`
- PyTorch 的容器类，如 `Module`, `ModuleList`, `Sequential` 等
- `torch.optim.Optimizer`（作为优化器的基类）
- 其他未被明确禁止的 PyTorch 定义

## 代码结构

- **主代码区**：`cs336_basics/*` 目录，所有从零编写的代码
- **适配器文件**：`adapters.py`，填充函数实现，调用自定义代码
- **测试文件**：`test.*.py`，通过 `adapters.py` 测试实现

## 实现计划

### 第一阶段：基础组件实现（高优先级）

#### 1. 修复nn_utils.py中的问题
- [ ] 确保Linear层正确实现
- [ ] 确保Embedding层正确实现
- [ ] 修复SwiGLU激活函数实现
- [ ] 确保RMSNorm归一化正确实现
- [ ] 修复多头自注意力实现
- [ ] 修复RoPE位置编码实现
- [ ] 确保softmax和交叉熵损失函数正确实现

#### 2. 完善transformer.py中的实现
- [ ] 修复TransformerBlock的实现，确保Pre-norm结构正确
- [ ] 完善TransformerLM，确保权重绑定和前向传播正确
- [ ] 确保与nn_utils.py中组件的集成

#### 3. 修复model.py中的函数实现
- [ ] 确保所有函数与适配器接口一致
- [ ] 修复RoPE实现，确保位置编码正确应用
- [ ] 确保函数式实现与面向对象实现的一致性

### 第二阶段：训练工具实现（中优先级）

#### 4. 完善train_utils.py中的训练工具
- [ ] 确保AdamW优化器正确实现
- [ ] 完善学习率调度函数
- [ ] 确保检查点保存和加载功能正常
- [ ] 实现梯度裁剪功能

#### 5. 实现文本生成功能
- [ ] 实现支持温度调节的采样方法
- [ ] 实现Top-p（nucleus）采样
- [ ] 创建文本生成接口
- [ ] 实现批量文本生成

### 第三阶段：测试与优化（中低优先级）

#### 6. 在TinyStories数据集上测试模型
- [ ] 准备TinyStories数据集
- [ ] 确保模型能正常训练
- [ ] 验证损失函数收敛
- [ ] 测试文本生成效果
- [ ] 计算困惑度

#### 7. 性能分析与优化
- [ ] 分析代码瓶颈
- [ ] 优化训练速度
- [ ] 确保在Apple Silicon或CPU上高效运行
- [ ] 内存使用优化

## 技术要点

### 模型架构
- **Transformer块结构**：使用Pre-norm结构（先归一化再进入子层）
- **归一化**：实现RMSNorm（均方根归一化）
- **激活函数**：使用SwiGLU（Swish门控线性单元）
- **位置编码**：应用RoPE（旋转位置编码）
- **注意力机制**：实现因果自注意力（掩码确保只看到前面的token）

### 训练策略
- **优化器**：使用AdamW（带权重衰减的Adam）
- **学习率调度**：实现余弦退火调度
- **梯度处理**：应用梯度裁剪防止梯度爆炸
- **检查点**：支持训练状态的保存和恢复

### 文本生成
- **采样方法**：实现温度调节和Top-p采样
- **解码策略**：支持贪心解码和采样解码
- **连贯性**：确保生成文本的连贯性和多样性

## 实现细节

### nn_utils.py 关键函数

#### 线性层
```python
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, x: Tensor) -> Tensor
```

#### 嵌入层
```python
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int)
    def forward(self, x: Tensor) -> Tensor
```

#### SwiGLU激活函数
```python
class SwiGLU(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None)
    def forward(self, x: Tensor) -> Tensor
```

#### RMSNorm归一化
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5)
    def forward(self, x: Tensor) -> Tensor
```

#### 多头自注意力
```python
class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, rope_theta: float = 10000.0)
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor
```

### transformer.py 关键类

#### Transformer块
```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0)
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor
```

#### Transformer语言模型
```python
class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int = 4096)
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor
```

### train_utils.py 关键函数

#### AdamW优化器
```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01, correct_bias: bool = True)
    def step(self, closure=None)
```

#### 学习率调度
```python
def get_lr_cosine_schedule(optimizer: torch.optim.Optimizer, max_lr: float, min_lr: float,
                          warmup_steps: int, total_steps: int) -> List[float]
```

#### 检查点管理
```python
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None
def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int
```

## 测试策略

1. **单元测试**：确保每个组件单独工作正常
2. **集成测试**：验证组件组合后的正确性
3. **端到端测试**：在简单数据集上验证完整流程
4. **性能测试**：在目标数据集上验证训练和生成效果

## 预期成果

1. **功能完整的BPE分词器**：能够训练词汇表、对文本进行编码和解码
2. **结构正确的Transformer语言模型**：包含所有必要组件
3. **完整的训练工具**：支持模型训练和评估
4. **文本生成器**：能够生成连贯的文本
5. **实验报告**：包含训练过程、超参数调优和模型性能分析

## 时间规划

- **第一周**：完成基础组件实现和测试
- **第二周**：完成Transformer模型和训练工具
- **第三周**：进行实验和性能优化
- **第四周**：完成实验报告和排行榜提交

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始Transformer论文
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - 旋转位置编码论文
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU相关论文
- [AdamW: Decoupled Weight Decay](https://arxiv.org/abs/1711.05101) - AdamW优化器论文