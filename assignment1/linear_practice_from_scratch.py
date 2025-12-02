import torch # 引入PyTorch库，这是必须的
import torch.nn.functional as F #引入后面做基础计算工具用，这是允许的
from torch import Tensor # 引入张量类型，这是允许的
from typing import Iterable, Optional # 引入可迭代类型和可选类型，这是允许的
import math # 数学库，这是允许的


class Linear(torch.nn.Module): # 继承自torch.nn.Module
    '''
    定义Linear类，继承自torch.nn.Module
    所有PyTorch神经网络层都必须继承torch.nn.Module
    这样才能自动追踪参数、支持梯度计算等
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        '''
        构造函数
        in_features: 输入特征维度（输入向量的大小）
        out_features: 输出特征维度（输出向量的大小）
        bias: 是否使用偏置项，默认为True
        '''
        super().__init__()
        '''
        调用父类构造函数
        必须调用父类torch.nn.Module的初始化方法
        这样才能正确注册参数和子模块
        '''

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        '''
        创建权重矩阵
        torch.randn(): 从标准正态分布N(0,1)随机初始化
        形状：(out_features, in_features) - 注意是转置的，转置的意思是矩阵转置
        例如：输入128维，输出64维 → 权重形状是(64, 128)
        这样转置后才能与输入相乘，两者相乘的结果形状是(batch_size, out_features)
        -----例子---
        # 输入样本：32个样本，每个128维
        x = torch.randn(32, 128)  # (batch_size, in_features)

        # 权重矩阵：64个输出神经元，每个连接128个输入
        W = torch.randn(64, 128)  # (out_features, in_features)

        # 计算输出：
        y = x @ W.T  # (32, 128) @ (128, 64) = (32, 64)
        # 结果：32个样本，每个64维输出


        '''
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            '''
            如果需要偏置，创建偏置向量
            偏置的形状是(out_features,) - 一维向量
            每个输出神经元有一个偏置值
            '''
        else:
            self.register_parameter('bias', None)
            '''
            如果不使用偏置项，则注册一个None参数
            '''
        
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        '''
        Kaiming均匀分布初始化权重
        Kaiming初始化（何凯明提出）：解决深度网络梯度消失/爆炸问题
        uniform_: 使用均匀分布（下划线表示in-place操作）
        a=math.sqrt(5): 负斜率参数，用于计算初始化范围
        '''
        if self.bias is not None: # 如果有偏置，初始化偏置
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            '''
            计算fan_in（扇入）
            fan_in: 输入连接数 = in_features
            fan_out: 输出连接数 = out_features（这里用_忽略）
            用于计算合适的初始化范围
            '''

            bound = 1 / math.sqrt(fan_in)
            '''
            计算偏置初始化的边界
            PyTorch默认的偏置初始化策略
            确保初始值不会太大或太小
            '''

            torch.nn.init.uniform_(self.bias, -bound, bound) # 在[-bound, bound]范围内均匀采样
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        前向传播函数
        输入x的形状：(batch_size, in_features) 或 (*, in_features)
        输出形状：(batch_size, out_features) 或 (*, out_features)
        '''
        return F.linear(x, self.weight, self.bias)
        '''
        执行线性变换
        F.linear(x, weight, bias): 计算 y = xW^T + b
        等价于：x @ self.weight.T + self.bias
        PyTorch优化过的实现，比手动矩阵乘法更快
        禁止使用的 torch.nn.functional
        指的是直接使用functional中的函数来替代你需要从零实现的组件
        F.linear(x, weight, bias) 是你实现Linear层的数学计算工具，而不是替代整个Linear层
        '''

"""
Embedding类，继承自torch.nn.Module
所有PyTorch神经网络层都必须继承torch.nn.Module
这样才能自动追踪参数、支持梯度计算等
"""
# 词嵌入层，相当于把token转化为词向量，这个词向量会在训练中迭代，理解语义关系
class Embedding(torch.nn.Module):

    # num_embeddings: 词汇表大小（有多少个不同的词）
    # embedding_dim: 嵌入维度（每个词用多少维向量表示）
    def __init__(self, num_embedding:int, embedding_dim: int):
        super().__init__()

        '''
        创建嵌入权重矩阵
        形状：(词汇表大小, 嵌入维度)
        每一行对应一个词的向量表示
        torch.randn(): 标准正态分布初始化
        '''
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        torch.nn.init.normal_(self.weight, mean=0.0, std=embedding_dim ** -0.5) # mean=0.0 表示分布的中心在0点。权重的初始化，**-0.5是开根号的意思
    
    '''
    x: 输入的词索引，形状如(batch_size, sequence_length)
    '''
    def forward(self, x:Tensor) -> Tensor: # 前向传播
        return F.embedding(x, self.weight) # F.embedding(): PyTorch内置的嵌入查找函数,根据索引x从self.weight中查找对应的词向量

'''
补充小课堂：

**嵌入维度** = 每个词用多少维的向量来表示

```python
# 举例说明：
# 嵌入维度 = 128：每个词用128维向量表示
# 嵌入维度 = 512：每个词用512维向量表示
# 嵌入维度 = 768：每个词用768维向量表示（如BERT-base）
```
# 查找过程：
# 输入词索引：[5, 123, 45]  # 3个词的索引
# 输出形状：(3, 128)  # 3个128维向量
embedded = self.weight[[5, 123, 45]]  # 索引查找

# 不同模型的嵌入维度：
Word2Vec: 300维
BERT-base: 768维  
GPT-2: 768维
GPT-3: 12288维（12K）

**嵌入维度**决定了：
1. **表达能力**：维度越高，能捕获的语义信息越丰富
2. **参数数量**：权重矩阵大小 = 词汇表大小 × 嵌入维度
3. **计算成本**：维度越高，计算和存储成本越大
4. **模型性能**：需要在表达能力和效率之间平衡

就像照片的分辨率一样：分辨率越高细节越丰富，但也需要更多存储空间！
'''

'''
分词器：负责映射（文字 ↔ 数字索引）
嵌入层：负责转换（数字索引 → 语义向量）
两者配合：完成从文本到数学向量的完整转换
维度一致性：分词器词汇表大小 = 嵌入层输入维度
'''

"""
SwiGLU激活函数，结合了Glu和Swish激活函数的优点
Glu：Gating Mechanism + Linear Unit 
Swish：Self-gating activation function
"""
class SwiGLU(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()

