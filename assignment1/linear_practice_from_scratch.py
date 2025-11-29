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
