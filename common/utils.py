import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import random


# 定义函数 对所有模块使用统一的随机种子
def seed_everything(seed):
    random.seed(seed)  # python内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # python哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed)  # 针对cpu上的随机种子
    torch.cuda.manual_seed(seed)  # 针对CPU的随即种子
    """
    科学研究：确保实验结果可被他人复现----禁用CuDNN中的非确定性算法
    模型调试：排查问题时消除随机性干扰
    超参数调优：公平比较不同超参数的效果
    论文实验：保证报告的结果稳定可靠
    """
    torch.backends.cudnn.deterministic = True  # 确保CuDNN的确定性
    # 大幅提升训练速度----但会带来一点点不确定性
    torch.backends.cudnn.benchmark = False  # 启用CuDNN的性能优化
