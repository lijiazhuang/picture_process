__all__ = ['ImageModel']

import torch
import torch.nn as nn


# 定义模型
class ImageModel(nn.Module):
    """
    定义模型
    去除噪声原理为，通过卷积池化，来提取主要特征，此时非主要特征（噪声）就会被过滤掉
    维度压缩：通过pool层的最大池化操作，压缩空间维度的同时保留主要特征
    噪声抑制：在特征空间中，噪声成分被压缩或丢弃，保留图像的主要结构信息
    语义特征学习：通过conv1层将3通道扩展到32通道，使网络能够学习更丰富的图像特征表示
    多尺度特征：不同通道可以捕捉图像的不同特征模式，如边缘、纹理、颜色分布等
    噪声模式识别：增加的通道提供了更多的参数空间，让网络能够学习和区分噪声模式
    特征分离：通过扩展通道数，将噪声信息和有效图像信息在更高维的特征空间中分离
    层次化特征：conv2继续压缩到16通道，conv3进一步到8通道，形成特征金字塔
    关键信息保留：在通道数变化过程中保留最重要的图像信息，滤除噪声成分
    """
    def __init__(self):
        super().__init__()
        # 编码器
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 解码器
        self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
        # 墨瞳卷积层
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 编码
        x=torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        # 解码:
        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        x = torch.relu(self.t_conv3(x))
        x = torch.sigmoid(self.conv_out(x))  # 将结果限制到  0-1之间，Sigmoid激活函数可将数值映射到0-1之间
        return x
