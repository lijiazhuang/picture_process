__all__ = ['ImageModel']

import torch
import torch.nn as nn


# 定义模型
class ImageModel(nn.Module):
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
