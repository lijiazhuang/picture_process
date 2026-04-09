__all__ = ['ImageModel']

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class ImageModel(nn.Module):
    """
    定义模型
    """

    def __init__(self):
        super().__init__()

        # --- 编码器 (Encoder) ---
        # 每一层输出都会保存，准备给解码器做拼接
        self.enc1 = self._block(3, 32)  # Output: (Batch, 32, H, W)
        self.pool1 = nn.MaxPool2d(2)  # Output: (Batch, 32, H/2, W/2)

        self.enc2 = self._block(32, 64)  # Output: (Batch, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2)  # Output: (Batch, 64, H/4, W/4)

        # 最底层 (Bottleneck)
        self.bottleneck = self._block(64, 128)  # Output: (Batch, 128, H/4, W/4)

        # --- 解码器 (Decoder) ---
        # 第一层上采样：输入 128 (来自底层)，上采样后拼接 64 (来自 enc2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._block(128 + 64, 64)

        # 第二层上采样：输入 64，上采样后拼接 32 (来自 enc1)
        #scale_factor=2 表示上采样的倍数，mode='bilinear' 表示使用双线性插值，align_corners=False 表示不对齐角点
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._block(64 + 32, 32)

        # 最终输出层：恢复到 3 通道
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self._initialize_weights()

    def _block(self, in_channels, out_channels):
        """基础卷积块：Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):#判断是否是BatchNorm2d
                nn.init.constant_(m.weight, 1)# 初始化权重为1
                nn.init.constant_(m.bias, 0)# 初始化偏置为0

    def forward(self, x):
        # 保存原始输入用于最后的残差学习
        identity = x

        # Encoder 阶段
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder 阶段 (带拼接)
        # 1. 向上采样 b，并与 e2 拼接
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)  # 在通道维度拼接
        d1 = self.dec1(d1)

        # 2. 向上采样 d1，并与 e1 拼接
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        # 输出层：残差逻辑
        # 网络学习的是图像的修正量
        out = self.conv_out(d2)
        return torch.sigmoid(identity + out)




    # def __init__(self):
    #     super().__init__()
    #     # 编码器
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # 解码器
    #     self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2, padding=0, output_padding=0)
    #     self.t_conv2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=0, output_padding=0)
    #     self.t_conv3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
    #     # 墨瞳卷积层
    #     self.conv_out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    #
    # def forward(self, x):
    #     # 编码
    #     x=torch.relu(self.conv1(x))
    #     x = self.pool(x)
    #     x = torch.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = torch.relu(self.conv3(x))
    #     x = self.pool(x)
    #     # 解码:
    #     x = torch.relu(self.t_conv1(x))
    #     x = torch.relu(self.t_conv2(x))
    #     x = torch.relu(self.t_conv3(x))
    #     x = torch.sigmoid(self.conv_out(x))  # 将结果限制到  0-1之间，Sigmoid激活函数可将数值映射到0-1之间
    #     return x
