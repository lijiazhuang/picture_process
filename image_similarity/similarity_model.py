__all__=['ConvEncoder','ConvDeconder']

import torch
import torch.nn as nn


#定义编码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        #初始化父类
        super().__init__()
        #卷积层1
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        #卷积层2
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        #卷积层3
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        # 卷积层4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # 卷积层5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        #第一层操作
        x=torch.relu(self.conv1(x))
        # print(f"第一层卷积后的形状:{x.shape}")
        x=self.pool(x)
        # print(f"第一层池化后的形状:{x.shape}")
        #第二层操作
        x=torch.relu(self.conv2(x))
        # print(f"第二层卷积后的形状:{x.shape}")
        x=self.pool(x)
        # print(f"第二层池化后的形状:{x.shape}")
        #第三层操作
        x=torch.relu(self.conv3(x))
        # print(f"第三层卷积后的形状:{x.shape}")
        x=self.pool(x)
        # print(f"第三层池化后的形状:{x.shape}")
        #第四层操作
        x=torch.relu(self.conv4(x))
        # print(f"第四层卷积后的形状:{x.shape}")
        x=self.pool(x)
        # print(f"第四层池化后的形状:{x.shape}")
        #第五层操作
        x=torch.relu(self.conv5(x))
        # print(f"第五层卷积后的形状:{x.shape}")
        x=self.pool(x)
        # print(f"第五层池化后的形状:{x.shape}")
        return x


class ConvDeconder(nn.Module):
    def __init__(self):
        #初始化父类
        super().__init__()
        #转置卷积1
        self.t_conv1=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,padding=0,output_padding=0)
        #转置卷积2
        self.t_conv2=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0,output_padding=0)
        #转置卷积3
        self.t_conv3=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,padding=0,output_padding=0)
        #转置卷积4
        self.t_conv4=nn.ConvTranspose2d(32,16,kernel_size=2,stride=2,padding=0,output_padding=0)
        #转置卷积5
        self.t_conv5=nn.ConvTranspose2d(16,3,kernel_size=2,stride=2,padding=0,output_padding=0)

    def forward(self,x):
        #第一层操作
        x=torch.relu(self.t_conv1(x))
        # print(f"第一层转置卷积后的形状:{x.shape}")
        #第二层操作
        x=torch.relu(self.t_conv2(x))
        # print(f"第二层转置卷积后的形状:{x.shape}")
        #第三层操作
        x=torch.relu(self.t_conv3(x))
        # print(f"第三层转置卷积后的形状:{x.shape}")
        #第四层操作
        x=torch.relu(self.t_conv4(x))
        # print(f"第四层转置卷积后的形状:{x.shape}")
        #第五层操作
        x=torch.sigmoid(self.t_conv5(x))
        # print(f"第五层转置卷积后的形状:{x.shape}")
        return x

if __name__ == '__main__':
    #创建编码器
    encoder=ConvEncoder()
    #创建解码器
    decoder=ConvDeconder()
    #创建输入数据
    x=torch.randn(1,3,64,64)

    #前向传播
    x=encoder(x)
    x=decoder(x)