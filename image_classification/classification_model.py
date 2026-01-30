__all__ = ['ClassificationModel']

import torch
import torch.nn as nn
class ClassificationModel(nn.Module):
    def __init__(self,n_classes=5):
        super().__init__()
        self.conv1=nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear=nn.Linear(4096,n_classes)
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=self.pool(x)
        # print(f"第二层输出：{x.shape}")
        x=torch.relu(self.conv2(x))
        # print(f"第三层输出：{x.shape}")
        x=self.pool(x)
        # print(f"第四层输出：{x.shape}")
        x=x.reshape(x.shape[0],-1)
        # print(f"第五层输入：{x.shape}")
        x=self.linear(x)
        # print(f"第六层输出：{x.shape}")
        return x

#测试模型
if __name__ == '__main__':
    x=torch.randn(1,3,64,64)
    model=ClassificationModel()
    print(model(x).shape)