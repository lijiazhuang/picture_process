import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from denoising_data import ImageDataset
from denoising_model import ImageModel
from denoising_config import *
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from common import utils
from denoising_engine import *

def test(model,test_loader,device):
    model.eval()
    with torch.no_grad():
        loader_iter = iter(test_loader)
        x, y = next(loader_iter)
        model = model.to(device)
        x = x.to(device)
        output = model(x)
        x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
        output = output.permute(0, 2, 3, 1).cpu().detach().numpy()
        y = y.permute(0, 2, 3, 1).cpu().detach().numpy()
        fig, axes=plt.subplots(3,10,figsize=(25,4))
        for imgs, row in zip([x, output, y], axes):
            for img, ax in zip(imgs, row):
                ax.imshow(img)
                ax.axis('off')
        plt.show()

if __name__ == "__main__":
    # 检测GPU是都可用并定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 指定随机种子，去除训练的不确定性
    utils.seed_everything(SEED)
    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])
    # 创建数据集
    print("--------------------创建数据集--------------------")
    dataset = ImageDataset(IMG_PATH, transform)
    # 划分数据集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])
    print("------------------数据集创建完成--------------------")
    # 创建数据加载器
    print("--------------------创建数据加载器--------------------")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("------------------数据加载器创建完成--------------------")
    # 创建模型
    # 定义模型，损失函数，优化器
    model = ImageModel()
    model.load_state_dict(torch.load(DENOiSER_MODEL_NAME))
    model.to(device)
    test(model,test_loader,device)
