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






if __name__ == "__main__":
    #检测GPU是都可用并定义设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #指定随机种子，去除训练的不确定性
    utils.seed_everything(SEED)
    #定义图像预处理操作
    transform=transforms.Compose([
        transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
        transforms.ToTensor()
    ])
    #创建数据集
    print("--------------------创建数据集--------------------")
    dataset=ImageDataset(IMG_PATH,transform)
    #划分数据集
    train_dataset,test_dataset=random_split(dataset,[TRAIN_RATIO,TEST_RATIO])
    print("------------------数据集创建完成--------------------")
    #创建数据加载器
    print("--------------------创建数据加载器--------------------")
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE)
    print("------------------数据加载器创建完成--------------------")
    #创建模型
    #定义模型，损失函数，优化器
    model=ImageModel()
    loss=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)

    model.to(device)

    #初始化最小误差
    min_test_loss=np.inf
    print("--------------------开始训练模型--------------------")
    for epoch in tqdm(range(EPOCHS)):
        #训练一轮
        train_loss=step_train(model,train_loader,optimizer,loss,device)
        print(f"Epoch:{epoch+1}/{EPOCHS}  Train Loss:{train_loss:.4f}")
        #进行测试，查看测试误差
        test_loss=step_test(model,test_loader,loss,device)
        print(f"Epoch:{epoch+1}/{EPOCHS}  Test Loss:{test_loss:.4f}")
        #进行判断，看当前测试误差，是否小于历史最小值，小于就保存模型参数，防止过拟合
        if test_loss<min_test_loss:
            print("测试误差已减小，保存模型参数!")
            min_test_loss=test_loss
            torch.save(model.state_dict(),DENOiSER_MODEL_NAME)
        else:
            print("测试误差未减小，不做保存!")
    print("训练结束！")


