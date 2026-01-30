import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from similarity_data import ImageDataset
from similarity_model import *
from similarity_config import *
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from common import utils
from similarity_engine import *






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
    #定义全数据集加载器
    full_loader=DataLoader(dataset,batch_size=FULL_BATCH_SIZE)
    print("------------------数据加载器创建完成--------------------")
    #创建模型
    #定义模型，损失函数，优化器
    encoder=ConvEncoder()
    decoder=ConvDeconder()

    loss=nn.MSELoss()#均方误差损失函数
    #合并编码器和解码器的参数列表传给优化器
    autoencoder_params=list(encoder.parameters())+list(decoder.parameters())
    #参1接受的是模型的参数列表，因此俩个模型参数拼接成一个列表后可以一起优化
    optimizer=optim.AdamW(autoencoder_params,lr=LEARNING_RATE)

    encoder.to(device)
    decoder.to(device)

    #初始化最小误差
    min_test_loss=np.inf
    print("--------------------开始训练模型--------------------")
    for epoch in tqdm(range(EPOCHS)):
        #训练一轮
        train_loss=step_train(encoder,decoder,train_loader,optimizer,loss,device)
        print(f"\nEpoch:{epoch+1}/{EPOCHS}  Train Loss:{train_loss:.4f}")
        #进行测试，查看测试误差
        test_loss=step_test(encoder,decoder,test_loader,loss,device)
        print(f"\nEpoch:{epoch+1}/{EPOCHS}  Test Loss:{test_loss:.4f}")
        #进行判断，看当前测试误差，是否小于历史最小值，小于就保存模型参数，防止过拟合
        if test_loss<min_test_loss:
            print("测试误差已减小，保存模型参数!")
            min_test_loss=test_loss
            torch.save(encoder.state_dict(),ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(),DECODER_MODEL_NAME)
        else:
            print("测试误差未减小，不做保存!")
    print("训练结束！")


    #加载训练好的最优模型
    encoder.state_dict=torch.load(ENCODER_MODEL_NAME,map_location=device)
    encoder.load_state_dict(encoder._state_dict)

    #生成图像的嵌入表达
    embeddings=create_embedding(encoder,full_loader,device)
    #把张量转为numpy的ndarray进行保存
    #(N,C,H,W)->(N,C*H*W)
    vec_embeddings=embeddings.detach().numpy().reshape(embeddings.shape[0],-1)
    #保存嵌入向量到文件
    np.save(EMBEODING_NAME,vec_embeddings)