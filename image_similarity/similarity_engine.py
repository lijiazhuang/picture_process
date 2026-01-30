__all__ = ['step_train', 'step_test','create_embedding']

import torch
#送一一个轮次的训练步骤
def step_train(encoder,deconder, train_loader, optimizer,loss, device):
    """

    :param encoder:编码器
    :param deconder:解码器
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param loss: 损失函数
    :param device: 设备
    :return: 当前伦茨的平均训练损失
    """
    #设置为训练模式
    encoder.train()
    deconder.train()
    #累计损失
    total_loss=0.0
    #遍历loader，按批次训练模型
    for train_img,target_img in train_loader:
        #将数据移动到设备
        train_img=train_img.to(device)
        target_img=target_img.to(device)
        #前向传播
        en_output=encoder(train_img)
        output=deconder(en_output)
        #计算损失
        loss_value=loss(output,target_img)
        #反向传播
        loss_value.backward()
        #优化参数
        optimizer.step()
        #梯度归零
        optimizer.zero_grad()
        #累计损失
        total_loss+=loss_value.item()
    return total_loss/len(train_loader)




def step_test(enconder,deconder, test_loader, loss, device):
    #设置为测试模式
    enconder.eval()
    deconder.eval()
    #定义测试误差
    total_loss=0.0
    with torch.no_grad():
        for test_img,target_img in test_loader:
            #将数据转移到设备
            test_img=test_img.to(device)
            target_img=target_img.to(device)
            #前向传播
            en_output=enconder(test_img)
            output=deconder(en_output)
            #计算损失
            loss_value=loss(output,target_img)
            #累计损失
            total_loss+=loss_value.item()
    return total_loss/len(test_loader)



def create_embedding(encoder,full_loader,device):
    """
    创建图片的嵌入向量
    :param encoder: 训练好的编码器
    :param full_loader: 完整的数据集加载器
    :param embeding_dim: 期望嵌入的维度
    :param device: 设备
    :return: 返回嵌入张量，形状:(N,C,H,W)-->(N,256,2,2)
    """
    encoder.eval()
    #定义嵌入张量，初始化为空，后续进行拼接
    embedings=torch.empty(0)
    with torch.no_grad():
        for train_img,target_img in full_loader:
            train_img=train_img.to(device)
            #前向传播
            encoded_img=encoder(train_img).cpu()
            #拼接这一批次的结果到嵌入张量中
            embedings=torch.cat((embedings,encoded_img),dim=0)
    return embedings
