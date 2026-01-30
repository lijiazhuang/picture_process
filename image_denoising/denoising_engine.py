__all__ = ['step_train', 'step_test']

import torch


def step_train(denoiser, train_loader, optimizer,loss, device):
    """

    :param denoiser:模型:降噪器
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param loss: 损失函数
    :param device: 设备
    :return: 当前伦茨的平均训练损失
    """
    #设置为训练模式
    denoiser.train()
    #累计损失
    total_loss=0.0
    #遍历loader，按批次训练模型
    for train_img,target_img in train_loader:
        #将数据移动到设备
        train_img=train_img.to(device)
        target_img=target_img.to(device)
        #前向传播
        output=denoiser(train_img)
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




def step_test(denoiser, test_loader, loss, device):
    #设置为测试模式
    denoiser.eval()
    #定义测试误差
    total_loss=0.0
    with torch.no_grad():
        for test_img,target_img in test_loader:
            #将数据转移到设备
            test_img=test_img.to(device)
            target_img=target_img.to(device)
            #前向传播
            output=denoiser(test_img)
            #计算损失
            loss_value=loss(output,target_img)
            #累计损失
            total_loss+=loss_value.item()
    return total_loss/len(test_loader)