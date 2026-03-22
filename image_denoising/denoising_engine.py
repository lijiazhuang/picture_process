__all__ = ['step_train', 'step_test']

import torch
def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1, img2: 图像张量，范围[0, max_val]，形状 [B, C, H, W]
        max_val: 像素最大值（如果是0-1范围则max_val=1.0）

    Returns:
        psnr值 (dB)
    """
    # 确保在CPU上计算（避免GPU内存累积）
    img1 = img1.detach().cpu()
    img2 = img2.detach().cpu()

    # 计算MSE
    mse = torch.mean((img1 - img2) ** 2)

    # 避免除零
    if mse == 0:
        return float('inf')

    # 计算PSNR
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()


def calculate_batch_psnr(outputs, targets, max_val=1.0):
    """
    计算一批图像的PSNR（取平均）
    """
    batch_psnr = 0
    for i in range(outputs.size(0)):
        batch_psnr += calculate_psnr(outputs[i], targets[i], max_val)
    return batch_psnr / outputs.size(0)




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
    total_loss = 0
    total_psnr = 0
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
            batch_psnr = calculate_batch_psnr(output, target_img)
            total_psnr += batch_psnr


        avg_loss = total_loss / len(test_loader)
        avg_psnr = total_psnr / len(test_loader)

        return avg_loss, avg_psnr
