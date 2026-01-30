__all__=['step_train','step_test']




def step_train(model,loader,optimizer,loss,device):
    model.train()
    total_loss=0.0
    for images,labels in loader:
        images=images.to(device)
        labels=labels.to(device)
        #前向传播
        output=model(images)
        #计算损失
        loss_value=loss(output,labels)
        #反向传播+优化+梯度归零
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss+=loss_value.item()
    return total_loss/len(loader)

def step_test(model, test_loader, loss, device):
    model.eval()#设置为测试模式
    total_loss,correct_num=0.0,0
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        #前向传播
        output=model(images)
        y_pre=output.argmax(dim=1)#预测标签---计算预测值最大的对应项
        correct_num+=y_pre.eq(labels).sum()#预测正确的数量
        #计算损失
        loss_value=loss(output,labels)
        total_loss+=loss_value.item()
    return total_loss/len(test_loader),correct_num