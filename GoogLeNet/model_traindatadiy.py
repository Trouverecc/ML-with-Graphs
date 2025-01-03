#针对自建数据集
import copy
import time
import torch
#from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import GoogLeNet,Inception #
import torch.nn as nn
import pandas as pd

#初始化定义

def train_val_data_process():
    #定义数据集路径
    ROOT_TRAIN = r'data\train'#绝对路径也可
    #归一化
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])#3通道
    #定义数据集处理方法的变量
    train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])#->tensor\0-1
    #加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)


    train_data,val_data= Data.random_split(train_data, [round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_dataloader= Data.DataLoader(dataset=train_data,
                                    batch_size=12,
                                    shuffle=True,
                                    num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=12,
                                     shuffle=True,
                                     num_workers=2)
    return train_dataloader, val_dataloader

train_val_data_process()
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Adam优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)#方便后续模型更新，学习率0.001,类梯度下降
    #损失函数为交叉熵函数
    criterion=nn.CrossEntropyLoss()
    #将模型放到训练设备中
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高精确度
    best_acc=0.0
    #训练集loss列表
    train_loss_all=[]
    #验证集loss列表
    val_loss_all = []
    # 训练集精度列表
    train_acc_all = []
    # 验证集精度列表
    val_acc_all = []

    since = time.time()
    #初始化OK，接下来训练模型，迭代过程，更新参数（反向传播）
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))#轮次从0开始，所以-1,0-99,打印训练过程
        print("-"*10)

        #初始化loss值
        train_loss=0.0
        #训练集准确度
        train_corrects =0

        #初始化验证集
        val_loss = 0.0
        #验证集准确度
        val_corrects = 0

        #训练集样本数量
        train_num =0
        #验证集样本数量
        val_num=0

        #对每一个mini-batch训练和计算
        for step, (b_x,b_y) in enumerate(train_dataloader):   #train_dataloader:所有数据，b_x128*28*28*1，b_y128*label
            #将标签放到训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为训练模式
            model.train()


            #前向传播过程，输入为一个batch，输出为一个batch对应的预测
            output=model(b_x)#前向传播，得结果
            #查找每一行最大值对应的行标
            pre_lab=torch.argmax(output,dim=1)

            loss = criterion(output,b_y)

            #将梯度值为0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #根据网络反向传播的梯度信息来更新网络参数，以降低函数loss值
            optimizer.step()

            #计算loss值
            train_loss += loss.item() * b_x.size(0)
            #若预测正确，准确度+1
            train_corrects+=torch.sum(pre_lab==b_y.data)


            train_num+=b_x.size(0)
        #验证过程
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将标签放到验证设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入为一个batch，输出为一个batch中的预测
            output=model(b_x)
            # 查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            # 计算loss值
            val_loss += loss.item() * b_x.size(0)
            # 若预测正确，准确度+1
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        #计算并保存每一次迭代的loss，准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        #训练over
        #最优模型保存
        #寻找最高准确度(权重）
        if val_acc_all[-1]>best_acc:
            #保存当前最高准确度
            best_acc=val_acc_all[-1]
            # 保存当前最高准确度模型参数
            best_model_wts=copy.deepcopy(model.state_dict())
        #计算训练耗时
        time_use=time.time()-since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))

    #选择最优参数
    #加载最高准确率下的模型参数

    #改路径
    torch.save(best_model_wts, 'D:/pythonProject/Pytorch-try/GoogLeNet/best_model.pth')
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all
                                       })
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)#一行两列的第一张图
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label = "train loss")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label = "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__=="__main__":
    #将模型实例化
    #改
    GoogLeNet = GoogLeNet(Inception)
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader,20)#20轮
    matplot_acc_loss(train_process)