import requests
from pathlib import Path
import gzip
import pickle
import matplotlib.pyplot as plt


# 加载数据集(解压）
with gzip.open(r"C:\Users\29048\Desktop\mnist\mnist.pkl.gz", 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

# 打印数据集形状
#print(x_train.shape)

# 显示示例图片
plt.imshow(x_train[5].reshape((28, 28)), cmap='gray')
#plt.show()


#构建网络架构

import torch
# 将x_train, y_train, x_valid, y_valid转换为torch.tensor类型
x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

# 定义网络架构
# 导入PyTorch的神经网络功能模块
#如果模型有可学习的参数，则必须使用torch.nn.Module的子类来定义模型。否则使用torch.nn.functional中的函数即可
import torch.nn.functional as F

loss_func = F.cross_entropy #损失函数；分类问题一般用交叉熵损失函数

'''bs = 64 #批量大小
xb = x_train[0:bs] #取前64个样本作为批量
yb = y_train[0:bs]
# 定义模型
weights = torch.randn([784, 10],dtype=torch.float, requires_grad=True) #随机初始化权重
bias = torch.zeros([10],dtype=torch.float, requires_grad=True) #初始化偏置
def mode(xb):
    #前向传播 预测值
    return xb.mm(weights) + bias

#print(loss_func(mode(xb), yb))'''


#创建一个更简单的模型
from torch import nn
#构建模型
class Mnist_NN(nn.Module):
    #继承nn.Module类
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128) #隐层1
        self.hidden2 = nn.Linear(128, 256) #隐层2
        self.out = nn.Linear(256, 10) #输出层

    def forward(self, xb):#前向传播

        x = F.relu(self.hidden1(xb)) #将输入数据通过隐层1，并使用ReLU激活函数
        x = F.relu(self.hidden2(x)) #将隐层1的输出通过隐层2，并使用ReLU激活函数
        x = self.out(x) #将隐层2的输出通过输出层
        return x #返回输出结果


'''net = Mnist_NN()
print(net)
optimizer = torch.optim.SGD(net.parameters()),# 获取模型的所有参数
closs = torch.nn.MSELoss(reduction='mean')
for name,param in net.named_parameters():
        print(name, param,param.size())'''


#使用tensorDataset :将测试数据或者验证数据集的x和y组合在一起，方便后续使用
#DataLoader ：将数据集转换为迭代器，方便后续使用
from torch.utils.data import TensorDataset, DataLoader

bs = 64 #批量大小
# 创建训练数据集
train_ds = TensorDataset(x_train, y_train)
# 创建训练数据加载器，设置批量大小为bs，并打乱数据
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# 创建验证集数据集
valid_ds = TensorDataset(x_valid, y_valid)
# 创建验证集数据加载器，batch_size为bs*2，验证集不需要打乱数据
valid_dl = DataLoader(valid_ds, batch_size=bs*2)


#1.定义训练函数
def get_data(train_ds, valid_ds, bs):
    return(
        # 创建训练数据集的DataLoader，设置batch_size为bs，并设置shuffle为True，即每次迭代时都会打乱数据集
        DataLoader(train_ds,batch_size=bs,shuffle=True),

        # 创建验证数据集的DataLoader，设置batch_size为bs*2
        DataLoader(valid_ds,batch_size=bs*2)
    )

#一般在训练模型上加上model.train()，这样会正常使用Batch Normalization和Dropout
# 测试的时候一般选择model.eval()，这样就不会使用Batch Normalization和Dropout


import numpy as np
#2. 训练函数得到平均损失
def fit(steps,model,loss_func, opt, train_dl, valid_dl):
    '''
    :param steps:  迭代次数
    :param model:   模型
    :param loss_func:   损失函数
    :param opt: 优化器
    :param train_dl: 训练集
    :param valid_dl:  验证集
    :return:
    '''
    #训练模型
    for step in range(steps):
        #训练模型
        model.train()
        for xb,yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        #验证模型
        # 将模型设置为评估模式
        model.eval() #模型评估
        with torch.no_grad():# 禁用梯度计算，用于推理阶段，以节省内存和计算资源
            #计算验证集损失
            # 计算验证集上的损失和样本数量
            losses, nums = zip(*[(loss_func(model(xb), yb).mean().item(),1) for xb,yb in valid_dl])
            '''# 计算验证集上的损失和样本数量
            # 使用zip函数将损失和样本数量打包成元组
            # 使用列表推导式遍历验证集数据
            # 对每个样本计算损失，并取平均值和item值
            losses, nums = zip(*[loss_func(model(xb), yb).mean().item() for xb,yb in valid_dl])'''
        #计算总损失
        total_loss = np.sum(np.multiply(losses, nums))
        #计算平均损失
        avg_loss = total_loss / np.sum(nums)
        print('当前step：'+str(step)+'，验证集平均损失：'+str(avg_loss))

#3.获取模型和优化器
from torch import optim
def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)
    # 返回模型和优化器

# 定义一个简单的神经网络模型
#4.计算损失并进行反向传播和参数更新
def loss_batch(model, loss_func, xb, yb, opt=None):
    # 计算损失
    loss = loss_func(model(xb), yb)

    # 如果有优化器，则进行反向传播和参数更新
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # 返回损失和样本数量
    return loss.item(), len(xb)

import torch.nn.functional as F
loss_func = F.cross_entropy#损失函数；分类问题一般用交叉熵损失函数
train_dl1,valid_dl1 = get_data(train_ds,valid_ds, bs=64)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl1, valid_dl1)
