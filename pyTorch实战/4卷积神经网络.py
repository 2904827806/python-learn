#1.前提
#计算机视觉的发展趋势
#卷积神经网络 ：检测任务，分类与检索，医学任务，字体识别，标志识别，无人驾驶 ，超分辨率重构等等
#卷积神经网络：局部感知，参数共享，平移不变性
#卷积神经网络包括：卷积层，池化层，全连接层，激活函数，损失函数，优化器


#2.卷积神经网络与传统神经网络的区别
'''
#传统神经网络：全连接层，参数多，计算量大，训练时间长，容易过拟合
#输入层为像素点（1列特征）

#卷积神经网络：卷积层，池化层，参数少，计算量小，训练时间短，不容易过拟合
#输入层为原始图像（三维数据）
'''


import torch.nn
#3.卷积神经网络的整体架构
#输入层>卷积层>激活函数>池化层>全连接层>输出层

#(1)卷积层：卷积核，卷积核大小，步长，填充 （主要做特征提取）,卷积核的第三个维度是通道数，卷积核的通道数与输入图像的通道数相同，
# 当使用n个不同的卷积核时，卷积核的个数等于后面图像的通道数

'''
卷积做了什么？
#先将图像进行分割，
卷积核在输入图像上滑动，计算卷积核与输入图像的局部区域的内积，得到输出'特征图'的一个像素值。（也可以看作权重参数矩阵）
'''

#图像的颜色通道（RGB）：3
#卷积核的通道数（卷积核的深度）：3
#卷积核的大小（卷积核的宽和高）：3*3
#每一个颜色通道先单独做卷积，再相加

#卷积层涉及参数,卷积之后要进行激活函数，激活函数之后要进行池化层，两次卷积一次池化
#常用卷积核大小：3*3，5*5，步长：1（图像数据），2（文本数据），填充：0，1

#带参数计算的才有卷积层，不带参数计算的只有池化层，最后一个池化层和全连接层之间要进行拉长操作

'''
滑动窗口步长：就是指移动多少单元格，步长比较小，寻找更精细的特征，得到特征更丰富，速度较慢，步长比较大，寻找更粗糙的特征，得到特征较少。

卷积核尺寸：就是指选择区域的大小即卷积核的宽度和高度，卷积核尺寸比较小，寻找更精细的特征，得到特征更丰富，卷积核尺寸比较大，寻找更粗糙的特征，得到特征较少。

边缘填充：就是指在图像边缘填充多少像素，填充可以使得卷积后特征图的尺寸不变。当不进行边缘填充时，边界明显利用次数少于中间数据，这样会显得中间数据重要，而边界数据不重要

卷积核个数：就是指卷积核的个数，卷积核个数越多，特征图越多，特征越丰富，但参数量也越大，计算量也越大。

卷积参数共享：就是指同一个卷积核在图像上滑动时，卷积核的参数是共享的，即卷积核的参数是固定的，这样就可以大大减少参数的数量，降低计算量。
'''


#(2)激活函数：ReLU，Sigmoid，Tanh
'''
激活函数做了什么？
#将卷积层输出的特征图进行非线性变换，引入非线性因素，使得卷积神经网络可以逼近任意复杂的函数。
'''


#(3)池化层：最大池化，平均池化，池化核大小，步长，填充 （主要作用是：降维，防止过拟合，也就是压缩特征,也可以说是下采样）
'''
池化做了什么？（下采样）筛选工作
#将特征图进行下采样，减少特征图的尺寸，降低参数数量，降低计算量，防止过拟合。

最大池化(max pooling)：取特征图的最大值，保留特征图中的最显著特征。将重要的特征留下来，不重要的特征被丢弃。效果较好

平均池化(mean pooling)：取特征图的平均值，保留特征图中的平均特征。

池化核大小：就是指池化窗口的宽度和高度，池化核尺寸比较小，保留特征图中的精细特征，池化核尺寸比较大，保留特征图中的粗糙特征。

步长：就是指移动池化窗口的步长，步长比较小，保留特征图中的精细特征，步长比较大，保留特征图中的粗糙特征。

填充：就是指在特征图边缘填充多少像素，填充可以使得池化后特征图的尺寸不变。当不进行边缘填充时，边界明显利用次数少于中间数据，这样会显得中间数据重要，而边界数据不重要。

'''


#(4)全连接层：全连接层，激活函数，损失函数，优化器

'''
全连接层做了什么？
将特征图进行分类，输出预测结果。
'''



#比较经典的卷积神经网络：
# LeNet，卷积神经网络，是最早的卷积神经网络之一，用于手写数字识别
# AlexNet，深度卷积神经网络，首次在ImageNet竞赛中取得好成绩
# VGGNet，深度卷积神经网络，通过增加卷积层的深度来提高性能
# GoogLeNet，深度卷积神经网络，通过增加Inception模块来提高性能
# ResNet，残差网络，解决了梯度消失和梯度爆炸问题（加上同等映射）当做特征提取网络,当有一层做的不好时，同等映射可以将其映射到下一层，这样下一层就可以学习到更好的特征
# DenseNet，密集连接网络，通过密集连接来提高性能
# SENet，Squeeze-and-Excitation网络，通过自适应调整通道的权重来提高性能
# EfficientNet，高效网络，通过复合缩放来提高性能
# Transformer，自注意力机制，用于自然语言处理和计算机视觉任务
# YOLO，You Only Look Once，实时目标检测算法
# Faster R-CNN，快速区域卷积神经网络，用于目标检测
# Mask R-CNN，在Faster R-CNN基础上增加了对实例分割的支持
# SSD，单阶段检测器，用于目标检测
# RetinaNet，Focal Loss，用于目标检测
# CenterNet，中心点检测，用于目标检测
# DETR，Transformer，用于目标检测
# ViT，Vision Transformer，用于计算机视觉任务
# MAE，Masked Autoencoders，用于图像生成
# CLIP，对比学习，用于图像和文本的匹配
# GPT，生成预训练变换器，用于自然语言处理任务
# BERT，双向编码器表示，用于自然语言处理任务
# T5，文本到文本的预训练模型，用于自然语言处理任务
# GPT-，生成预训练变换器的变体，用于自然语言处理任务


#感受域：
# 卷积核的大小决定了感受域的大小，感受域越大，
# 卷积核能看到的图像区域越大，能提取的特征越丰富，但计算量也越大。
# 感受域越小，卷积核能看到的图像区域越小，能提取的特征越粗糙，但计算量也越小。



#构建简单的卷积神经网络
import torch.nn as nn
import torch
import torch.optim as optim #创建优化器
import torch.nn.functional as F
import matplotlib as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
# 导入gzip模块，用于压缩和解压缩文件
import gzip
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

#1.读取数据

#定义超参数
input_size = 28 #图像的总尺寸为28*28
num_classes = 10 #10个类别，0-9
num_epochs = 3 #训练的轮数
batch_size = 64 #每个batch的大小,64张图片


#训练集 手写数据集
with gzip.open(r"C:\Users\29048\Desktop\mnist\mnist.pkl.gz", 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

#将数据转换为torch.tensor类型
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

#2.构建batch数据
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_valid, y_valid)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


#3.构建卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
            #定义网络结构
        #顺序性：nn.Sequential 中的模块是按照顺序执行的，因此需要注意模块的顺序，确保数据流能够正确传递。

        #nn.Conv2d 是 PyTorch 深度学习框架中的一个类，用于定义二维卷积层。

        # 卷积层是卷积神经网络（CNN）的核心组成部分，用于提取输入图像的局部特征。

        # 定义第一个卷积层，输入通道数为1，输出通道数为16，卷积核大小为5，步长为1，填充为2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, #（灰度图 ）#输入特征图个数
                                             out_channels=16, #输出特征图个数
                                             kernel_size=5, #卷积核大小
                                             stride=1, #步长
                                             padding=2,#边缘填充层数，如果希望卷积后大小跟原来一样，需要设置padding=（kernel_size-1)/2if stride = 1
                                             ),#输出特征图为（16，28，28）
                                    # 定义激活函数，使用ReLU
                                    nn.ReLU(),
                                    # 定义池化层，池化核大小为2，输出结果为（16，14，14）
                                    nn.MaxPool2d(kernel_size=2)) #最大池化

        # 定义第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为5，步长为1，填充为2

        #一般卷积层，relu层，池化层可以写成一个套餐
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=5,
                                             stride=1,
                                             padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))#输出（32，7，7）

        #全连接层
        self.out = nn.Linear(32*7*7, 10) #全连接层，输入大小为32*7*7，输出大小为10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #将数据展平 filatten操作，结果为（batch_size, 32*7*7）
        output = self.out(x)
        return output


#4.准确率作为评价指标
def accuracy(predictions, labels):
    '''
    :param predictions: 预测值
    :param labels: 标签
    '''
    pred = torch.max(predictions.data, 1)[1] #返回每一行中最大值的那个元素，且返回其索引
    rights = pred.eq(labels.data.view_as(pred)).sum() #返回张量的数目
    return rights, len(labels)

#5.训练模型
net = CNN()
devier = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(devier)
#损失函数
criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数
#优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)
#训练
for epoch in range(num_epochs):
    tain_right = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1,1,28,28) #将图片展开为二维数据
        images = images.to(devier)
        labels = labels.view(-1)
        labels = labels.to(devier)
        #print(images.shape)
        #前向传播 预测
        outputs = net(images)

        #计算损失
        loss = criterion(outputs, labels)

        #梯度清零
        optimizer.zero_grad()

        #反向传播
        loss.backward()

        #更新参数
        optimizer.step()

        right = accuracy(outputs, labels)
        tain_right.append(right)

        #每100个batch输出一次训练集准确率
        if (i+1) % 100 == 0:
            net.eval()
            val_right = []

            #遍历测试集
            for j, (images1, labels1) in enumerate(test_loader):
                images1 = images1.view(-1,1, 28, 28)
                images1 = images1.to(devier)
                labels1 = labels1.view(-1)
                labels1 =labels1.to(devier)

                #前向传播 预测
                outputs1 = net(images1)
                #计算损失
                right1 = accuracy(outputs1, labels1)
                val_right.append(right1)

            #准确率计算

            train_accuracy = sum([tup[0] for tup in tain_right]) / sum([tup[1] for tup in tain_right])
            val_accuracy = sum([tup[0] for tup in val_right]) / sum([tup[1] for tup in val_right])

            print('当前epoch: {}, [{}/{} ({:.0f}%]\t损失：{:.6f}\t训练集准确率：{:.2f}%\t测试集正确率:{:.2f}'.format(
                epoch, j * batch_size,len(train_loader.dataset),
                100. * j / len(train_loader),
                loss.item(),
                100 * train_accuracy,
                100 * val_accuracy))































