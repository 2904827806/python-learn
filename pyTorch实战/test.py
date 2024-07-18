import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

#1.基本使用方法
#1.1 创建一个5行3列的矩阵
a = torch.empty(5, 3)
#print(a)

#1.2展示矩阵大小
#print(a.size())

#1.3初始化一个全零点矩阵
x = torch.zeros(5, 3, dtype=torch.long)# 创建一个5行3列的全0张量，数据类型为torch.long
#print('全0矩阵',x)

#1.4初始化一个矩阵，矩阵中的元素按照均匀分布随机取值
y = torch.rand(5, 3)
#print('随机矩阵',y)

#1.5根据已有的数据初始化一个矩阵
b = torch.tensor([5.5, 3])
#print(b)

#1.6从已有的tensor中创建一个新的tensor
c = b.new_ones(5, 3, dtype=torch.double)
#print(c)


#索引
#print(c[:, 1])

#1.7直接从数据中创建一个tensor，并且不会复制数据
d = torch.from_numpy(np.ones(5))
#print('d',d)

#1.8 tensor和numpy之间的转换
e = d.numpy()
#print(e)

#矩阵加法
#print(x+y)


#view 操作可以改变数据的形状，但是不能改变数据的总数
x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8) # -1表示自动计算该维度大小

#print(x.size(), y.size(), z.size())


#与numpy的互操作性
# 创建一个包含5个元素的矩阵，每个元素的值都为1
a = torch.ones(5)
#print('4',a)
# 将矩阵a转换为numpy数组
b = a.numpy()
#print(b)
#print(a, b, sep='\n')
# 修改a的值，b的值也会随之改变，因为a和b共享内存
c = np.ones(5)
d = torch.from_numpy(c)
e = d.view(-1, 1)

'''二、常见的tensor格式'''

'''
# 标量
scalar :通常是一个数值
# 向量 
vector ： 通常是一维数组;在深度学习中通常指特征，例如词向量特征，某一维度特征等。
# 矩阵
matrix ： 通常是一个二维数组，矩阵
# n维张量 
n-dimensional tensor ：通常是一个n维数组
'''

'''import torch
from torch import tensor
# 创建一个标量
x1 = tensor(42.)
print(x1)

# 创建一个向量
x2 = tensor([1,2,3])
print(x2)

#创建一个矩阵
x3 = tensor([[1,2,3],[4,5,6]])
print(x3)

# 创建一个四维张量
x4 = tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(x4)'''















