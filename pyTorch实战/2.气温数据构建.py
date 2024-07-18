#hub模块，作用是方便我们下载预训练模型，方便我们使用预训练模型

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim



features = pd.read_csv(r"C:\Users\29048\Desktop\temps.csv")

#打印前五行数据
#print(features.head())

#print('wd',features.shape)

#处理时间数据
import datetime

#分别得到年月日
years = features['year']
months = features['month']
days = features['day']

#将年月日转换为时间格式
dates = [str(year)+'-'+str(month)+'-'+str(day) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

#year  month  day  week  temp_2  temp_1  average  actual  friend

#准备画图
#指定默认风格
plt.style.use('fivethirtyeight')

#设置画布大小
plt.figure(figsize=(15,12))

plt.subplot(221)
plt.plot(dates,features['actual'],label='Actual')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual Max Temp')
plt.legend()

plt.subplot(222)
plt.plot(dates,features['temp_1'],label='temp_1')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature (temp_1)')
plt.legend()

plt.subplot(223)
plt.plot(dates,features['temp_2'],label='temp_2')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature (temp_2)')
plt.legend()

plt.subplot(224)
plt.plot(dates,features['friend'],label='Friend')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature (Friend)')
plt.legend()

plt.show()


#数据编码pd.get_dummies() #将分类变量转换为虚拟变量
# 将features中的分类变量转换为虚拟变量
features = pd.get_dummies(features)
#print(features.head())

#标签
labels = np.array(features['actual'])

##删除标签列
features = features.drop('actual',axis=1)#按列进行操作
#print(features.shape)
#名单单独保存，以备后患
feature_list = list(features.columns) #x的标签

#print(len(feature_list))

#转换为numpy格式
features = np.array(features)
#print(features)


#数据标准化
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing #数据预处理模块
input_features = preprocessing.StandardScaler().fit_transform(features)#标准化
#print(input_features[0])
#构建网络模型
#print(input_features[0])
#1 将数据转换为tensor格式
x = torch.tensor(input_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float) #标签

#2.初始化权重参数
#输入层（384，14）
#y = wx + b

'''#隐层1
weights = torch.randn((14,128),dtype=torch.float,requires_grad=True) #14个特征，1个标签
biass = torch.randn(128,dtype=torch.float,requires_grad=True)
#隐层2
weights2 = torch.randn((128,1),dtype=torch.float,requires_grad=True)
biass2 = torch.randn(1,dtype=torch.float,requires_grad=True)

#3.定义模型
learning = 0.001
losses = [] #损失值

for i in range(1000):
    #计算隐层1
    hidden = x.mm(weights) + biass #矩阵相乘
    #加入激活函数torch.relu(hidden)
    hidden = torch.relu(hidden)
    #预测结果
    predictions = hidden.mm(weights2) + biass2
    #计算损失
    loss = torch.mean((predictions-y)**2)
    losses.append(loss.data.numpy())  #转化为numpy格式，并记录

    #打印损失值
    if i % 100 == 0:
        print('loss:',loss)

    #反向传播
    loss.backward()

    #更新参数
    # 更新权重
    # 将权重数据减去学习率乘以权重梯度数据,作用是进行权重更新
    weights.data.add_(- learning * weights.grad.data)
    # 更新偏置
    biass.data.add_(-learning * biass.grad.data)
    # 更新第二层权重
    weights2.data.add_(- learning * weights2.grad.data)
    # 更新第二层偏置
    biass2.data.add_(- learning * biass2.grad.data)


    #每次更新后，清零梯度
    weights.grad.data.zero_()
    biass.grad.data.zero_()
    weights2.grad.data.zero_()
    biass2.grad.data.zero_()'''


#使用更简单的构造网络模型
input_size = features.shape[1] #输入层
hidden_size = 128 #隐层
output_size = 1 #输出层
bath_size = 16
my_nn = torch.nn.Sequential(
      torch.nn.Linear(input_size,hidden_size), #定义输入层到隐层的线性变换
      torch.nn.Sigmoid(), #定义隐层的激活函数为Sigmoid
      torch.nn.Linear(hidden_size,output_size) #定义隐层到输出层的线性变换
    )
cost = torch.nn.MSELoss(reduction='mean') #定义损失函数为均方误差
optimizer = torch.optim.Adam(my_nn.parameters(),lr=0.001) #定义优化器为Adam，学习率为0.001


#训练模型
losss = []
for i in range(1000):
    batch = []
    #MINI-BATCH方法来训练
    # 遍历input_features列表，每次步长为bath_size
    for start in range(0,len(input_features),bath_size): #目的是取部分数据
        # 如果start + bath_size小于len(input_features)，则end为start + bath_size
        # 计算本次批次的结束位置，如果结束位置超过输入特征长度，则取输入特征长度
        end = start + bath_size if start + bath_size < len(input_features) else len(input_features)
        # 将本次批次的输入特征转换为torch.tensor类型，并指定数据类型为float
        xx = torch.tensor(input_features[start:end],dtype=torch.float,requires_grad=True)
        # 将本次批次的标签转换为torch.tensor类型，并指定数据类型为float
        yy = torch.tensor(labels[start:end],dtype=torch.float,requires_grad=True)

        # 使用神经网络进行预测
        prediction = my_nn(xx)
        # 计算损失值
        loss = cost(prediction,yy)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        # 记录损失值
        batch.append(loss.data.numpy())
        # 打印损失值
    if i % 100 == 0:
        losss.append(np.mean(batch))
        print(i,'loss:',np.mean(batch))

x = torch.tensor(input_features, dtype=torch.float) #将数据转换为torch.tensor类型
predict = my_nn(x).data.numpy() #预测结果值并转换为numpy类型

#绘图
#创建一个表格；来存放日期和其对应的标签
true_data = pd.DataFrame(data={'date': dates, 'temperature': labels})

#将日期设置为索引
#true_data.set_index('date', inplace=True)
#创建一个表格；来存放日期和其对应的预测值
predict_data = pd.DataFrame(data={'date': dates, 'temperature': predict.reshape(-1)})

#真实值和预测值绘图
plt.plot(true_data['date'], true_data['temperature'], label='true data')

plt.plot(predict_data['date'], predict_data['temperature'], label='predict data')
plt.legend()
plt.show()
