#简单逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import os
pl.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
# datas = {'Exam 1':[34.623660,30286711,38.847409,60.182599,79.032736],'Exam 2':[78.024693,43.894998,72.902198,86.308552,75.344376],'Admitted':[0,0,0,1,1]}
pd.set_option('display.unicode.east_asian_width', True)  # 解决列不对齐
fill = r"C:\Users\29048\Desktop\逻辑回归1.xls" #数据地址
datas = pd.read_excel(fill, sheet_name='数据')#读取数据
data1 = pd.DataFrame(datas)#将数据转为二维数据
a = [1 for i in range(len(data1['评分']))] # 获取1列全为1的数据
data1.insert(1, '1', 1)#往第一列添加全为1的列
data1.drop(columns='电影', axis=1, inplace=True)#删除索引为’电影‘的列，并更新数据
# print(data)
#print(data1.shape) #查看数据（数量，列数）
# print(data)
# 评分  评价人数  是否选择
positive = data1[data1['是否选择'] == 1] #当是否选择=1时候将这些数据归为positive
# print(positive)
negative = data1[data1['是否选择'] == 0]#当是否选择=0时候将这些数据归为positive
pl.figure(figsize=(10, 5))  # 设置画布大小
pl.scatter(positive['评分'], positive['评价人数'], c='b', marker='o', label='选择') #绘制散点图
pl.scatter(negative['评分'], negative['评价人数'], c='r', marker='x', label='不选择')
pl.legend() #添加图列
pl.xlabel('评分 Score') #x轴标签
pl.ylabel('评价人数（万人） Score')#y轴标签
pl.show()


#绘制决策边界
#目标：建立分类器（求解出三个参数（a0,a1,a2)
#设定阈值，根据阈值判断录取结果


#要完成的模块，sigmoid：映射到概率的函数 1 / 1+e**(-z)
#model :返回预测结果值 1/1+e**(-(x*theta.T))
#cost：根据参数计算损失 dx = -ylog(ha(x) - (1-y)log(1-ha(x))
#gradient：计算每个参数的梯度方向
#descent：进行参数跟新
#accuracy：计算精度

#设定阈值
#yz = 0.5
#sigmoid 函数
def sigmoid(z):
    #sigmoid：映射到概率的函数 1 / 1+e**(-z)
    from math import e,pi,sqrt
    gz = 1/(1+e**(-z))
    return gz
def model(x,theta):
    # model :返回预测结果值 1/1+e**(-(x*theta.T))
    import numpy as np
    pf = sigmoid(np.dot(x,theta.T)) #np.dot 进行矩阵乘法
    return pf
data = data1.to_numpy()
#print(data)
#print(data)

cols = data1.shape[1]
x = data1.iloc[:,0:cols-1].values #提取除了最后一列之外的所有列作为特征矩阵。
y = data1.iloc[:,cols-1:cols].values #提取最后1列作为结果值
theta = np.zeros([1,3]) #构造一个二维0矩阵1行3列
#print(x)
#print(y)

#损失函数cost：根据参数计算损失
def cost(x,y,theta):
    #损失函数 dx = -ylog(ha(x) - (1-y)log(1-ha(x))
    #损失平均值 jx = sum(dx)/len(x)
    left = np.multiply(-y,np.log(model(x,theta))) #计算左边值 (np.multiply乘法）
    right = np.multiply(1-y,np.log(1-model(x,theta)))#计算右边值
    dax = np.sum(left-right)/len(x)#计算平均损失 np.sum 累加和
    return dax
#print(cost(x,y,theta))

#gradient：计算每个参数的梯度方向
def gradient(x,y,theta):
    #公式 - 1/m * np.sum(yi - ha(xi))xij
    grad = np.zeros(theta.shape) #定义一个梯度
    #print(grad)
    error = (model(x,theta)-y).ravel()#ravel() 是 NumPy 库中的一个方法
    #ravel() ，用于将多维数组展平（flatten）成一维数组 ==-(yi-h0(xi))
    # - np.sum(yi - ha(xi))

    for j in range(len(theta.ravel())):
        #print(j)
        term = np.multiply(error,x[:,j]) #表示 - np.sum(yi - ha(xi))xij x[:,j]取第j列
        grad[0,j] = np.sum(term)/len(x)
    return grad
a = gradient(x,y,theta)
print(a)

#比较3种不同的梯度下降方法:根据迭代次数
S_i = 0
S_c = 1
S_g = 2
def stopCriterion(type,value,threshold):
    #设定三种不同的停止策略
    if type == S_i:
        return value > threshold
    elif type == S_c:
        return abs(value[-1]-value[-2]) < threshold
    elif type == S_g:
        return np.linalg.norm(value) < threshold

import numpy.random

#洗牌
def shuffleData(data):
    #是 NumPy 库中的一个函数，用于对数组中的元素进行随机排序
    # （就地，即直接修改原数组，不返回新数组）。
    np.random.shuffle(data) #数据重新洗牌 打乱数据排序
    cols = data.shape[1]
    x = data1.iloc[:, 0:cols - 1].values  # 提取除了最后一列之外的所有列作为特征矩阵。
    y = data1.iloc[:, cols - 1:].values  # 提取最后1列作为结果值
    return x,y

import time
#descent：进行参数跟新
def descent(data,theta,batchSize,stopType,thresh,alpha):
    #梯度下降求解
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    x,y = shuffleData(data)
    #grad = np.zeros(theta.shape) #计算梯度
    costs = [cost(x,y,theta)] #计算损失值
    while True:
        grad = gradient(x[k:k+batchSize],y[k:k+batchSize],theta) #gradient：计算每个参数的梯度方向
        k += batchSize #取batch数量个数
        if k >= n:
            k = 0
            x,y = shuffleData(data) #重新洗牌
        theta = theta-alpha*grad #参数更新
        costs.append(cost(x, y, theta)) #计算新的损失
        i += 1
        if stopType == S_i:
            value = i
        elif stopType == S_c:
            value = costs
        elif stopType == S_g:
            value = grad
        if stopCriterion(stopType,value,thresh):
            break
    return theta,i-1,costs,grad,time.time(),-init_time

'''def runExpe(data,theta,batchSize,stopType,thresh,alpha):
    theta,lter,costs,grad,dur,a = descent(data,theta,batchSize,stopType,thresh,alpha)
    if (data[:,1]>2).sum() > 1:
        name = 'Original'
    else:
        name = 'Scaled'
    name += 'data - learning rate:{}'.format(alpha)
    if batchSize == n:
        strDescType = 'Gradient'
    elif batchSize == 1:
        strDescType = 'Stochastic'
    else:
        strDescType = 'Mini-batc({})'.format(batchSize)
    name += strDescType + 'descent - stop：'
    if stopType == S_i:
        strStop = '{} iterations'.format(thresh)
    elif stopType == S_c:
        strStop = 'costs change < {}'.format(thresh)
    else:
        strStop = 'gradient norm <{}'.format(thresh)
    name += strStop
    print('***{}\nTheta:{} - Iter:{} - Last cost:{:0.2f} - Duration:{:0.2f}s'.format(
        name,theta,lter,costs[-1],dur))
    pl.figure(figsize=(12,4))
    pl.plot(np.arange(len(costs)),costs,'r')
    pl.xlabel('Iterations')
    pl.ylabel('Cost')
    pl.title(name.upper() + '- Error vs Iteration')
    pl.show()'''


import numpy as np
import matplotlib.pyplot as plt  # 确保导入了正确的模块

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iterations, costs, grad, dur, _ = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = 'Scaled' if (data[:, 1] > 2).sum() <= 1 else 'Original'  # 假设data是DataFrame，且我们基于第二列的值来判断是否缩放
    name += ' data - Learning rate: {}'.format(alpha)
    if batchSize == data.shape[0]:
        strDescType = 'Gradient'
    elif batchSize == 1:
        strDescType = 'Stochastic'
    else:
        strDescType = 'Mini-batch({})'.format(batchSize)
    name += strDescType + ' descent - Stop: '
    if stopType == S_i:
        strStop = '{} iterations'.format(thresh)
    elif stopType == S_c:
        strStop = 'cost change < {}'.format(thresh)
    else:
        strStop = 'gradient norm < {}'.format(thresh)
    name += strStop
    print('***{}\nTheta: {} - Iterations: {} - Last cost: {:03.2f} - Duration: {:03.2f}s'.format(
        name, theta, iterations, costs[-1], dur))
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(costs)), costs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(name.upper() + ' - Error vs Iteration')
    plt.show()
    print(theta)


# 假设data和theta已经被正确定义和初始化
# n是数据集的大小，您应该在之前定义它，例如 n = data.shape[0]
# runExpe(data, theta, n, S_i, thresh=5000, alpha=0.0000001)
n = 251
runExpe(data,theta,n,S_i,thresh=500000,alpha=0.0000001)
#print(theta)

#精度
def predict(x,theta):
    return [1 if x >= 0.5 else 0 for x in model(x,theta)]

x = data1.iloc[:,0:cols-1].values
y = data1.iloc[:,cols-1:cols].values
predictions = predict(x,theta)
corect = [1 if ((a == 1 and b ==1) or (a == 0 and b == 0 )) else 0 for (a,b) in zip(predictions,y)]
accuracy = (sum(map(int,corect)) / len(corect))
print(accuracy*100)