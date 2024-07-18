"""
一、主成分分析（无监督）
1.用途：降维中常用的一种手段
2.目标：提取最有价值的信息（基于方差）
3.问题：降维后数据的意义？

二、向量的表示及基变换
1.内积：（a1,a2,....,an)**T*(b1,b2...,bn)**T = a1b1+a2b2+...+anbn
2.解释：A.B = |A||B|cos（a)
设向量B的模为1，则A与B的内积值等于A向B所在直线投影灯矢量长度
向量（3，2）实际上表示线性组合：x(1,0)**T+y(0,1)**T
基：（1，0）和(0,1)叫做二维空间的一组基

三、基变换
基是正交的（内积为0，或相互垂直）
要求：线性无关
基变换：数据与一个基做内积运算，结果作为第一个新的坐标分量，
然后与第二个基做内积运算，结果作为第二个新坐标的分量

将数据映射到基坐标中

四、协方差矩阵 -1 - 1 越接近1，变化越小
1.方向：如何选择这个方向，使得尽量保存最多的原始信息（直观说：希望投影后投影值尽可能分散）
2.方差：var = 1/m * sum(ai-u)**2
3.寻找一个一维基，使得所有数据变换为这个基上坐标表示后，方差值最大
4.协方差（假设均值为0时）：cov = 1/m*sum（ai*bi）
5.为了让两个字段尽可能表示更多的原始信息，不希望它们之间存在线性相关性。
当协方差=0时，表示两个字段完全独立，因此最终选择的两个基一定是正交的。

五、优化目标
1.将一组N维向量降为K维，目标是选择K个单位正交基，使得原始数据映射到该组基上后
两两间协方差为0，方差尽可能大

2.协方差矩阵对角线上的两个元素分别是两个字段的方差，而其他元素师a和b的协方差
3.协方差矩阵对角化，且按对角线进行排序

数据-协方差矩阵-特征值、特征向量-对角化-降维
"""

import pandas as pd
import numpy as np
#加载数据
df = pd.read_table(r"C:\Users\29048\Desktop\iris.data.txt",sep=',',encoding='utf-8')

#为数据添加列索引
df.columns = ['sepal_len','sepal_wid','petal_len','petal_wid','class']
#print(df)

#

x = df[['sepal_len','sepal_wid','petal_len','petal_wid']]
y = df['class']

#可视化展示
from matplotlib import pyplot as pl
import math
label_dict = {1:'Iris-setosa',
              2:'Iris-versicolor',
              3:'Iris-virginica'
}
feature_dict = {0:'sepal length(cm)',
                1:'sepal width(cm)',
                2:'petal length(cm)',
                3:'petal width(cm)'
}

pl.figure(figsize=(8,6)) #设置画布大小
for i,cnt in enumerate(['sepal_len','sepal_wid','petal_len','petal_wid']):
    pl.subplot(2,2,i+1)  # 设置放置位置
    for lab in label_dict.values():
        # 注意这里使用 loc 索引器按类别筛选数据
        select_data = x.loc[y==lab,cnt]
        pl.hist(select_data,label=lab,bins=10,alpha=0.3)
    pl.xlabel(feature_dict[i])
    pl.legend(loc='upper right', fancybox=True, fontsize=8)
pl.tight_layout()
#pl.show()

'''import pandas as pd
import matplotlib.pyplot as plt

# 假设 df 已经是加载好的鸢尾花数据集 DataFrame  

label_dict = {'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2
              }  # 这里的键是类别标签，值是用于绘图的标签（可以是整数或其他标识符）  
feature_dict = {0: 'sepal length (cm)',
                1: 'sepal width (cm)',
                2: 'petal length (cm)',
                3: 'petal width (cm)'
                }

plt.figure(figsize=(8, 6))  # 设置画布大小  
for cnt, feature in enumerate(x.columns, start=0):  # 从0开始计数，因为索引也是从0开始的  
    plt.subplot(2, 2, cnt + 1)  # 设置放置位置（因为subplot索引从1开始）  
    for lab in label_dict:  # 直接遍历类别标签  
        # 使用布尔索引选择对应类别的数据  
        selected_data = x.loc[y == lab, feature]
        if not selected_data.empty:  # 确保有数据才绘制直方图  
            plt.hist(selected_data, label=lab, bins=10, alpha=0.3)
    plt.xlabel(feature_dict[cnt])  # 设置x轴标签  
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
plt.show()'''
#x数据标准化
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)
print(x_std)