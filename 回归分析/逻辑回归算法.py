import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

### 神奇的sigmoid函数：
def sigmodel():
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.title('Figure 4-21. Logistic function')
    plt.show()

### 加载sklearn内置数据集
from sklearn import datasets
iris = datasets.load_iris()
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

'''对于传统逻辑回归要对标签做变换，也就是属于当前类别为1其余类别为0'''
x = iris['data'][:,3:]
print(x.shape)
#将某种花设置为1，其他全为0
y = (iris['target'] == 2).astype(np.int_)

#astype(np.int_)转换为整数序列。在这个转换中，True会变成1，False会变成0。
from sklearn.linear_model import LogisticRegression #逻辑回归模型
#实例化
log_reg = LogisticRegression()
log_reg.fit(x,y)

x_new = np.linspace(0,3,1000).reshape(-1,1)

#预测概率值
y_prdict_prob = log_reg.predict_proba(x_new)
plt.figure(figsize=(10,5))
#边界
decision_boundary = x_new[y_prdict_prob[:,1]>=0.5][0]

#print(decision_boundary)
plt.plot([decision_boundary,decision_boundary],(-1,2),'k:')
#指定箭头x,y,dx:指向>0右，<0左，dy
plt.arrow(decision_boundary[0],-0.8,-0.3,0,head_width=0.05,head_length=0.1,fc='b',ec='b')
plt.arrow(decision_boundary[0],1.6,0.3,0,head_width=0.05,head_length=0.1,fc='g',ec='g')
#添加说明
plt.text(decision_boundary[0]+0.02,-0.6,'Decision Boundary',fontsize = 16,color = 'k',ha='center')
plt.plot(x_new,y_prdict_prob[:,1],'g',label='Iris-Virginica')
plt.plot(x_new,y_prdict_prob[:,0],'b--',label='Not Iris-Virginica')
plt.xlabel('peta width（cm)')
plt.ylabel('y_prdict_prob')
#plt.axis([0,3,-0.02,1.02])
plt.legend(loc = 'center left',fontsize = 16)
plt.show()

#绘制决策边界
#1.选择数据
X = iris['data'][:,(2,3)] #2维
Y = (iris['target']==2).astype(np.int_)

#2.构建模型
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,Y)

#3.构建坐标数据
x_min = np.min(X[:,0])
x_max = np.max(X[:,0])
y_min = np.min(X[:,1])
y_max = np.max(X[:,1])
#np.meshgrid 是 NumPy 库中主要用于生成坐标矩阵
x0,x1 = np.meshgrid(np.linspace(2.9,7,500).reshape(-1,1),np.linspace(0.8,2.7,200).reshape(-1,1))
#将数据拼接在一起形成测试数据
X_new = np.c_[x0.ravel(),x1.ravel()]

#4.预测概率值
Y_pro = log_reg.predict_proba(X_new)

#绘制等高线
plt.figure(figsize=(10,4))
#print(X[Y==0,0]) 当y==0时候选择x的第一列
#print(X[Y==0,1])，当y==0时候选择x的第二列
plt.plot(X[Y==0,0],X[Y==0,1],'bs')
plt.plot(X[Y==1,0],X[Y==1,1],'g^')

Y5 = pd.DataFrame(Y_pro)

#绘制等高线
z = Y_pro[:,1].reshape(x0.shape) #维度与x相同

contour = plt.contour(x0,x1,z,cmap=plt.cm.brg)
plt.clabel(contour,inline = 1)
'''
plt.clabel(...): 这是用于给等高线添加标签的函数。
contour: 是前面plt.contour函数返回的等高线对象。
inline=1: 这是一个可选参数,如果inline是True或1，则标签会被裁剪到它们的线段上，
这样可以避免标签重叠或覆盖其他线条。
'''
plt.text(3.5,1.5,'NOT Vir',fontsize = 16,color = 'b')
plt.text(6.5,2.3,'Vir',fontsize = 16,color = 'g')
plt.show()

xx = iris['data'][:,(2,3)]
yy = iris['target']

#处理多类别分类问题
softmax_reg = LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
'''
multi_class = 'multinomial'：
这个参数决定了当 LogisticRegression 用于多类别分类时应该使用哪种策略。
'multinomial' 选项表示使用多项式逻辑回归

solver = 'lbfgs'：
这个参数决定了用于优化问题的算法。
'lbfgs' 是一个优化算法
'''

softmax_reg.fit(xx,yy)

x2,x3 = np.meshgrid(np.linspace(0,8,500).reshape(-1,1),np.linspace(0,3.5,200).reshape(-1,1))
X_new2 = np.c_[x2.ravel(),x3.ravel()]
y_prob = softmax_reg.predict_proba(X_new2)
y_predict = softmax_reg.predict(X_new2)


zz = y_predict.reshape(x2.shape)
zz1 = y_prob[:,1].reshape(x2.shape)
plt.figure(figsize=(10, 4))
#绘制散点图
plt.plot(xx[yy==2, 0], xx[yy==2, 1], "g^", label="Iris-Virginica")
plt.plot(xx[yy==1, 0], xx[yy==1, 1], "bs", label="Iris-Versicolor")
plt.plot(xx[yy==0, 0], xx[yy==0, 1], "yo", label="Iris-Setosa")

#导入自定义颜色映射：
from matplotlib.colors import ListedColormap
#创建了一个名为custom_cmap的自定义颜色映射对象，
# 该对象包含了三个颜色：#fafab0（一种浅黄色）、
# #9898ff（一种蓝紫色）和#a0faa0（一种绿色）。
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

#使用contourf函数绘制填充的等高线图：
plt.contourf(x2, x3, zz, cmap=custom_cmap)

#绘制等高线
contour = plt.contour(x2, x3, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
