import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings

import pylab as pl

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
"""
'axes.labelsize': 这设置了坐标轴标签（如x轴和y轴的标签）的字体大小。
在你的代码中，它被设置为14。
'xtick.labelsize': 这设置了x轴刻度标签（即x轴上的各个点的标签）的字体大小。
在你的代码中，它被设置为12。
'ytick.labelsize': 这设置了y轴刻度标签（即y轴上的各个点的标签）的字体大小。
同样，在你的代码中，它被设置为12。
"""

warnings.filterwarnings('ignore')
#忽略（即不显示）之后代码中产生的所有警告。

np.random.seed(0) #设置随机因子
from sklearn.datasets import fetch_openml

#数据读取
#mnist = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\2-线性回归代码实现\线性回归-代码实现\data\server-operational-params.csv")
#x,y = mnist[["Latency (ms)","Throughput (mb/s)"]], mnist["Anomaly"]
'''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)'''

#1.线性回归方程实现
#机器学习中核心的思想是迭代更新

#随机生成数据
import numpy as np
x = 2*np.random.rand(100,1)
y = 4+ 3*x +np.random.randn(100,1)
plt.plot(x,y,'b.')
plt.xlabel('X_1')#设置坐标轴名称
plt.ylabel('y')
plt.axis([0,2,0,15])#置坐标轴范围。
#plt.show()
import statsmodels.api as sms
#x = sms.add_constant(x)

x = pd.DataFrame(x,columns=['x'])
y = pd.DataFrame(y,columns=['y'])
#在数据中加上一列1
x.insert(0,'1',1)

#np.linalg.inv(x.T.dot(x))：这是计算x.T.dot(x)的逆矩阵
therat = np. linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
#print(therat)
y1 = np.dot(x,therat)
y1 = pd.DataFrame(y1)
plt.subplot(121)
plt.plot(x.x,y,'bo')
plt.subplot(122)
plt.plot(x.x,y,'bo')
plt.plot(x.x,y1,'r')
#plt.show()
#print(x)
#plt.show()
from sklearn.linear_model import LinearRegression #线性回归模型
from sklearn.model_selection import train_test_split

res = LinearRegression() #实例化模型
res.fit(x,y)
plt.scatter(x.x,y)
plt.plot(x.x,res.predict(x),'g')
#plt.show()

'''
### sklearn api文档：
不用背，用到的时候现查完全够用的。
https://scikit-learn.org/stable/modules/classes.html
'''

#2.梯度下降效果
#问题:1.步长过长，2.步长过短

#数据标准化
eta = 0.1#学习率
n_iterations = 1000#迭代次数
m = 100  #选择样本个数
theta = np.random.randn(2,1) #随机生出参数
theta = pd.DataFrame(theta)

x1 = 2*np.random.rand(100,1)
X_b = np.c_[np.ones((100,1)),x1]
'''#1.批量梯度下降'''
#梯度 = 2 / m * np.dot(x.T,(np.dot(x,theta1) - y))
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y) #计算梯度下降
    theta = theta - eta * gradients #更新参数
print(theta)
"""theta1 = pd.DataFrame(theta)
for iteration in range(n_iterations):
    gradients = 2 / m * np.dot(x.T,(np.dot(x,theta1) - y))
    theta1 = theta1 - eta * gradients
print(theta1)"""

'''#不同学习率对结果的影响'''
theta_path_bgd =[]
def plot_gradient_descent(theta,eta,theta_path = None):
    m = len(x)
    plt.plot(x.x,y,'b.') #绘制初始数据散点图
    n_iterations = 1000  #迭代次数
    for iteration in range(n_iterations):
        y_predict = np.dot(x,theta)  #计算预测结果
        plt.plot(x.x,y_predict,'r')  #绘制回归曲线
        gradients = 2 / m * np.dot(x.T, (np.dot(x, theta) - y)) #计算梯度下降结果
        theta = theta - eta * gradients #更新参数
        #theta_path_bgd.append(theta)
        if theta_path is not None:  #保存参数
            theta_path.append(theta)
    plt.xlabel('X_1')
    plt.axis([0, 2, 0, 15])
    plt.title('eta = {}'.format(eta))


theta1 = np.random.rand(2,1)
#print(theta1.shape)
plt.figure(figsize=(10,4))
plt.subplot(131) #设置子图位置
plot_gradient_descent(theta1,eta=0.02) #设置不同的学习率
plt.subplot(132)
plot_gradient_descent(theta1,eta=0.1,theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta1,eta=0.5)
plt.show()

#随机梯度下降
#核心解决方案，不光在线性回归中能用上，
# 还有其他算法中能用上，比如神经网络
theta_path_sgd =[]
m = len(x)
n_epochs = 50
t0 = 5
t1 = 50
theta2 = np.random.randn(2,1) #随机生出参数

#衰减策略：学习率不断减小
#学习率应当尽可能小，随着迭代的进行应当越来越小。
def learing_schedule(t):#t表示迭代次数
    return t0/(t1+t)
for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            plt.title(f'{epoch}:{i}')
            y_predict = np.dot(x, theta2)  # 计算预测结果
            plt.plot(x.x, y_predict, 'r')  # 绘制回归曲线

        random_index = np.random.randint(m)
        xi = X_b[random_index]
        xi = pd.DataFrame([xi])
        #print(xi)
        #print(xi)
        yi = y.loc[random_index,]
        yi = pd.DataFrame(yi)
        #计算梯度
        gradients = 2 * np.dot(xi.T, (np.dot(xi, theta2) - yi))  # 计算梯度下降结果
        eta = learing_schedule(epoch * m + i) #学习率更新
        theta2 = theta2 - eta * gradients  # 更新参数
        theta_path_sgd.append(theta2)

plt.plot(x.x, y, 'g.')  # 绘制初始数据散点图
plt.xlabel('X_2')
plt.axis([0, 2, 0, 15])
plt.show()

#小批量梯度下降
theta_path_mgd =[]
n_epochs2 = 50
minibatch = 16
theta3 = np.random.randn(2,1) #随机生出参数
t = 0
#np.random.seed(0)
for epoch in range(n_epochs2):
    #数据洗牌
    shuffled_index = np.random.permutation(m)
    x_b_Shuffled = x.loc[shuffled_index,]
    #print(x_b_Shuffled)
    y_shuffled = y.loc[shuffled_index]
    #print(y_shuffled)
    for i in range(0,minibatch):
        plt.title(f'{epoch}:{i}')
        y_predict = np.dot(x, theta3)  # 计算预测结果
        plt.plot(x.x, y_predict, 'r')  # 绘制回归曲线
        t +=1
        xi = x_b_Shuffled.loc[i:i+minibatch,]
        xi = pd.DataFrame(xi)
        #print(xi)
        yi = y_shuffled.loc[i:i+minibatch,]
        yi =pd.DataFrame(yi)
        #print(xi)
        gradients = 2/minibatch * np.dot(xi.T, (np.dot(xi, theta3) - yi))  # 计算梯度下降结果
        eta = learing_schedule(t)
        theta3 = theta3 - eta * gradients  # 更新参数
        theta_path_mgd.append(theta3)
plt.plot(x.x, y, 'g.')  # 绘制初始数据散点图
plt.show()
#3种策略的对比实验
theta_path_mgd = np.array(theta_path_mgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_bgd = np.array(theta_path_bgd)
#print(theta_path_bgd)
plt.figure(figsize=(10,4))
plt.plot(theta_path_sgd[:,0],theta_path_sgd[:,1],'r-s',linewidth=1,label='Sgd')
plt.plot(theta_path_mgd[:,0],theta_path_mgd[:,1],'g-+',linewidth=2,label='Mgd')
plt.plot(theta_path_bgd[:,0],theta_path_bgd[:,1],'b-o',linewidth=3,label='Bgd')
plt.legend(loc='upper left')

plt.show()

'''
实际当中用minibatch比较多，一般情况下选择batch数量应当越大越好。
'''

#多项式回归
#获取数据 #250个点
m = 100
x = 6*np.random.rand(m,1) - 3
y = 0.5*x**2+x+np.random.randn(m,1)
plt.plot(x,y,'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([-3,3,-5,10])
#PolynomialFeatures类。这个类用于生成多项式特征
from sklearn.preprocessing import PolynomialFeatures
#创建多项式特征转换器:
poly_features = PolynomialFeatures(degree=2,include_bias=False)
#对原始数据进行多项式特征转换
x_poly = poly_features.fit_transform(x)
#导入并初始化线性回归模型:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#用多项式特征训练线性回归模型:
lin_reg.fit(x_poly,y)
#创建新的数据点并进行预测:
x_new = np.linspace(-3,3,100).reshape(100,1)
#对这些点进行多项式特征转换
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.plot(x_new,y_new,'r--',label='prediction')
plt.legend()
plt.show()


#Pipeline将多个数据处理步骤（如特征选择、缩放、编码等）和模型训练步骤整合到一个单一的流程中。
#流程化工具
from sklearn.pipeline import Pipeline

#标准化模块
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12,6)) #设置画布大小
for style,width,degree in (('g-',1,100),('b--',1,2),('r-+',1,1)):
    poly_features = PolynomialFeatures(degree=degree,include_bias = False)
    std = StandardScaler()
    lin_reg = LinearRegression()

    #流程化工具
    polynomial_reg = Pipeline([('poly_features',poly_features),
             ('StandardScaler',std),
             ('lin_reg',lin_reg)])
    polynomial_reg.fit(x,y)
    y_new_2 = polynomial_reg.predict(x_new)#获取预测值
    plt.plot(x_new,y_new_2,style,label='degree   '+str(degree),linewidth = width)
plt.plot(x,y,'b.')
plt.axis([-3,3,-5,10])
plt.legend()
plt.show()
def cs():
    #对自己的数据进行多项式回归曲线绘制
    data = pd.read_csv(
        r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\2-线性回归代码实现\线性回归-代码实现\data\non-linear-regression-x-y.csv")

    x = data['x'].values.reshape(-1, 1)  # 保x是二维的，因为fit_transform需要二维输入
    y = data['y'].values

    plt.plot(x, y, 'b.')
    plt.xlabel('X_1')
    plt.ylabel('y')

    # PolynomialFeatures类。这个类用于生成多项式特征
    from sklearn.preprocessing import PolynomialFeatures
    # 创建多项式特征转换器:
    poly_features = PolynomialFeatures(degree=8, include_bias=False)
    # 对原始数据进行多项式特征转换
    x_poly = poly_features.fit_transform(x)
    # 导入并初始化线性回归模型:
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()

    # 用多项式特征训练线性回归模型:
    lin_reg.fit(x_poly, y)
    # 创建新的数据点并进行预测:
    x_new = np.linspace(x.min(), x.max(), 250).reshape(-1, 1)
    # 对这些点进行多项式特征转换
    x_new_poly = poly_features.transform(x_new)
    y_new = lin_reg.predict(x_new_poly)
    plt.plot(x_new, y_new, 'r--', label='prediction')
    #plt.show()
'''
特征变换的越复杂，得到的结果过拟合风险越高，不建议做的特别复杂。
'''

#数据样本数量对结果的影响
from sklearn.metrics import mean_squared_error #均方误差模块
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

m = 100
X = 6*np.random.rand(m,1) - 3
y = 0.5*X**2+X+np.random.randn(m,1)
def plot_learning_curves(model,X,y):
    #划分测试集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=100)
    train_errors,val_errors = [],[] #均方误差列表
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m]) #获取预测值
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict[:m]))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label = 'train_error')
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label='val_error')
    plt.xlabel('Trainsing set size')
    plt.ylabel('RMSE')
    plt.legend()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg,X,y)
plt.axis([0,80,0,3.3])
plt.show()

'''
数据量越少，训练集的效果会越好，但是实际测试效果很一般。
实际做模型的时候需要参考测试集和验证集的效果。
'''

#5.过拟和与欠拟和
#多项式回归的过拟合风险
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
#流程化工具
polynomial_reg = Pipeline([('poly_features',PolynomialFeatures(degree=10,include_bias=False)),
             ('lin_reg',LinearRegression())])
plot_learning_curves(polynomial_reg,X,y)
plt.axis([0,80,0,5])
plt.show()

#6.正则化的作用：解决过拟和风险
'''
对权重参数进行惩罚，让权重参数尽可能平滑一些，
有两种不同的方法来进行正则化惩罚:岭回归Ridge 和Lasso L1正则化
'''
#公式：j(e) = MSE（e) + a*(1/2)*sum((ei)**2)
from sklearn.linear_model import Ridge
#岭回归是一种用于特征之间存在高度相关性的线性回归方法。（
# 通常是L2正则化项）来防止过拟合
np.random.seed(42)
m1 = 20
x2 = 3*np.random.rand(m1,1)
y2 = 0.5*x2 + np.random.rand(m1,1)/1.5 +1
x2_new = np.linspace(0,3,100).reshape(100,1)

def plot_model(model_calss,polynomial,alphas,**model_kargs):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    #对比实验
    for alpha,style in zip(alphas,('b-','g--','r:')):
        model = model_calss(alpha,**model_kargs)
        if polynomial:
            model = Pipeline([('poly_features', PolynomialFeatures(degree=8,include_bias=False)),
                                       ('StandardScaler', StandardScaler()),
                                       ('lin_reg',model)])
        model.fit(x2,y2)
        y_new_regul = model.predict(x2_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(x2_new, y_new_regul, style, linewidth=lw, label='alpha = {}'.format(alpha))
    plt.plot(x2, y2, 'b.', linewidth=3)
    plt.legend()

plt.figure(figsize=(14,6))
plt.subplot(121)
plot_model(model_calss=Ridge,polynomial=False,alphas=(0,10,100))
plt.subplot(122)
plot_model(model_calss=Ridge,polynomial=True,alphas=(0,10**(-5),1))
plt.show()

from sklearn.linear_model import Lasso
#Lasso (最小绝对收缩和选择算子)
#Lasso是一个用于实现线性回归的类,它执行L1正则化
plt.figure(figsize=(14,6))
plt.subplot(121)
plot_model(model_calss=Lasso,polynomial=False,alphas=(0,0.1,1))
plt.subplot(122)
plot_model(model_calss=Lasso,polynomial=True,alphas=(0,10**(-1),1))
plt.show()



#7.提取停止策略