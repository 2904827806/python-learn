"""import numpy as np
import matplotlib.pyplot as pl
import statsmodels.api as sm #用于估计许多不同统计模型的参数，进行统计测试，
# 以及统计数据的探索。
from sklearn import linear_model as lm #从机器学习库中导入线性模型
import pandas as pd

nsample = 200 #设置随机因子数
x1 = np.linspace(0,10,nsample) #设置初始值，终值和个数
#一元线性回归
x = sm.add_constant(x1) #将x转换为包含一列全为1的二维数据
#等同于
data = pd.DataFrame(x1)
data.insert(0,'1',1)
#a0,a1分别设置为2，5
bata = np.array([2,5]) #参数设置
#误差项
e = np.random.normal(size=nsample) #np.random.normal: 这是 numpy 库中的一个函数，
# 用于生成从标准正态分布（均值为0，标准差为1）中抽取的随机数。个数为200

#y的实际值
y = np.dot(x,bata) +e #np.dot 矩阵乘法公式
#y1 = np.dot(data,bata) +e #np.dot 矩阵乘法公式

#最小二乘法
model = sm.OLS(y,x) #于拟合普通最小二乘法（Ordinary Least Squares，简称OLS）线性回归模型。
#拟和数据
res = model.fit() #fit() 方法来拟合模型，并得到一个包含模型估计结果的对象
#打印全部结果:res.summary()
#获取回归系数 ：res.params
xs = res.params
a0 = xs[0]
a1 = xs[1]
y2 = a0 + a1*x1 #拟和函数
#拟和的估计值
y_ = res.fittedvalues
#设置画布大小
pl.figure(figsize=(8,6))
#绘制图形
pl.plot(x1,y,'o',label='data') #原始数据
pl.plot(x1,y_,'r--',label='test') #拟和数据
#pl.scatter(x1,y2,c='k')
pl.legend()#图列
pl.show()
"""
import random

#高阶回归
"""#y = 5+2x+3x**2
import numpy as np
import matplotlib.pyplot as pl
import statsmodels.api as sm #用于估计许多不同统计模型的参数，进行统计测试，
# 以及统计数据的探索。
from sklearn import linear_model as lm #从机器学习库中导入线性模型
import pandas as pd

nsample = 200
x1 = np.linspace(0,10,nsample) #设置初始值，终值和个数
#一元线性回归
x = np.column_stack((x1,x1**2)) #来将两个数组（或列表、矩阵等）沿着列方向堆叠起来。
x = sm.add_constant(x) #将x转换为包含一列全为1的二维数据
#等同于
data = pd.DataFrame(x1)
data.insert(0,'1',1)
#a0,a1分别设置为2，5
bata = np.array([5,2,3]) #参数设置
#误差项
e = np.random.normal(size=nsample) #np.random.normal: 这是 numpy 库中的一个函数，
# 用于生成从标准正态分布（均值为0，标准差为1）中抽取的随机数。个数为20

#y的实际值
y = np.dot(x,bata) +e #np.dot 矩阵乘法公式
#y1 = np.dot(data,bata) +e #np.dot 矩阵乘法公式

#最小二乘法
model = sm.OLS(y,x) #于拟合普通最小二乘法（Ordinary Least Squares，简称OLS）线性回归模型。
#拟和数据
res = model.fit() #fit() 方法来拟合模型，并得到一个包含模型估计结果的对象
#打印全部结果:res.summary()
#获取回归系数 ：res.params
xs = res.params
a0 = xs[0]
a1 = xs[1]
a2 = xs[2]
y2 = a0 + a1*x1 + a2*(x1**2) #拟和函数
#print(res.summary())
#拟和的估计值
y_ = res.fittedvalues
#设置画布大小
pl.figure(figsize=(8,6))
#绘制图形
pl.plot(x1,y,'o',label='data') #原始数据
pl.plot(x1,y_,'r--',label='test') #拟和数据
pl.legend()#图列
pl.show()"""

#多元回归分析
"""#y = a0+a1x1+a2x2
import numpy as np
import matplotlib.pyplot as pl
import statsmodels.api as sm #用于估计许多不同统计模型的参数，进行统计测试，
# 以及统计数据的探索。
from sklearn import linear_model as lm #从机器学习库中导入线性模型
import pandas as pd

nsample = 210
groups = np.zeros(nsample,int) #设置初始值，终值和个数
groups[70:140] = 1
groups[140:] = 2
dummy = pd.get_dummies(groups,dtype=int) 
#将分类变量（通常是字符串或类别类型）转换为一系列二进制列，也称为“虚拟变量”或“哑变量”。
#每个唯一的分类值都会成为新的列，并在原始数据中该分类值出现的位置上标记为1，其他位置为0

#y = 5+2x+3z1+6z2+9z3

x1 = np.linspace(0,20,nsample)
x = np.column_stack((x1,dummy)) #来将两个数组（或列表、矩阵等）沿着列方向堆叠起来。
x = sm.add_constant(x) #在第1列插入1
beta = np.array([5,2,3,6,9])
e = np.random.normal(size=nsample)
#从标准正态分布（均值为0，标准差为1）中抽取的随机数。个数为nsample
y = np.dot(x,beta) + e
rusalt = sm.OLS(y,x).fit()
rusalt1 = rusalt.params
a1 = rusalt1[0]
a2 = rusalt1[1]
a3 = rusalt1[2]
a4 = rusalt1[3]
a5 = rusalt1[4]
pl.figure(figsize=(8,6))
pl.scatter(x1,y,label='data')
pl.plot(x1,rusalt.fittedvalues,'r--',label='OLS')
pl.legend()
pl.show()"""

'''#身高和体重
import pandas as pd
import statsmodels.api as sm #用于估计许多不同统计模型的参数，进行统计测试，
#plotly是一个交互式的图表库
from plotly.offline import init_notebook_mode,iplot
#iplot函数用于在Jupyter notebook中显示plotly图表。
#init_notebook_mode函数用于初始化plotly在Jupyter notebook中的交互模式。
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as pl
#plotly.graph_objs模块的别名，用于创建各种图表对象。
pd.set_option('display.unicode.east_asian_width', True)  # 解决列不对齐
datas = pd.read_excel(r"C:\\Users\29048\Desktop\逻辑回归1.xls",sheet_name=0)
datas.drop('电影',axis=1,inplace=True)
#data = pd.DataFrame(datas)
a1 = datas['评价人数']
a2 = datas['评分']
x = sm.add_constant(a1)

bate = [1,2.8]
e = np.random.normal(size=len(a2))
y = a2
#最小二乘法
res1 = sm.OLS(y,x)
res = res1.fit()
a = res.params
pl.figure(figsize=(8,6))
pl.scatter(a1,y)
pl.plot(a1,res.fittedvalues,c='r')
pl.show()
print(res.summary())
print('y = {}+{}x'.format(a[0],a[1]))'''
"""#评分	评价人数
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import statsmodels.api as sm
from random import random
pd.set_option('display.unicode.east_asian_width', True)  # 解决列不对齐
pd.set_option('display.max_columns',3000)
data = pd.read_excel(r"C:\\Users\29048\Desktop\双色球.xls")
data.dropna(inplace=True)
x1 = data['一']
x2 = data['二']
x3 = data['三']
x4 = data['四']
x5 = data['五']
x6 = data['六']
x7 = data['七']
data = sm.OLS(x7,x6)
res = data.fit()
print(res.summary())
print(res.params)
"""
"""
#汽车价格的预测
#使用sklearn库
#根据多个特征学习得到
#包含类别属性，连续指标
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns #。Seaborn是一个基于matplotlib的数据可视化库
import missingno as msno #专门用于探索和处理数据中的缺失值的库

#stats
from statsmodels.distributions.empirical_distribution import ECDF
#statsmodels是一个Python库，提供了描述性统计、统计模型估计和推断等功能。
# ECDF是（经验累积分布函数）的缩写，它用于表示一组数据的累积分布。
from sklearn.metrics import mean_squared_error,r2_score
"""
sklearn.metrics模块中导入两个常用的评估指标函数：
mean_squared_error（均方误差）和
r2_score（决定系数，也称为R²分数）。
"""

#machine learning
from sklearn.preprocessing import StandardScaler
"""
导入了StandardScaler类，用于数据标准化。标准化是一种预处理技术，
通过将数据变换为均值为0，标准差为1的分布，
可以消除量纲对模型训练的影响。
"""
from sklearn.linear_model import Lasso,LassoCV
"""
导入了两个线性回归模型：Lasso和LassoCV。
Lasso是一种线性回归模型，它使用L1正则化（即绝对值之和）来减少过拟合。
LassoCV是Lasso的一个变体，它使用交叉验证来选择最佳的L1正则化参数。
"""
from sklearn.model_selection import train_test_split,cross_val_score
"""
这行代码导入了两个用于模型选择和评估的函数：
train_test_split：用于将数据集分割为训练集和测试集。
cross_val_score：用于执行交叉验证，评估模型的性能。
"""
from sklearn.ensemble import RandomForestClassifier
"""
RandomForestClassifier类，用于构建随机森林分类器。
"""
import statsmodels.api as sm
seed = 123 #创建随机种子,控制随机因素
#数据读取
fill= r"C:\Users\29048\Desktop\二手车.xls"
pd.set_option('display.unicode.east_asian_width', True)  # 解决列不对齐
data = pd.read_excel(fill,sheet_name=0) #导入数据
#print(data.columns)#查看数据的列索引
print(data.describe())
#print(data.dtypes)#查看数据的类型
x = sm.add_constant(data) #往数据中加入1列全为1的数据
#print(data)
#data.describe()获取数据中数值类型的一些相应统计信息

#缺失值处理
#1.缺失值低于10%的可以直接去掉  data.dropna()
#2.用该数据中的均值，重数，中位数来填充
#3.构建回归模型来填充
pl.rcParams['font.sans-serif'] = ['SimHei']  #解决中文乱码
sns.set(style='ticks')
#style='ticks'意味着绘图时将使用白色网格线，并且坐标轴标签将具有小的刻度标记。
msno.matrix(data.sample(50))        #查看缺失值.sample(120)随机显示120个数据
# matrix函数来创建一个热图，用于可视化data数据集中的缺失值
#pl.show()      #显示绘制缺失值视图
#缺失值填充

#绘制缺失值情况
pl.figure(figsize=(12,5)) #设置画布大小
#使用ECDF
pl.subplot(121) #subplot 函数的三个参数通常表示子图的布局和当前激活的子图位置
#pl.subplot(121): 这行代码表示要创建一个1行2列的子图网格，
# 并将当前的绘图焦点设置到第一个子图
cdf = ECDF(data['jg'])#来创建 data['jg'] 列的经验累积分布函数对象。
pl.plot(cdf.x,cdf.y,label='statmodels',c='r') #数据累计结果

#绘制了ECDF。cdf.x 和 cdf.y 分别代表ECDF的x和y坐标值。
#ECDF主要用于描述和分析数据样本的分布情况。
#直观地展示出随机变量的分布情况
pl.xlabel('jg')
pl.ylabel('DCDF')

pl.subplot(122)
pl.hist(data['jg'].dropna(),bins=int(np.sqrt(len(data['jg']))),color='b') #数据分别情况
pl.show()  # 添加pl.show()来显示图形
#bins=int(np.sqrt(len(data['jg']))): 这部分设置了直方图中柱子的数量。
# np.sqrt(len(data['jg'])) 计算了 'jg' 列中数据点数量的平方根，
# 并使用 int() 函数将其转换为整数。
# 基于数据点的数量来自动选择合适的柱子数量，
# 以便在图中展示数据的分布情况。

#基于’groupby'将数据进行分组jg，并计算每个组中'ghcs'列的统计描述。
data1 = data.groupby('ghcs')['jg'].describe() #获取数据统计相关信息
print(data1)

#用中位数填充，
#print(data[pd.isnull(data['jg'])])#查看缺失值
#print(data1)
#jb	ryjh	fdj	csys	qdfs	bsx	lx	ck	lcs	pl	ghcs	jg

#替换缺失值 (用分组的均值进行填充）
data = data.dropna(subset=['jb','ryjh','fdj','csys','qdfs','bsx','lx','ck','lcs','pl']) #去除对应列名的缺失值
data['jg'] = data.groupby('ghcs')['jg'].transform(lambda x: x.fillna(x.mean()))
print(data)
#ransform函数用于对每个分组应用一个函数，并将结果广播回原始的形状
#x.mean(): 计算分组中'jg'列的非缺失值的平均值。
##x.fillna(x.mean()): 使用上面计算出的平均值来填充分组中'jg'列的缺失值
data2 = data.drop(['jb','ryjh','fdj','csys','qdfs','bsx','lx','ck'],axis=1) #删除相应列

#查看特征相关性
cormatrix = data2.corr()
#print(cormatrix)

#返回上三角矩阵
cormatrix *= np.tri(*cormatrix.values.shape,k=-1).T #设置下三角全为0
#下三角区域的元素将被设置为0。
#print(cormatrix)
#print(cormatrix)
#整合数据,整合为二维数据
cormatrix = cormatrix.stack()
#print(cormatrix)

#获取相关性最大的值(排序）
cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
print(cormatrix)

#重新指定列名
cormatrix.columns = ['d1','d2','d3']
#print(cormatrix)

#绘制相关性热力图
d = data2.corr() #特征相关性

mask = np.zeros_like(d,dtype=np.bool_)
#这行代码创建了一个与d具有相同形状的全零布尔数组mask

mask[np.triu_indices_from(mask)] = True
#np.triu_indices_from(mask)函数返回mask数组上三角部分的索引。
# 然后，这些索引被用来将mask数组的上三角部分设置为True。
f,ax= pl.subplots(figsize=(11,9))

sns.heatmap(d,mask=mask,square=True,linewidths=5,ax=ax,cmap='BuPu')
#使用Seaborn库的heatmap函数来绘制相关性热图。
"""* `d`: 要绘制的数据（相关性矩阵）。  
* `mask=mask`: 使用之前创建的`mask`来遮盖热图的上三角部分。  
* `square=True`: 使每个单元格为正方形。  
* `linewidths=5`: 设置单元格之间的线条宽度。  
* `ax=ax`: 在之前创建的坐标轴上绘制热图。  
* `cmap='BuPu'`: 设置颜色映射为'BuPu'，这是一种从蓝色到紫色的颜色映射。"""

sns.pairplot(data2,hue='ghcs',palette='BuPu')
#绘制数据集中成对变量之间的关系。pairplot函数会生成一个矩阵图，
# 其中对角线上的图是每个变量的直方图或KDE（Kernel Density Estimation）图，
# 非对角线上的图则是两个变量之间的散点图或回归图。
# 颜色的深浅主要是为了区分不同的类别或组，而不是表示具体的数量或度量值的大小

#查看两列数据之间的关系
data5 = pd.read_excel(r"C:\Users\29048\Desktop\逻辑回归1.xls") #导入数据

sns.lmplot(x='评分',y='评价人数',data=data5,hue='是否选择',col='是否选择',palette='plasma',fit_reg=True)
"""
'x': 用于绘制散点图的 x 轴变量。
'y': 用于绘制散点图的 y 轴变量。
data2: 包含你要绘制的数据的 Pandas DataFrame。
hue='': 可选参数，用于根据某一列的值将数据分组，并在图中以不同的颜色表示。
col='': 可选参数，用于将数据分成多个子图，每个子图对应 col 参数指定列的一个唯一值。
row='': 可选参数，也用于将数据分成多个子图，但这次是沿垂直方向。每个子图对应 row 参数指定列的一个唯一值。
palette='': 可选参数，用于设置 hue 参数指定的分组使用的颜色调色板。例如，你可以使用 'BuPu'、'plasma' 等 Seaborn 提供的调色板。
"""
pl.show()

#数据预处理
#当一个特征的方差比其他的要大得多，那么他可能支配目标函数，使估计者不能
#预期那样正确地从其他特征中学习，这需要我们先对数据进行缩放
#traget feature
target = data.jg #获取数据中jg列数据
regressors = [x for x in data.columns if x not in ['jg']] #获取数据列名
features = data.loc[:,regressors] #查找数据
#print(features)
#'ck','lcs','pl','ghcs','jg',
num = ['lcs','pl','ghcs'] #连续数据的索引名

#scale the data #连续值特征缩放
#StandardScaler 是从 scikit-learn中的 preprocessing 模块导入的一个类，用于特征缩放。
# StandardScaler 会将特征数据（即列或变量）缩放到均值为 0，标准差为 1 的分布。
standard_scale = StandardScaler()#创建一个 StandardScaler 对象的实例，并将其赋值给变量 standard_scale
features[num] = standard_scale.fit_transform(features[num])
#fit_transform 方法首先计算 features[num] 中数据的均值和标准差（即“拟合”数据）
#将缩放后的数据被重新赋值给 features[num]
#print(features)

#对分类属性进行one-hot编码*****
classes = ['jb','ryjh','fdj','csys','qdfs','lx']
dummies = pd.get_dummies(features[classes],dtype=int) # 创建独热编码的DataFrame
das = features.drop(classes,axis=1)# 从features中删除分类特征
das2 = dummies.join(das)# 将独热编码的数据与删除分类特征后的数据合并
features = das2# 将合并后的数据赋值回features
#print(features)
#划分数据集，分为训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=seed)

#Lasso回归
#多加了一个绝对值项来惩罚过大的系数，alphas=0那就是最小二乘法
pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
alphas = 2 ** np.arange(2,12)
scores = np.empty_like(alphas)
#创建了一个空的NumPy数组scores，其形状与alphas数组相同。
#msno.matrix(features.sample(50))        #查看缺失值.sample(120)随机显示120个数据
for i,a in enumerate(alphas):
    lasso = Lasso(random_state=seed)#创建了一个Lasso回归模型的实例
    #random_state=seed确保每次运行代码时，模型的随机性（如权重初始化）是一致的
    lasso.set_params(alpha=a)#我们为当前的Lasso模型实例设置了正则化强度为a
    lasso.fit(x_train,y_train)#使用训练数据x_train和y_train来拟合Lasso回归模型
    scores[i] = lasso.score(x_test,y_test)
    #使用测试数据x_test和y_test来评估拟合后的Lasso模型的表现，
    # 并将得分（通常是R^2得分，即决定系数）存储在scores数组的对应位置i上
    #print(scores)
lassocv = LassoCV(cv=10,random_state=seed)#创建了一个LassoCV对象
lassocv.fit(features,target)#使用features（特征矩阵）和target（目标变量向量）来拟合LassoCV模型
lassocv_sores = lassocv.score(features,target)#计算了模型在训练集x_train和y_train上的得分（通常是R^2得分，即决定系数）
lassocv_alpha = lassocv.alpha_#
print(lassocv_alpha)
pl.figure(figsize=(10,4))
pl.plot(alphas,scores,'-ko')
pl.axhline(lassocv_sores,color = 'b')
pl.xlabel('alpha')
pl.ylabel('score')
pl.xscale('log')
sns.despine(offset=15)
#print(lassocv_sores,lassocv_alpha)

#绘制Lasso回归模型中系数的条形图(影响的程度)
coefs = pd.Series(lassocv.coef_,index=features.columns)
coefs = pd.concat([coefs.sort_values().head(5),coefs.sort_values().tail(5)])
#前5个（head(5)）和后5个（tail(5)）系数
#使用pd.concat将这两个序列合并成一个新的序列。
# 这通常是为了可视化目的，展示那些具有最大和最小绝对值的系数。
pl.figure(figsize=(10,4))
coefs.plot(kind='barh',color='c')
#pl.show()

#建模
# 导入必要的库
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as pl

# 假设alphas, seed, x_train, y_train, x_test已经正确定义并准备好

# 创建LassoCV模型并拟合训练数据
model_ll = LassoCV(alphas=alphas, cv=10, random_state=seed)
model_ll.fit(x_train, y_train)
# 使用predict方法对测试集x_test进行预测，预测结果存储在y_pred_ll中
y_pred_ll = model_ll.predict(x_test)

# 绘制残差点散点图
pl.rcParams['figure.figsize'] = (6.0, 6.0)
# 这里使用训练集数据来绘制残差点图，通常更推荐用测试集或验证集来评估模型
preds = pd.DataFrame({'preds': model_ll.predict(x_train), 'true': y_train})
preds['residuals'] = preds['true'] - preds['preds']
preds.plot(x='preds', y='residuals', kind='scatter', color='r')



# 定义NES函数来计算均方误差
def NES(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


# 定义R2函数来计算R^2分数
def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2


# 计算测试集的均方误差和R^2分数
mse_test = NES(y_test, y_pred_ll)
r2_test = R2(y_test, y_pred_ll)

# 打印结果
#print(f"Test MSE: {mse_test}")
#print(f"Test R^2: {r2_test}")

# 创建包含测试集真实值和预测值的DataFrame
d = {"true": list(y_test), 'p': y_pred_ll}
c = pd.DataFrame(d)

# 通常这里你会想要对DataFrame 'c' 进行一些操作，比如查看、分析或保存，但这里没有后续代码
#print(c)
pl.show()