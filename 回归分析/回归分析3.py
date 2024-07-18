import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as pl
pl.rcParams['axes.labelsize'] = 14
pl.rcParams['xtick.labelsize'] = 12
pl.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42) #设置随机因子
from sklearn.datasets import fetch_openml #机器学习中自动的数据

#数据读取
#mnist = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\2-线性回归代码实现\线性回归-代码实现\data\server-operational-params.csv")
#x,y = mnist[["Latency (ms)","Throughput (mb/s)"]], mnist["Anomaly"]
from sklearn.datasets import make_classification #获取数据的方法
from sklearn.model_selection import train_test_split #划分验证集和测试集

x, y = make_classification(
    n_samples=100000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)
#x = pd.DataFrame(X)
#y = pd.DataFrame(y)

'''import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
x = MinMaxScaler().fit_transform(X)'''
#划分测试数据和验证数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#进行写牌
shffle_index = np.random.permutation(70000) #随机打乱序号
#print(shffle_index)
x_train,y_train = x_train[shffle_index],y_train[shffle_index]

#交叉验证
from sklearn.model_selection import KFold

from sklearn.linear_model import SGDClassifier
#SGDClassifier 是一个使用随机梯度下降进行优化的线性分类器。
#实列化对象
sgd_clf = SGDClassifier(max_iter=5,random_state=42)
sgd_clf.fit(x_train,y_train)
#获取预测数据
data = sgd_clf.predict(x_test)

#对模型进行评价
from sklearn.model_selection import cross_val_score #模型准确率
a = cross_val_score(sgd_clf,x_train,y_train,cv=3,scoring='accuracy')
print(a)

from sklearn.model_selection import StratifiedKFold
#用于将数据集分割成多个训练/测试子集，
# 同时确保每个子集（或“fold”）中每个类别的样本比例与完整数据集中的比例大致相同
from  sklearn.base import clone #克隆

sKFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
'''n_splits=3将数据划分为三个子集，
shuffle=True，划分数据前先进行写牌操作
'''

for train_index, test_index in sKFold.split(x_train, y_train):
    # 克隆模型，确保每次迭代都是从头开始训练
    clone_clf = clone(sgd_clf)

    # 提取训练集和验证集
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_val_folds = x_train[test_index]  # 注意这里我们使用x_train而不是x_test
    y_val_folds = y_train[test_index]  # 同样，我们使用y_train而不是y_test

    # 训练模型
    clone_clf.fit(x_train_folds, y_train_folds)
    #获取预测值
    y_pred = clone_clf.predict(x_val_folds)
    #获取查  准率
    n_correct = sum(y_pred == y_val_folds)
    ps = n_correct/len(y_pred)
    #print(ps)

#混淆矩阵
# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels_name, title="Confusion Matrix",  is_norm=True,  colorbar=True, cmap=pl.cm.Blues):
    #* `cm`: 混淆矩阵，通常是一个二维数组。
    #* `labels_name`: 类别标签的列表或数组。
    #* `title`: 图的标题，默认为 "Confusion Matrix"。
    #* `is_norm`: 是否要对混淆矩阵进行归一化，默认为 True。
    #* `colorbar`: 是否显示颜色条，默认为 True。
    #* `cmap`: 用于图形的颜色映射，默认为蓝色系。

    if is_norm==True:
        #is_norm 为 True，那么混淆矩阵会按行（即每个类别的真实样本数）进行归一化，并保留两位小数。
        #求概率
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)  # 横轴归一化并保留2位小数

    pl.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):#在混淆矩阵的每个单元格上添加数字:
        for j in range(len(cm)):
            pl.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center') # 默认所有值均为黑色
            #annotate 函数在每个单元格上添加其对应的值。
            # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center') # 将对角线值设为白色
    if colorbar:
        pl.colorbar() # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    pl.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    pl.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    pl.title(title)  # 图像标题
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show() # plt.show()在plt.savefig()之后
    #pl.close()

from sklearn.model_selection import cross_val_predict#交叉验证预测的函数

#生成每个样本在交叉验证过程中的预测值。
y_train_pred = cross_val_predict(sgd_clf,x_train,y_train,cv=3)
from  sklearn.metrics import confusion_matrix#导入混淆矩阵
da = confusion_matrix(y_train,y_train_pred) #获取混淆矩阵
class_names = [0,1]
pl.figure()
plot_confusion_matrix(da,class_names)
#pl.show()
from sklearn.metrics import precision_score,recall_score,f1_score#准确性、召回率和F1分数。
#精度：准确率
p_S = precision_score(y_train,y_train_pred)
#召回率：查全率
r_s = recall_score(y_train,y_train_pred)
#print('ps',p_S)
#print('rs',r_s)
#调和平均数
f1 = f1_score(y_train,y_train_pred)
#print(f1)

#阈值对结果都影响

#获得得分值
dec_Df = sgd_clf.decision_function(x_train)
#print(dec_Df)

#设置阈值
t = 0.1
#划分类型
for (i,y_scoress) in enumerate(dec_Df):
    if y_scoress > t:
        dec_Df[i] = 1
    else:
        #dec_Df[y_scores] = 0
        dec_Df[i] = 0


#绘制roc曲线

y_scores = cross_val_predict(sgd_clf,x_train,y_train,cv=3,method='decision_function')
#print(y_scores)

#用于计算给定真实标签和预测概率的精度-召回曲线的数据点。这个函数在评估分类器的性能时特别有用
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train,y_scores)
#precision：对应于不同阈值的精度值。
#recall：对应于不同阈值的召回率值。
#thresholds：用于计算精度和召回率的阈值

#绘制精度-召回曲线。
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    pl.plot(thresholds,
             precisions[:-1],
             "b--",
             label="Precision")

    pl.plot(thresholds,
             recalls[:-1],
             "g-",
             label="Recall")
    pl.xlabel("Threshold", fontsize=16)
    pl.legend(loc="upper left", fontsize=16)
    pl.ylim([0, 1])


pl.figure(figsize=(8, 4)) #设置画布大小
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
pl.xlim([thresholds.min(),thresholds.max()]) #设置x轴的界限
pl.show()

#精度（Precision）对召回率（Recall）的曲线
def plot_precision_vs_recall(precisions, recalls):
    pl.plot(recalls,
             precisions,
             "b-",
             linewidth=2)

    pl.xlabel("Recall", fontsize=16)
    pl.ylabel("Precision", fontsize=16)
    pl.axis([0, 1, 0, 1])


pl.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
pl.show()


#绘制roc曲线
from sklearn.metrics import roc_curve
fpr,tpr,threshold = roc_curve(y_train,y_scores)
#真正率（TPR）和假正率（FPR），以及阈值
def plot_roc_curve(fpr, tpr, label=None):
    pl.plot(fpr, tpr, linewidth=2, label=label)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.axis([0, 1, 0, 1])
    pl.xlabel('False Positive Rate', fontsize=16)
    pl.ylabel('True Positive Rate', fontsize=16)

pl.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
pl.show()
from sklearn.metrics import roc_auc_score
rom = roc_auc_score(y_train,y_scores)
print(rom)
