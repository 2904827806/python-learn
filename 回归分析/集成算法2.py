import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
'''
投票机制：
1.硬投票：少数服从多数
2.软投票：各自分类器的概率值进行加权平均
'''

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
#获取数据集
X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
#划分验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha=0.6)
plt.plot(X[:,0][y==1],X[:,1][y==1],'bs',alpha=0.6)
plt.show()

#软投票和硬投票结果对比
#1.导入模块
'''决策树分类模块'''
from sklearn.tree import DecisionTreeClassifier
'''逻辑回归模块'''
from sklearn.linear_model import LogisticRegression
'''支持向量机模块'''
from sklearn.svm import SVC
'''RandomForestClassifier随机森林和VotingClassifier投票器模块'''
from sklearn.ensemble import RandomForestClassifier,VotingClassifier

#2.创建模型，并实例化
log_rg = LogisticRegression(random_state=42)
svc_rg = SVC(random_state=42,probability=True)
rand_tree = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
'''voting 参数的值为 'hard'，硬投票（即多数投票）。
软投票 voting 参数设置为 'soft'，'''
voc = VotingClassifier(estimators=[('lr',log_rg),('rand_tree',rand_tree),('svc',svc_rg),('rf',rf)],voting='hard')
voc.fit(X_train,y_train)

from sklearn.metrics import accuracy_score#模型预测的准确率
for cf in (log_rg,svc_rg,rf,rand_tree):
    cf.fit(X_train,y_train)
    y_prdct = cf.predict(X_test)
    print(f'{cf}模型的准确率为:  {accuracy_score(y_test,y_prdct)}')

y_1 = voc.predict(X_test)
print(accuracy_score(y_test,y_1))
#软投票
voc1 = VotingClassifier(estimators=[('lr',log_rg),('rand_tree',rand_tree),('svc',svc_rg),('rf',rf)],voting='soft')
voc1.fit(X_train,y_train)
y_12 = voc1.predict(X_test)
print(accuracy_score(y_test,y_12))

'''
Bagging策略
多个模型取平均值
- 首先对训练数据集进行多次采样，保证每次得到的采样数据都是不同的
- 分别训练多个模型，例如树模型
- 预测时需得到所有模型结果再进行集成
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_rf = BaggingClassifier(DecisionTreeClassifier(),
                n_estimators=500,#创建500个基分类器（即决策树）
                 max_samples = 100,#从原始数据集中抽取的最大样本数量100。
                  bootstrap = True,#当bootstrap为True时，采样过程是有放回的，
                  n_jobs = -1,#这个参数决定了用于计算的CPU核心数量。
                  random_state = 42)#这个参数设置了随机种子，
bag_rf.fit(X_train,y_train)
y_prdct1 = bag_rf.predict(X_test)
print(accuracy_score(y_test,y_prdct1))

te = DecisionTreeClassifier(random_state=42)
te.fit(X_train,y_train)
y_prdct12 = te.predict(X_test)
print(accuracy_score(y_test,y_prdct12))


#决策边界
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
   #1.构建数据
    xl1 = np.linspace(axes[0],axes[1],1000)
    xl2 = np.linspace(axes[2], axes[3], 1000)
    # 创建网格
    x1,x2 = np.meshgrid(xl1,xl2)
    # 将网格展平，并合并成新的数组
    x_new = np.c_[x1.ravel(),x2.ravel()]
   #2.获取预测值
    y_new = clf.predict(x_new).reshape(x1.shape)
   #3.设置渲染颜色
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
   #4.绘制等高线
    plt.contourf(x1,x2,y_new,cmap=custom_cmap,alpha=0.3)
   #5.绘制散点图
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0],'bo',alpha=0.6)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1],'ys',alpha=0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')
plt.figure(figsize = (12,5))
plt.subplot(121)
plot_decision_boundary(te,X,y)
plt.title('Decision Tree')
plt.subplot(122)
plot_decision_boundary(bag_rf,X,y)
plt.title('Decision Tree With Bagging')
plt.show()


'''### OOB策略 ：袋外数据'''
bag_rf1 = BaggingClassifier(DecisionTreeClassifier(),
                n_estimators=500,#创建500个基分类器（即决策树）
                 max_samples = 100,#从原始数据集中抽取的最大样本数量100。
                  bootstrap=True,#当bootstrap为True时，采样过程是有放回的，
                  n_jobs = -1,#这个参数决定了用于计算的CPU核心数量。
                  random_state = 42,#这个参数设置了随机种子，
                  oob_score=True )
bag_rf1.fit(X_train,y_train)
y_prdct13 = bag_rf1.predict(X_test)
#print(accuracy_score(y_test,y_prdct13))
'''#oob_score_ 属性，它表示袋外（Out-of-Bag）估计的分数'''
#print(bag_rf1.oob_score_)
'''属于各个类别的概率值'''
#print(bag_rf1.oob_decision_function_)

#随机森林
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

'''特征重要性'''
from sklearn.datasets import load_iris
iris = load_iris()
rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rf_clf.fit(iris['data'],iris['target'])
for name,score in zip(iris['feature_names'],rf_clf.feature_importances_):
    print (name,score)

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

rf_clf5 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rf_clf5.fit(mnist['data'],mnist['target'])
print(mnist['data'].shape)
#特征重要性热力图
def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image,cmap=matplotlib.cm.hot)
    plt.axis('off')

plot_digit(rf_clf5.feature_importances_)
char = plt.colorbar(ticks=[rf_clf5.feature_importances_.min(),rf_clf5.feature_importances_.max()])
char.ax.set_yticklabels(['Not important','Very important'])
plt.show()
# 导入sklearn库中的Pipeline模块,作用是将多个步骤组合成一个步骤
from sklearn.pipeline import Pipeline