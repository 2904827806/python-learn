'''AdaBoost : 加权平均值
跟上学时的考试一样，这次做错的题，是不是得额外注意，下次的时候就和别错了！'''
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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier

X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=0.6)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', alpha=0.6)
plt.show()

'''决策边界'''
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,cmap = custom_cmap,alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,cmap = custom_cmap2,alpha=0.8)
    plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
    plt.plot(X[:,0][y==1],X[:,1][y==1],'bs',alpha = 0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')

'''加权模型'''
m = len(X_train)
for subplot,learning_rate in ((121,1),(122,0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel='rbf',C=0.05,random_state=42)
        svm_clf.fit(X_train,y_train,sample_weight = sample_weights)
        '''sample_weight=sample_weights: 这是一个可选参数，
        允许你为训练集中的每个样本指定一个权重。'''
        y_pred = svm_clf.predict(X_train)
        #权重更新
        sample_weights[y_pred != y_train] *= (1+learning_rate)
        plot_decision_boundary(svm_clf,X,y)
        plt.title('learning_rate = {}'.format(learning_rate))
    if subplot == 121:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                   n_estimators = 200,
                   learning_rate = 0.5,
                   random_state = 42
)
ada_clf.fit(X_train,y_train)
plot_decision_boundary(ada_clf,X,y)
plt.show()

'''
Gradient Boosting
算法提升流程
'''
np.random.seed(42)
x0 = np.random.rand(100,1) - 0.5
y0 = 3*x0[:,0]**2 + 0.05*np.random.randn(100)
from sklearn.tree import DecisionTreeRegressor
tre_c1 = DecisionTreeRegressor(max_depth = 2)
tre_c1.fit(x0,y0)
y1 = y0 - tre_c1.predict(x0)
tre_c2 = DecisionTreeRegressor(max_depth = 2)
tre_c2.fit(x0,y1)
y2 = y1 - tre_c2.predict(x0)
tre_c3 = DecisionTreeRegressor(max_depth = 2)
tre_c3.fit(x0,y2)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tre_c1,tre_c2,tre_c3))
print(y_pred)
'''#回归图形'''


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pre = sum(regressor.predict(x1.reshape(-1,1))for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pre, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


plt.figure(figsize=(11,11))
plt.subplot(321)
plot_predictions([tre_c1], x0, y0, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tre_c1], x0, y0, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tre_c2], x0, y1, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tre_c1, tre_c2], x0, y0, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tre_c3], x0, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tre_c1, tre_c2, tre_c3], x0, y0, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.show()


from sklearn.ensemble import GradientBoostingRegressor,AdaBoostClassifier
grd = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0,
    random_state=41
)
grd.fit(x0,y0)
grd_1 = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 3,
                          learning_rate = 0.1,
                          random_state = 41
)
grd_1.fit(x0,y0)
grd_2 = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 200,
                          learning_rate = 0.1,
                          random_state = 41
)

grd_2.fit(x0,y0)
plt.figure(figsize = (11,4))
plt.subplot(121)
plot_predictions([grd],x0,y0,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(grd.learning_rate,grd.n_estimators))

plt.subplot(122)
plot_predictions([grd_1],x0,y0,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(grd_1.learning_rate,grd_1.n_estimators))
plt.figure(figsize = (11,4))

plt.subplot(121)
plot_predictions([grd_1],x0,y0,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(grd_1.learning_rate,grd_1.n_estimators))

plt.subplot(122)
plot_predictions([grd_2],x0,y0,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(grd_2.learning_rate,grd_2.n_estimators))
plt.show()


'''提前停止策略'''
from sklearn.metrics import mean_squared_error
X_train,X_val,y_train,y_val = train_test_split(x0,y0,random_state=49)
gbrt = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 120,
                          random_state = 42
)
gbrt.fit(X_train,y_train)

errors = [mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors) #找到最小均方误差的索引
#print(bst_n_estimators)
gbrt_best = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = bst_n_estimators,
                          random_state = 42
)
gbrt_best.fit(X_train,y_train)

min_error = np.min(errors)

plt.figure(figsize = (11,4))
plt.subplot(121)
plt.plot(errors,'b.-')
plt.plot([bst_n_estimators,bst_n_estimators],[0,min_error],'k--')
plt.plot([0,120],[min_error,min_error],'k--')
plt.axis([0,120,0,0.01])
plt.title('Val Error')

plt.subplot(122)
plot_predictions([gbrt_best],x0,y0,axes=[-0.5,0.5,-0.1,0.8])
plt.title('Best Model(%d trees)'%bst_n_estimators)
plt.show()

gbrt = GradientBoostingRegressor(max_depth = 2,
                             random_state = 42,
                                 warm_start =True
)

error_going_up = 0
min_val_error = float('inf') #无限大

for n_estimators in range(1,120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train,y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val,y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up == 5:
            break
#print (gbrt.n_estimators)
'''### Stacking（堆叠集成）'''
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
#划分测试集和验证集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
#随机森林
random_forest_clf = RandomForestClassifier(random_state=42)
#极度随机树
extra_trees_clf = ExtraTreesClassifier(random_state=42)
#支持向量机（SVM）的分类器，但它被设计用于处理线性可分的问题。
svm_clf = LinearSVC(random_state=42)
#多层感知器（MLP）分类器
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for indx,estimator in enumerate(estimators):
    estimator.fit(X_train,y_train)
    #print(estimator)
    X_val_predictions[:, indx] = estimator.predict(X_val)
print(X_val_predictions)

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)