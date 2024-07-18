import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris #数据
from sklearn.tree import DecisionTreeClassifier,plot_tree #决策树分类算法
#导入数据
iris = load_iris()
#获取x，y数据
X = iris.data[:,2:] # petal length and width
y = iris.target
#创建决策树模型
tree_clf = DecisionTreeClassifier(max_depth=2)
#实例化模型对象
tree_clf.fit(X,y)
from sklearn.tree import export_graphviz
import graphviz
ase = export_graphviz(
    tree_clf,
    out_file=None,
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
#graphviz.Source(ase)
#graphviz.render('12512152152145214521452152145214521')
'''
export_graphviz函数
将训练好的决策树模型导出为 Graphviz 的 DOT 格式，
以便你可以使用 Graphviz 工具或其他兼容的图形工具来可视化这个决策树。
'''
#显示树状模型
from sklearn.tree import export_graphviz
import graphviz
dot = export_graphviz(
    tree_clf,#对应树模型
    out_file=None,#存放位置
    feature_names=iris.feature_names[2:],#特征名称
    class_names=iris.target_names,#标签
    rounded=True,
    filled=True
)
plot_tree(tree_clf)
#plt.show()
# 导出决策树为 Graphviz 的 .dot 格式
# 使用 graphviz 的 Python 库来渲染决策树
graph = graphviz.Source(dot)
graph.render("iris_tree")  # 这会生成一个名为 'iris_tree.png.pdf' 的 PDF 文件

#绘制决策边界
#导入渲染颜色模块
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    #1.构造新的数据:根据x地维度创建新的数据
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    #2.获取预测数据值
    y_pred = clf.predict(X_new).reshape(x1.shape)
    #3.设置渲染颜色
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    #4.绘制等高线
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        #5.绘制散点图
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.title('Decision Tree decision boundaries')

#plt.show()


#概率估计
'''
**估计类概率**
输入数据为：花瓣长5厘米，宽1.5厘米的花。
 相应的叶节点是深度为2的左节点，因此决策树应输出以下概率：
* Iris-Setosa 为 0％（0/54），
* Iris-Versicolor 为 90.7％（49/54），
* Iris-Virginica 为 9.3％（5/54）。
'''
#数据预测
y_pro = tree_clf.predict_proba([[5,1.5]])
print(y_pro)
y_predict = tree_clf.predict([[5,1.5]])
print(y_predict)

'''
决策树中的正则化
**DecisionTreeClassifier类**还有一些其他参数类似地限制了决策树的形状：
* min_samples_split（节点在分割之前必须具有的最小样本数），
* min_samples_leaf（叶子节点必须具有的最小样本数），
* max_leaf_nodes（叶子节点的最大数量），
* max_features（在每个节点处评估用于拆分的最大特征数）。
* max_depth(树最大的深度)
'''
from sklearn.datasets import make_moons
#make_moons用于生成一个非线性可分的数据集，形状类似于两个半月形
X,y = make_moons(n_samples=100,noise=0.25,random_state=53)
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4,random_state=42)
tree_clf1.fit(X,y)
tree_clf2.fit(X,y)

plt.figure(figsize=(12,4))
plt.subplot(121)
plot_decision_boundary(tree_clf1,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title('No restrictions')

plt.subplot(122)
plot_decision_boundary(tree_clf2,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title('min_samples_leaf=4')
plt.show()

#对数据的敏感
np.random.seed(6) #设置随机因子
Xs = np.random.rand(100, 2) - 0.5 #随机获取数据
#将数据进行转换，将满足条件的转为1其余全部转为0
ys = (Xs[:, 0] > 0).astype(np.float32) * 2
#数据旋转角度
angle = np.pi / 4
#旋转矩阵
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#旋转之后的数据
Xsr = Xs.dot(rotation_matrix)

tree_clf_s = DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs, ys)
tree_clf_sr = DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr, ys)
plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title('Sensitivity to training set rotation')

plt.subplot(122)
plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title('Sensitivity to training set rotation')

plt.show()

'''回归任务'''
#1.构建数据
np.random.seed(42)
m=200
X5=np.random.rand(m,1)
y5 = 4*(X5-0.5)**2
y5 = y5 + np.random.randn(m,1)/10
#2.创建模型:回归模型
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
tree_clf_hg = DecisionTreeRegressor(max_depth=3)
tree_clf_hg.fit(X5,y5)
#plot_tree(tree_clf_hg)
#3.展示树状图
from sklearn.tree import export_graphviz
import graphviz
dot1 = export_graphviz(
        tree_clf_hg,
        out_file=None,
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
# 使用 graphviz 的 Python 库来渲染决策树
a = graphviz.Source(dot1)
a.render('hg')

#
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X5, y5)
tree_reg2.fit(X5, y5)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)

plot_regression_predictions(tree_reg1, X5, y5)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)

plot_regression_predictions(tree_reg2, X5, y5, ylabel=None)
#绘制
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)

plt.show()
plt.figure(figsize=(14,8))
tree_reg3 = DecisionTreeRegressor(random_state=42)
tree_reg3.fit(X5, y5)
tree_reg4 = DecisionTreeRegressor(random_state=42,min_samples_leaf=10)
tree_reg4.fit(X5, y5)
plt.subplot(121)
plot_regression_predictions(tree_reg3, X5, y5)
plt.subplot(122)
plot_regression_predictions(tree_reg4, X5, y5, ylabel=None)
#绘制
plt.show()