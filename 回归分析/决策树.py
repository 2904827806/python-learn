#一，决策树算法
"""
(一）树模型
1.决策树：从根节点开始一步步走到叶子节点
（1）所有数据最终都会落到叶子节点上，既可以做分类也可以做回归
（2）树的组成
                |  #根节点（最重要的，效果最好的）
         |      |      |  #中间过程
      |   |  |  |   |   |  | #叶节点

（3）节点
   增加节点相当于在数据中切了一刀

2.训练和测试
（1）训练阶段：从给定的训练集构造出一棵树
（2）测试阶段：根据构造出来的树模型从上到下走一遍

（3）如何切分特征
问题：根节点的选择该用什么特征
目标：通过一种衡量标准，来计算通过不同的特征进行分支选择后的分类情况，
找出最好的那个当根节点，以此类推

3.衡量标准：熵（表示随机变量的不确定性的度量）
import numpy as np
公式：H(x) = - sum(pk*np.log2pk)
熵值越小，集合的纯净度越高(集合中类别越少)

4.信息增益：表示特征x使得类Y的不确定性减少点程度。
（分类后的专一性，希望分类后的结果是同类在一起）
公式：
Gain(D,a) = H(x) - sum((Dv/D)*H(DV))
(Dv/D)表示分支节点的权重，H（Dv)表示Dv的信息熵

5.信息增益率
6.基尼系数

连续值怎么办：
    贪婪算法
        排序，离散化：二分

7.决策树剪枝策略
（1）剪枝策略：预剪枝，后剪枝
    预剪枝：边建立决策树边进行剪枝的操作（更实用）
    后剪枝：当建立完决策树后进行剪枝操作
"""
"""import matplotlib.pyplot as pl
import pandas as pd

data = pd.read_excel()
#print(data.columns.values)
#['outlook' 'temperature' 'humidity' 'windy' 'play']
from sklearn import tree #树模型
dtr = tree.DecisionTreeClassifier(max_depth=2)
#max_depth用于设置决策树的最大深度。深度为2意味着决策树最多有两层分裂
classes = data.columns.values
#print(classes)
data = pd.get_dummies(data[classes],dtype=int)
from sklearn.model_selection import train_test_split
x = data.loc[:,data.columns != 'play']
y = data.loc[:,data.columns == 'play']
x_tn,x_ts,y_tn,y_ts = train_test_split(x,y) #实验和测试样本划分
dtr.fit(x,y)
#可视化展示
dot_data = tree.export_graphviz(dtr,out_file=None,feature_names=data.columns.values,filled=True,impurity=False,rounded=True)
"""
def tree():
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    from sklearn import tree
    from sklearn.datasets import load_iris  # 机器学习库中包含的数据
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    iris = load_iris()
    x = iris.data
    x = pd.DataFrame(x, columns=['v1', 'v2', 'v3', 'v4'])  # 添加列索引
    # print(x)
    y = iris.target
    y = pd.DataFrame(y, columns=['jg'])  # 添加列索引
    # print(y)
    name = [i for i in x.columns.values]  # 获取索引名称

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)  # 划分实验集和验证集
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)  # 实例化对象
    clf.fit(x_train, y_train)  # 模型拟和
    print(clf.score(x_test, y_test))  # 输出模型准确值
    # print(iris.data[:,[0,1,2,3]])
    tree.plot_tree(clf)  # 查看树状图
    plt.show()

    # 决策树可视化
    '''dor_data = tree.export_graphviz(clf,out_file=None,feature_names=name,filled=True,impurity=False,rounded=True)
    import pydotplus
    graph = pydotplus.graph_from_dot_data(dor_data)
    from  IPython.display import Image,display
    # 创建PNG格式的图像，并使用IPython的Image来显示
    image = Image(graph.create_png())  # 注意这里调用了create_png()方法
    display(image)  # 使用display函数来显示图像'''

    '''clf = DecisionTreeClassifier(max_depth=2, random_state=0) #实例化对象
    clf.fit(X_train, y_train)  #模型拟和
    n_nodes = clf.tree_.node_count #这个属性表示决策树中节点的总数
    children_left = clf.tree_.children_left #这是一个数组，表示每个非叶子节点（即决策节点）的左孩子节点的索引
    children_right = clf.tree_.children_right#表示每个非叶子节点的右孩子节点的索引
    feature = clf.tree_.feature #这是一个数组，表示每个非叶子节点用于分割数据的特征的索引
    threshold = clf.tree_.threshold#这是一个数组，表示每个非叶子节点用于分割数据的阈值。
    values = clf.tree_.value#表示每个叶子节点中每个类别的样本数

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=values[i]
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=values[i],
                )
            )

    tree.plot_tree(clf)
    plt.show()'''

    # 自动参数选择
    from sklearn.ensemble import RandomForestRegressor
    # 用于回归问题的随机森林算法的实现
    from sklearn.tree import DecisionTreeRegressor
    # 用于回归问题的决策树算法的实现
    dtr = DecisionTreeRegressor(random_state=42)  # 实例化决策树模型
    dtr.fit(x_train, y_train)  # 模型拟和
    # print(dtr.score(x_test,y_test))
    rfr = RandomForestRegressor(random_state=42)  # 实例化随机森林
    rfr.fit(x_train, y_train)  # 模型拟和
    # print(rfr.score(x_test,y_test))

    # 自动参数选择
    '''from sklearn.model_selection import GridSearchCV
    tree_pargram_grid = {'max_samples_split':list((3,6,9)),'n_estimators':list((10,50,100))}
    grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_pargram_grid,cv=5)
    grid.fit(x_train,y_train)
    print(grid.scorer_,grid.best_params_,grid.best_score_)'''

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV  # 寻找最佳参数组合模块
    # 更改参数网格，移除不适用的'max_samples_split'，可以替换为其他有效参数，如'max_features'
    tree_param_grid = {'max_features': ['auto', 'sqrt', 'log2'], 'n_estimators': [10, 50, 100],
                       'min_samples_split': [i for i in range(10)]}

    # 初始化GridSearchCV，传入随机森林回归器和参数网格
    grid = GridSearchCV(rfr, param_grid=tree_param_grid, cv=5)  # cv表示交叉验证几次，param_grid想要调剂参数

    # x_train和y_train是已经准备好的训练数据和标签
    # 在实际环境中，取消下面这行的注释来拟合数据
    grid.fit(x_train, y_train)  # 模型拟和

    # 打印默认的评分器、最佳参数组合和最高得分
    # 注意：这些值只有在调用grid.fit()之后才有效
    a = [grid.best_params_, grid.best_score_]
    print(a)

#集成算法 ---- 随机森林

#集成算法
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
"""
目的：让机器学习效果更好，
bagging：训练多个分类器取平均 fx = 1/m * sum(fm(x))
典型代表：随机森林
    随机：数据采样随机，特征选择随机
    由于二重随机性，使得每个树基本上都不会一样，最终的结果也会不一样
    优点:1.能处理高维数据，且不用做特征选择
         2.主要是训练完之后能给出那些数据比较重要
         3.可视化展示，便于分析
    
boosting:从弱学习器开始加强，通过加强来进行训练
典型代表：AdaBoost、Xgboost
    AdaBoost：会根据前一次的分类效果调整数据权重
    1.如果某一个数据在这次分错了，下次就给它更大的权重
    2.每个分类器会根据自身的准确性来确定各自的权重，在合体

stacking：聚合多个分类或回归模型(可以分阶段来做)
#很暴力，各种分类器进行组合
可以堆叠各种各样的分类器
分阶段：第一阶段得出各自结果，第二阶段再用前一阶段结果训练
为了结果不择手段
"""
