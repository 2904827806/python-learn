"""
一、聚类概念：
1.无监督分类（没有标签）
2.聚类：相似的东西分到一组
3.难点：如何评估，如何调参


二、K-MEANS算法
1.基本概念：
要得到簇的个数，需要指定k值
质心：均值，即向量各维取平均即可
距离的度量：常用欧几里得距离和余弦相似度（先标准化） :进行分类
优化目标：min.sum.sum（dist(c,x)**2)

2.工作流程
（1）：导入数据
（2）：随机根据簇的k值，生成不同的质心
（3）：遍历每一个点到不同质心的距离，进行分类
（4）：根据分类结果重新生成质心，更新质心位置
（5）：重复3，4步骤，直到质心位置不在发送变化

3.优缺点
优势：简单，快速，适合规范数据集
缺点：k值难确定，复杂度与样本呈现线性关系，很难发现任意形状的簇

三、DBSCAN算法
1.基本概念：
核心对象：若某个点的密度达到算法设定的阈值则其为核心点
领域的距离阈值：设定的半径r
直接密度可达:若某点p在点q的r领域内，则q是核心点则p-q直接密度可达
密度可达：若有一个点的序列q0,q1,....qk，对任意qi-qi-1是直接密度可达的，
则称q0-qk密度可达，这实际上是直接密度可达的’传播‘
密度相连：从某个核心点p出发，点q和点k都是密度可达，责称点q和点k三密度相连
边界点：属于某个类的非核心点，不能发展下线了
噪声点：不属于任何一个类的簇，从任一个核心点出发都是密度不可达

2.工作流程
参数：输入数据集
参数E：指定半径
minpts：密度阈值
半径E，可以根据k距离来设定：找突变点k距离：给定数据集P={p(i),i=0,1,2...n},
计算P（i）到集合D的子集s中所有点之间的距离，距离从小到大排序，d（k）就称为k距离
minpts：k距离中的k值，一般取小一些，多次尝试

3.优缺点
优点：不用指定簇个数，可以发现任意形状的簇，擅长找到离群点，两个参数就可以

缺点：高维数据有些困难，参数难选择，sklearn中效率慢
"""

#聚类
# 导入必要的库
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.cluster import KMeans #导入聚类模块
from sklearn.datasets import make_blobs
from pandas.plotting import scatter_matrix

# 随机生成三组二元正态分布随机数
def datas():
    np.random.seed(1234)
    mean1 = [0.5, 0.5]
    cov1 = [[0.3, 0], [0, 0.3]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
    """mean1 应该是一个指定均值（或期望）的数组或列表，
cov1 应该是一个指定协方差矩阵的二维数组，它必须是正定的，并且其形状为 (d, d)。
1000 是你想要生成的样本数量（n）。
np.random.multivariate_normal(mean1, cov1, 1000) 会返回一个形状为 (1000, d) 的数组。
转置操作实际上将两个特征分别作为行和列分开"""

    x, y = [], []
    mean2 = [0, 8]
    cov2 = [[1.5, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
    mean3 = [8, 4]
    cov3 = [[1.5, 0], [0, 1]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 1000).T
    for i in range(len(x1)):
        x.append(x1[i])
        y.append(y1[i])
    for i in range(len(x1)):
        x.append(x2[i])
        y.append(y2[i])
    for i in range(len(x1)):
        x.append(x3[i])
        y.append(y3[i])
    return x,y


# 导入数据
data = pd.read_excel(r"C:\\Users\29048\Desktop\啤酒数据.xls")
# 选择特征进行聚类
x = data[['calories', 'sodium', 'alcohol', 'cost']]

# 使用KMeans进行聚类
km = KMeans(n_clusters=3).fit(x)
km2 = KMeans(n_clusters=2).fit(x)

# 获取每个KMeans对象得到的簇标签并添加到数据集中
data['cluster'] = km.labels_
data['cluster2'] = km2.labels_

# 将数据进行排序（按cluster列）
data.sort_values('cluster', inplace=True)

# 打印数据
print(data.columns)

# 获取聚类中心
cluster_centers = km.cluster_centers_
#cluster_centers = pd.DataFrame(cluster_centers)
cluster_centers_2 = km2.cluster_centers_
#print(cluster_centers)
# 打印聚类中心的平均值（对于每个cluster）
#data.groupby('cluster').mean()
#data.groupby('cluster2').mean()

#获取中心点
#centers = data.groupby('cluster').mean().reset_index()
#设置画布大小
pl.rcParams['font.size'] = 14
#设置颜色
colors = np.array(['red','green','blue','yellow'])
#绘制聚类结果
pl.scatter(data['calories'],data['alcohol'],c=colors[data['cluster']])

#绘制中心点
pl.scatter(cluster_centers[:,0],cluster_centers[:,2],marker='*',c='black',s=300)

# 绘制散点图矩阵（仅针对聚类1）查看两两之间的分类
scatter_matrix(data[['calories', 'sodium', 'alcohol', 'cost']],c=colors[data['cluster']], alpha=1, figsize=(10, 10), s=100)
pl.suptitle('Scatter plot matrix with KMeans (n_clusters=3)')

pl.show()

#数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_scaler = pd.DataFrame(x_scaler,columns=['calories', 'sodium', 'alcohol', 'cost'])

#标准化数据之后继续进行聚类
km3 = KMeans(n_clusters=3).fit(x_scaler)
data['scaler_cluster'] = km3.labels_
x_scaler['scaler_cluster'] = km3.labels_
# 将数据进行排序（按scaler_cluster列）
data.sort_values('scaler_cluster', inplace=True)
x_scaler.sort_values('scaler_cluster', inplace=True)
print(data)
#获取中心点
cluster_centers3 = km3.cluster_centers_

#绘制聚类结果
pl.scatter(x_scaler['calories'],x_scaler['alcohol'],c=colors[x_scaler['scaler_cluster']])

#绘制中心点
pl.scatter(cluster_centers3[:,0],cluster_centers3[:,2],marker='*',c='black',s=300)
pl.show()

#轮廓系数:聚类评估
#有关轮廓系数的计算，可以直接调用sklearn子模块metrics中的函数，即silhouette_score。
# 需要注意的是，该函数接受的聚类簇数必须大于等于2。
# 下面基于该函数重新自定义一个函数，用于绘制不同k值下对应轮廓系数的折线图

from sklearn import metrics
score_scaled = metrics.silhouette_score(x,data.scaler_cluster)
score = metrics.silhouette_score(x,data.cluster)

scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(x).labels_
    score = metrics.silhouette_score(x,labels)
    scores.append(score)


pl.plot(list(range(2,20)),scores)
pl.title('轮廓系数')
pl.show()

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10,min_samples=2).fit(x)
data['cluster_db'] = db.labels_
data.sort_values('cluster_db',inplace=True)
print(data)


#DBSCAN算法
# encoding:utf-8
import matplotlib.pyplot as pl
import pandas as pd
import random
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')


# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt((np.power((t1[0] - t2[0]), 2) + np.power((t1[1] - t2[1]), 2)))
    return dis

# DBSCAN算法，Data参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    '''
    Data参数为数据集，
    Eps为指定半径参数，
    MinPts为制定邻域密度阈值
    '''
    num = len(Data)  # 点的个数
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]  # C为输出结果，默认是一个长度为num的值全为-1的列表
    k = -1  # 用k来标记不同的簇，k = -1表示噪声点
    while len(unvisited) > 0:
        p = random.choice(unvisited)  # 随机选择一个unvisited对象
        unvisited.remove(p) #从未访问列表中删除数据
        visited.append(p)#访问过列表中加入数据
        N = []  # N为p的epsilon邻域中的对象的集合
        for i in range(num): #遍历数据编号
            if (dist(Data[i], Data[p]) <= Eps):  # #如果距离小于等于指定半径
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts: #如果p领域内对象大于等于指定数量
            k = k + 1 #指定为簇k
            C[p] = k #将p划分为第k簇
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N: #遍历p直接密度可达点
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi]) <= Eps):
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
    return C

if __name__ == '__main__':
	# 数据集二：788个点
	dataSet = pd.read_csv(r"C:\Users\29048\Desktop\project\788points.txt")
	dataSet = dataSet.values.tolist()
	C = dbscan(dataSet, 2, 14)
	x, y = [], []
	for data in dataSet:
	    x.append(data[0])
	    y.append(data[1])
	pl.figure(figsize=(8, 6), dpi=80)
	pl.scatter(x, y, c=C, marker='o')
	pl.show()


#线性判别分析
#指定列名
feature_data = {i:label for i,label in zip(
    range(4),
    ('sepal length in cm',
     'sepal width in cm',
     'petal length in cm',
     'petal width in cm'
    )
)}

#导入数据
df = pd.read_table(r"C:\Users\29048\Desktop\iris.data.txt",header=None,sep=',')

#将上列指定列名赋予数据
df.columns = [l for i,l in sorted(feature_data.items())] + ['class label']
print(df.tail()) #打印后五位数据

from sklearn.preprocessing import LabelEncoder
#用于将标签（即分类变量）转换为从 0 到 n_classes-1 的整数

x = df[['sepal length in cm','sepal width in cm','petal length in cm', 'petal width in cm']].values
y = df['class label'].values

#将标签（即分类变量）转换为从 0 到 n_classes-1 的整数
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)+1

np.set_printoptions(precision=4)

#获取每类的均值
mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(x[y == cl],axis=0))
    print('Mean vector %s:%s\n' % (cl,mean_vectors[cl-1]))

#计算两个4x4维矩阵：类内散布矩阵（s_w）和类间散布矩阵(s-b)

#类内散布矩阵（s_w）
s_w = np.zeros((4,4))
for cl,mv in zip(range(1,4),mean_vectors):
    class_sc_mat = np.zeros((4,4))
    for row in x[y == cl]:
        row,mv = row.reshape(4,1),mv.reshape(4,1)
        class_sc_mat +=(row-mv).dot((row-mv).T)
    s_w += class_sc_mat
print('within-class scatter matrix:\n',s_w)

#类间散布矩阵(s-b)
overall_mean = np.mean(x,axis=0)
s_b = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):
    n = x[y == i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1)
    overall_mean = overall_mean.reshape(4,1)
    s_b += n*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
print('between-class scatter matrix:\n',s_b)

#求解矩阵特征值
eig_vals,eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
"""
s_w,s_b均为方阵
np.linalg.inv 是 NumPy 库中的一个函数，用于计算矩阵的逆。
np.linalg.eig 是 NumPy 库中的一个函数，用于计算给定方阵的特征值和特征向量

这段代码首先计算了 s_w 的逆矩阵与 s_b 的乘积，
然后计算了这个乘积矩阵的特征值和特征向量，
并将它们分别存储在 eig_vals 和 eig_vecs 中
"""
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('\nEigenvector{}:\n{}'.format(i+1,eigvec_sc.real))
    print('\nEigenvector{}:{:.2e}'.format(i + 1, eig_vals[i].real))
# 创建了一个列表 eig_pairs，它包含了特征值的绝对值（作为键）和对应的特征向量（作为值）的元组
eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i])for i in range(len(eig_vals))]

# 使用 sorted 函数根据特征值的绝对值进行排序（降序）
eig_pairs = sorted(eig_pairs,key=lambda k: k[0],reverse=True)

print('Eigenvalues in decreasing order:\n')

#根据特征值进行排序
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
#求特征值所占百分比
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue{0:}:{1:.2%}'.format(i+1,(j[0]/eigv_sum).real))

#选择前两维特征
w = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1)))
print('Matrix w:\n',w.real)

x_lda = x.dot(w)
assert x_lda.shape ==(150,2),'the matrix is not 150x2 dimensional'

'''def plot_step_lda():
    ax = pl.subplots(111)
    for label,marker,color in zip(range(1,4),('','s','o'),('blue','red','green')):
        pl.scatter(x=x_lda[:,0].real[y==label],y=x_lda[:,1].real[y==label],marker=marker,c=color,alpha=0.5,label=label_encoder[label][0])

    pl.xlabel('LD1')
    pl.xlabel('LD2')
    leg = pl.legend(loc='upper right',fancybox=True)
    leg.get_frame().set_alpha(0.5)
    pl.text('LDA:Iris projection onto the first 2 linear discriminants')

    pl.tick_params(axis='both',which='both',bottom='off',top='off',
                   labelbottom='on',left='off',right='off',labelleft='on')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    pl.grid()
    pl.tight_layout
    pl.show()
plot_step_lda()'''

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sklearn_lad = LDA(n_components=2)
x_lda_sklearn = sklearn_lad.fit_transform(x,y)

import matplotlib.pyplot as plt


def plot_step_lda(x, title):
    fig, ax = plt.subplots()  # 如果需要子图，这里可以创建，但这里只创建一个图形即可
    label_dict = ['setosa', 'versicolor', 'virginica']
    for label, marker, color in zip(range(1, 4), ('', 's', 'o'), ('blue', 'red', 'green')):
        ax.scatter(x=x[:, 0][y == label - 1], y=x[:, 1][y == label - 1], marker=marker, c=color, alpha=0.5,
                   label=label_dict[label - 1])

    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD2')
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax.text(0.5, 0.1, 'LDA: Iris projection onto the first 2 linear discriminants', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='off', right='off',
                   labelleft='on')
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    ax.grid(True)
    plt.tight_layout()
    plt.title(title)
    plt.show()
plot_step_lda(x_lda_sklearn,title='Default LDA via scikit-learn')
