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
from sklearn.datasets import make_blobs

'''#中心点'''
blob_centers = np.array(
    [[0.2,2.3],
     [-1.5,2.3],
     [-2.8,1.8],
     [-2.8,2.8],
     [-2.8,1.3]])


'''#为中心点的标准偏差'''
blob_std =np.array([0.4,0.3,0.1,0.1,0.1])
'''
生成随机的样本数据:make_blobs
make_blobs 函数的主要参数包括：
n_samples（整数，默认为100）：
要生成的样本数量。

centers（整数或数组形状为 [n_centers, n_features]，默认为3）：
要生成的簇的中心数量或中心坐标。

cluster_std（浮点数或数组形状为 [n_centers,]，默认为1.0）：
每个簇的标准偏差。

random_state（整数、RandomState实例或无，默认为None）：
确定随机数生成的随机数种子。
'''
X,y = make_blobs(n_samples=2000,centers=blob_centers,
                     cluster_std = blob_std,random_state=7)

plt.figure(figsize=(16,8))
def plot_clusters(X, y=None):
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
#plt.figure(figsize=(8, 4))  #设置画布大小
plot_clusters(X)
#plt.show()

from sklearn.cluster import KMeans #导入聚类模块
k = 5 #簇数
km = KMeans(n_clusters=k,random_state=42) #实例化模块
km.fit(X)
'''#获取预测值'''
y_predict = km.predict(X)
centers = km.cluster_centers_
centers_label = km.labels_
colors = np.array(['red','green','blue','yellow'])

def plot_clusters(X):
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1],c=centers_label, s=1)
    #plt.plot(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='r')
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
#plt.figure(figsize=(8, 4))  #设置画布大小
plot_clusters(X)
plt.scatter(centers[:,0],centers[:,1],marker='*',c='r')
#print(y_predict)
#print(km.labels_)
#print(km.cluster_centers_)

x_new = np.array([[0,2],[3,2],[-3,3],[-3,2.5]])
#print(km.predict(x_new))
#plt.show()

#绘制决策边界
#绘制散点数据
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='r', cross_color='k'):
    #绘制中心点数据
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)

'''#绘制决策边界'''
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()]) #预测结果值
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
#%%
plt.figure(figsize=(8, 4))
plot_decision_boundaries(km, X)
#plt.show()


'''#算法流程'''
km_1 = KMeans(n_clusters=6,init='random',n_init=1,max_iter=1,random_state=1)
km_2 = KMeans(n_clusters=6,init='random',n_init=1,max_iter=2,random_state=1)
km_3 = KMeans(n_clusters=6,init='random',n_init=1,max_iter=3,random_state=1)
'''
n_clusters=6：类别个数
init='random'：随机初始位置
n_init=1,#跑几次
max_iter=1#最大迭代次数
random_state=1：随机种子
'''
km_1.fit(X)
km_2.fit(X)
km_3.fit(X)
plt.figure(figsize=(10,8))
plt.subplot(321)
plot_data(X) #绘制散点
plot_centroids(km_1.cluster_centers_, circle_color='r', cross_color='k')
plt.title('update cluster_centers')

plt.subplot(322)
plot_decision_boundaries(km_1, X,show_xlabels=False, show_ylabels=False)
plt.title('Label')

plt.subplot(323)
plot_decision_boundaries(km_1, X,show_xlabels=False, show_ylabels=False)
plot_centroids(km_2.cluster_centers_)
plt.title('update cluster_centers')

plt.subplot(324)
plot_decision_boundaries(km_2, X,show_xlabels=False, show_ylabels=False)
plt.title('Label')

plt.subplot(325)
plot_decision_boundaries(km_2, X,show_xlabels=False, show_ylabels=False)
plot_centroids(km_3.cluster_centers_)
plt.title('Label')

plt.subplot(326)
plot_decision_boundaries(km_3, X,show_xlabels=False, show_ylabels=False)
plt.title('Label')
#plt.show()

'''#不稳定的结果'''
def plo_clusterer_comparision(c1,c2,X):
    c1.fit(X)
    c2.fit(X)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_decision_boundaries(c1,X)
    plt.subplot(122)
    plot_decision_boundaries(c2, X)

c1 = KMeans(n_clusters = 6,init='random',n_init = 1,random_state=11)
c2 = KMeans(n_clusters = 6,init='random',n_init = 1,random_state=19)
plo_clusterer_comparision(c1,c2,X)
#plt.show()

'''#评估方法:km.inertia_'''
#print(km.inertia_)
x_dist = km.transform(X)
#print(x_dist)
#print(km.labels_)
a = x_dist[np.arange(len(x_dist)),km.labels_]

#print(a)
dist_sum = np.sum(a**2)
#print(dist_sum)
#print(km.score(X))

'''#如何找到合适的K值'''
'''找到最佳簇数'''
kmeans_per_k = [KMeans(n_clusters = k).fit(X) for k in range(1,20)]
inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(8,4))
plt.plot(range(1,20),inertias,'bo-')
plt.axis([1,30,0,1300])
plt.show()

'''轮廓系数'''
'''
 ai: 计算样本i到同簇其他样本的平均距离ai。
 ai 越小，说明样本i越应该被聚类到该簇。
 将ai 称为样本i的簇内不相似度。
 
 bi: 计算样本i到其他某簇Cj 的所有样本的平均距离bij，
 称为样本i与簇Cj 的不相似度。
 定义为样本i的簇间不相似度：bi =min{bi1, bi2, ..., bik}
 
 s(i) = (b(i）-a(i))/max{a(i),b(i)}
 s(i)接近1，说明分类合理
  s(i)接近-1，说明分类不合理
   s(i)接近0，说明样本在两个簇的边界上
'''

'''轮廓系数'''
from sklearn.metrics import silhouette_score
si = silhouette_score(X,km.labels_)
silhouette_scores = [silhouette_score(X,model.labels_) for model in kmeans_per_k[1:]]
plt.figure(figsize=(8,4))
plt.plot(range(2,20),silhouette_scores,'bo-')
#plt.show()


'''Kmeans存在的问题'''
#1.先生成数据
x1,y1 = make_blobs(n_samples=1000,centers=((4,-4),(0,0)),random_state=42)
x1_ero = np.array([[0.374, 0.95], [0.732, 0.598]])
x1 = np.dot(x1,x1_ero)
x2,y2 = make_blobs(n_samples=250,centers=1,random_state=42)
x2 = x2 +[6,-8]
x3 = np.r_[x1,x2]#创建或连接数组
y3 = np.r_[y1,y2]
plot_data(x3)
#plt.show()
#2.对比实验：初始族点
kms_god = KMeans(n_clusters=3,init=np.array([[-1.5,2.5],[0.5,0],[4,0]]),n_init=1,random_state=42)
kms_bad = KMeans(n_clusters=3,random_state=42)
kms_god.fit(x3)
kms_bad.fit(x3)
#3.绘图
plt.figure(figsize=(14,8))
plt.subplot(121)
plot_decision_boundaries(kms_god,x3)
plt.title(f'good kmeans{kms_god.inertia_}')

plt.subplot(122)
plot_decision_boundaries(kms_bad,x3)
plt.title(f'bad kmeans{kms_bad.inertia_}')
#plt.show()

'''图像分割小例子'''
#1.读取图像数据
from matplotlib.image import imread
image = imread(r"C:\Users\29048\Desktop\红嘴鸥.jpg")
X = image.reshape(-1,image.shape[2])
#2.创建聚类
'''kmeans = KMeans(n_clusters=8,random_state=42)
kmeans.fit(X) #实例化对象
c5 = kmeans.cluster_centers_

#3.展示数据
#(760, 971, 4)
#根据标签找到中心点
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
#对比实验'''
segmented_imgs = []
n_Colors = [10,8,6,4,2]
for n_cluster in n_Colors:
    km9 = KMeans(n_clusters=n_cluster,random_state=42)
    km9.fit(X)
    segmented_img = km9.cluster_centers_[km9.labels_].reshape(image.shape)
    segmented_imgs.append(segmented_img.reshape(image.shape))
#4.绘图
plt.figure(figsize=(10,5))
plt.subplot(231)
plt.imshow(image)
plt.title('original image')
for idx,n_cluster in enumerate(n_Colors):
    plt.subplot(232+idx)
    plt.imshow(segmented_imgs[idx])
    plt.title('{}colors'.format(n_cluster))

plt.show()

