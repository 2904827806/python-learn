from sklearn.datasets import make_moons
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
#构建数据集
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
#绘制数据散点图
plt.plot(X[:,0],X[:,1],'b.')
#plt.show()
print(X.shape)
#创建dbscn算法
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.05,min_samples=5)
'''
eps=0.05:半径
min_samples=5：最小点
'''
dbscan.fit(X)
#获取类别
lab = dbscan.labels_
#获取核心对象缩影
cent = dbscan.core_sample_indices_
d = dbscan.components_


dbscan2 = DBSCAN(eps=0.2,min_samples=5)
dbscan2.fit(X)

#绘图操作
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    #用于标记DBSCAN中的核心样本
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    #标记噪声点
    anomalies_mask = dbscan.labels_ == -1
    #用于标记既不是核心样本也不是噪声点的样本。
    non_core_mask = ~(core_mask | anomalies_mask)

    #获取核心样本的中心点
    cores = dbscan.components_
    #print(cores.shape)
    #获取噪声点
    anomalies = X[anomalies_mask]
    #print(anomalies.shape)
    #获取既不是核心样本也不是噪声点的点。边界点
    non_cores = X[non_core_mask]
    #print(non_cores.shape)

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker="x")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=1)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=6, show_ylabels=False)
plt.show()
