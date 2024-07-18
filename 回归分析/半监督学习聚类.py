'''
首先，让我们将训练集聚类为50个集群， 然后对于每个聚类，
让我们找到最靠近质心的图像。 我们将这些图像称为代表性图像：
'''
from sklearn.datasets import load_digits
# 加载手写数字数据集
X_digits,y_digits = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
# 划分训练集和测试集
# 默认情况下，test_size=0.25，表示25%的数据将用作测试集，剩下的75%用作训练集
# random_state参数用于确保每次分割的结果都是一样的（可重复性）
X_train,X_test,y_train,y_test = train_test_split(X_digits,y_digits,random_state=42)

#加载逻辑回归模块
from sklearn.linear_model import LogisticRegression
#标签数目
n_labeled = 50
#创建逻辑回归模型
log_reg = LogisticRegression(random_state=42)
#实例化模型
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
#获取准确率
log_reg_score = log_reg.score(X_test, y_test)
print(log_reg_score)
#创建聚类模型
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#设置k值
k = 50
km = KMeans(n_clusters=k,random_state=42)
#获取每个点到聚类中心的距离
x_digits_dist = km.fit_transform(X_train)
#获取距离簇中心距离最小点的索引值
representative_digits_idx = np.argmin(x_digits_dist,axis=0)

#获取测试样本中的数据
X_representative_digits = X_train[representative_digits_idx]
#现在让我们绘制这些代表性图像并手动标记它们：
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

#plt.show()
#现在我们有一个只有50个标记实例的数据集，
# 它们中的每一个都是其集群的代表性图像，
# 而不是完全随机的实例。 让我们看看性能是否更好：
y_representative_digits = np.array([
                 9,2,6,0,1,7,3,9,5,8,
                 9,6,1,0,1,2,7,1,7,2,
                 1,5,9,8,8,3,5,3,2,4,
                 8,0,7,6,2,8,3,9,0,3,
                 2,7,2,3,4,7,1,2,8,4])
log = LogisticRegression(random_state=42)
log.fit(X_representative_digits,y_representative_digits)
lor_scor = log.score(X_test,y_test)
print(lor_scor)

#但也许我们可以更进一步：
# 如果我们将标签传播到同一群集中的所有其他实例，该怎么办？
y_train_propagated = np.empty(len(X_train), dtype=np.int32)

for i in range(k):
    y_train_propagated[km.labels_ == i] = y_representative_digits[i]
log_res1 = LogisticRegression(random_state=42,max_iter=5000, solver='sag')

log_res1.fit(X_train,y_train_propagated)
lor_sco2 = log_res1.score(X_test,y_test)
print(lor_sco2)

#只选择前20个来试试
nps = 20
x_cluster_dist = X_digits[np.array(len(X_train)),km.labels_]
for i in range(k):
    #找到的与聚类标签相同的数据
    in_cluster = (km.labels_ == i)
    # 选择属于当前簇的所有样本
    cluster_dist = x_cluster_dist[in_cluster]
    # 排序找到前20个
    cutoff_distance = np.percentile(cluster_dist, nps)
    # False True结果
    above_cutoff = (x_cluster_dist > cutoff_distance)
    x_cluster_dist[in_cluster & above_cutoff] = -1
partially_propagated = (x_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
log_reg3 = LogisticRegression(random_state=42,max_iter=10000, solver='sag')
log_reg3.fit(X_train_partially_propagated, y_train_partially_propagated)
lor_scor3 = log_reg3.score(X_test,y_test)
print(lor_scor3)