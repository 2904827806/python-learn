import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
class KMeans:
    def __init__(self,data,num_clustres):
        '''
        :param data:  数据
        :param num_clustres: 族数==类别数
        '''
        self.data = data #数据
        self.num_clustres = num_clustres #类别数

    def train(self,max_iterations):
        #max_iterations 迭代次数
        #初始化质心，随机选择k个中心点
        # 1.先随机选择K个中心点
        centroids = KMeans.centroids_init(self.data,self.num_clustres)
        #计算不同点到中心点的距离
        # 2.开始训练
        num_example = self.data.shape[0]
        #选择最近中心点
        closest_centeroids_ids = np.empty((num_example,1))
        for _ in range(max_iterations):
            # 3得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centeroids_ids = KMeans.centroids_distict(self.data,centroids)
            ##4.进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data,closest_centeroids_ids,self.num_clustres)
        return centroids,closest_centeroids_ids


    @staticmethod
    def centroids_init(data,num_clustres):
        '''初始化中心点 '''
        num_example = data.shape[0]
        random_ids = np.random.permutation(num_example) #数据洗牌
        centroids = data[random_ids[:num_clustres],:]
        return centroids
    @staticmethod
    def centroids_distict(data,centroids):
        '''计算最小距离中心点的id'''
        num_example = data.shape[0] #数据个数
        num_centroids = centroids.shape[0] #中心点个数
        closest_centroids_ids = np.zeros((num_example,1))
        for id in range(num_example):
            distance = np.zeros((num_centroids,1))
            for kid in range(num_centroids):
                distance_diff = data[id,:] - centroids[kid,:]
                distance[kid] = np.sum(distance_diff**2)
            closest_centroids_ids[id] = np.argmin(distance)
            #np.argmin ：获取最小值所的索引
        return closest_centroids_ids
    @staticmethod
        #质心更新
    def centroids_compute(data,closest_centeroids_ids,num_clustres):
        num_feature = data.shape[1]
        #初始化质心数据
        centroids = np.zeros((num_clustres,num_feature))
        #循环不同的类别
        for centroid_id in range(num_clustres):
            closest_ids = closest_centeroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(),:],axis=0)
        return centroids


data = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\2-线性回归代码实现\线性回归-代码实现\data\iris.csv")
#'SETOSA' 'VERSICOLOR' 'VIRGINICA'
#sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'
# print(data.columns)
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']
x_axis ='petal_length'
y_axis ='petal_width'
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('known bq')
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class']==iris_type],data[y_axis][data['class']==iris_type],label = iris_type)
plt.legend()
plt.subplot(122)
plt.title('dont known bq')
plt.scatter(data[x_axis][:],data[y_axis][:])
plt.show()


num_example = data.shape[0]
x_train = data[[x_axis,y_axis]].values.reshape((num_example,2))
num_clusters = 3
max_iteritions = 50
km = KMeans(x_train,num_clusters)
centroids,closest_centeroids_ids = km.train(max_iteritions)

# 对比结果
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class']==iris_type],data[y_axis][data['class']==iris_type],label = iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1,2,2)
#进行聚类之后的结果
for centroids_id ,centroid in enumerate(centroids):
    a = (closest_centeroids_ids == centroids_id).flatten()
    plt.scatter(data[x_axis][a],data[y_axis][a],label=centroids_id)
for centroids_id ,centroid in enumerate(centroids):
    plt.scatter(centroid[0],centroid[1], label=centroids_id,marker='*')
plt.show()
#print(closest_centeroids_ids)

