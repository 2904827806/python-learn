#k_means 聚类分析
import pandas as pd
#导入数据
pd.set_option('display.max_rows',10)          #设置显示数据的行数最大为10行
dataA = pd.read_excel(r'C:\Users\29048\Desktop\DaPy_data.xlsx','BSdata')
dataB = dataA['身高']
dataD = dataA['体重']
dataF = dataA['支出']
print(dataB)


#转换数据
import numpy as np
data1 = np.array(dataB)   #将数组转换成列表
data2 = np.array(dataD)
data3 = np.array(dataF)

data = np.vstack((data1,data2,data3)).T
print(data)

'''dataC = [[i] for i in data ]   #将列表遍历 只有单要素需要进行
print(dataC)'''


from sklearn.cluster import KMeans
julei = KMeans(n_clusters=4)           #n_clusters 是用于聚类算法的参数,表示要将数据分为多少个簇(clusters)
julei.fit(data)                       #将数据dataC进行聚类

label = julei.labels_               #获得聚类标签
center = julei.cluster_centers_         #聚类中心
print(label)
print(center)

#有效性评价
#轮廓系数评价
from sklearn.metrics import silhouette_samples
lkxs = silhouette_samples(data,label)                 #求轮廓系数评价
print(lkxs)
means =np.mean(lkxs)
print(means)

#求解最佳聚类系数

def juleipingjia(n):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    julei = KMeans(n_clusters=n)  # n_clusters 是用于聚类算法的参数,表示要将数据分为多少个簇(clusters)
    julei.fit(data)  # 将数据dataC进行聚类
    label = julei.labels_  # 获得聚类标签
    center = julei.cluster_centers_  # 聚类中心
    lkxs = silhouette_samples(data, label,metric='euclidean')  # 求轮廓系数评价
    means = np.mean(lkxs)
    return means

y = []
for n in range(2,50):
    means = juleipingjia(n)
    y.append(means)
print(y)
print(y.index(max(y)))

