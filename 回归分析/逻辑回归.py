import numpy as np
from scipy.optimize import minimize
'''#scipy.optimize.minimize用于找到无约束或约束优化问题的最小值。
# 它提供了多种算法来求解不同类型的优化问题，
# 如线性规划、非线性规划、约束优化等。'''
from utils.features import prepare_for_training
#数据预处理

from utils.hypothesis import sigmoid
#sigmoid函数

class LogisticRegression: #定义逻辑回归类
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        #数据预处理
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=False)

        self.data = data_processed #x值
        self.labels = labels #y值

        # 找到数组labels中的所有唯一元素，并按排序顺序返回它们
        self.unique_labels = np.unique(labels)

        #平均值
        self.features_mean = features_mean

        #标准差
        self.features_deviation = features_deviation

        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        #计算有多少个特征
        num_features = self.data.shape[1]

        #分类个数=标签个数
        num_unique_labels = np.unique(labels).shape[0]

        # 初始化特征矩阵
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000): #输入最大迭代次数
        # 训练模块，执行梯度下降
        cost_histories = [] #不同的损失值列表
        num_features = self.data.shape[1] #特征个数
        #训练多个二分类模型
        for label_index, unique_label in enumerate(self.unique_labels):
            #获取当前类别的特征值
            current_initial_theta = np.copy(self.theta[label_index].ravel())
            #重新定义标签
            '''如果self.labels中的某个值与unique_label相等，则结果为True，
            然后转换为浮点数1；否则，结果为False，然后转换为浮点数0。'''
            current_lables = (self.labels == unique_label).astype(float)
            #astype(float) 指定数据类型为浮点数

            #梯度下降算法（从LogisticRegression类中调用）来训练当前类别的模型。
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data, current_lables,
                                                                                current_initial_theta, max_iterations)
            #更新参数和存储损失值
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
        #返回更新后的参数（所有类别的）和损失值列表
        return self.theta, cost_histories
    @staticmethod
    #梯度下降
    def gradient_descent(data,labels,current_initial_theta,max_iterations):
        '''
        :param data:  数据
        :param labels: 标签
        :param current_initial_theta: 初始化权重参数值
        :param max_iterations: 最大迭代次数
        :return:
        '''
        cost_history = []
        num_features = data.shape[1] #特征个数
        result = minimize(
            # 要优化的目标：fun 损失函数
            #列子：fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
            # lambda current_theta:LogisticRegression.cost_function(data,labels,current_initial_theta.reshape(num_features,1)),
            lambda current_theta: LogisticRegression.cost_function(data,
                                                                   labels,current_theta.reshape(num_features, 1)),
            # 初始化的权重参数
            current_initial_theta,
            # 选择优化策略
            #CG共轭梯度下降：
            method='CG',

            # 梯度下降迭代计算公式
            # jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_initial_theta.reshape(num_features,1)),
            jac=lambda current_theta: LogisticRegression.gradient_step(data,
                                                                       labels,current_theta.reshape(num_features, 1)),
            # 记录结果：记录损失值
            callback=lambda current_theta: cost_history.append(
                LogisticRegression.cost_function(data, labels, current_theta.reshape((num_features, 1)))),
            # 迭代次数:字典形式
            options={'maxiter': max_iterations}
        )
        if not result.success:#检查优化结果的成功性
            #如果优化过程没有成功（即 result.success 为 False），
            # 则代码会抛出一个 ArithmeticError 异常
            raise ArithmeticError('Can not minimize cost function' + result.message)
        #重新整形优化后的参数，并将其形状更改为 (num_features, 1)
        optimized_theta = result.x.reshape(num_features, 1)
        #返回优化后的参数和成本列表
        return optimized_theta, cost_history
    @staticmethod
    #损失函数
    def cost_function(data, labels,theat):
        #总数据量：m
        num_example = data.shape[0]
        #预测值
        predictions = LogisticRegression.hypothesis(data,theat)
        #类别为1的损失值
        y_is_set_cost = np.dot(labels[labels == 1],np.log(predictions[labels == 1]))
        # 类别为0的损失值
        y_is_not_set_cost = np.dot((1-labels[labels == 0]),np.log((1-predictions[labels == 0])))
        #损失值
        cost = (-1/num_example)*(y_is_set_cost+y_is_not_set_cost)
        return cost

    #预测值公式：h(x) = 1/(1+e**(-np.dot（data,theat))
    @staticmethod
    def hypothesis(data,theat):
        predictions = sigmoid(np.dot(data,theat))
        return predictions

    #梯度下降公式： (1/m) * (h(x)-y)*x
    @staticmethod
    def gradient_step(data,label,theat):
        num_example = data.shape[0]
        predict = LogisticRegression.hypothesis(data,theat)
        yi = predict - label
        gradients = (1/num_example)*np.dot(data.T,yi)
        #gradients.T.flatten()先把数据进行转至，再返回一个形状为 (m*n，）的一维数组。
        return gradients.T.flatten()

    def predict(self,data):#判断数据的类别
        num_examples = data.shape[0]
        #数据预处理
        data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        #预测概率值
        prob = LogisticRegression.hypothesis(data_processed,self.theta.T)
        #找到最大概率大索引
        max_prob_index = np.argmax(prob, axis=1)
        #将索引转换为类别标签
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            '''
            使用一个循环，遍历 self.unique_labels
            （这应该是一个包含所有可能类别的列表或数组），
            并使用枚举（enumerate）来同时获取索引和标签。
            '''
            class_prediction[max_prob_index == index] = label
        #返回预测结果：
        return class_prediction.reshape((num_examples, 1))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\5-逻辑回归代码实现\逻辑回归-代码实现\data\iris.csv")
#'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'
#print(np.unique(data['class']))
#花的类别
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']
x_axis = 'petal_length'
y_axis = 'petal_width'

for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[y_axis][data['class'] == iris_type],
                label=iris_type)
plt.show()
#数据总数
num_examples = data.shape[0]
#测试数据
x_train = data[[x_axis,y_axis]].values.reshape((num_examples,2))

y_train = data['class'].values.reshape((num_examples,1))
#最大迭代次数
max_iterations = 1000

polynomial_degree = 0
sinusoid_degree = 0

#类的实例化
logistic_regression = LogisticRegression(x_train,y_train,polynomial_degree,sinusoid_degree)
(theta, cost_histories) = logistic_regression.train(max_iterations)
labels = logistic_regression.unique_labels

#绘制loss曲线
plt.plot(range(len(cost_histories[0])),cost_histories[0],label=labels[0])
plt.plot(range(len(cost_histories[1])),cost_histories[1],label=labels[1])
plt.plot(range(len(cost_histories[2])),cost_histories[2],label=labels[2])
plt.show()

#获取预测结果
y_train_predctions = logistic_regression.predict(x_train)
print(y_train_predctions)
#准确率
precision = np.sum(y_train_predctions == y_train)/y_train.shape[0]*100
print(f'模型准确率：{precision}%')

#绘制决策边界
x_min = np.min(x_train[:,0])
x_max = np.max(x_train[:,0])
y_min = np.min(x_train[:,1])
y_max = np.max(x_train[:,1])

sampls = 150 #样本数
X = np.linspace(x_min,x_max,sampls)
Y = np.linspace(y_min,y_max,sampls)

#'SETOSA','VERSICOLOR','VIRGINICA'
#初始化z值
Z_SETOSA = np.zeros((sampls,sampls))
Z_VERSICOLOR = np.zeros((sampls,sampls))
Z_VIRGINICA = np.zeros((sampls,sampls))

#预测
for x_index,x in enumerate(X):
    for y_index,y in enumerate(Y):
        #预测
        data = np.array([[x,y]])
        #print(data.shape)
        prediction = logistic_regression.predict(data)[0][0]
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] = 1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1

for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(),0],
        x_train[(y_train == iris_type).flatten(),1],
        label = iris_type
                )

#绘制等高图 == 决策边界
plt.contour(X,Y,Z_SETOSA)
plt.contour(X,Y,Z_VERSICOLOR)
plt.contour(X,Y,Z_VIRGINICA)
plt.show()

#非线性决策边界
data1 = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\5-逻辑回归代码实现\逻辑回归-代码实现\data\microchips-tests.csv")

#类别标签
validities = [0,1]

#选择两个特征'param_1', 'param_2'
x_axis = 'param_1'
y_axis = 'param_2'

#绘制散点图

for validitie in validities:
    plt.scatter(
        data1[x_axis][data1['validity'] == validitie],
        data1[y_axis][data1['validity'] == validitie],
        label=validitie
    )
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')
plt.show()

#获取数据总数
num_examples = data1.shape[0]
#获取实验数据
x_train = data1[[x_axis,y_axis]].values.reshape((num_examples,2))
y_train = data1['validity'].values.reshape((num_examples,1))

#调试参数
#最大迭代次数
max_iterations = 100000
regularization_param = 0
polynomial_degree = 5 #对特征进行变换
sinusoid_degree = 0

#逻辑回归
logisticRegression = LogisticRegression(data=x_train,labels=y_train,polynomial_degree=polynomial_degree,sinusoid_degree=sinusoid_degree)

#训练
(thetas, costs) = logisticRegression.train(max_iterations)
colums = []

for theta_index in range(0,thetas.shape[1]):
    colums.append('thheta'+ str(theta_index))

#获取训练结果
labels = logisticRegression.unique_labels

#绘制损失下降结果
plt.plot(range(len(costs[0])),costs[0],label=labels[0])
plt.plot(range(len(costs[1])),costs[1],label=labels[1])

plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.legend()
plt.show()

#准确性
y_prediction = logisticRegression.predict(x_train)
pro = np.sum(y_prediction == y_train)/y_train.shape[0]*100
print('准确性:{}'.format(pro))

#绘制决策边界
num_examples = x_train.shape[0]
x_min = np.min(x_train[:,0])
x_max = np.max(x_train[:,0])
y_min = np.min(x_train[:,1])
y_max = np.max(x_train[:,1])

sampls = 150 #样本数
X = np.linspace(x_min,x_max,sampls)
Y = np.linspace(y_min,y_max,sampls)
z = np.zeros((sampls,sampls))
#预测
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        #为z的第x_index行，y_index列赋值
        z[x_index][y_index] = logisticRegression.predict(data)[0][0]

positives = (y_train == 1).flatten()
negatives = (y_train == 0).flatten()

#将决策边界映射到二维图像当中
plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='0')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='1')

plt.contour(X, Y, z)

plt.xlabel('param_1')
plt.ylabel('param_2')
plt.title('Microchips Tests')
plt.legend()

plt.show()









