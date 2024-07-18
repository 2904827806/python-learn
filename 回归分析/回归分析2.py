#导入模块
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
import statsmodels.api as sms
def prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
    #计算样本总数
   """ num_examples = data.shape[0]
    data_processed = np.copy(data)

    #预处理4
    features_mean = 0
    features_deviation =0
    data_normalized = data_processed
    if normalize_data:
        pass
        #data_processed = data_normalized"""


pd.set_option('display.unicode.east_asian_width',True)#解决列不对齐
a = {'工资':[0.4,0.8,0.5,0.75,1.2],'年龄':[25,30,28,33,40],'额度':[2,7,3.5,5,8.5]}
data = pd.DataFrame(a)
b = [1,1,1,1,1]
#data = sms.add_constant(data)
data.insert(0,'a',b)
#print(data.shape)
#数据预处理
class LinearRegression:
    #初始化操作，传入相应的数据
    def __init__(self,data,labels,polynomiual_degree=0,sinusoid_degree = 0,normalize_data=True):
        #数据预处理（没有预处理代码，暂时忽略）
        # 1.对数据进行预处理
        # 2.获取所有特征个数
        # 3.初始化参数矩阵
        #prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0)
        self.data = data #x
        self.labels = labels #y
        self.sinusoid_degree = sinusoid_degree
        self.polynomiual_degree = polynomiual_degree
        self.normalize_data = normalize_data

        #获取列数
        num_features = self.data.shape[1]
        #print(num_features)
        #初始化特征矩阵
        self.theta = np.zeros((num_features))
        #print(self.theta)

    def train(self,alpha,num_iterations=500):
        # 训练模块，执行梯度下降
        # alpha学习率，num_iterations迭代次数
        '''
        :param alpha:  学习率
        :param num_iterations: 迭代次数
        '''
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history #返回theta值和损失值

    def gradient_descent(self,alpha,num_iterations):
        '''
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        '''
        #实际迭代
        cost_history = [] #损失值
        for i in range(num_iterations):
            self.gradient_Step(alpha) #参数更新
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_Step(self,alpha):
        '''
        :param alpha: 学习率
        '''
        #进行参数更新
        num_examples = self.data.shape[0]  #样本个数
        prediction = LinearRegression.hypothesis(self.data,self.theta) #预测值
        delta = prediction - self.labels #预测值减去真实值
        #参数跟新：批量梯度下降公式
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * np.dot(delta, self.data)
        self.theta = theta

    def cost_function(self,data,labels):
        #计算损失函数：误差的目标函数
        num_examples = data.shape[0]  # 样本个数
        dalta = LinearRegression.hypothesis(data,self.theta) - labels
        cost = (1 / 2) * np.dot(dalta.T, dalta)
        return cost[0][0] if cost.ndim == 2 else cost  # 处理一维或二维数组的情况


    @staticmethod
    def hypothesis(data,theta):
        #获取预测值
        prediction = np.dot(data,theta)
        return prediction

    def grt_cost(self,data,labels):
        #数据预处理
        return self.cost_function(data,labels)

    def predict(self):
        #预测得到回归值
        prediction = LinearRegression.hypothesis(data,self.theta)
        #print(prediction)





x1 = '工资'
x2 = '年龄'
y = '额度'
#获取训练和实验数据
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

#划分测试数据和验证数据
train_data = data.sample(frac=0.8) #随机选取百分之80的数据
test_data = data.drop(train_data.index)#选取剩下0.2的数据
x_train = train_data[[x1]].values
x_test = test_data[[x1]].values
#print(x.shape)
y_train = train_data['额度'].values
y_test = test_data['额度'].values

#获取数据列数
num_features = x_train.shape[1] #获取x的列数
#print(num_features)
#初始化特征矩阵
theta = np.zeros((num_features))

#print(theta)
#获取预测值
def hypothesis(data,theta):
    '''
    :param data:  x
    :param theta: 系数
    '''
    prediction = np.dot(data,theta)
    return prediction #预测值

#参数跟新
def gradient_step(alpha):
    global theta
    num_examples = x_train.shape[0] #获取x有多少个数据
    predictions = hypothesis(x_train,theta) #预测值
    dalta = predictions - y_train
    theta = theta - alpha*(1/num_examples)*np.dot(dalta,x_train)
    return theta
#计算损失函数
def cost_function(data,labels):
    num_examples = x_train.shape[0]
    dalta1 = hypothesis(data,theta) - labels
    cost = (1/2)*np.dot(dalta1.T,dalta1) #误差
    return cost[0][0] if cost.ndim == 2 else cost

#进行迭代计算
def gradient_descent(alpha,num_iteratioms):
    '''
    :param alpha: 学习率
    :param num_iteratioms: 迭代次数
    '''
    global theta #系数项
    cost_history = []
    for i in range(num_iteratioms):
        theta = gradient_step(alpha)
        cost_history.append(cost_function(x_train,y_train))
    print(cost_history)
    return cost_history
#梯度下降
def train(alpha,num_iterationns=500):
    cost_history = gradient_descent(alpha,num_iterationns)
    return theta,cost_history


'''plot_training_trace = go.Scatter(
    x=x_train[:,0].flatten(),
    y=x_train[:,1].flatten(),
    z=y_train.flatten(),
    name='s',
    mode='markers',
    marker={
        'size':10,
        'opacity':1,
        'line':{
            'color':'rgb(255,255,255)',
            'width':1
        }
    }
)'''
import matplotlib.pyplot as pl
#pl.scatter(x,y)

#pl.show()
'''pl.scatter(x_train,y_train,label='TRAIN DATA')
pl.scatter(x_test,y_test,label='TEST DATA',c='r')
pl.xlabel(input_param_name)
pl.ylabel(output_param_name)
pl.title('Happy')
pl.legend()


num_iterations = 500
alpha = 0.01
(thetas1,cost_history1) = train(alpha=alpha,num_iterationns=num_iterations)

linearRegression = LinearRegression(x_train,y_train)
(thetas,cost_history) = linearRegression.train(alpha=alpha,num_iterations=num_iterations)
prediction = hypothesis(x_train,thetas)
pl.plot(x_train,prediction)

pl.show()
print(cost_history1[0])
print(cost_history1[-1])
pl.plot(cost_history,c='r') #绘制损失曲线
pl.show()

#测试数量
prediction_num = 100
x_predictions =np.linspace(x_train.min(),x_train.max(),prediction_num)
y_predictions = np.linspace(y_train.min(),y_train.max(),prediction_num)
ls = LinearRegression(x_predictions,y_predictions)
thetas2,cost_history2 = ls.train(alpha=alpha,num_iterations=num_iterations)

pr = hypothesis(x_predictions,thetas2)
pl.scatter(x_predictions,y_predictions)
pl.plot(x_predictions,pr)
pl.show()'''

num_iterations = 50000
lenrning = 0.02
polynomial_degree = 15
sinusoid_degree = 15
normaliz_Data = True
linear_regression = LinearRegression(x_train,y_train,polynomiual_degree=polynomial_degree,sinusoid_degree=sinusoid_degree,normalize_data=normaliz_Data)
(theta1,cost_history) = linear_regression.train(lenrning,num_iterations)

'''def generate_sinysoids(dataset,sinusoid_degree):
    num_example = dataset.shape[0]
    sinusoids =np.empty((num_example,0))
    for degree in range(1,sinusoid_degree+1):
        sinusoid_feature = np.sin(degree*dataset)
        sinusoids = np.concatenate((sinusoids,sinusoid_feature),axis=1)
    return sinusoids'''