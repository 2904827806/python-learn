import numpy as np
from utils.hypothesis import sigmoid,sigmoid_gradient #sigmoid的倒数函数
from utils.features import prepare_for_training #归一化
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        #1.数据预处理
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.label = labels
        #神经网络的层数
        self.layers = layers  #784像素点，25：隐层神经元，10 ：分类个数
        self.normalize_data = normalize_data #是否进行初始化
        #（1）权重参数初始化
        self.thetas = MultilayerPerceptron.thetas_init(layers)

    def train(self,max_iteration=1000,alpha=0.1):
        '''
        #2.训练模块
        :param max_iteration: 最大迭代次数
        :param alpha: 学习率
        '''
        #（1）数据矩阵拉长
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)
        #print(unrolled_theta)
        #（2）梯度下降
        (optimized_theta,cost_history) = MultilayerPerceptron.gradient_descnt(self.data,self.label,unrolled_theta,self.layers,max_iteration,alpha)
        self.thetas = MultilayerPerceptron.thetas_unroll(optimized_theta)
        #print('a')
        return self.thetas,cost_history
    def predict(self,data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        theta = MultilayerPerceptron.theta_roll(self.thetas,self.layers)
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed,theta,self.layers)
        return np.argmax(predictions,axis=1).reshape((num_examples,1))
    @staticmethod
    #梯度下降
    def gradient_descnt(data,label,unrolled_theta,layers,max_iteration,alpha):
        '''梯度下降'''
        optimized_theta = unrolled_theta #参数
        cost_history = [] #损失值集合
        #print('b')
        for _ in range(max_iteration):
            #损失函数
            cost = MultilayerPerceptron.cost_function(data,label,MultilayerPerceptron.theta_roll(optimized_theta,layers),layers)
            cost_history.append(cost)
            #梯度
            #print('p',optimized_theta)
            theta_gradient = MultilayerPerceptron.gradient_step(data,label,optimized_theta,layers)
            #参数更新
            optimized_theta = optimized_theta - alpha*theta_gradient
        return optimized_theta,cost_history

    @staticmethod
    #计算梯度
    def gradient_step(data,label,optimized_theta,layers):
        '''
        计算梯度
        :param data:数据
        :param label:标签
        :param optimized_theta:标准化的参数
        :param layers: 层数
        '''
        # 数据拉长的反变换
        theta = MultilayerPerceptron.theta_roll(optimized_theta,layers)
        #print('c')
        '''(1)反向传播走一次'''
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data,label,theta,layers)
        #print('f',thetas_rolled_gradients)
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        #print('d')
        return thetas_unrolled_gradients

    @staticmethod
    #反向传播
    def back_propagation(data, label, theta, layers):
        '''
             反向传播
        :param data: 数据
        :param label: 标签
        :param theta: 参数
        :param layers: 层数
        '''
        num_layers = len(layers)
        num_example = data.shape[0]
        num_features = data.shape[1]
        num_labels_types = layers[-1]
        deltas = {}
        for layer_index in range(num_layers-1):
            # 输入层
            in_count = layers[layer_index]
            # 输出层
            out_count = layers[layer_index + 1]
            #初始化数据
            deltas[layer_index] = np.zeros((out_count,in_count+1))
        for example_index in range(num_example):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index,:].reshape((num_features,1))
            layers_activations[0] = layers_activation
            #前向传播 ：逐层计算
            for layer_index in range(num_layers-1):
                #得到当前的权重参数
                layer_theta = theta[layer_index]
                #矩阵计算
                layers_input = sigmoid(np.dot(layer_theta,layers_activation))
                #完成激活函数并添加偏置参数
                layers_activation = np.vstack((np.array([[1]]),layers_input))
                #后一层计算结果
                layers_inputs[layer_index + 1] = layers_input
                # 后一层经过激活函数的计算结果
                layers_activations[layer_index + 1] = layers_activation
            output_layer_activation = layers_activation[1:,:]
            delta = {}
            #定义标签
            bitwise_label = np.zeros((num_labels_types,1))
            bitwise_label[label[example_index][0]] = 1
            # 计算输出层和最终的差异
            delta[num_layers-1] = output_layer_activation - bitwise_label
            #遍历循环L L-1 L-2 ..... 2
            for layer_index in range(num_layers - 2,0,-1):
                layer_theta = theta[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array([[1]]),layer_input))
                #按照公式进行计算
                delta[layer_index] = np.dot(layer_theta.T,next_delta)*sigmoid_gradient(layer_input)
                #过滤偏置参数
                delta[layer_index] = delta[layer_index][1:,:]
            #计算梯度值
            for layer_index in range(num_layers-1):
                layer_delta = np.dot(delta[layer_index+1],layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta
        for layer_index in range(num_layers-1):
            deltas[layer_index] = deltas[layer_index] * (1/num_example)
        return deltas

    @staticmethod
    #计算损失值
    def cost_function(data,label,theta_roll,layers):
        '''
        :param data: 数据
        :param label: 标签
        :param theta_roll:恢复后的参数
        '''
        num_layers = len(layers)
        num_example = data.shape[0] #特征个数
        num_labels = layers[-1] #类别个数
        '''(1)前向传播走一次'''
        predictions = MultilayerPerceptron.feedforward_propagation(data,theta_roll,layers)
        #print(predictions)
        '''(2)制作标签，每一个样本的标签都得是one_hot'''
        bitwise_labels = np.zeros((num_example,num_labels))
        for example_index in range(num_example):
            #制作标签
            bitwise_labels[example_index][label[example_index][0]] = 1
        #标签为1的损失值
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        # 标签为0的损失值
        bie_not_set_cost = np.sum(np.log(1-predictions[bitwise_labels == 0]))
        cost = (-1/num_example)*(bit_set_cost+bie_not_set_cost)
        #print(cost)
        return cost

    @staticmethod
    #前向传播
    def feedforward_propagation(data,thetas,layers):
        '''前向传播'''
        num_layers = len(layers)
        num_example = data.shape[0]
        #(1)输入数据
        in_layer_activation = data

        '''1.#逐层计算'''
        for layer_index in range(num_layers-1):
            theta = thetas[layer_index]
            #（2）隐层结果
            out_layer_activation = sigmoid(np.dot(in_layer_activation,theta.T))
            #（3）考虑偏置项，输出层
            out_layer_activation = np.hstack((np.ones((num_example,1)),out_layer_activation))
            #（4）转变输入赋值
            in_layer_activation = out_layer_activation
            #print(in_layer_activation)
        #(5)返回输出层结果(省去偏置项）
        return in_layer_activation[:,1:]

    @staticmethod
    # 数据拉长的反变换
    def theta_roll(unrolled_thetas,layers):
        '''
        #数据拉长的反变换
        :param thetas: 拉长之后的参数
        :param layers: 网络层数
        '''
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers-1):
            #print(layers)
            in_count = layers[layer_index]            # 输入层
            out_count = layers[layer_index + 1]            # 输出层
            thetas_width = in_count + 1            #列数
            thetas_height = out_count            #行数
            thetas_volunm = thetas_height * thetas_width  #总数
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volunm
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            #print('250',layer_theta_unrolled)
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height,thetas_width))
            unrolled_shift = unrolled_shift + thetas_volunm
        return thetas

    @staticmethod
    # 数据矩阵拉长
    def thetas_unroll(thetas):
        #数据矩阵拉长
        #(25,725) -- 25*725
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            #数据拼接
            unrolled_theta = np.hstack((unrolled_theta,thetas[theta_layer_index].flatten()))
        return unrolled_theta
    @staticmethod
    # 初始化权重参数
    def thetas_init(layers):
        #初始化权重参数
        num_layers = len(layers) #层的个数
        thetas = {}
        for layer_index in range(num_layers-1):
            '''
            输入layers为（784，25，10）
            会执行两次，得到两组参数矩阵 25*785 ，10*26
            '''
            #输入层
            in_count = layers[layer_index]
            # 输出层
            out_count = layers[layer_index+1]
            #这里需要考虑偏置项,要添加一列偏置系数，记住偏置个数与输出结果一致
            thetas[layer_index] = 0.05*np.random.rand(out_count,in_count+1)
        return thetas


data = pd.read_csv(r"C:\Users\29048\Desktop\机器学习\02.机器学习算法课件资料\部分代码资料\2-线性回归代码实现\线性回归-代码实现\data\mnist-demo.csv")
numbers_to_display = 25
#math.ceil取整
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for plot_index in range(numbers_to_display):
    #获取图像像素值
    digit = data[plot_index:plot_index+1].values
    digit_label = digit[0][0] #标签
    digit_pixels = digit[0][1:] #像素点值
    image_size = int(math.sqrt(digit_pixels.shape[0])) #图像大小
    frame = digit_pixels.reshape((image_size,image_size))
    plt.subplot(num_cells,num_cells,plot_index+1)
    plt.imshow(frame,cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.8,hspace=0.5) #间距
plt.show()

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
train_data = train_data.values
test_data = test_data.values
#划分数据
num_training_examples = 5000
x_train = train_data[:num_training_examples,1:]
y_train = train_data[:num_training_examples,[0]]
x_test = test_data[:,1:]
y_test = test_data[:,[0]]

layers = [784,25,10]
normalize = True
max_iterations = 500
alpha = 0.5
multilayerPerceptron = MultilayerPerceptron(data=x_train,labels=y_train,layers=layers,normalize_data=normalize)
(thetas,costs) = multilayerPerceptron.train(max_iteration=max_iterations,alpha=alpha)
plt.plot(range(len(costs)),costs)
plt.ylabel('Grident steps ')
plt.xlabel('costs')
plt.show()

y_train_predict = multilayerPerceptron.predict(x_train)
y_test_predict = multilayerPerceptron.predict(x_test)
train_p = (np.sum(y_train_predict == y_train) / y_train.shape[0]) * 100
test_p = (np.sum(y_test_predict == y_test) / y_test.shape[0]) * 100

print('训练集准确率',train_p)
print('测试集准确率',test_p)

#展示

plt.figure(figsize=(15,15))
for plot_index in range(numbers_to_display):
    #获取图像像素值
    digit_label = y_test[plot_index,0] #标签
    digit_pixels = x_test[plot_index,:] #像素点值
    predict_label = y_test_predict[plot_index][0]
    image_size = int(math.sqrt(digit_pixels.shape[0])) #图像大小
    frame = digit_pixels.reshape((image_size,image_size))
    color_map = 'Greens' if predict_label == digit_label else 'Reds'
    plt.subplot(num_cells,num_cells,plot_index+1)
    plt.imshow(frame,cmap=color_map)
    plt.title(predict_label)
plt.subplots_adjust(wspace=0.8,hspace=0.5) #间距
plt.show()
