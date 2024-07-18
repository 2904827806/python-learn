import torch
import numpy as np
#框架最厉害的一件事就是帮我们把反向传播全部计算好了
# 自动求导机制
import time

#tqdm(range(100)) #进度条
'''from tqdm import tqdm
for tqdm in tqdm(range(100)):
    time.sleep(0.1)'''

'''#1.需要求导的，可以手动定义
#方法1
x = torch.rand(3,4,requires_grad=True) #需要计算梯度的一个属性requires_grad=True
print(x)

#方法2
x = torch.rand(3,4)
x.requires_grad_(True) #在原地修改requires_grad属性
print(x)

b = torch.rand(3,4,requires_grad=True)
t = x + b
print(t)
y = t.sum()
print(y)
y.backward() #反向传播
# 计算y关于x的导数
print(x.grad) #y关于x的导数
#b.grad() #计算梯度

print(x.requires_grad,b.requires_grad,t.requires_grad,y.requires_grad) #True True True True'''


#列子

#计算流程
'''x = torch.rand(1)
b = torch.rand(1,requires_grad=True)
w = torch.rand(1,requires_grad=True)
y = x * w
z = y + b
print(x.requires_grad,b.requires_grad,w.requires_grad,y.requires_grad,z.requires_grad)


#反向传播计算
z.backward(retain_graph=True) #如果不加retain_graph=True，计算完一次梯度后，梯度会被清空
print(b.grad) #计算梯度
print(w.grad) #计算梯度'''



#做一个线性回归的例子

#数据准备
x_values = [i for i in range(11)]
x_train = np.array(x_values,dtype=np.float32)
x_train = x_train.reshape(-1,1)
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values,dtype=np.float32)
y_train = y_train.reshape(-1,1)
t1 = time.time()

#线性回归模型
# 导入PyTorch神经网络模块
import torch.nn as nn

# 定义一个线性回归模型
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        # 调用父类的初始化方法
        super(LinearRegression,self).__init__()
        # 定义一个线性回归模型，输入维度为input_dim，输出维度为output_dim
        self.linear = nn.Linear(input_dim,output_dim) #线性回归模型

    def forward(self,x):
        # 对输入x进行线性变换
        out = self.linear(x) #前向传播方式
        return out


input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim) #实例化模型
#print(model) #打印模型

#指定好参数和损失函数
epochs = 1000 #迭代次数
learning_rate = 0.01#学习率
optimizer = torch.optim.SGD(model.parameters(),# 获取模型的所有参数
lr=learning_rate) #创建一个优化器，作用是更新参数
criterion = nn.MSELoss() #损失函数

#训练模型

for epoch in range(epochs):
    epoch += 1

    #注意把数据转换为tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    #梯度要清零，每此迭代
    optimizer.zero_grad()  # 清空梯度

    #前向传播
    y_predicted = model(inputs) #预测值

    #计算损失
    loss = criterion(labels,y_predicted) #计算损失

    #反向传播
    loss.backward() #反向传播

    #更新权重参数
    optimizer.step() #更新参数

    if epoch % 50 == 0:
        print('epoch: ',epoch,'loss: ',loss.item())

#预测
predicted = model(inputs).data.numpy()# 使用模型对训练数据进行预测，并将预测结果转换为numpy数组
print(predicted)

#模型的保存与读取
torch.save(model.state_dict(),'model.pkl') #保存模型
model.load_state_dict(torch.load('model.pkl')) #读取模型
t2 = time.time()
print(t2-t1)


#model.eval() #评估模式
#model.train() #训练模式
#model(x_train) #预测
#model.zero_grad() #清空梯度
#model.parameters() #获取模型参数
#model.state_dict() #获取模型状态
#torch.save(model,'model.pth') #保存整个模型
#model = torch.load('model.pth') #读取整个模型


t3 = time.time()
#使用CPU进行训练
#只需要把数据和模型放到CPU上即可
import numpy as np
# 定义一个线性回归模型
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        # 调用父类的初始化方法
        super(LinearRegression,self).__init__()
        # 定义一个线性回归模型，输入维度为input_dim，输出维度为output_dim
        self.linear = nn.Linear(input_dim,output_dim) #线性回归模型



    def forward(self,x):
        # 对输入x进行线性变换
        out = self.linear(x)
        return out
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim) #实例化模型

#使用GPU；注意要把模型和数据都传入GPU
# 检查是否有可用的GPU设备
device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#print(model) #打印模型

#指定好参数和损失函数
epochs = 1000 #迭代次数
learning_rate = 0.01#学习率
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #优化器
criterion = nn.MSELoss() #损失函数

#训练模型

for epoch in range(epochs):
    epoch += 1
    #注意转换为tensor
    inputs = inputs.to(device)
    labels = labels.to(device)

    #梯度要清零，每次迭代
    optimizer.zero_grad()  # 清空梯度

    #前向传播
    y_predicted = model(inputs) #预测值

    #计算损失
    loss = criterion(labels,y_predicted) #计算损失

    #反向传播
    loss.backward() #反向传播

    #更新权重参数
    optimizer.step() #更新参数

    if epoch % 50 == 0:
        print('epoch: ',epoch,'loss: ',loss.item())

#预测
predicted = model(inputs).data.numpy() #将预测结果转化为numpy数组
print(predicted)
t4 = time.time()
print(t4-t3)