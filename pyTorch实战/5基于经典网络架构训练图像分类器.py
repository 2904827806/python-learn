#安装各种库
#PyTorch库的主模块
import time
import torch
#这是PyTorch的神经网络模块，包含了构建神经网络所需的各种层和工具，如全连接层、卷积层、池化层等
from torch import nn, optim
#Dataset用于定义数据集，DataLoader用于批量加载数据。
from torch.utils.data import DataLoader, Dataset
#数学函数和数组操作功能
import numpy as np
# 用于数据分析和处理的库
import pandas as pd
# 用于绘制图表的库
import matplotlib.pyplot as plt
#transforms用于对图像进行预处理，datasets提供了常用的视觉数据集，models提供了预训练的视觉模型。
from torchvision import transforms, datasets, models
#用于读取和写入图像文件的库
import imageio
#用于处理警告信息
import warnings
#提供了生成随机数的功能
import random
#提供了对Python解释器及其环境的访问
import sys
#提供了深拷贝和浅拷贝的功能
import copy
#用于处理JSON数据
import json
#是一个用于处理图像的库。它提供了丰富的图像处理功能，如打开、保存、显示、转换、裁剪、缩放、旋转
from PIL import Image
import os

#一，数据预处理
#1.数据读取与预处理
data_dir = r"C:\Users\29048\Desktop\flower_data\flower_data" #数据存储位置
train_dir = data_dir + '\\'+'train'
valid_dir = data_dir + '\\'+ 'valid'


#2.制作数据源
#数据增强，扩大数据量
data_transforms = {
    #训练集做数据增强 ：扩大数据量
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转,-45--45度之间随机选择
        transforms.CenterCrop(224),  # 中心裁剪为224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        transforms.RandomGrayscale(p=0.025),  # 随机转换为灰度图像，概率为0.025 ,3通道加上R=G=B
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),  # 标准化，均值为[0.485,0.456,0.406]，标准差为[0.229,0.224,0.225]

    #验证集不做数据增强
    'valid': transforms.Compose([transforms.Resize(256),  # 调整大小为256x256
             transforms.CenterCrop(224),  # 中心裁剪为224x224
             transforms.ToTensor(),  # 转换为张量
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])}  # 标准化，均值为[0.485,0.456,0.406]，标准差为[0.229,0.224,0.225]


#2.制作训练数据集
batch_size = 32

# 创建一个字典，将训练集和验证集的数据集和对应的转换函数传入
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

# 创建一个字典，将训练集和验证集的数据集和对应的批量大小传入，并设置随机打乱
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train','valid']}

# 创建一个字典，用于存储训练集和验证集的大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','valid']}
#print(dataset_sizes)
# 获取训练数据集的类别名称
claa_name = image_datasets['train'].classes

#读取标签文件，获取类别名称
with open(r"C:\Users\29048\Desktop\flower_data\cat_to_name.json", 'r') as f:
    cat_to_name = json.load(f)

'''
标签：cat_to_name 
类别：claa_name
数据集大小：dataset_sizes
数据加载器：dataloaders
数据集：image_datasets
'''

'''#3.展示数据集的图片和标签
def im_convert(tensor):
 
    展示数据
    tensor: 图像数据

    # 将tensor转换为cpu，并克隆和分离
    image = tensor.to('cpu').clone().detach()
    # 将tensor转换为numpy数组，并去掉维度为1的维度
    image = image.numpy().squeeze()
    # 将numpy数组的维度从(3, H, W)转换为(H, W, 3)
    image = image.transpose(2, 1, 0) #改变维度顺序 由（0，1，2）--->(1,2,0)
    #print(image.shape)
    # 将图像的像素值从[0, 1]转换为[0, 255]，并乘以RGB通道的标准差，加上均值
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # 将图像的像素值限制在[0, 1]之间
    image = image.clip(0, 1)
    # 返回处理后的图像
    return image


# 创建一个大小为20x20的图像
fig = plt.figure(figsize=(20, 20))
# 设置图像的列数为4
colums = 8
# 设置图像的行数为2
rows = 4

# 创建一个迭代器，用于遍历验证集数据
dataiter = iter(dataloaders['valid'])
# 获取迭代器的下一个数据
inputs,classes = next(dataiter)
#print(inputs)
#print(classes)
# 遍历图像的每个位置
for idx in range(colums*rows):
    # 在图像中添加一个子图
    ax = fig.add_subplot(rows,colums,idx+1,xticks=[],yticks=[])
    # 设置子图的标题为类别名称
    # 检查是否存在对应的类别名
    class_idx = str(classes[idx].item())
    if class_idx in cat_to_name:
        ax.set_title(cat_to_name[class_idx])
    else:
        ax.set_title("未知类")
    plt.imshow(im_convert(inputs[idx]))
# 显示图像
#plt.show()'''


#二、加载models中的预训练模型，并且使用训练好的参数进行初始化
model_name = 'resnet'

#是否使用人家训练好的特征来做
feature_extract = True

#是否使用GPU
'''train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('GPU不可用')
else:
    print('GPU可用')'''

#三、定义训练函数

def set_parameters(model, feature_extracting):
    # 如果需要特征提取
    if feature_extracting:
        # 遍历模型的所有参数
        for param in model.parameters():
            # 将参数的requires_grad属性设置为False，即不需要计算梯度
            param.requires_grad = False
    # 否则
    else:
        # 遍历模型的所有参数
        for param in model.parameters():
            # 将参数的requires_grad属性设置为True，即需要计算梯度
            param.requires_grad = True


# 定义一个ResNet-152模型  152层的网络
model_ft = models.resnet152() #实例化模型
#print(model_ft)


#
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    '''
    :param model_name: 模型名称
    :param num_classes: 类别个数
    :param feature_extract: 是否使用人家训练好的特征来做
    :param use_pretrained:
    '''

    #选择合适的模型，不同模型店初始化方法稍微有点不同
    model_ft = None #初始化模型为无
    input_size = 0
    if model_name == "resnet":
        """ 
        Resnet152
        """
        # 获取预训练模型
        model_ft = models.resnet152(pretrained=use_pretrained) #use_pretrained = True决定是否使用预训练权重

        # 设置参数
        set_parameters(model_ft, feature_extract) #是否需要计算梯度

        # 获取模型的最后一层的输入特征数，也就是获取全连接层的输入特征数
        num_ftrs = model_ft.fc.in_features

        # 修改模型的最后一层为全连接层，输出特征数为num_classes
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """
        Alexnet
        """
        # 获取预训练模型
        model_ft = models.alexnet(pretrained=use_pretrained)
        # 设置参数
        set_parameters(model_ft, feature_extract)
        # 获取模型的最后一层的输入特征数
        num_ftrs = model_ft.classifier[6].in_features
        # 修改模型的最后一层为全连接层，输出特征数为num_classes
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """
        VGG11_bn
        """
        # 获取预训练模型
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        # 设置参数
        set_parameters(model_ft, feature_extract)
        # 获取模型的最后一层的输入特征数
        num_ftrs = model_ft.classifier[6].in_features
        # 修改模型的最后一层为全连接层，输出特征数为num_classes
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'inception':
        """
        Inception v3        
        Be careful, expects (299,299) sized images and has auxiliary output 
        """

        # 获取预训练模型
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # 设置参数
        set_parameters(model_ft, feature_extract)
        # 获取模型的最后一层的输入特征数
        num_ftrs = model_ft.AuxLogits.fc.in_features
        # 修改模型的最后一层为全连接层，输出特征数为num_classes
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        # 打印错误信息
        print("Invalid model name, exiting...")
        # 退出程序
        exit()

    return model_ft, input_size

#三、设置哪些层需要训练

model_fts, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

#gpu计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_fts = model_fts.to(device=device)

#模型保存
filename = 'checkpoint.pth'

#是否训练所有层
#获取模型参数
params_to_update = model_fts.parameters()
print('Params to learn:')

if feature_extract:
    # 如果是特征提取，则只更新最后一层
    params_to_upda = []
    for name,param in model_fts.named_parameters():
        if param.requires_grad == True:
            params_to_upda.append(param)
            print("\t",name)

else:
    # 如果是微调，则更新所有层
    for name,param in model_fts.named_parameters():
        if param.requires_grad == True:
                print("\t",name)

#四、优化器的设置
# 定义Adam优化器，参数为params_to_update，学习率为0.001
optimizer = torch.optim.Adam(params_to_update, lr=0.001)
# 定义学习率调度器，每7个epoch将学习率乘以0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# 定义损失函数为负对数似然损失函数
criterion = nn.NLLLoss()

# 五、训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False,filename=filename):
    """
    训练模型
    :param model: 模型
    :param dataloaders: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练轮数
    :param is_inception: 是否是Inception模型
    :param filename: 模型保存文件名
    """
    # 记录训练时间
    since = time.time()
    best_acc = 0

    # 将模型移动到指定的设备上
    model = model.to(device=device)

    # 用于存储验证集准确率的历史记录
    val_acc_history = []
    # 用于存储训练集准确率的历史记录
    train_acc_history = []
    # 用于存储训练集损失的历史记录
    train_losses = []
    # 用于存储验证集损失的历史记录
    valid_losses = []
    # 用于存储学习率的历史记录
    LRs = [optimizer.param_groups[0]['lr']]

    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs): #循环多少次
       # 打印当前epoch和总epoch数
       print('Epoch {} /  {}'.format(epoch, num_epochs))
       print('-' * 10)

       #  训练和验证模式
       for phase in ['train', 'val']:
           if phase == 'train':
               model.train()  # 设置模型为训练模式
           else:
               model.eval()   # 设置模型为评估模式、
           runing_loss = 0.0 #损失值
           runing_corrects = 0 #预测正确个数

           # 遍历每个batch
           for inputs, labels in dataloaders[phase]:#获取数据

               # 将输入和标签移动到GPU上
               inputs = inputs.to(device=device)
               labels = labels.to(device=device)

               # 清空梯度
               optimizer.zero_grad()

               preds = None

               #只有训练的时候计算和更新梯度
               with torch.set_grad_enabled(phase == 'train'): #使用torch.set_grad_enabled来控制是否启用梯度计算
                   # 如果是Inception模型，需要将输出转换为logits
                   # 如果is_inception为真且phase为'train'，则执行以下代码
                   if (is_inception == True) and (phase == 'train'):
                       # 使用模型对输入进行预测，得到outputs和aux_outputs
                       outputs, aux_outputs = model(inputs)
                       # 使用criterion计算outputs和labels之间的损失
                       loss1 = criterion(outputs, labels)
                       # 使用criterion计算aux_outputs和labels之间的损失
                       loss2 = criterion(aux_outputs, labels)
                       # 将loss1和0.4倍的loss2相加，得到最终的损失
                       loss = loss1 + 0.4*loss2

                   # 如果is_inception为假，则执行以下代码
                   else:
                       # 使用模型对输入进行预测
                       outputs = model(inputs) #前向传播

                       loss = criterion(outputs, labels) #计算损失值

                       # 如果是训练阶段，则计算梯度并更新模型参数
                   _, preds = torch.max(outputs, 1)

                   #训练阶段更新权重
                   if phase == 'train':
                       loss.backward() #反向传播
                       optimizer.step() #参数更新
                   #计算损失

               runing_loss += loss.item() * inputs.size(0)
               #print('数据',preds == labels)
               runing_corrects += torch.sum(preds == labels.data)

               #计算每个epoch的损失和准确率
               epoch_loss = runing_loss / dataset_sizes[phase]
               epoch_acc = runing_corrects.double() / dataset_sizes[phase]

               #计算每个epoch的运行时间
               time_em = time.time() - since
               print('Time elapsed: {:.0f}m {:.0f}s'.format(time_em // 60, time_em % 60))
               print('{}loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

               #得到最好的模型
               if phase == 'val' and epoch_acc > best_acc:
                   best_acc = epoch_acc
                   best_model_wts  = copy.deepcopy(model.state_dict())
                   state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'best_acc':best_acc}
                   torch.save(state,'./model.pth')
               if phase == 'val':
                   val_acc_history.append(epoch_acc)
                   valid_losses.append(epoch_loss)
                   scheduler.step(epoch_loss)
               if phase == 'train':
                   train_acc_history.append(epoch_acc)
                   train_losses.append(epoch_loss)
           print('Optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
           LRs.append(optimizer.param_groups[0]['lr'])
           print()
       time_em = time.time() - since
       print('Time elapsed: {:.0f}m {:.0f}s'.format(time_em // 60, time_em % 60))
       print('Best val Acc: {:4f}'.format(best_acc))

       #训练完后用最好的模型进行测试
       model.load_state_dict(best_model_wts)
       return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


#开始训练模型
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,criterion, optimizer, num_epochs=10)


#再继续训练所有层
for param in model_ft.parameters():
    param.requires_grad = True

#再训练所有的参数，学习率设置为0.0001
optimizer_fy = torch.optim.Adam(params_to_update, lr=0.0001) #优化器
scheduler_fy = torch.optim.lr_scheduler.StepLR(optimizer_fy, step_size=7, gamma=0.1) #学习率衰减策略

#损失函数
criterion = nn.NLLLoss()

#Load the checkpoint
checkpoint = torch.load(filename)
batch_acc = checkpoint['batch_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer_fy.load_state_dict(checkpoint['optimizer'])

model_fts, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,criterion, optimizer_fy, num_epochs=10)


#五、测试模型

#加载模型
model_ftss,input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

model_ftss = model_ftss.to(device)

filename = 'seriouscherjpoint.pth'
#测试模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ftss.load_state_dict(checkpoint['state_dict'])


#测试数据预处理
def process_image(img):
    # 打开图片
    img = Image.open(img)
    # 如果图片宽度大于高度，则将图片缩放到宽度为1000000，高度为256
    if img.size[0] > img.size[1]:
        img.thumbnail((1000000, 256))
    # 否则，将图片缩放到宽度为256，高度为1000000
    else:
        img.thumbnail((256, 1000000))
        # 计算左边距
        left_margin = (img.width - 224) / 2
        # 计算底部边距
        bottom_margin = (img.height - 224) / 2
        # 计算右边距
        right_margin = left_margin + 224
        # 计算顶部边距
        top_margin = bottom_margin + 224
        # 裁剪图片
        img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
        # 将图片转换为numpy数组，并归一化到0-1之间
        img = np.array(img) / 255
        # 定义均值
        mean = np.array([0.485, 0.456, 0.406])
        # 定义标准差
        std = np.array([0.229, 0.224, 0.225])
        # 将图片标准化
        img = (img - mean) / std
        # 将图片的维度从(H, W, C)转换为(C, H, W)
        img = img.transpose((2, 0, 1))



    return img

#展示数据
def imshow(image, ax=None, title=None):
    # 如果ax为空，则创建一个新的图像
    if ax is None:
        fig, ax = plt.subplots()
    # 将图像的通道顺序从（H,W,C）转换为（C,H,W）
    image = image.transpose((1, 2, 0))
    # 定义图像的均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 将图像的像素值从[0,1]转换为[-1,1]
    image = std * image + mean
    image = np.clip(image, 0, 1)
    # 在图像上显示处理后的图像
    ax.imshow(image)
    ax.set_title(title)
    return ax

#测试图片
image_path = r"C:\Users\29048\Desktop\flower_data\flower_data\valid\14\image_06082.jpg"
img = process_image(image_path)
imshow(img)







