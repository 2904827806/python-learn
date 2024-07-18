import numpy as np
import matplotlib.pyplot as pl #画图模块
import pandas as pd
from openpyxl import Workbook
wb = Workbook()
pd.set_option('display.unicode.east_asian_width', True)  # 解决列不对齐
#导入数据
r = pd.read_csv(r"C:\Users\29048\Desktop\creditcard.csv",delimiter=',',index_col=None,encoding='gbk')
data = pd.DataFrame(r)
#print(data.shape) #查看数据维度
#print(data.info()) #查看数据类型

#数据的列索引
"""['Time' 'V1' 'V2' 'V3' 'V4' 'V5' 'V6' 'V7' 'V8' 'V9' 'V10' 'V11' 'V12''V13' 'V14' 'V15' 'V16' 'V17'
'V18' 'V19' 'V20' 'V21' 'V22' 'V23' 'V24' 'V25' 'V26' 'V27' 'V28' 'Amount' 'Class']"""
#class = 1表示异常
#利用逻辑回归构建分类边界线

#统计数据中的class列的0，1数据量
count_class = pd.Series(data['Class']).value_counts(sort=True).sort_index()
pl.grid(axis='y')
count_class.plot(kind='bar',color='r') #kind= ’bar'表示绘制柱状图
#pl.show()


#当数据极度不均衡时
     #下采样，过采样
     #下采样，使样本一样少
     #过采样，使样本一样多

#数据进行缩放(归一化和标准化）
from  sklearn.preprocessing import StandardScaler
#StandardScale数据标准化模块，preprocessing数据预处理模块
stadac = StandardScaler()
data['newAmount'] = stadac.fit_transform(data['Amount'].values.reshape(-1,1))
#values.reshape(-1,1)则将该列重塑为一个二维数组，
# 其中每一行都包含一个来自'Amount'列的值。这是必要的，因为StandardScaler需要二维输入。
data = data.drop(['Time', 'Amount'],axis=1) #按列删除不用数据
data.insert(0,'1',1)#数据添加1列全为1的列

#获取自变量与因变量
x = data.loc[:,data.columns != 'Class']
y = data.loc[:,data.columns == 'Class']

#下采样
#1获取class= 1异常数据的个数和行索引
number_records_fraud = len(data[data.Class == 1]) #获取class=1的个数
fraud_indices = np.array(data[data.Class == 1].index)#获取class=1的行索引

#2正常数据的行索引
normal_indices = data[data.Class == 0].index#获取class=0的行索引

#3数据随机选择使正常和异常数据个数相同
random_normal_indices = np.random.choice(normal_indices,number_records_fraud,replace=False)
#normal_indices 数据源
# number_records_fraud选择数据个数
# replace=Falsereplace=False，则每个被选中的元素都不会被再次选择。这确保了选出的索引是不重复的。
random_normal_indices = np.array(random_normal_indices)#将获取的索引放到一维数组中

#4将选取的数据整合到列表中
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

#选取数据
under_sample_data = data.loc[under_sample_indices,:]
under_sample_data = pd.DataFrame(under_sample_data)

#重新选择自变量和因变量
x_undersample = under_sample_data.loc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:,under_sample_data.columns == 'Class']

#print(len(under_sample_data[under_sample_data.Class==0])/len(under_sample_data))
#print(len(under_sample_data[under_sample_data.Class==1])/len(under_sample_data))
#print(len(under_sample_data))


#交叉验证
#1选取测试样本和验证样本
from sklearn.model_selection import train_test_split #测试和验证选
#对原数据
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#下采样之后的数据
x_train_sample,x_test_sampl,y_train_sampl,y_test_sampl = train_test_split(x_undersample,y_undersample,test_size=0.3,random_state=0)

#建模操作
from sklearn.linear_model import LogisticRegression
#实现逻辑回归的类

from sklearn.model_selection import cross_val_score,KFold
#* `cross_val_score`：该函数用于执行交叉验证，以评估模型的性能。交叉验证是一种评估模型性能的统计方法，
# 其中数据被划分为k个“折”或子集，模型在k-1折上进行训练，然后在剩余的1折上进行评估。这个过程重复k次，每次选择不同的折作为测试集。
#* `KFold`：这是一个类，用于生成k折交叉验证的索引。通过`KFold`，你可以将数据分成k个不相交的子集，每个子集的大小大致相等。

from sklearn.metrics import confusion_matrix,recall_score,classification_report
#* `confusion_matrix`：该函数用于计算混淆矩阵，混淆矩阵是展示模型分类结果的一个表格，它显示了每个类别的真实标签和预测标签的数量。
#* `recall_score`：这个函数计算召回率（也叫查全率）。召回率是分类模型正确预测为正样本的实例占所有真正样本的比例。是一个重要的指标。
#* `classification_report`：这个函数返回关于主要分类指标的文本报告，包括每个类别的精确度、召回率、F1分数等。

def printing_Kfold_scores(x_train_data  ,y_train_data):
    fold = KFold(n_splits=5, shuffle=False) #获取交叉验证的类
    #len(y_train_data): 这是指定数据集中样本的总数。
    #5: 这是指定你想要创建的折的数量。在这里，数据将被分成5个部分，每次使用4个部分进行训练，剩下的1个部分进行验证。
    #shuffle=False: 这表示在创建折之前，数据不会被随机打乱。

    #惩罚系数:正则化惩罚项
    c_param_range = [0.01,0.1,10,100]

    #初始化结果表:
    results_table = pd.DataFrame(index=range(len(c_param_range)),columns=['C_parameter','Mean recall score'])
    print('r',results_table)
    results_table['C_parameter'] = c_param_range

    j = 0#循环找到最好的查询力度
    for c_param in c_param_range:
        print('-----------------------------------------------')
        print('惩罚系数:',c_param)
        print('------------------------------------------------')


        recall_accs = [] #查全率表
        for iteration,indices in enumerate(fold.split(x_train_data),start=1):
            # 使用特定的C参数调用逻辑回归模型
            # 参数 solver=’liblinear’ 消除警告
            # 出现警告：模型未能收敛 ，请增加收敛次数
            #  增加参数 max_iter 默认1000


            #这里实例化了一个逻辑回归模型对象lr
            lr = LogisticRegression(C=c_param,penalty='l1',solver='liblinear',max_iter=10000)

            #训练逻辑回归模型。
            #print('x0',indices[1])
            lr.fit(x_train_data.iloc[indices[0],:].values,y_train_data.iloc[indices[0],:].values.ravel())

            #这行代码使用训练好的模型lr来预测
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            #计算了预测结果y_pred_undersample与真实标签y_train_data.loc[indices[1],:].values之间的召回率
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('iteration:',iteration,'查全率:',recall_acc)
            #
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('-*'*25)
        print('平均查全率',np.mean(recall_accs))
        print()

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    best_c1 = results_table.loc[results_table['Mean recall score'].idxmax()]['Mean recall score']
    print('*************************************************************************************')
    print('最佳模型的C参数值:', best_c)
    print('最佳模型平均查全率:',best_c1)
    print('*************************************************************************************')

    return best_c

# 使用函数之前，请确保已经导入了所需的库，并且x_train_data和y_train_data是有效的数据。


# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels_name, title="Confusion Matrix",  is_norm=True,  colorbar=True, cmap=pl.cm.Blues):
    #* `cm`: 混淆矩阵，通常是一个二维数组。
    #* `labels_name`: 类别标签的列表或数组。
    #* `title`: 图的标题，默认为 "Confusion Matrix"。
    #* `is_norm`: 是否要对混淆矩阵进行归一化，默认为 True。
    #* `colorbar`: 是否显示颜色条，默认为 True。
    #* `cmap`: 用于图形的颜色映射，默认为蓝色系。

    if is_norm==True:
        #is_norm 为 True，那么混淆矩阵会按行（即每个类别的真实样本数）进行归一化，并保留两位小数。
        #求概率
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)  # 横轴归一化并保留2位小数

    pl.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):#在混淆矩阵的每个单元格上添加数字:
        for j in range(len(cm)):
            pl.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center') # 默认所有值均为黑色
            #annotate 函数在每个单元格上添加其对应的值。
            # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center') # 将对角线值设为白色
    if colorbar:
        pl.colorbar() # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    pl.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    pl.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    pl.title(title)  # 图像标题
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    #pl.show() # plt.show()在plt.savefig()之后
    #pl.close()

best_c = printing_Kfold_scores(x_train_sample,y_train_sampl)
import itemloaders
lr = LogisticRegression(C=best_c,penalty='l1',solver='liblinear',random_state=0)
lr.fit(x_train_sample.values,y_train_sampl.values.ravel())
y_pred_undersample = lr.predict(x_test.values)

cnf_matrix = confusion_matrix(y_test,y_pred_undersample)  #混淆矩阵
np.set_printoptions(precision=2)

class_names = [0,1]
pl.figure()
plot_confusion_matrix(cnf_matrix,class_names)


ir = LogisticRegression(C=best_c,penalty='l1',solver='liblinear',random_state=0)
ir.fit(x_train_sample.values,y_train_sampl.values.ravel())
y_pred_undersample = ir.predict_proba(x_test_sampl.values)

theat = [i/10 for i in range(1,10)]

pl.figure(figsize=(15,15))
j = 1
for i in theat:
    y_test_predictions_high_recall = y_pred_undersample[:,1] > i
    #print(y_test_predictions_high_recall)
    pl.subplot(3,3,j)
    j += 1
    cnf_matrix = confusion_matrix(y_test_sampl, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    print('rc',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, class_names,title=f'theat>{i}')

#pl.show()


#SMOTE算法
from imblearn.over_sampling import SMOTE  #过采样模块
#过采样
oversample = SMOTE(random_state=0) #实例化对象
lables = data['Class']

os_x,os_y = oversample.fit_resample(x_train,y_train)
print('*'*30)
os_x = pd.DataFrame(os_x)
os_y = pd.DataFrame(os_y)
print(len(os_y[os_y.values == 1]))
#best_c2 = printing_Kfold_scores(os_x,os_y)

jr = LogisticRegression(C=best_c,penalty='l1',solver='liblinear',random_state=0)
jr.fit(os_x.values,os_y.values.ravel())
y_pred_undersample = jr.predict(x_test.values)

cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
np.set_printoptions(precision=2)

class_names = [0,1]
pl.figure()
plot_confusion_matrix(cnf_matrix,class_names)
pl.show()







