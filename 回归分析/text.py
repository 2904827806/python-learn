# 回归分析
# 回归分析是寻找存在相关关系变量之间的数学表达式，并进行统计推断的一种统计方法

# 线性回归 :根据数据反馈一个值 #找到最合适的拟和曲线
# 分类：反馈一个类别
import pandas as pd

'''import matplotlib.pyplot as pl
import xlrd2 #读取数据模块
from openpyxl import Workbook #创建数据模块
def input_data():
    #通过字典添加数据
    a = [['工资','年龄','额度'],[4000,25,20000],[8000,30,70000],[5000,28,35000],[7500,33,50000],[12000,40,85000]]
    wb = Workbook()
    sh = wb.create_sheet('数据',0)
    del wb['Sheet']
    for i in a:
        sh.append(i)#往excl中写入数据
    wb.save('线性回归1.xls')

def huq():
    wr = xlrd2.open_workbook(r"C:\\Users\29048\Desktop\线性回归1.xls") #读取数据
    sh = wr.sheet_by_index(0)
    for i in range(sh.nrows):
        for j in range(sh.ncols):
            a = sh.cell_value(i,j)
            print(a,end='\t')
        print()

def data():
    #设置对齐
    pd.set_option('display.unicode.east_asian_width',True)#解决列不对齐
    re = pd.read_excel(r"C:\\Users\29048\Desktop\线性回归1.xls",sheet_name='数据') #读取数据
    data = pd.DataFrame(re,index=None,columns=None)#将数据转为二维数据
    a = [1,1,1,1,1]
    data.insert(0,'1',a) #在第一列插入数据
    return data
'''
"""
#列子：预测（找到最合适的一条线来最好的拟合曲线）
数据：工资和年龄（2个特征）自变量
目标：预测银行会贷款多少钱给我（标签）因变量
考虑：工资和年龄都会影响贷款，那么他们的影响有多大（参数） 
f(x) = a1x1 +a2x2 +a0  :c是误差
整合f(x) = (a**t) *x
加入一列全为1的数据是为了整合a0

#真实值与预测值之间肯定存在差异（用误差表示）
    误差是'独立并具有相同的分布'，并且服从均值为0，方差为&^2的高斯分布
    似然函数：什么样的参数跟我们的数据组合后恰好是真实值. AQ2 （用数据去推测参数值）
    #极大似然估计，最接近真实值的概率
    对数似然：将似然函数里面的乘法转换为加法 （）
    展开化解：目标：让似然函数越大越好（最小二乘法）
    展开目标函数，奇偏导，当偏导=0时，得到最值点

评估方法：
    #最常用的评估项：R^2：(残差平方和/类似方差项)
    R^2的取值越接近1，模型拟合的越好
#要在数据前行加入1


#逻辑回归
    目的：分类（经典的二分类算法）
    机器学习算法选择，先逻辑回归在用复杂的，能简单还是用简单的
    逻辑回归的决策边界：可以是非线性的（高阶）
    sigmoid函数:
        公式：g（z) = 1/1+e^(-z)
        自变量取值为任意实数，值域【0，1】
        就是：将任意的输入值映射到[0,1]区间，这样在线性回归中可以得到一个预测值，
        再将映射到sigmoid函数中，完成:值到概率的转换，就是分类任务
        对于二分类任务（0，1），
        整合后：P(y/x;a) = (ha(x))**y*(1ha(x))**(1-y)
        y取0指保留（1-h0(x)^(1-y)）
        y取1指保留（h0(x))^y
 
        #逻辑回归求解：
            将似然函数转为对数似然，此时应用梯度上升求解最大值，引入J（a) = - (1/m)*l(a)转换为梯度下降任务
"""


# 高斯分布
def gshs():
    import matplotlib.pyplot as pl
    from math import pi, e, sqrt, sin, log
    a = log(4, 2)
    pl.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
    xs = [i / 100 for i in range(-400, 400)]
    px = []
    bs = []
    for x in xs:
        fx = (1 / (sqrt(2 * pi))) * e ** (-x ** 2 / 2)
        gx = 1 / (1 + e ** (-x))
        px.append(fx)
        bs.append(gx)

    pl.plot(xs, bs, c='r')
    pl.plot(xs, px)
    pl.legend('数据类型')
    pl.show()
def sigmoid():
    from math import e
    import matplotlib.pyplot as pl
    x = [i / 100 for i in range(-600, 600)]
    y = []
    for j in x:
        gx = 1 / (1 + e ** (-j))
        y.append(gx)
    pl.figure(figsize=(6, 4))
    pl.plot(x, y, c='r')
    pl.show()



