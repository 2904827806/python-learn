#什么是数据可视化
#数据可视化是指通过绘图工具和方法将数据集中的数据以图形图像的形式表现出来，
#并利用数据分析发现其中未知信息都处理过程
import random


def numpy(): #numpy的相关介绍
    import numpy as np
    # numpy 数据包 常用于机器学习

    # 数组
    # 一维数组：一行或者一列数据
    # 二维以上的数组也可以称为矩阵
    # 二维数组：包括行和列的数据，灰度图像是二维数组
    # 三维数组：维数为三点数据，彩色图像是三维数组
    # 多维数组：三维以上的数组

    # 轴
    # 二维数组:轴包括行和列，0表示行，1表示列

    # numpy的数据类型
    # 布尔类型：np.bool
    # 整型：np.int
    # 浮点型：np.float
    # 对象型：np.object
    # 字符串类型：np.string
    # 复数：np.complex
    # import numpy as np #导入numpy模块
    # print(np.int_(3.14))
    # print(np.float_(3.14))
    # print(np.complex64(3.14))
    # print(np.bool_(3.14))
    # print(np.str_(3.14))

    # ndarray 数组对象
    # numpy.array(object,dtype,copy,order,subok,ndmin)
    # object 表示数组或者嵌套序列的对象
    # dtype 数组所需的数据类型
    # copy对象是否需要拷贝
    # order指定数组的内存布局，c行排列，f列排列，a任意方向
    # subok默认返回一个与基类类型一致的数组
    # ndmin指定生成数组的最小维度

    # 例子
    """import numpy as np
    a = [1,2,3,4,5,6,7,8,9]
    b = np.array(a)
    print(b) #打印内容
    print(b.dtype) #打印类型
    print(b.shape)#打印数组的形状
    print(b.ndim) #打印数组维度
    print(b.size) #打印数组的长度"""

    # dtype数据类型对象
    # numpy.dtype（obj[,align,copy])
    # object参数是要转换的数据类型对象
    """
    import numpy as np
    a = np.random.random(6) #随机生成浮点类型数组
    print(a.dtype) #查看数组a的类型对象
    #定义一个复数数组（数组的列一定要相同列）
    b = np.array([[1,2,3,4,5],[4,5,6,7,8]],dtype=complex)
    print(b)
    print(b.dtype)"""

    # 一创建数组的多种方法

    # 1.zeros()函数，创建元素均为0的数组,默认为浮点数据
    a = np.zeros(6)
    b = np.zeros(6, dtype=int)
    c = np.zeros((6, 6), dtype=int)  # 创建6行6列的数组，类型为整数
    print(c)

    # 2.ones函数，创建元素均为1的数组,默认为浮点数据
    d = np.ones(6)
    print(d)
    print(d.dtype)

    # 3.arange()函数，通过指定起始值，终止值和步长来创建一个一维数组,
    # 创建的数组中包含起始值不包含终止值
    a1 = np.arange(1, 9, 2)
    print(a1)

    # 4.linspace（）通过指定起始值，终止值和元素个数来创建一个一维数组,
    # 创建的数组中包含起始值和含终止值
    # 每一个元素之间的间隔 = 终止值除以元素个数
    b1 = np.linspace(1, 9, 9, dtype=int)
    print(b1)

    # 5.logspace（）函数
    # 用于创建等比数列，其中起始值和终止值均为10的幂，元素个数不变
    # 如果需要将基数修改为其他数字，可以通过指定base参数来实现
    c2 = np.logspace(0, 9, 10)
    c3 = np.logspace(0, 9, 10, base=2)  # 改为2的幂函数
    print(c2)
    print(c3)

    # 6.eye()函数，用于生成对角线元素为1，其他元素均为0，类似于对角矩阵
    d1 = np.eye(9)  # 内设置行列数
    print(d1)

    # 7.diag（）函数，可以指定对角线上的元素为其他值，其余元素均为0
    d2 = np.diag([3, 4, 5, 6, 7])  # 需要几行几列就设置对角线元素为几个数
    print(d2)

    # 二，生成随机数
    # 1.rand()函数，生成任意维数的数组，数组元素在0-1上面均匀分布
    # 没有设置参数的情况下，则生成1个数
    a3 = np.random.rand(1, 2, 3)
    # 第一个维度（或轴）的大小为1，表示这个维度有1个元素。
    # 第二个维度的大小为2，表示这个维度有2个元素。
    # 第三个维度的大小为3，表示这个维度有3个元素。
    # print(a3)

    # 2.randint（）函数用于生成指定范围内的随机数 #整数
    c4 = np.random.randint(3, 5, size=(2, 5, 2))  # size表示数组的形状（不同的维度包含几个数字）
    print(c4)

    # 3.random（）函数，随机生成0-1浮点型随机数组，当填写单个数字时
    # 将随机生成对应数量的元素数组，在指定数组形状时候需要设置为形状
    a37 = np.random.random(6)
    v = np.random.random((1, 2, 2))
    print(a37)
    print(v)

    # (三)，切片与索引 与Python列表中的索引类似
    a14 = np.arange(10)
    # 通过索引访问多维数组
    s = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(s[..., 1])  # 打印第二列的所有数据
    print(s[1, ...])  # 打印第二行的所有数据


def pandas():#pandas及相关介绍
    import pandas as pd
    #什么是pandas (数据处理，数据分析，数据可视化)工具

    #pandas家族成员的核心对象 series对象和dataframe对象

    #一、Series对象（带索引的一维数组结构（一列数据））
    #包含一些属性和函数，主要用来对每一列数据中的字符串数据进行操作，包括查找，替换，切分等
    #通过pandas的series类来创建
    s = pd.Series([1,2,3,4,5,6],['a','b','c','d','e','f'])
    #print(s)
    #注：当data参数是多维数组时候，index长度必须与data数据长度一致
    #如果没有指定index参数，将自动创建数值型索引(从0-data长度-1）
    #使用字典创建series对象,key=索引，value=类元素
    s2 = pd.Series({1:'d',"d":1})
    #print(s2)
    #创建一列物理成绩
    s3 = pd.Series({'姓名':'成绩','a':25,'b':33})
    #print(s3)

    #二、dataFrame对象（带索引的二维数组结构（表格型数据，包括行和列））
    #主要用于对表格型对象进行操作，如底层数据和属性（行数，列数，数据维数等等）
    #index行标签，columns列标签
    #set_option()函数可以帮助解决列不对其或者多行多列显示不全的问题
    #pd.set_option(‘display.unicode.east_asian_width’，True)解决列不对齐
    #pd.set_option("display.width",100)显示宽度
    #pd.set_option("display.max_columns",100)显示最大列数
    import xlrd2
    wb = xlrd2.open_workbook(r"C:\Users\29048\Desktop\工资.xls")
    sh = wb.sheet_by_index(0)
    a = []
    for i in range(sh.nrows):
        b = []
        for j in range(sh.ncols):
            b.append(sh.cell_value(i, j))
        a.append(b)
    pd.set_option('display.unicode.east_asian_width',True)#解决列不对齐
    pd.set_option("display.max_columns", 20)  # 显示最大列数
    pd.set_option("display.max_rows", 20)  # 显示最大列数
    b = pd.DataFrame(a,columns=['名称','评分','','','',''])
    print(b)

    #通过字典创建成绩表
    s = {'a':[1,2,3,4,6],'b':[2,5,8,9,6],'c':[1,3,4,6,7],'d':[7,8,9,6,5],'f':[7,4,1,2,5]}
    se = pd.DataFrame(s)
    print(se)

pandas()



