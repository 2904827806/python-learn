import pandas as pd
import matplotlib.pyplot as pl
#设置对齐
pd.set_option('display.unicode.east_asian_width',True)
#一、绘制柱形图 mb.bar
#fill = r"C:\Users\29048\Desktop\daat\电影数据.xlsx"
#open_data = pd.read_excel(fill,sheet_name=0)
#data = pd.DataFrame(open_data)
#pl.bar(x=data.index.values,height=data['身高'].values)


#基本柱状图
"""fill1 = r"E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\销售表.xlsx"
open_data = pd.read_excel(fill1,sheet_name=0)
data1 = pd.DataFrame(open_data)
#print(data1)
pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
x = data.index.values
height = data['身高'].values
pl.grid(axis='y',which='major') #设置网格线
pl.xlabel('年份') #x轴标签
pl.ylabel('元')
#设置图标标题
pl.title('2013-2019分析图')
pl.bar(x,height,width=0.5,align='center',color='b',alpha=0.5)
#设置每个柱子的文本标签
for a,b in zip(x,height):
    pl.text(a,b,format(b,','),ha= 'center',va='bottom',fontsize=9,color='b')
pl.legend(['销售码洋'])

pl.show()"""

#多柱形图
fill = r'E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\books.xlsx'
daat = pd.read_excel(fill,sheet_name='Sheet2')

#print(daat)
#
"""
width = 0.25
pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
x = daat['年份']
y1 = daat['京东']
y2 = daat['天猫']
y3 = daat['自营']
pl.grid(axis='y',which='major')
pl.ylabel('销售额（元）')
pl.title('销售分析')
pl.bar(x,y1,width=width,color='b')
pl.bar(x+width,y2,width=width,color='r')
pl.bar(x+2*width,y3,width=width,color='g')
#设置柱子文本标签
for a,b in zip(x,y1):
    pl.text(a,b,format(b,','),ha= 'center',va='bottom',fontsize=9)
for a,b in zip(x,y2):
    pl.text(a+width,b,format(b,','),ha= 'center',va='bottom',fontsize=9)
for a,b in zip(x,y3):
    pl.text(a+2*width,b,format(b,','),ha= 'center',va='bottom',fontsize=9)
pl.legend(['京东','天猫','自营'])

pl.show()"""


#(二）绘制折线图
"""f = r'E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\data.xls'
d = pd.read_excel(f)
y = d['语文']
x = d['姓名']
y1 = d['英语']
y2 = d['数学']
pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
#pl.rcParams['ytick.direction'] = 'in' #y轴刻度向外显示
#pl.rcParams['xtick.direction'] = 'out'#x刻度向内显示‘
pl.title('语数外成绩大比拼',fontsize=18)
pl.plot(x,y,label='语文',color='r')
pl.plot(x,y1,label='英语',color='b',marker='.',mfc='r',ms=8)
pl.plot(x,y2,label='数学',color='g',linestyle='-.',marker='*')
pl.grid(axis='y')
pl.legend(['语文','英语','数学'])
pl.show()"""

#（三）绘制散点图 scatter()函数
"""pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
f = r'E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\JDcar.xls'
f1 = r'E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\JDdata.xls'
datas = pd.read_excel(f)
datas1 = pd.read_excel(f1)
df1 = datas1[['业务日期','金额']]
df2 = datas[['投放日期','支出']]
#(数据清洗）
#去除日期和金额为空的记录
df1 = df1[df1['业务日期'].notnull() & df1['金额'] != 0]
df2 = df2[df2['投放日期'].notnull() & df2['支出'] != 0]
#将日期替换为
df1['业务日期'] = pd.to_datetime(df1['业务日期'])
df2['投放日期'] = pd.to_datetime(df2['投放日期'])
#将日期作为索引
dfData = df1.set_index('业务日期',drop=True)
dfDc = df2.set_index('投放日期',drop=True)
#按月统计并显示销售金额
dfData_m = dfData.resample('M').sum().to_period('M')
#按月统计并显示广告费
dfDc_m = dfDc.resample('M').sum().to_period('M')

x = pd.DataFrame(dfDc_m['支出'])
y = pd.DataFrame(dfData_m['金额'])

pl.scatter(x,y)
pl.show()"""

#绘制饼图pie()函数
"""pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
x = [2,5,12,70,2,9]
pl.pie(x,autopct='%1.1f%%')
pl.show()"""

#基础饼形图
pl.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
fill2 = r'E:\Pythonproject\2. TM（实例源码+习题答案）\21\datas\data3.xls'
rd = pd.read_excel(fill2)
pl.figure(figsize=(5,3)) #设置画布大小
label = rd['省']
size = rd['销量']
#设置饼形图每块颜色
colors = ['r','y','b','g','b','r','y','r','r','g']
a = [1,2,1.2,1.5,1.6,0.9,0.8,1.02,1.5,2,1]
#内环
pl.pie(size,#绘图数据
       #labels=label,#标签
       colors=colors,#填充颜色
       labeldistance=1.2,#标签到圆心的距离
       autopct='%.1f%%',#百分比格式保留一位小说
       startangle=180,  #初始角度=90
       radius=0.29,#饼图半径
       textprops={'fontsize': 9, 'color': 'k'},#文本标签的属性值
       pctdistance=0.8,#百分比标签与圆心的距离
       #explode=[0.1,0,0,0,0.04,0,0.05,0,0,0],#设置分离部分
       #shadow=True,#设置阴影
       wedgeprops={'width':0.2,'edgecolor':'k'}#设置环形图width数值愈小，空心愈大
        )
#外环
pl.pie(size,#绘图数据
       labels=label,#标签
       colors=colors,#填充颜色
       labeldistance=1.2,#标签到圆心的距离
       autopct='%.1f%%',#百分比格式保留一位小说
       startangle=180,  #初始角度=90
       radius=0.5,#饼图半径
       textprops={'fontsize': 9, 'color': 'k'},#文本标签的属性值
       pctdistance=0.8,#百分比标签与圆心的距离
       explode=[0.1,0,0,0,0.04,0,0.05,0,0,0],#设置分离部分
       #shadow=True,#设置阴影
       wedgeprops={'width':0.2,'edgecolor':'k'}#设置环形图width数值愈小，空心愈大
        )
#设置x,y轴刻度一样，保证饼图为圆形
pl.axis('equal')
pl.legend(label,title='地区',frameon=False,bbox_to_anchor=(0.25,0.5))#bbox_to_anchor=(0.25,0.5)设置图列位置
pl.show()
