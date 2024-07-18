"""#数据操作
import pandas as pd
#一、数据的增删改查
#1数据增加
pd.set_option('display.unicode.east_asian_width',True)
pd.set_option('display.max_columns',200)
fill = r"C:\Users\29048\Desktop\daat\电影数据.xlsx"
read_data = pd.read_excel(fill,sheet_name=0)
data = pd.DataFrame(read_data)
#print(data)
#(1) 按列增加数据
#直接为dataframe对象赋值（一般不建议）
# 但要注意赋予的值一定要对应其行数
data['划线'] = ['a','b','c','d','e','f','j','e']
#print('*'*60)
#print(data)
#print('*'*60)
#使用loc属性在dataframe对象的最后行后增加1列 loc[:,索引名] = 插入数据
# (列数要等于以有列数)
data.loc[:,'马化腾'] = ['1','2','3','4','5','6','7','8']
#print(data)
#在指定位置插入1列 ：inser(位置，索引名，插入数据)方法
#print('*'*60)
del data['马化腾']
wl = ['1','2','3','4','5','6','7','8']
data.insert(1,'物理',wl)
#print(data)

#（2）按行增加数据
#使用loc属性在dataframe对象的最后行后增加1行
data.loc[9] = ['a1','51','8','9','152','75','9','152','75']
#print(data)
#print('*'*60)
#增加多行数据，使用字典，_append（）方法
a = {'电影名': ['1', '2', ' 5'], '物理': ['48', '85', '69'], '电影评分': ['7', '8', '9'], '等级': ['1', '5', '25'],
     '学号': ['1', '5', '25 '], '身高': ['1', '5', '25'], '体重': ['1', '5', '25'], '支出': ['1', '5', '25'],
     '划线': ['1', '5', '25']}
data1 = pd.DataFrame(data=a,index=['加','减','除'])
d = data._append(data1)

#2删除数据
# dataframe.drop（label=None,axis=0,index=None,
# columns=Nne,level=None,inplace=False,errors ='raise')
#labels表示行标签或者是列标签
#axis=0表示按行删除，axis=1表示按列删除，默认值为0
b = data.drop(columns='物理')
data.drop(9,axis=0,inplace=True)

#3.修改数据
#(1)修改列名
data.rename(columns={'身高': 'ssl'},inplace=True)
#(2）修改行标题
data.rename(index={7: 'ssl'},inplace=True)
#print(data)
#（3）修改数据
data.loc[3,'ssl'] = 555
#print(data)
#4.查找数据
a = data.loc[3,'ssl']
#查找某个值的行索引
a = data[data['ssl'] >= 175].index
for i in a:
     print(i)
     data.drop(index=i,inplace=True)
print(data)
print(data[data['ssl'] >= 171])"""
