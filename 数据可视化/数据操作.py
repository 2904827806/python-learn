#一、处理缺失值
#缺失值:由于某种原因导致数据为空，造成数据为空的情况
#1.人为因素导致数据丢失
#2.数据采集过程中，没有采集到相应数据
#3.系统或者设备出现故障

#缺失值:NaN \ NOne\ Nat

#二，缺失值查看
#1.info（）方法
"""
查看索引，数据有多少列，每一列的数据类型，非空值的数量和内存使用量
"""
#2.isnull（）方法
"""
空值返回True，非空值返回False
"""
#3.notnull（）方法
"""
空值返回False，非空值返回True
"""

#例1，查看数据概况
import pandas as pd
fill = r"C:\Users\29048\Desktop\工资.xls"
re = pd.read_excel(fill)
df = pd.DataFrame(re)
print(df)

#print(re.info())

#例2，查看数据是否缺失
#print(re.isnull()) #空值为True
#print(re.notnull())#空值为False

#df[df.isnull()=False],会将所有不是缺失值的数据找出来，只针对Series对象


#三、缺失值处理
#1.缺失值删除 dropna（）方法，删除包含缺失值的行
b = re.dropna()
#print(b)
#当数据包含缺失值但不影响处理,一般低于30%不做处理
#将NaN填充为其他数据 fillna（）方法实现
df['肖申克的救赎'] = df['肖申克的救赎'].fillna(0)
#print(df)




