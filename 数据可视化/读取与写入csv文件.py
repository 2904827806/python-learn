import pandas as pd
import numpy as np

#读取csv文件 pd.read_csv()
fill = r"C:\Users\29048\Desktop\工资.csv"
pd.set_option('display.unicode.east_asian_width',True)#解决列不对齐
re = pd.read_csv(fill,delimiter=',',encoding='gbk',header=None) #编码格式为gbk
a = pd.DataFrame(re)


fill1 = r"C:\Users\29048\Desktop\工资.xls"
se = pd.read_excel(fill1)
print(se)
#将数据写入csv文件：to_csv()
se.to_csv( r"C:\Users\29048\Desktop\data.csv",index=False,sep='?')
#分割符
#to_csv(,sep='?')
#替换空值
#to_csv(,ba_rep='NA')
#数据格式化，保留两位小数
#to_csv(,float_format = '%.2f')
#index=0不保存行索引，header=0不保存列名


