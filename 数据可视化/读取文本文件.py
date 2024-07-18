import pandas as pd

#读取文本文件
#read_table()函数，默认\t分割文件数据,用法与to_csv类似
#read_csv()函数，默认，分割文件数据

rw = pd.read_table(r"C:\Users\29048\Desktop\项目\tree.txt",sep='?')
print(rw)
