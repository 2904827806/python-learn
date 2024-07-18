import time
import requests
import os
from lxml import etree
import pandas as pd

df = pd.DataFrame()

#读取网页文件：read_html()
#table类型的表格
#pandas可以简单爬取网页

urls = []
for i in range(1,4):
    url = r'https://www.espn.com/nba/salaries/_/page/'+str(i)
    urls.append(url)

for url1 in urls:
    df = df._append(pd.read_html(url1), ignore_index=True)
#print(df)

#筛选排除没用数据
df = df[[x.startswith('$')for x in df[3]]]
df.to_excel('玩意.xlsx',header=None)
print('完成')

