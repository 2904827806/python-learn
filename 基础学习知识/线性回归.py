import pandas as pd                       #导入数据模块
pd.set_option('display.max_rows',10)
data = pd.read_excel(r'C:\Users\29048\Desktop\DaPy_data.xlsx','BSdata')
print(data)
from matplotlib import pyplot as plt                     #绘图模块
plt.scatter(data['身高'],data['体重'])               #绘制散点图
plt.show()
data['xm']=1

from statsmodels.regression.linear_model import OLS              #拟和
mode = OLS(data['体重'],data[['xm','身高']]).fit()
b = mode.summary()       #mode的拟和结果
print(b)
plt.scatter(mode.fittedvalues,mode.resid_pearson)       #横轴预测值，纵轴标准化残差
plt.show()

#绘制散点图的回归线
from statsmodels.graphics.api import abline_plot

fig= abline_plot(model_results=mode)         #绘制回归线
plt.scatter(data['身高'],data['体重'])         #原始散点图
plt.show()

#多个自变量的回归分析
import seaborn as sns
colums = ['身高','体重','支出']
sns.pairplot(data[colums],kind='reg',diag_kind='kde')
