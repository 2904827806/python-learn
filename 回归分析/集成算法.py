#集成算法
#bagging方法
#http://ml-ensemble.com/  工具包
import xml.sax.xmlreader
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

seed = 222  # 随机因子
np.random.seed(seed)
# NumPy的随机数生成器的种子为之前定义的seed值

# 导入数据
df = pd.read_csv(r"C:\Users\29048\Desktop\creditcard.csv", delimiter=',', encoding='gbk')
print(df)
from sklearn.model_selection import train_test_split  # 训练集和验证集划分
from sklearn.metrics import roc_auc_score  # 用于计算特征曲线（ROC 曲线）
from sklearn.preprocessing import StandardScaler #数据标准化

def get_train_test(test_size=0.95): #获取训练集和验证集
    y = df.loc[:,df.columns == 'Class']
    x = df.drop(['Class','Time'],axis=1)
    st = StandardScaler()
    #x = pd.get_dummies(x,sparse=True)
    #sparse=True：这个参数指示 pandas 使用稀疏数据结构来存储结果。
    # 稀疏数据结构在存储大量零值时更加高效，因为它们只存储非零值，从而节省了大量的内存和存储空间。
    x['Amount'] = st.fit_transform(x['Amount'].values.reshape(-1,1)) #数据标准化
    #print(y)
    #print(x)
    return train_test_split(x,y,test_size=test_size,random_state=seed) #返回训练集和验证集
#ROC与AUC
#ROC x轴：TPR = TP/(TP+FN)  y轴：FPR= FP/(TN+FP)
#AUC定义为ROC曲线下的面积，值一般在0.5-1
df_pl = pd.Series(df['Class']).value_counts(normalize=True).sort_index() #统计数量
pl.grid(axis='y')
df_pl.plot(kind='bar',color='r') #kind= ’bar'表示绘制柱状图
#pl.show()

import pydotplus
from IPython.display import Image
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz,plot_tree

def print_graph(clf,featur_name):
    """
    决策树可视化
    dor_data = tree.export_graphviz(clf,out_file=None,feature_names=name,filled=True,impurity=False,rounded=True)
    import pydotplus
    graph = pydotplus.graph_from_dot_data(dor_data)
    # 创建PNG格式的图像，并使用IPython的Image来显示
    image = Image(graph.create_png())  # 注意这里调用了create_png()方法
    """
    graph = export_graphviz(clf,precision=True,impurity=False,out_file=None,feature_names=featur_name,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(graph)
    return Image(graph.create_png())

x_train,x_test,y_train,y_test = get_train_test()
t1 = DecisionTreeClassifier(max_depth=3,random_state=seed) #实例化决策树对象
t1.fit(x_train,y_train)#模型拟和
p1 = t1.predict_proba(x_test)[:,1] #预测数据

#打印auc指标值
print('Decisions tree ROC_AUR_score: %.3f' % roc_auc_score(y_test,p1))

x_name = [i for i in x_train.columns]
plot_tree(t1)
#pl.show()
#print_graph(t1,x_name)

x_trains = pd.DataFrame(x_train)
x_tests = pd.DataFrame(x_test)
x_trains.drop('V17',axis=1,inplace=True)
x_tests.drop('V17',axis=1,inplace=True)
t3 = DecisionTreeClassifier(max_depth=3,random_state=seed)
t3.fit(x_trains,y_train)
p2 = t3.predict_proba(x_tests)[:,1]
print('Decisions tree ROC_AUR_score: %.3f' % roc_auc_score(y_test,p2))
p = np.mean([p1,p2],axis=0)#求不同模型预测的平均值
print('Decisions tree ROC_AUR_score: %.3f' % roc_auc_score(y_test,p))
plot_tree(t3)
#pl.show()


#随机森林
from sklearn.ensemble import RandomForestClassifier #随机森林
rf = RandomForestClassifier(n_estimators=10,max_features=3,random_state=seed) #实例化随机森林
rf.fit(x_train,y_train) #拟和模型
p3 = rf.predict_proba(x_test)[:,1] #获取预测值

print('Decisions tree ROC_AUR_score: %.3f' % roc_auc_score(y_test,p3))


#
from sklearn.svm import SVC,LinearSVC  #支持向量机
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem,RBFSampler
from sklearn.pipeline import make_pipeline
def ger_models(): #建立一个集合模型
    nb = GaussianNB()
    svc = SVC(C=100,probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100,random_state=seed)
    nn = MLPClassifier((80,100),early_stopping=False,random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=100,random_state=seed)
    rf = RandomForestClassifier(n_estimators=10,max_features=3,random_state=seed)
    models = {
        'svm':svc,
        'knn':knn,
        'naive bayes':nb,
        'mlp-nn':nn,
        'random forest':rf,
        'gbm':gb,
        'logistic':lr,
    }
    return models

def train_predict(models): #获取预测值和索引
    p = np.zeros((y_test.shape[0],len(models)))
    p = pd.DataFrame(p)

    print('Fitting models')
    cols = list()
    for i,(name,m) in enumerate(models.items()):
        print('%s ...' % name,end='',flush=False)
        m.fit(x_train,y_train)
        p.iloc[:,i] = m.predict_proba(x_test)[:,1]
        cols.append(name)
        print('done')
    p.columns = cols
    print('Done\n')
    return p

def score_models(p,y):
    print('Scoring models')
    for m in p.columns:
        score = roc_auc_score(y,p.loc[:,m])
        print('%-26s:%.3f' % (m,score))
    print('Done')

models = ger_models()
p = train_predict(models)
a = score_models(p,y_test)
#print(p)
#print(a)

#展示数据
"""from mlens.visualization import corrmat
corrmat(p.corr,inflate=False)
pl.show()"""

#roc曲线
"""from sklearn.metrics import roc_curve

def pl_roc_curve(y_test,p_base_learners,p_ensemble,labels,ens_labe):
    pl.figure(figsize=(10,8))
    pl.plot([0,1],[0,1],'k--')

    cm = [pl.cm.get_cmap()]

    for i in [np.linspace(0,1.0,p_base_learners.shape[1]+1)]:
        p = p_base_learners[:,i]
        fpr,tpr = roc_curve(y_test,p)
        pl.plot(fpr,tpr,labels=labels[i],c=cm[i+1])

    fpr,tpr, _ =roc_curve(y_test,p_ensemble)
    pl.plot(fpr,tpr,labels=ens_labe,c=cm[0])
    pl.xlabel('F P R')
    pl.ylabel('T P R')
    pl.legend(fractions=False)
    pl.show()
cas = pl_roc_curve(y_test,p.values,p.mean(axis=1),list(p.columns),'ensemble')"""


p = p.apply(lambda x:1*(x >= 0.5).value_counts(normalize=True))
p.index = ['den','rep']
p.plot(kind='bar')
pl.axhline(0.25,color='k',linewidth=0.5)
pl.text(0,0.23,'Truess')
#pl.show()


#定义基础模型
base_learner = ger_models()

#定义权重分配模型
mata_learner= GradientBoostingClassifier(
    n_estimators=1000,
    loss='exponential',
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005,
    random_state=seed
)

#划分数据
x_train_b,x_test_b,y_train_b,y_test_b = train_test_split(x_train,y_train,test_size=0.5,random_state=seed)

#训练基础模型
def train_base_learners(base_learner,inp,out,verbose=True):
    #inp,x训练集 out y训练集，base_learner基础模型
    if verbose:
        print('Fitting models')
    for i, (name, m) in enumerate(base_learner.items()):
        # name：模型名称，m对应的模型
        if verbose:
            print('%s....' % name, end='', flush=False)
        m.fit(inp, out)  # 模型拟和
        if verbose:
            print('done')
train_base_learners(base_learner,x_train_b,y_train_b) #base_learner基础模型

#准备二阶段权重分配分类器的训练数据
def train_predict_base(pred_base_learner,inp,verbose=True): #获取预测值和索引
    #pred_base_learner，基础模型,inp，x训练集,
    P = np.zeros((inp.shape[0],len(pred_base_learner))) #创建一个二维0数据集

    if verbose:
        print('Generating base learner predictions')
    for i, (name, m) in enumerate(base_learner.items()):
        #i索引号, name：模型名称, m：模型
        if verbose:
            print('%s....' % name, end='', flush=False)
        p = m.predict_proba(inp)#获取预测值
        P[:,i] = P[:,1]# 假设我们只对正类的概率感兴趣
        if verbose:
            print('done')
    return p # 返回所有模型的预测结果
P_base = train_predict_base(base_learner,x_train_b)

#训练二阶段得出分类结果
aswe = mata_learner.fit(P_base,y_test_b)
def es_prs(base_learners,mode_learn,inp,verbose=True):
    ##定义权重分配模型mode_learn
    #base_learners 集成模型
    #inp x实验集
    p_pred = train_predict_base(base_learners,inp,verbose=verbose)
    return p_pred,mode_learn.predict_proba(p_pred)[:,1] #返回预测结果和定义权重分配模型的预测结果
p_pred,p = es_prs(base_learner,mata_learner,x_test)

print('\nEnsemble ROC-AUC score: %.3f' % roc_auc_score(y_test,p))

from sklearn.base import clone

def stacking(base_learners,meta_learner,x,y,generator):
    #交叉验证（Cross-Validation）过程中的一部分，特别是关于如何分割数据、训练基础模型（base learners）并对测试集进行预测。
    #base_learners：集成模型,
    # meta_learner：定义权重分配模型,
    # x：x数据
    # y：y数据
    # generator：generator 是一个交叉验证数据生成器

    print('F f b l',end='')
    train_base_learners(base_learners,x,y,verbose=False) #训练基础模型
    print('deon')

    print('G C P...')
    cv_preds,cv_y = [],[] #设置两个参数空列表
    # 遍历交叉验证的每一折
    for i,(train_idx,test_idx) in enumerate(generator.split(x)):#split 方法会返回训练集和测试集的索引
       # 使用当前折的训练集索引从 x 和 y 中选择数据
        fold_xtrain,fold_ytrain = x[train_idx,:],y[train_idx]
       # 使用当前折的测试集索引从 x 和 y 中选择数据
        fold_xtest, fold_ytest = x[test_idx, :], y[test_idx]

       # 为当前折克隆基础模型字典，确保每折使用相同但独立的模型实例
        fold_base_learner = {name:clone(model) for name,model in base_learner.items()}

       # 使用当前折的训练数据训练基础模型
        train_base_learners(fold_base_learner,fold_xtrain,fold_ytrain,verbose=False)

       # 使用训练好的基础模型对当前折的测试数据进行预测
        fold_P_base = train_predict_base(fold_base_learner,fold_xtest,verbose=False)

       # 将预测结果添加到 cv_preds 列表中
        cv_preds.append(fold_P_base)

       # 将测试集的真实标签添加到 cv_y 列表中
        cv_y.append(fold_ytest)
        print('fold %i done' %(i+1))

    print('CV-predictions done')

    cv_preds = np.vstack(cv_preds)
    # 使用列表推导式和numpy的concatenate来合并所有折的真实标签
    cv_y = np.concatenate(cv_y)

    print('Fitting meta learner...',end='')
    meta_learner.fit(cv_preds,cv_y)
    print('done')

    return base_learners,meta_learner

from sklearn.model_selection import KFold

cv_base_learners,cv_meta_learner = stacking(
    ger_models(),clone(mata_learner),x_train.values,y_train.values,KFold(2))

P_pred,p = es_prs(cv_base_learners,cv_meta_learner,x_test,verbose=False)
print('\n电视Ensemble ROC-AUC score: %.3f' % roc_auc_score(y_test,p))
pl.show()
