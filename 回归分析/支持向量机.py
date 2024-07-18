import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
'''#与传统模型比较'''
#1.一般模型
x0 = np.linspace(0.5,5,200)
pred_1 = 5*x0-20
pred_2 = x0-1.8
pred_3 = 0.1*x0+0.5


def plot_svc_decision_boundary(svm,xmin,ymax,sv=True):
    '''绘制支持向量机决策边界线'''
    w = svm.coef_ #权重参数
    b = svm.intercept_ # 偏执参数
    x0 = np.linspace(xmin,ymax,2000).reshape(-1,1)
    decision_boundary = -w[0][0]/w[0][1]*x0 - b/w[0][1] ##决策边界
    margin = 1/w[0][1]
    gutter_up = decision_boundary+margin
    gutter_down = decision_boundary - margin
    if sv:
        svs = svm.support_vectors_ #支持向量
        plt.scatter(svs[:,0],svs[:,1],s=180,c='r')
    plt.plot(x0,decision_boundary,'k')
    plt.plot(x0, gutter_up, 'k--')
    plt.plot(x0, gutter_down, 'k--')
from sklearn.svm import SVC
#导入支持向量机模型
from sklearn.datasets import load_iris
iris = load_iris()
x = iris['data'][:,(2,3)]
y = iris['target']

#将数据分为两类
st = (y==0) | (y==1)
x = x[st]
y = y[st]
svm_clf = SVC(kernel='linear',C=99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999)
svm_clf.fit(x,y)
#print(svm_clf.coef_)
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.title('普通模型')
plt.plot(x[:,0][y==0],x[:,1][y==0],'bs')
plt.plot(x[:,0][y==1],x[:,1][y==1],'ys')
plt.plot(x0,pred_1,'g')
plt.plot(x0,pred_2,'r')
plt.plot(x0,pred_3,'y')
plt.axis([0,6,-1,3])
plt.subplot(122)
plt.title('支持向量机模型')
plot_svc_decision_boundary(svm_clf,0,5.5)
plt.plot(x[:,0][y==0],x[:,1][y==0],'bs')
plt.plot(x[:,0][y==1],x[:,1][y==1],'ys')
plt.axis([0,6,-1,3])
#plt.show()


'''
#2.数据标准化的影响
#3.软间隔
'''
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
#iris = load_iris()
X = iris['data'][:,(2,3)]
y1 = (iris['target'] == 2).astype(np.float64)
svm_clf1 = Pipeline((
    ('std',StandardScaler()),
    ('lsvc',LinearSVC(C=1))
))
svm_clf1.fit(X,y1)
#预测数据
s = svm_clf1.predict([[5.5,1.7]])
#print(s)
'''对比不同c值带来的效果差异'''
scaler = StandardScaler()
scaler.fit(X,y1)
svm_clf2 = LinearSVC(C=1,random_state=42)
svm_clf3 = LinearSVC(C=100,random_state=42)
'''svm_clf2.fit(X,y1)
svm_clf3.fit(X,y1)'''
scaler_svm_clf1 = Pipeline([
    ('std',StandardScaler()),
    ('lsvc',svm_clf2)
])
scaler_svm_clf2 = Pipeline([
    ('std',StandardScaler()),
    ('lsvc',svm_clf3)
])
scaler_svm_clf1.fit(X,y1)
scaler_svm_clf2.fit(X,y1)
b1 = svm_clf2.decision_function([-scaler.mean_/scaler.scale_])
b2 = svm_clf3.decision_function([-scaler.mean_/scaler.scale_])
w1 = svm_clf2.coef_[0]/scaler.scale_
w2 = svm_clf3.coef_[0]/scaler.scale_
svm_clf2.intercept_=np.array([b1])
svm_clf3.intercept_=np.array([b2])
svm_clf2.coef_=np.array([w1])
svm_clf3.coef_=np.array([w2])
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.title(f'C = {svm_clf2.C}',fontsize=32)
plt.plot(X[:,0][y1==1],X[:,1][y1==1],'bs')
plt.plot(X[:,0][y1==0],X[:,1][y1==0],'ys')
plot_svc_decision_boundary(svm_clf2,4,6,sv=False)
plt.axis([4,6,0.8,2.8])
plt.subplot(122)
plt.title(f'C = {svm_clf3.C}',fontsize=32)
plt.plot(X[:,0][y1==0],X[:,1][y1==0],'bs')
plt.plot(X[:,0][y1==1],X[:,1][y1==1],'ys')
plot_svc_decision_boundary(svm_clf3,4,6,sv=False)
plt.axis([4,6,0.8,2.8])
plt.show()

'''4.非线性支持向量机'''
x1 = np.linspace(-4,4,9).reshape(-1,1)
x2 = np.c_[x1,x1**2]
y2 = np.array([0,0,1,1,1,1,1,0,0])
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.grid(True,which='both')
plt.axhline(y=0,color='k')
plt.plot(x1[:,0][y2==1],np.zeros(5),'bs')
plt.plot(x1[:,0][y2==0],np.zeros(4),'ys')
plt.gca().get_yaxis().set_ticks([])
plt.subplots_adjust(right=1)
#plt.axis([4.5,4.5,-0.2,0.2])
#plot_svc_decision_boundary(svm_clf2,4,6,sv=False)
plt.subplot(122)
plt.grid(True,which='both')
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.plot(x2[:,0][y2==0],x2[:,1][y2==0],'bs')
plt.plot(x2[:,0][y2==1],x2[:,1][y2==1],'ys')
plt.gca().get_yaxis().set_ticks([0,4,8,12,16])
plt.plot([-4.5,4.5],[6.5,6.5],'r--',linewidth=3)
#plot_svc_decision_boundary(svm_clf3,4,6,sv=False)
#plt.axis([4.5,4.5,-1,1.7])
plt.subplots_adjust(right=1)
plt.show()

from sklearn.datasets import make_moons
x3,y3 = make_moons(n_samples=100,noise=0.15,random_state=42)

def plot_data(x,y,axes):
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], 'bs')
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], 'ys')
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel('x1',fontsize=20)
    plt.ylabel('x2',fontsize=20,rotation=0)
plot_data(x3,y3,[-1.5,2.5,-1,1.5])
plt.show()

from sklearn.preprocessing import PolynomialFeatures#多项式特征模块
p_svc_clf = Pipeline([('poly',PolynomialFeatures(degree=10)),
     ('scale',StandardScaler()),
     ('svc_clf',LinearSVC(C=10,loss='hinge'))
     ])
p_svc_clf.fit(x3,y3)

pse = SVC(kernel='rbf',C=10)
pse.fit(x3,y3)
pse1 = SVC(kernel='linear',C=10)
pse1.fit(x3,y3)
from matplotlib.colors import ListedColormap
def plot_predict(clf,axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X_new = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X_new).reshape(x0.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x0, x1, y_pred, cmap=custom_cmap, alpha=0.3)
plt.subplot(121)
plot_predict(p_svc_clf,[-1.5,2.5,-1,1.5])
plot_data(x3,y3,[-1.5,2.5,-1,1.5])

plt.subplot(122)
plot_predict(pse,[-1.5,2.5,-1,1.5])
plot_data(x3,y3,[-1.5,2.5,-1,1.5])
plt.show()


'''三、SVM中如何运用核函数'''
'''
kernel='poly'：这指定了SVM的核函数为多项式核
degree=3：degree 参数定义了多项式中的最高次数。
coef0=1：偏置项或截距项,在多项式核中，它决定了多项式中的常数项
C=5：它控制了对误分类的惩罚程度与决策边界的平滑度之间的权衡。
'''

poly_kernel_svm_vlf = Pipeline([
    ('scal',StandardScaler()),
    ('sv_clf',SVC(kernel='poly',degree=3,coef0=100,C=5))
])
poly_kernel_svm_vlf.fit(x3,y3)

poly100_kernel_svm_vlf = Pipeline([
    ('scal',StandardScaler()),
    ('sv_clf',SVC(kernel='poly',degree=10,coef0=100,C=5))
])
poly100_kernel_svm_vlf.fit(x3,y3)
plt.subplot(121)
plot_predict(poly_kernel_svm_vlf,[-1.5,2.5,-1,1.5])
plot_data(x3,y3,[-1.5,2.5,-1,1.5])

plt.subplot(122)
plot_predict(poly100_kernel_svm_vlf,[-1.5,2.5,-1,1.5])
plot_data(x3,y3,[-1.5,2.5,-1,1.5])
plt.show()






'''绘制高斯函数变化图'''
def gaussian_rbf(x,landmark,gamma):
    return np.exp(-gamma*np.linalg.norm(x-landmark,axis=1)**2)

gamma = 0.1
x1s = np.linspace(-4.5,4.5,200).reshape(-1,1)
x2s = gaussian_rbf(x1s,-2,gamma)
x3s = gaussian_rbf(x1s,1,gamma)
xk = np.c_[gaussian_rbf(x1,-2,gamma),gaussian_rbf(x1,1,gamma)]
yk = np.array([0,0,1,1,1,1,1,0,0])

plt.figure(figsize=(11,4))

plt.subplot(121)
plt.grid(True,which='both')
plt.axhline(y=0,color='k')
plt.scatter(x=[-2,1],y=[0,0],s=150,alpha=0.5,c='red')
plt.plot(x1[:,0][yk==1],np.zeros(5),'bs')
plt.plot(x1[:,0][yk==0],np.zeros(4),'ys')
plt.plot(x1s,x2s,'g--')
plt.plot(x1s,x3s,'b:')
plt.gca().get_yaxis().set_ticks([])
plt.annotate(r'$\mathbf{x}$',
             xy=(x1[3,0],0),
             xytext=(-0.5,0.20),
             ha='center',
             arrowprops=dict(facecolor='black',shrink=0.1)
)
plt.text(-2,0.9,'$x_2$',ha='center',fontsize=20)
plt.text(1,0.9,'$x_3$',ha='center',fontsize=20)
plt.axis([-4.5,4.5,-0.1,1.1])
plt.subplot(122)
plt.grid(True,which='both')
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.plot(xk[:,0][yk==1],xk[:,1][yk==1],'bs')
plt.plot(xk[:,0][yk==0],xk[:,1][yk==0],'ys')
plt.annotate(r'$\mathbf{x}$',
             xy=(x1[3,0],0),
             xytext=(0.65,0.50),
             ha='center',
             arrowprops=dict(facecolor='black',shrink=0.1),
             fontsize=18
)
plt.plot([-0.1,1.1],[0.57,-0.1],'r--',linewidth=3)
plt.axis([-0.1,1.1,-0.1,1.1])
plt.subplots_adjust(right=1)
plt.show()


rbf_clf = Pipeline([
    ('str',StandardScaler()),
    ('rbf',SVC(kernel='rbf',gamma=5,C=0.001))
])
rbf_clf.fit(X,y1)

from sklearn.svm import SVC
gamma1,gamma2 = 0.1,5
C1,C2 = 0.001,1000
hyperparams = ((gamma1,C1),(gamma1,C2),(gamma2,C1),(gamma2,C2))
svm_clfs = []
for gamma,C in hyperparams:
    rbf_svm_clF = Pipeline([
        ('str',StandardScaler()),
        ('svm',SVC(kernel='rbf',gamma=gamma,C=C))
    ])
    rbf_svm_clF.fit(x3,y3)
    svm_clfs.append(rbf_svm_clF)

plt.figure(figsize=(12,10))

for i ,svm_clf in enumerate(svm_clfs):
    plt.subplot(221+i)
    plot_predict(svm_clf,[-1.5,2.5,-1,1.5])
    plot_data(x3,y3,[-1.5,2.5,-1,1.5])
    gamma,C = hyperparams[i]
    plt.title(r'$\gamma = {},C = {}'.format(gamma,C,fontsize=16))

plt.show()
