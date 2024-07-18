class Djia_zte:
    """代价曲线与期望总体代价"""
"""
    output = list(range(12)) #函数输出打分
    y = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1] #正确分类
    # 设定p,结合中正列比例，范围是0-100
    p = list(range(0, 101, 10))
    p = [i / 100 for i in p]
    # 设定代价
    c01 = 3
    c02 = 2
    # 一个阈值
    theta = [i + 0.5 for i in range(12)]

    # print(theta)
    # 函数判断输出
    def a(outp, theta):
        aso = []
        for i in range(len(outp)):
            if outp[i] < theta:
                aso.append(0)
            else:
                aso.append(1)

        return aso

    bas = []
    for i in range(len(theta)):
        ab = a(output, theta[i])
        bas.append(ab)
    # print(ab)
    # 统计正数和反数个数
    import pandas as pd
    def pdgs(y):
        import pandas as pd
        result = pd.value_counts(y)
        m_positive = result[1]
        m_negative = result[0]
        return m_negative, m_positive

    f, z = pdgs(y)

    # 计算混淆矩阵
    cs = []
    Pcostss = []
    Postzs = []
    for a in bas:
        def c_confusion(y, a):
            con1 = 0
            con2 = 0
            con3 = 0
            con4 = 0
            for i in range(len(y)):
                if y[i] == 1:
                    if y[i] == a[i]:
                        con1 += 1
                    else:
                        con2 += 1
                else:
                    if y[i] == a[i]:
                        con4 += 1
                    else:
                        con3 += 1

            return con1, con2, con3, con4

        con1, con2, con3, con4 = c_confusion(y, a)
        cs.append([con1, con2, con3, con4])

        # print(con1,con2,con3,con4)

        # 奇比例
        def c_FNR_FPR(con1, con2, con3, con4):
            fnr = round(con2 / (con1 + con2), 4)
            fpr = round(con3 / (con4 + con3), 4)
            return fnr, fpr

        FNR, FPR = c_FNR_FPR(con1, con2, con3, con4)

        # print(FNR,FPR)
        # 正概率代价
        def c_Pcost(p, c01, c02):
            Pcosts = []
            for i in range(len(p)):
                Pcost = round(p[i] * c01 / (p[i] * c01 + (1 - p[i]) * c02), 4)
                Pcosts.append(Pcost)
            return Pcosts

        Pcosts = c_Pcost(p, c01, c02)
        Pcostss.append(Pcosts)

        # print(Pcosts)

        # 归一化总概率
        def c_Postz(p, c01, c02, FNR, FPR):
            Postz = []
            for i in range(len(p)):
                Post = round((FNR * p[i] * c01 + FPR * (1 - p[i]) * c02) / (p[i] * c01 + (1 - p[i]) * c02), 4)
                Postz.append(Post)
            return Postz

        Postz = c_Postz(p, c01, c02, FNR, FPR)
        # print(Postz)
        Postzs.append(Postz)
    import matplotlib.pyplot as pl
    pl.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
    pl.xlabel('正概率代价')
    pl.ylabel('归一化代价')
    pl.grid(axis='y',which='major')
    pl.title('代价曲线与期望总体代价')
    for i in range(len(Pcostss)):
        pl.plot(Pcostss[i], Postzs[i], marker='.', mfc='b', ms=5, color='r')
        # pl.plot(p, Postzs[i], linestyle='-.')
    pl.show()
"""


#二项分布
from scipy.special import comb
e_all = 0.3
m_t =10
m_t_error = 6
def calculate_p(m_t,m_t_error):
    p = (comb(m_t,m_t_error))*(e_all**m_t_error)*(1-e_all)**(m_t-m_t_error)
    p = round(p,4)
    return p
a = calculate_p(m_t,m_t_error)

#出现每个都概率
def calculate_ps(m_t):
    m_t_errors1 = [-i for i in range(1,m_t+1) ]+ list(range(m_t+1))
    m_t_errors = list(range(m_t+1))
    ps = []
    for i in range(len(m_t_errors)):
        m_T_error = m_t_errors[i]
        p = (comb(m_t, m_T_error)) * (e_all ** m_T_error) * (1 - e_all) ** (m_t - m_T_error)
        p = round(p, 4)
        ps.append(p)
    return m_t_errors ,ps
sb,as1 = calculate_ps(m_t)
#a = list(reversed(as1))[1:]
#as2 = sorted(a) + sorted(as1,reverse=True)
#print(as2)
#画出分布图
import matplotlib.pyplot as pl
def plot_h(x,y):
    pl.scatter(x,y,s=20,c='r')
    pl.plot(x,y)
    pl.bar(x,y)
    pl.show()

plot_h(sb,as1)