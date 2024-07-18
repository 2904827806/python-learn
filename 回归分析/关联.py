import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
'''data = {'Id':[1,2,3,4,5,6],
        'Basket':[['beer','diaper','pretzels','chips','aspirin'],
                  ['diaper','beer','chips','lotion','juice','babyfood','milk'],
                  ['soda','chips','milk'],
                  ['soup','beer','diaper','milk','icecream'],
                  ['soda','coffee','milk','bread'],
                  ['beer','chips']]}

data = pd.DataFrame(data)

retail = data[['Id','Basket']]
#print(retail)
retail_id = retail.drop('Basket',axis=1)
#print(retail_id)
retail_basket = retail.Basket.str.join(',')
#print(retail_basket)
retail_basket = retail_basket.str.get_dummies(',')
#print(retail_basket)
retail = retail_id.join(retail_basket)
#print(retail)

from mlxtend.frequent_patterns import apriori,association_rules

dae = apriori(retail.drop('Id',axis=1),use_colnames=True)
dat = association_rules(dae,metric='lift')
'''

def loaddataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    c1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
            else:
                continue
    c1.sort()
    return list(map(frozenset,c1))

#扫描数据集


def scanD(dataSet,CK,minsupport):
    ssCnt = {}
    for transaction in dataSet:
        for can in CK:
            if can.issubset(transaction):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    nums = float(len(dataSet))
    retlist = []
    dels = []
    supportData = {}
    for item in ssCnt:
        supportData[item] = (ssCnt[item]/nums)
        if (ssCnt[item]/nums) < minsupport:
            dels.append(item)
        else:
            retlist.insert(0,item)
    for i in dels:
        del ssCnt[i]
    return retlist,supportData

#组合
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2: #前k-2个元素相同
                retList.append(Lk[i] | Lk[j]) #求并集
    return retList


def apriori(dataset,minsuoppr=0.5):
    c1 = createC1(dataset)
    L1,supporData = scanD(dataset,c1,minsuoppr)
    L = [L1]
    k = 2
    while (len(L[k-2])>0):
        CK  = aprioriGen(L[k-2],k)
        LK,supk = scanD(dataset,CK,minsuoppr)
        supporData.update(supk)
        L.append(LK)
        k += 1
    return L,supporData
#规则生成函数
def generateRules(L,supporData,minsuoppr=0.6):
    retlist = []
    for i in range(1,len(L)):
        for freeqset in L[i]:
            H1 = [frozenset([item]) for item in freeqset]
            rulessfrom (freeqset,H1,supporData,retlist,minsuoppr)

def rulessfrom (freeqset,H,supporData,retlist,minsuoppr=0.6):
    m = len(H[0])
    while (len(freeqset) > m):
        H = calConf(freeqset,H,supporData,retlist,minsuoppr)
        if (len(H)>1):
            aprioriGen(H,m)
            m += 1
        else:
            break
def calConf(freeqset,H,supporData,retlist,minsuoppr = 0.6):
    prunedh = []
    for i in H:
        conf = supporData[freeqset]/supporData[freeqset-i]
        if conf >= minsuoppr:
            print(freeqset-i,'-->',i,'conf:',conf)
            retlist.append((freeqset-i,i,conf))
            prunedh.append(i)
    return prunedh

if __name__ == '__main__':
    dataset = loaddataset()
    L,support = apriori(dataset)
    i = 0
    for freq in L:
        if freq:
            print('项数',i+1,':',freq)
            i += 1
        else:
            continue
        rule = generateRules(L,support,minsuoppr=0.5)



