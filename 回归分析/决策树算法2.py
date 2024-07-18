# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
#FontProperties 类是 font_manager 模块中设置字体的各种属性的模块
import matplotlib.pyplot as plt
from math import log
import operator
#operator 模块提供了一系列内置运算符的函数版本


def createDataSet():

    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels
def createTree(dataset,labels,featlabels):
    """
    :param dataset: 数据集
    :param labels: 标签
    :param featlabels: 按顺序排列标签
    """
    #迭归调用
    #获取每个数据的判断列表
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        #如果列表中全是同一类别，返回结果值
        return classlist[0]
    if len(dataset[0]) == 1:
        #如果数据中只剩下标识数据时，返回数据
        return majoritycnt(classlist)
    #选择最好的特征索引进行分割
    bestFeat = chooseBestFeatureToSplit(dataset)
    #print(bestFeat)
    #获取当前索引对应的标签
    bestFeatlabel = labels[bestFeat]
    #print(bestFeatlabel)
    featlabels.append(bestFeatlabel)
    myTree = {bestFeatlabel:{}} #放置树状分支
    del labels[bestFeat] #删除数据
    featValue = [example[bestFeat] for example in dataset]
    #print(featValue)
    #统计要分多少支
    uniquevals = set(featValue)
    for value in uniquevals:
        sublabels = labels[:]
        myTree[bestFeatlabel][value] = createTree(splitDataSet(dataset,bestFeat,value),sublabels,featlabels)
    return myTree

def majoritycnt(classlist):#
    #统计数量，找到数量最多的类
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        else:
            classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def chooseBestFeatureToSplit(dataset):
    '''选择最好的特征索引'''
    #获取特征个数
    numFeatures = len(dataset[0]) - 1
    #1.基础熵值
    baseEntropy = clacShannoEnt(dataset)#计算初始熵值
    bestInfoGain = 0
    baseFeatur = -1 #初始化特征序列
    for i in range(numFeatures): #循环特征序列
        featList = [example[i] for example in dataset] #获取每个特征的数据列表
        uniqueVals = set(featList) #获取特征值列表的唯一值
        newEntropy = 0
        for val in uniqueVals:
            #遍历计算数据中的每一个分支
            #切分数据集
            subDataSet = splitDataSet(dataset,i,val)
            #print(subDataSet)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * clacShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            baseFeatur = i
    #print(baseFeatur)
    return baseFeatur


def splitDataSet(dataset,axis,val):
    #数据切分
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == val:
            #print(dataset)
            #去除第一个最优特征
            reducedFeatVec = featVec[:axis]
            #print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet



def clacShannoEnt(dataset):
    '''计算熵值'''
    numexamples = len(dataset)
    labelcount = {}
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelcount.keys():
            labelcount[currentlabel] = 0
        labelcount[currentlabel] += 1
    shannoEnt = 0 #设置初始熵值为0
    for key in labelcount:
        prob = float(labelcount[key])/numexamples
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontsize=12)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataset,labels,featLabels)
    createPlot(myTree)


