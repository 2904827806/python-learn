import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse

def cv_show(name,image):
    '''图像展示'''
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

images = []


fille = r"C:\Users\29048\Desktop\Scan\images"
#1文件导入
for ima in os.listdir(r"C:\Users\29048\Desktop\Scan\images"):
    wj = fille + '\\'+ima
    image = cv2.imread(wj)
    '''(h, w) = image.shape[:2]
    r = 300 / float(w)
    dim = (300, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)'''
    images.append(image)
    #cv_show('wj',image)

def order_point(pts):
    #一共四个坐标点
    #1.创建一个四行两列的0矩阵
    rect = np.zeros((4,2),dtype='float32')

    #2.按顺序找到对应坐标0123分别是左上，左下，右下，右上
    #（1）计算左上，右下
    s = pts.sum(axis=1) #按行计算，求和
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    #(2)计算右上和左下
    diff = np.diff(pts,axis=1)#等价于y-x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    #3.导出数据
    return rect

def four_point_transform(image,pts):
    '''透视变换'''
    #获取输入坐标点
    rect = order_point(pts)
    (tl,tr,br,bl) = rect

    #计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxwidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxheight = max(int(heightA), int(heightB))

    #变换后对应坐标位置
    dst = np.array([
        [0,0],
        [maxwidth - 1,0],
        [maxwidth - 1,maxheight - 1],
        [0,maxheight - 1]],dtype='float32')
    #计算变换矩阵
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxwidth,maxheight))

    #返回变换后的结果
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


#1.边缘检测
imgs = images[0]
ratio = imgs.shape[0] / 500.0
orig = imgs.copy()
imgs = resize(orig,height=500)

#1.1灰度图
imgs_gray = cv2.cvtColor(imgs,cv2.COLOR_BGRA2GRAY)

#高斯模糊
imgs_gus = cv2.GaussianBlur(imgs_gray,(5,5),0)
edged = cv2.Canny(imgs_gus,50,200)
#cv_show('edged',edged)


#1.2绘制二值图
ret1,thresh = cv2.threshold(imgs_gus,127,255,cv2.THRESH_BINARY)

#1.3进行边界检查
contoure,hiersit = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#1.4在元图上面绘制轮廓
imgs_copy = imgs.copy()

#2.1轮廓检测
locs = []

for (i,c) in enumerate(contoure):
    #计算外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    imgs_copy1 = imgs.copy()
    cv2.drawContours(imgs_copy1, c, -1, (0, 0, 255), 1)
    #cv_show('ig',imgs_copy1)
    if (100 < w) and (50 < h):
        if c not in locs:
            locs.append(c)
        else:
            continue
    else:
        continue
#cv_show('ig',imgs_copy)

#2.2遍历轮廓
screenCnt = None
for c in locs:

    #计算轮廓近似值
    peri = cv2.arcLength(c,True)
    #c表示输入的点集
    #epsilon表示从原始轮廓到近似轮廓的最大距离，它是遥感准确度参数
    #True表示封闭的
    approx = cv2.approxPolyDP(c,0.02*peri,True)

    #4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

#2.3展示结果
imgs_copy1 = imgs.copy()
cv2.drawContours(imgs_copy1,[screenCnt], -1, (0, 0, 255), 2)
cv_show('ig',imgs_copy1)


#3.变换
#3.1透视变换
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
#warped = resize(warped,height=500)
#cv_show('warped',warped)

#3.2二值处理
warped_gray = cv2.cvtColor(warped,cv2.COLOR_BGRA2GRAY)
'''kernerl = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
kerner2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
tophart = cv2.morphologyEx(warped_gray,cv2.MORPH_TOPHAT,kernerl)
cv_show('warped',tophart)'''
thresh1 = cv2.threshold(warped_gray,137,255,cv2.THRESH_BINARY)[1]
thresh1 = cv2.medianBlur(thresh1,3)
#thresh1 = resize(thresh1,height=500)
cv_show('warped',thresh1)

#3.3文字提取
# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# tesseract -v进行测试
# tesseract XXX.png 得到结果
# pip install pytesseract
# anaconda lib site-packges pytesseract pytesseract.py
# tesseract_cmd 修改为绝对路径即可
from PIL import Image, ImageDraw, ImageFont
import pytesseract  # 用于识别图像中的文本
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  # 修改为你的 Tesseract 安装路径

text = pytesseract.image_to_string(thresh1)
print(text)









