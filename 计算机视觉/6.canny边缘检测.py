#1.使用高斯滤波，以平滑图像，消除噪声
#2.计算图像中每个像素点的梯度强度和方向
#3.应用非极大值抑制，以消除边缘检测带来的杂散响应
#4.应用双阈值检测来确定真实的和潜在的边缘
#5.通过抑制孤立的弱边缘最终完成边缘检测
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def cv_show(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imge = cv2.imread(r"C:\Users\29048\Desktop\tx\car.png",cv2.IMREAD_GRAYSCALE)
#cv_show('len',imge)
'''#1.使用高斯滤波，以平滑图像，消除噪声'''

gauss = cv2.GaussianBlur(imge,(3,3),1)

'''#2.计算图像中每个像素点的梯度强度和方向(sobel算子）'''
sobelx = cv2.Sobel(gauss,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx) #转回uint8
sobely = cv2.Sobel(gauss,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
direction = np.arctan2(sobely,sobelx) #np.arctan2避免除以0的情况

'''#3.应用非极大值(Non-Maximum Suppression)抑制，以消除边缘检测带来的杂散响应
非极大值抑制：
（1）线性插值法：设g1的梯度幅度值M（g1），g2的梯度幅度值M（g2）则
dtmp1可以得到M（dtmp1） = w*M(g2)+(1-w)*M(g1)
其中w = distance(dtmp1,g2)/distance(g1,g2)。

（2）把一个像素的梯度方向离散为八个方向，这样就只需要计算前后就可，不用插值了。
'''


'''#4.应用双阈值检测来确定真实的和潜在的边缘'''
a = cv2.Canny(gauss,50,150)
b = cv2.Canny(imge,50,150)
res = np.hstack((a,b))
cv_show('a',res)
#cv2.threshold(imge,127,255,cv2.THRESH_BINARY)

'''#5.通过抑制孤立的弱边缘最终完成边缘检测'''


'''
###图像轮廓
cv2.findContours(img,mode,method)
mode:轮廓检索模式
RETR_EXTERNAL ：只检索最外面的轮廓；
RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
##最常用：RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;
method:轮廓逼近方法
CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。'''
image = cv2.imread(r"C:\Users\29048\Desktop\tx\contours.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY) #灰度图像cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY) #二值化图像
#cv_show('a',thresh)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #轮廓图

'''#绘制轮廓'''
#传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
#注意需要copy不然原图会变
draw_img = image.copy()
# 在draw_img图像上绘制contours轮廓，颜色为红色，线条宽度为2
ress = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
#cv_show('res',ress)

'''轮廓特征'''
cnt = contours[0]
#面积cv2.contourArea(cnt)
print(cv2.contourArea(cnt))
#周长cv2.arcLength(cnt,True)
print(cv2.arcLength(cnt,True))

'''轮廓近似'''
img = cv2.imread(r"C:\Users\29048\Desktop\tx\contours2.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
ret1,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours1,hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt1 = contours1[0]

draw_img1 = img.copy()
res1 = cv2.drawContours(draw_img1,[cnt1],-1,(0,0,255),2)
cv_show('aa',res1)
draw_img2 = img.copy()
# 计算轮廓的周长
epsikon = 0.1*cv2.arcLength(cnt1,True)
# 对轮廓进行多边形逼近
#cv2.approxPolyDP(cnt1,epsikon,True)
approx = cv2.approxPolyDP(cnt1,epsikon,True)
res2 = cv2.drawContours(draw_img2,[approx],-1,(0,0,255),2)
cv_show('aa',res2)


#外接矩形
x,y,w,h = cv2.boundingRect(cnt1)
imag = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv_show('aa',imag)