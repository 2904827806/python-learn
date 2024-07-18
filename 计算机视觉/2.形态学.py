#一形态学-腐蚀操作erode
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"C:\Users\29048\Desktop\dige.png")
def cv_show(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#cv_show('img',img)
#腐蚀操作，去除多出来的毛刺

# 创建一个 5x5 的结构元素，全为 1的卷积核 np.ones((5,5),np.uint8) 或者cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel = np.ones((5,5),np.uint8)


# 进行腐蚀操作，指定迭代次数(iterations）为 1（您可以根据需要调整这个值）cv2.erode(图像,卷积核,iterations=1：迭代次数)
erosion = cv2.erode(img,kernel,iterations=1)
#cv_show('erosion',erosion)




pie = cv2.imread(r"C:\Users\29048\Desktop\pie.png")
#cv_show('pie',pie)
kernel1 = np.ones((30,30),np.uint8) #卷积核
erosion_1 = cv2.erode(pie,kernel1,iterations=1)
erosion_2 = cv2.erode(pie,kernel1,iterations=2)
erosion_3 = cv2.erode(pie,kernel1,iterations=3)
res = np.hstack((erosion_1,erosion_2,erosion_3))
#cv_show('res',res)


#二、膨胀操作
#1.先腐蚀去毛刺
#2.膨胀:cv2.dilate(图像,卷积核,iterations=1：迭代次数)

kernels = np.ones((3,3),np.uint8)
dige_dilate = cv2.dilate(erosion,kernels,iterations=1)
#cv_show('dilate',dige_dilate)
kernels1 = np.ones((30,30),np.uint8)
erosion_1s = cv2.dilate(pie,kernels1,iterations=1)
erosion_2s = cv2.dilate(pie,kernels1,iterations=2)
erosion_3s = cv2.dilate(pie,kernels1,iterations=3)
ress = np.hstack((erosion_1s,erosion_2s,erosion_3s))
#cv_show('ress',ress)

#三，开运算与闭运算

#1.开：先腐蚀，在膨胀 相当于 dilate(image) ---> erode(image)
#cv2.morphologyEx(图像,cv2.MORPH_OPEN,卷积核)

imge = cv2.imread(r"C:\Users\29048\Desktop\dige.png") #读入图像
krenel5 = np.ones((5,5),np.uint8) #创建一个 5x5 的结构元素，全为 1的卷积核  也可以使用 cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
opening = cv2.morphologyEx(imge,cv2.MORPH_OPEN,krenel5)
cv_show('open',opening)

#2.闭：先膨胀，再腐蚀(毛刺可能不会消失）
#cv2.morphologyEx(图像,cv2.MORPH_CLOSE,卷积核) 相当于 erode(image) ---> dilate(image)

imge = cv2.imread(r"C:\Users\29048\Desktop\dige.png") #读入图像
krene5 = np.ones((5,5),np.uint8)#创建卷积核
clossing = cv2.morphologyEx(imge,cv2.MORPH_CLOSE,krene5)
cv_show('close',clossing)

#四,梯度运算（减肥操作）
#梯度 = 膨胀-腐蚀

pies = cv2.imread(r"C:\Users\29048\Desktop\pie.png")
kerneel = np.ones((7,7),np.uint8)
dialet = cv2.dilate(pies,kerneel,iterations=5) #膨胀
erosio = cv2.erode(pie,kerneel,iterations=5) #腐蚀
res_v = np.hstack((dialet,erosio))
cv_show('res_v',res_v)
gradient = cv2.morphologyEx(pies,cv2.MORPH_GRADIENT,kerneel)
cv_show('gradient',gradient)


#五礼帽和黑帽
#1.礼帽 = 原始输入-开运算结果cv2.MORPH_TOPHAT 可以用来突出比周围区域更明亮的区域
im = cv2.imread(r"C:\Users\29048\Desktop\dige.png")
cv_show('im',im)
tophat = cv2.morphologyEx(im,cv2.MORPH_TOPHAT,kerneel)
cv_show('tophat',tophat)
#2.黑帽 = 闭运算 - 原始输入cv2.MORPH_BLACKHAT 可以用来突出比周围区域更暗的区域
im = cv2.imread(r"C:\Users\29048\Desktop\dige.png")
blackhat = cv2.morphologyEx(im,cv2.MORPH_BLACKHAT,kerneel)
cv_show('blackhat',blackhat)


'''cv2.erode(img,kernel,iterations=1) 腐蚀运算，图像，卷积核，迭代次数
cv2.dilate(img,kernel,iterations=1) 膨胀运算，图像，卷积核，迭代次数
cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) 开运算，图像，卷积核
cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)  闭运算，图像，卷积核
cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)  梯度运算，图像，卷积核
cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel) 礼帽运算，图像，卷积核 突出白的
cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel) 黑帽运算，图像，卷积核 突出黑的
'''