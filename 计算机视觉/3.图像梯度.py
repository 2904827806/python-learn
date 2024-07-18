#一，图像梯度——sobel算子
import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread(r"C:\Users\29048\Desktop\pie.png")

def cv_show(name,image):
    #展示图像
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('pie',image)

'''#sobel算子：右-左，下-上  再合并cv2.addWeighted(sboelx,0.5,sboely,0.5,0)
det = cv2.Sobel(src,ddepth,dx,dy,ksize)
src：输入图像
ddepth：图像深度
dx：水平方向
dy：竖直方向
ksize：sobel算子的大小
'''

sboel = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
cv_show('sboel',sboel)
'''白到黑色整数，黑到白是负数，所有负数会被截断为0，所以要取绝对值'''
sboelx = cv2.convertScaleAbs(sboel)
cv_show('sboel',sboelx)

sboely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
cv_show('sboel',sboel)
'''白到黑色整数，黑到白是负数，所有负数会被截断为0，所以要取绝对值'''
# 将sboely转换为绝对值
sboely = cv2.convertScaleAbs(sboely)
# 显示sboelx图像
cv_show('sboel',sboelx)

#分别计算x,和y,再求和
sboelxy = cv2.addWeighted(sboelx,0.5,sboely,0.5,0)
cv_show('sboel',sboelxy)

def sbole_show(image):
    sboel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    #cv_show('sboel', sboel)
    '''白到黑色整数，黑到白是负数，所有负数会被截断为0，所以要取绝对值'''
    sboelx = cv2.convertScaleAbs(sboel)
    #cv_show('sboel', sboelx)
    sboely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    #cv_show('sboel', sboel)
    '''白到黑色整数，黑到白是负数，所有负数会被截断为0，所以要取绝对值'''
    sboely = cv2.convertScaleAbs(sboely)
    #cv_show('sboel', sboelx)
    # 分别计算x,和y,再求和
    sboelxy = cv2.addWeighted(sboelx, 0.5, sboely, 0.5, 0)
    sbole_show(sboelxy)

ims = cv2.imread(r"C:\Users\29048\Desktop\lena.jpg",cv2.IMREAD_GRAYSCALE)



#二、scharr算子#与laplacian算子
#scharr算子可以看作sobel算子的增强版，在图像边缘检测方面比sobel算子更准确
# 对图像ims进行Scharr算子计算，得到x方向和y方向的梯度图像
scharrx = cv2.Scharr(ims,cv2.CV_64F,1,0)
scharry = cv2.Scharr(ims,cv2.CV_64F,0,1)
# 将梯度图像转换为无符号8位整型
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
# 将x方向和y方向的梯度图像进行加权相加，得到最终的梯度图像
scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

#laplacians算子可以用来检测图像中的边缘，它是一种二阶导数算子，可以突出图像中的边缘和纹理
laplacian = cv2.Laplacian(ims,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sboelxy,scharrxy,laplacian))
cv_show('res',res)


