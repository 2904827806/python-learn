'''一、高斯金字塔
1.向下采样（缩小）
2.向上采样（放大）：
（1）将图像在每个方向扩大为原来的两倍，新增的行和列用0填充
（2）使用先前同样的内核（乘以4）与放大后的图像卷积，获得近似值
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def cv_show(name, image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread(r"C:\Users\29048\Desktop\hzo.jpg",0)

#cv_show('image',image)

#上采样
up = cv2.pyrUp(image)
#cv_show('image',up)
#下采样
done = cv2.pyrDown(image)
#cv_show('image',done)



'''二、拉普拉斯金字塔
1低通滤波
2.缩小尺寸 cv2.pyrDown()
3.放大尺寸 cv2.pyrUp()
4.图片相减
'''
'''done = cv2.pyrDown(image)
douw_up = cv2.pyrUp(done)
l_l = image - douw_up
cv_show('l-l',l_l)
done1 = cv2.pyrDown(l_l)
douw_up1 = cv2.pyrUp(done1)
l_2 = l_l - douw_up1
cv_show('l-l',l_2)
done2 = cv2.pyrDown(l_2)
douw_up2 = cv2.pyrUp(done2)
l_3 = l_2 - douw_up2
cv_show('l-l',l_3)
done3 = cv2.pyrDown(l_3)
douw_up3 = cv2.pyrUp(done3)
l_4 = l_3 - douw_up3
cv_show('l-l',l_4)'''

images = [image]
for i in range(20):
    img = images[i]
    #print(img)
    done = cv2.pyrDown(img)
    douw_up = cv2.pyrUp(done)
    douw_up = cv2.resize(douw_up,(images[i].shape[1],images[i].shape[0]))
    l_i = images[0] - douw_up
    images.append(l_i)
    cv_show(f'l-{i}', l_i)



