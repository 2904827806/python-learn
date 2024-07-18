import cv2
import matplotlib.pyplot as plt
import numpy as np
imge = cv2.imread(r"C:\Users\29048\Desktop\tx\lenaNoise.png")


def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#cv_show('imeg',imge)

#一、不同的
#1.均值滤波 :简单的平均卷积操作
blue = cv2.blur(imge,(3,3))
#cv_show('bule',blue)

#2.方框滤波 ：基本上和均值相同，可以选择归一化,但容易越界
box = cv2.boxFilter(imge,-1,(3,3),normalize=True)
#cv_show('box',box)

box1 = cv2.boxFilter(imge,-1,(3,3),normalize=False)
#cv_show('b',box1)

#3.高斯滤波：高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视的
#离得越近，重视程度更大
gaussian = cv2.GaussianBlur(imge,(3,3),1)

#4.中值滤波：相当于用中值代替
md = cv2.medianBlur(imge,5)
res = np.hstack((box,gaussian,md))
cv_show('res',res)