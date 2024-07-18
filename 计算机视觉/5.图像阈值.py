import cv2
import matplotlib.pyplot as plt
import numpy as np
imge_gray = cv2.imread(r"C:\Users\29048\Desktop\tx\lenaNoise.png",cv2.IMREAD_GRAYSCALE)
imge_gray = cv2.medianBlur(imge_gray,5)
#ret,dst = cv2.threshold(imge,thresh,maxval,type)
# 对灰度图像进行二值化处理

ret,thresh1 = cv2.threshold(imge_gray,127,255,cv2.THRESH_BINARY)
# 对灰度图像进行反二值化处理
ret,thresh2 = cv2.threshold(imge_gray,127,255,cv2.THRESH_BINARY_INV)
# 对灰度图像进行截断处理
ret,thresh3 = cv2.threshold(imge_gray,127,255,cv2.THRESH_TRUNC)
# 对灰度图像进行零处理
ret,thresh4 = cv2.threshold(imge_gray,127,255,cv2.THRESH_TOZERO)
# 对灰度图像进行反零处理
ret,thresh5 = cv2.threshold(imge_gray,127,255,cv2.THRESH_TOZERO_INV)
print(ret)

def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
titles = ['oringinal image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images=[imge_gray,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(len(titles)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()