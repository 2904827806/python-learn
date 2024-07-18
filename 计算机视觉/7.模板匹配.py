import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread(r"C:\Users\29048\Desktop\lena.jpg",0)
template = cv2.imread(r"C:\Users\29048\Desktop\face.jpg",0)
h,w = template.shape[:2]


moths = ['cv2.TM_CCORR','cv2.TM_CCOEFF_NORMED','cv2.TM_CCOEFF','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
'''cv2.matchTemplate 函数用于在一张大图中查找模板图的位置。这个函数通过在大图上滑动模板图，并计算每个位置的相似度（或差异度），
来找到模板图在大图中的最佳匹配位置。cv2.minMaxLoc 函数则用于找到这个相似度（或差异度）矩阵（由 cv2.matchTemplate 返回）中的最小值和最大值，以及它们对应的位置。'''
'''res = cv2.matchTemplate(img,template,1)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)'''
'''
cv2.matchTemplate 函数返回的响应图像res（相似度矩阵）。
min_val：响应图像中的最小值。
max_val：响应图像中的最大值。
min_loc：最小值的位置（在响应图像中的坐标）。
max_loc：最大值的位置（在响应图像中的坐标）。
'''

for meth in moths:
    img2 = img.copy()

    #匹配方法的真值
    method = eval(meth)
    print(method)
    re = cv2.matchTemplate(img,template,method)
    min_va, max_va, min_lo, max_lo = cv2.minMaxLoc(re)

    #如果是平方匹配'cv2.TM_SQDIFF'或者归一化平方匹配'cv2.TM_SQDIFF_NORMED']，取最小值
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_lo
    else:
        top_left = max_lo
    bottom_right = (top_left[0]+w,top_left[1]+h)

    #画矩形
    cv2.rectangle(img2,top_left,bottom_right,(0,0,255),2)
    plt.subplot(121)
    plt.imshow(re,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(meth)
    plt.show()


'''#匹配多个对象'''
img_rgb = cv2.imread(r"C:\Users\29048\Desktop\txcl\mario.jpg")
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGRA2GRAY)

template_rbg = cv2.imread(r"C:\Users\29048\Desktop\txcl\mario_coin.jpg",0)
h1,w1 = template_rbg.shape[:2]
#模板匹配
res = cv2.matchTemplate(img_gray,template_rbg,cv2.TM_CCOEFF_NORMED)
#取匹配程度大于0.8的坐标
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    #print(pt)
    bottom_rights = (pt[0]+w1,pt[1]+h1)
    cv2.rectangle(img_rgb,pt,bottom_rights,(0,0,255),1)
cv2.imshow('imgrgb',img_rgb)
cv2.waitKey(0)


im = cv2.imread(r"C:\Users\29048\Desktop\102.png")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGRA2GRAY)
tm = cv2.imread(r"C:\Users\29048\Desktop\102(1).png",0)
print(im.shape)
h,w = tm.shape[:2]
re1 = cv2.matchTemplate(im_gray,tm,cv2.TM_CCOEFF_NORMED)
locs = np.where(re1 >= 0.7)

for ps in zip(*locs[::-1]):
    bty = (ps[0]+w,ps[1]+h)

    cv2.rectangle(im,ps,bty,(0,255,0),1)
cv2.imshow('rs',im)
cv2.waitKey(0)


