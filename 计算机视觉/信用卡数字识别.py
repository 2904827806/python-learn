import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
import myutils

plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
fr = r"C:\Users\29048\Desktop\template-matching-ocr\images"
images = []
templet = []
for i in os.listdir(fr):
    img = fr + '\\' + i
    if i != 'ocr_a_reference.png':
        images.append(img)
    else:
        templet.append(img)

#指定信用卡的类型
FIRST_NUMBER = {
    '3':'American Express',
    '4':'Visa',
    '5':'MasterCard',
    '6':'Discover Card'
}

#绘图展示
def cv_show(name, image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imge = cv2.imread(templet[0])
#cv_show('tmplate',imge)
#灰度图
imge_gray = cv2.cvtColor(imge,cv2.COLOR_BGRA2GRAY)
#cv_show('imge_gray', imge_gray)
#二值图
ret,thrath = cv2.threshold(imge_gray,10,255,cv2.THRESH_BINARY_INV)
#cv_show('thrath', thrath)
#边缘检测 ：RETR_EXTERNAL外轮廓
contours1, hierarchy = cv2.findContours(thrath.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #cv2.RETR_EXTERNAL表示只检测外轮廓，
# cv2.CHAIN_APPROX_SIMPLE，使得一个矩形轮廓只需4个点来保存轮廓信息

draw_img = imge.copy()
res = cv2.drawContours(draw_img, contours1, -1, (0, 0, 255), 3)
#cv_show('thrath', res)

#排序
#print(len(contours))
contours1 = myutils.sort_contours(contours1,method='left-to-right')[0]
digits = {}

#遍历每一个轮廓：获取每一个模板
for (i,c) in enumerate(contours1):
    #计算外接矩形并且resize成合适大小(将模板切分）
    x,y,w,h = cv2.boundingRect(c)
    roi = thrath[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))
    #每一个数字对应每一个模板
    digits[i] = roi
    #cv_show('im', roi)


#初始化卷积核

rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqkernerl = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读取输入图像，预处理
for im in images:
    image = cv2.imread(im)
    #cv_show('im',image)
    if image is None:
        print(f"Error: Unable to load image {im}")
        continue
    image = myutils.resize(image,width=300) #更改图像显示大小

    #灰度图
    gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    #cv_show('gray', gray)

    #礼帽操作，突出更亮区域
    tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectkernel)
    #cv_show('tophat', tophat)

    #梯度sobel算子：用来突出边缘
    gradx = cv2.Sobel(tophat,cv2.CV_32F,1,0,ksize=-1) #KSIZE = -1 相当于用3*3的
    # 计算梯度幅值的绝对值
    gradx = np.absolute(gradx)
    # 获取梯度幅值的最小值和最大值
    (minVal,maxVal) = (np.min(gradx),np.max(gradx))
    # 将梯度幅值归一化到0-255之间，并转换为uint8类型
    gradx = (255 * ((gradx-minVal) / (maxVal-minVal))).astype('uint8')
    #cv_show('gradx', gradx)
    '''grady = cv2.Sobel(tophat, cv2.CV_32F, 0, 1, ksize=-1)  # KSIZE = -1 相当于用3*3的
    grady = np.absolute(grady)
    (minValy, maxValy) = (np.min(grady), np.max(grady))
    grady = (255 * ((maxValy -grady) / (maxValy - minValy))).astype('uint8')
    gradxy = cv2.addWeighted(gradx,0.5,grady,0.5,0)
    #cv_show('gradx', gradxy)'''

    #闭操作:先膨胀，再腐蚀
    gradx = cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,rectkernel)
    #cv_show('gradxy', gradxy)

    #cv2.THRESH_OTSU:会自动寻找合适的阈值，适合双峰，需要把阈值参数设置为0
    thresh = cv2.threshold(gradx,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv_show('thresh', thresh)

    #再来一个闭操作
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernerl)
    #cv_show('thresh', thresh)

    #计算轮廓
    contour,hierarch = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = contour

    #绘制轮廓
    cur_imag = image.copy()
    cv2.drawContours(cur_imag,cnts,-1,(0,0,255),1)
    cv_show('cur_imag', cur_imag)

    locs = []
    #遍历轮廓
    for (i,c) in enumerate(cnts):
        #计算外接矩形
        (x,y,w,h) = cv2.boundingRect(c)
        ar = w/float(h) #长宽比
        mj = w*float(h)

        #选择合适的区域来执行任务
        if (2.8 < ar < 3.8) and (200.0 < mj < 2000.0):
            if (40 < w < 55) and (10 < h < 21):
                cur_imag = image.copy()
                cur_imag = cv2.rectangle(cur_imag, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv_show('aa', cur_imag)
                locs.append((x,y,w,h))

    #将符合的轮廓从左往右排列
    locs = sorted(locs,key=lambda x: x[0])
    output = []

    #遍历每一个轮廓中的数字
    for (i,(gx,gy,gw,gh)) in enumerate(locs):
        groupOutput = []

        #根据坐标提取每一组
        group = gray[gy-5:gy+gh+5,gx-5:gx+gw+5]

        #预处理
        group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv_show('group', group)

        #边缘检测
        contours2, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #排序
        contours2 = myutils.sort_contours(contours2,method='left-to-right')[0]

        #计算轮廓中的每一个数值
        for c in contours2:
            (x,y,w,h) = cv2.boundingRect(c)
            roi1 = group[y:y+h,x:x+w]
            roi1 = cv2.resize(roi1,(57,88))
            #cv_show('roi1', roi1)

            #计算匹配得分
            scores = []
            for key in digits:
                template_rbg = digits[key]
                #print(key)
                #cv_show(f'{key}', template_rbg)
                res = cv2.matchTemplate(roi1, template_rbg, cv2.TM_CCOEFF)
                (min_va, max_va, min_lo, max_lo) = cv2.minMaxLoc(res)
                scores.append(max_va)
            #得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))

        #绘制图像
        #给原图加边框
        cv2.rectangle(image,(gx-5,gy-5),(gx+gw+5,gy+gh+5),(0,0,255),1)
        #将获取的数字添加上去 #cv2.FONT_HERSHEY_SIMPLEX: 这是文本使用的字体
        #将y坐标减去15可能是为了将文本放置在稍微偏上的位置
        cv2.putText(image,''.join(groupOutput),(gx,gy-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)


        #得到结果
        output.extend(groupOutput)
    #打印结果
    print('Gredit Card Type:{}'.format(FIRST_NUMBER[output[0]]))
    print('Gredit Card # :{}'.format(''.join(output)))
    cv_show('roi1', image)


























