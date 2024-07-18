import cv2
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\29048\Desktop\cat.jpg") #读取图像数据
def cv_show(name,image):
    #绘图操作
    cv2.imshow(name, image)
    cv2.waitKey(0) #等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()  # 点击关闭时关闭所有窗口

#cv_show('image',image)




print(image.shape)
#cv2.IMREAD_COLOR :彩色图像
#cv2.IMREAD_GRAYSCALE:灰度图像
#读取灰度图像
img = cv2.imread(r"C:\Users\29048\Desktop\cat.jpg",cv2.IMREAD_GRAYSCALE)
#cv_show('hd',img)


#读取视频
simage = cv2.VideoCapture(r"C:\Users\29048\Desktop\test.mp4")

#检查是否正确打开
'''if simage.isOpened():
    open,frame = simage.read()
else:
    open = False
while open:
    ret,frame = simage.read()
    if frame is None:
        break
    if ret == True:
        #设置显示色彩为灰色
        gray = cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
        cv2.imshow('ret',gray)
        if cv2.waitKey(10) & 0xFF == 27:
            break
simage.release()
cv2.destroyAllWindows()
'''
#截取部分图像数据
cat = image[0:200,0:200]
#cv_show('car',cat)

#颜色通道提取
b,g,r = cv2.split(image)
print(b)

'''cut = image.copy()
cut[:,:,0] = 0
cut[:,:,1] = 0
cv_show('r',cut)
cut1 = image.copy()
cut1[:,:,1] = 0
cut1[:,:,2] = 0
cv_show('r',cut1)
cut2 = image.copy()
cut2[:,:,0] = 0
cut2[:,:,2] = 0
cv_show('r',cut2)'''
replicate = cv2.copyMakeBorder(image,50,50,50,50,borderType=cv2.BORDER_REPLICATE) #复制法，复制最边缘像素
reflect = cv2.copyMakeBorder(image,50,50,50,50,borderType=cv2.BORDER_REFLECT)#反射法，对感兴趣的图像中的像素在两边进行复制
reflect101 = cv2.copyMakeBorder(image,50,50,50,50,borderType=cv2.BORDER_REFLECT101)#反射法，以最边缘像素为轴，对称
wrap = cv2.copyMakeBorder(image,50,50,50,50,borderType=cv2.BORDER_WRAP)#外包装法
constan = cv2.copyMakeBorder(image,50,50,50,50,borderType=cv2.BORDER_CONSTANT,value=0)#常值法，常数值填充
#边界填充
plt.subplot(231)
plt.imshow(image,'gray')
plt.title('original')
plt.subplot(232)
plt.imshow(replicate,'gray')
plt.title('REPl')
plt.subplot(233)
plt.imshow(reflect,'gray')
plt.title('FLC')
plt.subplot(234)
plt.imshow(reflect101,'gray')
plt.title('101')
plt.subplot(235)
plt.imshow(wrap,'gray')
plt.title('WA')
plt.subplot(236)
plt.imshow(constan,'gray')
plt.title('CT')
plt.show()


#数值计算
img_cat = cv2.imread(r"C:\Users\29048\Desktop\cat.jpg")
ima_dog = cv2.imread(r"C:\Users\29048\Desktop\dog.jpg")
ima_cat2 = img_cat + 10

#维度相同才能直接想加，超过0-256会自动取余压缩好展示
a = (img_cat + ima_cat2)[:5,:,0]
print(a)

#图像融合
#cv2.addWeighted(img_cat,0.4,ima_dog,0.6,0)

#列子，猫加狗
#"C:\Users\29048\Desktop\cat.jpg"+"C:\Users\29048\Desktop\dog.jpg"

#改变融合图像的大小cv2.resize(ima_dog,(500,414))
#cv2.resize(img_cat,(500,414)
ma_dog = cv2.resize(ima_dog,(500,414))

#融合cv2.addWeighted(ima_dog,0.4,img_cat,0.6,0)

#gamma值，这是一个可选参数，用于调整结果的亮度
#alpha,beta  都是权重参数
print(ima_dog.shape)
print(img_cat.shape)
res = cv2.addWeighted(img_cat,0.4,ima_dog,0.6,0)
plt.imshow(res)
plt.show()

#按照倍数进行放缩
res1 = cv2.resize(img_cat,(0,0),fx=3,fy=1)
plt.imshow(res1)
plt.show()

#保存图像
cv2.imwrite('my.png',res)
