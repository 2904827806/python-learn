# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# tesseract -v进行测试
# tesseract XXX.png 得到结果 
# pip install pytesseract
# anaconda lib site-packges pytesseract pytesseract.py
# tesseract_cmd 修改为绝对路径即可
from PIL import Image, ImageDraw, ImageFont
import pytesseract #用于识别图像中的文本
import cv2
import os


preprocess = 'blur' #thresh

image = cv2.imread(r"C:\Users\29048\Desktop\Scan\scan.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
    
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
    
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
# 创建一个可以在给定图像上绘图的对象
draw = ImageDraw.Draw(image)

# 设置字体和大小（注意：这里需要你的系统中已经安装了该字体）
# 如果不指定字体，PIL将使用默认字体，但可能不支持中文
# 这里我们使用'arial.ttf'作为示例，你需要替换为你的字体文件路径
font = ImageFont.truetype("arial.ttf", 36)  # 替换为你的字体文件路径和大小

# 设置文本颜色和位置
text_color = (255, 255, 255)  # 白色
position = (10, 10)  # 文本开始的位置，左上角为(0,0)

# 在图像上绘制文本
draw.text(position, text, fill=text_color, font=font)

# 显示图像（或保存图像）
image.show()  # 这将使用默认的图像查看器打开图像
# 或者，保存图像到文件
# img.save('path_to_save_image.jpg')
