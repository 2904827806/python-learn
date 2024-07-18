import tkinter as tk
from tkinter import messagebox
import glob #读取工具
from PIL import Image,ImageTk   #图片转换工具
root = tk.Tk()
root.title('图片查看器')
root.geometry('1024x2048+100+100')
#root.resizable(0,0)
photos = glob.glob(r'C:\Users\29048\Desktop\*.png')    #读取图片
photos1 = [ImageTk.PhotoImage(Image.open(photo)) for photo in photos]
current_photo_no = 0
def prev(a):
    global current_photo_no,b
    current_photo_no += a
    if current_photo_no <= 0:
        current_photo_no = 0
        photo_lb.configure(image=photos1[current_photo_no])#重新定义照片
    elif current_photo_no > len(photos1) :
        current_photo_no = len(photos1)
        photo_lb.configure(image=photos1[-1])
    else:
        photo_lb.configure(image=photos1[current_photo_no])
    global number_var
    if current_photo_no == len(photos1):
        number_var.set(f'{current_photo_no} of {len(photos1)}')
    else:
        number_var.set(f'{current_photo_no+1} of {len(photos1)}')
photo_lb = tk.Label(root,image=photos1[current_photo_no],width=2000,height=900) #显示照片
photo_lb.pack() #布局
number_var = tk.StringVar()
number_var.set(f'{current_photo_no+1} of {len(photos1)}')
tk.Label(root,textvariable=number_var,bd=1,relief=tk.SUNKEN,anchor=tk.CENTER).pack(fill=tk.X)
#Frame（框架）组件是在屏幕上的一个矩形区域。
# Frame 主要是作为其他组件的框架基础，或为其他组件提供间距填充。
button_frame = tk.Frame(root)
button_frame.pack()
prev_photo = tk.Button(button_frame,text='上一页')
prev_photo.pack(side=tk.LEFT,anchor=tk.CENTER)
next_photo = tk.Button(button_frame,text='下一页')
next_photo.pack(side=tk.RIGHT,anchor=tk.CENTER)
prev_photo.config(command=lambda:prev(-1))
next_photo.config(command=lambda:prev(1))
root.mainloop()