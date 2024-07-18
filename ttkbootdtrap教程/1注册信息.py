import time
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

#root = tk.Tk()
root = ttk.Window(themename='litera') #使用ttk创建窗口
#root = ttk.Window(themename='united')   #改变窗口的常数
root.geometry('500x400')
root.resizable(0,0)
root.title('注册页面')
root.wm_attributes('-topmost',1)

#b1 = ttk.Button(root,text='按钮1',bootstyle=SUCCESS) #创建按钮 bootstyle=SUCCESS设置颜色
#b1.pack(side=LEFT,padx=5,pady=10) #布局


#b2 = ttk.Button(root,text='按钮2',bootstyle=(SUCCESS,OUTLINE))
#b2.pack(side=LEFT,padx=5,pady=10)

"""print(root.style.colors) #打印颜色的样式

for color in root.style.colors:
    b = ttk.Button(root,text=color,bootstyle=color)
    b.pack(side=LEFT,padx=5,pady=10)"""

#ttk.Button(root,text='Button 2',bootstyle=(INFO,)).pack(side=LEFT,padx=5,pady=10)
#ttk.Button(root,text='Button 3',bootstyle=(INFO,OUTLINE)).pack(side=LEFT,padx=5,pady=10)
#ttk.Button(root,text='Button 4',bootstyle=("info","outline")).pack(side=LEFT,padx=5,pady=10)
#ttk.Button(root,text='Button 5',bootstyle=("outline-info")).pack(side=LEFT,padx=5,pady=10)


username = tk.StringVar()
password = tk.StringVar()
#0女1男-1保密
gender_str_var = tk.IntVar() #使用 IntVar() 创建一个整数变量
#兴趣爱好
hobby_list = [
    [tk.IntVar(),'吃'],
    [tk.IntVar(),'喝'],
    [tk.IntVar(),'玩'],
    [tk.IntVar(),'乐']
]
#用户信息
tk.Label(root,width=10).grid()
tk.Label(root,text='用户名：').grid(row=1,column=1,sticky=W,pady=10)
tk.Entry(root,textvariable=username,width=20).grid(row=1,column=2,columnspan=2,sticky=W)
tk.Label(root,text='密  码：').grid(row=2,column=1,sticky=W,pady=10)
tk.Entry(root,textvariable=username,width=20).grid(row=2,column=2,columnspan=2,sticky=W)
#设置单选框
tk.Label(root,text='性  别：').grid(row=3,column=1,sticky=W,pady=10)
radio_frame = tk.Frame()
radio_frame.grid(row=3,column=2,sticky=W)
ttk.Radiobutton(radio_frame,text='男',variable=gender_str_var,value=0).pack(side=tk.LEFT,padx=5)
ttk.Radiobutton(radio_frame,text='女',variable=gender_str_var,value=1).pack(side=tk.LEFT,padx=5)
ttk.Radiobutton(radio_frame,text='保密',variable=gender_str_var,value=-1).pack(side=tk.LEFT,padx=5)
#设置多选框
tk.Label(root,text='爱  好：').grid(row=4,column=1,sticky=W,pady=10)
duo_fram = tk.Frame()
duo_fram.grid(row=4,column=2,sticky=W)
ttk.Checkbutton(duo_fram,text='吃',variable=hobby_list[0][0]).pack(side=tk.LEFT,padx=5)
ttk.Checkbutton(duo_fram,text='喝',variable=hobby_list[1][0]).pack(side=tk.LEFT,padx=5)
ttk.Checkbutton(duo_fram,text='玩',variable=hobby_list[2][0]).pack(side=tk.LEFT,padx=5)
ttk.Checkbutton(duo_fram,text='乐',variable=hobby_list[3][0]).pack(side=tk.LEFT,padx=5)
tk.Label(root,text='生  日:').grid(row=5,column=1,sticky=W,pady=10)
data_entry = ttk.DateEntry(width=16)
data_entry.grid(row=5,column=2,sticky=W,pady=10)
tk.Label(root,text='').grid(row=6,column=2,sticky=W,pady=10)
button = tk.Button(root,text='提交',width=17)
button.grid(row=6,column=2,sticky=W)

root.mainloop() #显示