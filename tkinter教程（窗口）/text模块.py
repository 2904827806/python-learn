#tkinter 模块（窗口模块）
import tkinter as tk

from tkinter import messagebox
from random import random
"""root.title('标题名')    　　 　　修改框体的名字,也可在创建时使用className参数来命名；
root.resizable(0,0)   　　 　　框体大小可调性，分别表示x,y方向的可变性；
root.geometry('250x150')　　指定主框体大小；
root.quit()        　　　　 　　 退出；
root.update_idletasks()
root.update()        　　　　　刷新页面；
Button        　　按钮；
Canvas        　　绘图形组件，可以在其中绘制图形；
Checkbutton      复选框；  
Radiobutton      单选框
Entry        　　 文本框（单行）；Entry
Text             文本框（多行）； Text 
Frame         　　框架，将几个组件组成一组 Frame
Label        　　 标签，可以显示文字或图片； Label 
Listbox      　　 列表框； Listbox
Menu     　　     菜单；Menu
Menubutton       它的功能完全可以使用Menu替代； 
Message          与Label组件类似，但是可以根据自身大小将文本换行；
Radiobutton      单选框；
Scale      　　   滑块；允许通过滑块来设置一数字值 Scale
Scrollbar        滚动条；配合使用canvas, entry, listbox, and text窗口部件的标准滚动条；
Toplevel         用来创建子窗口窗口组件。toplevel
（在Tkinter中窗口部件类没有分级；所有的窗口部件类在树中都是兄弟。）

"""
root = tk.Tk()   # 生成主窗口

root.geometry('1000x600+100+100')  #设置主窗口的大小 # (宽度x高度)+(x轴+y轴)
root.resizable(0, 0)  # 设置窗口大小不可变
root.title('百宝箱') #设置窗口名称’标题‘
frame1 = tk.Frame(root)    #窗口1
frame1.pack() #布局显示
tk.Label(frame1,text='尊敬的导师：',font=('黑体',24),padx=30,pady=30).pack(side= tk.LEFT,anchor=tk.N)  #生成标签
            #label.pack()        #将标签添加到主窗口
#file = r'C:\Users\29048\Desktop\2582.jpg'        #图片所在的位置
#img = tk.PhotoImage(file =file)                #加载图片
img_abe = tk.Label(frame1,text='拜拜,老子不想干了',font=('黑体',35),justify=tk.CENTER,height=20
         ,fg='red',pady=300)  #图片布局
img_abe.pack(side= tk.LEFT,anchor=tk.CENTER)  #相对布局
tk.Label(frame1, text='申请人：王**：',height=30, font=('黑体', 14), padx=30, pady=30, anchor=tk.S).pack(side=tk.LEFT)  # 生成标签 ，因为处于整个容器下方，所以anchor处于内部
#yes_img = tk.PhotoImage(file=r'C:\Users\29048\Desktop\02145.png')
#no_img = tk.PhotoImage(file=r'C:\Users\29048\Desktop\25852.png')
yes_btn = tk.Button(frame1,text='同意',bd=0,font=25,bg='blue')    #定义按钮
yes_btn.place(relx=0.4,rely=0.8,anchor=tk.CENTER)                #绝对布局
no_btn = tk.Button(frame1,text='不同意',bd=0,font=25,bg='green')
no_btn.place(relx=0.7,rely=0.8,anchor=tk.CENTER)          #位置居中

frame2 = tk.Frame(root)    #窗口2
frame2.pack()
tk.Label(frame2,
             text='导师大人，臣退了,\n这一退，可能就是一辈子 \n！！！ 拜拜',
         font=('黑体',35),justify=tk.CENTER,height=20
         ,fg='red',pady=300
         ).pack()    #justify 是存在多行文本时的对其方式
tc = tk.Button(frame2,text='退出')   #command 给予他一个命令
tc.place(relx=0.95,rely=0.8)
def on_eixt():
    messagebox.showwarning(title='提示',message='此路不通')
root.protocol('WM_DELETE_WINDOW',on_eixt)  # 重置事件，退出事件WM_DELETE_WINDOW 抓取退出的x键

#随机移动
def move(event):    #event表示响应事件event对象（def function(event)）：
    no_btn.place(relx=random(),rely=random(),anchor=tk.CENTER)

no_btn.bind('<Enter>',move)       #bind关联事件
def text(event):   #执行与鼠标有关的事件一定要在对象中加入event
    if messagebox.askyesno('退出', '确定要退出吗？'): #如果用户点击“是”，这个函数会返回True；如果用户点击“否”，它会返回False。
        #messagebox模块中的一个函数，用于显示一个带有“是”和“否”按钮的消息框。
        root.quit()
tc.bind('<Button-1>',text)
def yest(event):
    return frame1.pack_forget()



yes_btn.bind('<Button-1>',yest)
#yes_btn.config(command=)

root.mainloop()        #进入消息循环，展示窗口

#打包
#先在终端输入 pip freeze >  .txt
#开始打包
#多文件打包

#
