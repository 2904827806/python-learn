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
"""
#b1 = ttk.Button(root,text='按钮1',bootstyle=SUCCESS) #创建按钮 bootstyle=SUCCESS设置颜色
#b1.pack(side=LEFT,padx=5,pady=10) #布局


#b2 = ttk.Button(root,text='按钮2',bootstyle=(SUCCESS,OUTLINE))
#b2.pack(side=LEFT,padx=5,pady=10)

print(root.style.colors) #打印颜色的样式

for color in root.style.colors:
    b = ttk.Button(root,text=color,bootstyle=color)
    b.pack(side=LEFT,padx=5,pady=10)

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
"""

"""
import ttkbootstrap as ttk
from ttkbootstrap.constants import * #加载常数
root = ttk.Window()  #设置窗口
style = ttk.Style()   #创建一个样式对象，用于设置和查询小部件的样式
theme_names = style.theme_names()#获取 ttkbootstrap 支持的所有主题名称列表
theme_selection = ttk.Frame(root, padding=(10, 10, 10, 0))
#padding=(10, 10, 10, 0) 是一个元组，用于设置框架的内边距。
#这个元组的四个值分别对应上、右、下、左四个方向的内边距大小，单位是像素
theme_selection.pack(fill=X, expand=True)
lbl = ttk.Label(theme_selection, text="选择主题:")
#定义一个组合盒,用于创建一个下拉列表框
#style.theme.name获取当前主题的名称
#这设置了Combobox下拉列表中的所有可选值
theme_cbo = ttk.Combobox(
        theme_selection,
        text=style.theme.name,
        values=theme_names,
)
theme_cbo.pack(padx=10, side=RIGHT)
#调用这个方法后，Combobox会显示出列表中对应索引位置的项，并且这个项会被视为用户当前选中的项。
theme_cbo.current(theme_names.index(style.theme.name))
#这行代码的作用是设置Combobox（下拉列表框）的当前选中项

lbl.pack(side=RIGHT)
def change_theme(event):
    theme_cbo_value = theme_cbo.get()
    style.theme_use(theme_cbo_value)
    #这行代码使用ttk模块的style对象来更改应用程序的主题。
    #theme_use方法接受一个主题名称作为参数，并应用这个主题到整个应用程序。
    #这里，它使用了用户从下拉列表框中选择的主题名称。
    theme_selected.configure(text=theme_cbo_value) #重新定义
    theme_cbo.selection_clear()
theme_cbo.bind('<<ComboboxSelected>>', change_theme)      #绑定下拉框列表
#设置名称
theme_selected = ttk.Label(
        theme_selection,
        text="litera",
        font="-size 24 -weight bold"
)
theme_selected.pack(side=LEFT)
root.mainloop()"""

"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
#获取一定对象
sty = ttk.Style()
#获取支持的所有主题的名称
names = sty.theme_names()
te = ttk.Frame(root,padding=(10,10,10,10))
te.pack(fill=X,expand=True)
#设置一个下啦框
lib = ttk.Label(te,text='选择主题')
box = ttk.Combobox(te,text=sty.theme.name,values=names)

box.pack(side=RIGHT,padx=10)
box.current(names.index(sty.theme.name))#调用这个方法后，Combobox会显示出列表中对应索引位置的项，
# 并且这个项会被视为用户当前选中的项。设置Combobox（下拉列表框）的当前选中项
lib.pack(side=RIGHT)

def show(event):
    ab = box.get()
    sty.theme_use(ab)
    name_lib.configure(text=ab)
    box.selection_clear()

box.bind("<<ComboboxSelected>>",show)    #绑定下拉框列表
#设置名称
name_lib = ttk.Label(te,text='litera',font=24)#
name_lib.pack(side=LEFT)
root.mainloop()"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
ttk.Button(root, text="Button 1", bootstyle=SUCCESS).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 2", bootstyle=(INFO, OUTLINE)).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 3", bootstyle=(PRIMARY, "outline-toolbutton")).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 4", bootstyle="link").pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 5", bootstyle="success-link").pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 6", state="disabled").pack(side=LEFT, padx=5, pady=10) #在禁用状态下创建按钮
root.mainloop()
"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
#为按钮添加点击事件
#法一
def button1():
    print("Button1点击了一下！")
ttk.Button(root,text="Button1", bootstyle=(PRIMARY, "outline-toolbutton"),command=button1).pack(side=LEFT, padx=5, pady=10)
#法二
def button2(event): #这里要加一个参数，不然会报错
    print("Button2点击了一下！")
    button_text = event.widget["text"] #得到按钮上的文本
    print(button_text)
b = ttk.Button(root,text="Button2", bootstyle=(PRIMARY, "outline-toolbutton"))
b.pack(side=LEFT, padx=5, pady=10)
b.bind("<Button-1>", button2) #<Button-1>鼠标左键
root.mainloop()"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
tle = ttk.Style()
names = tle.theme_names()
#print(names)
them_select = ttk.Frame(root,padding=(20,20,20,20))
them_select.pack(fill=X,expand=True)
lab = ttk.Label(them_select,text='选择主题')
them_cob = ttk.Combobox(them_select,text=tle.theme.name,values=names)
them_cob.pack(side=RIGHT,padx=10)
them_cob.current(names.index(tle.theme.name))
lab.pack(side=RIGHT)
def chang_them(event):
    them_cob_use = them_cob.get()
    them_selectd.configure(text=them_cob_use)
    tle.theme_use(them_cob_use)

them_selectd = ttk.Label(them_select,text='liter',font=24)
them_selectd.pack(side=LEFT)

them_cob.bind("<<ComboboxSelected>>",chang_them)

root.mainloop()"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
root.geometry('1000x500')
e1 = ttk.Entry(root)
e1.insert("0",'默认内容')
e1.grid(row=5,column=1,sticky=W,padx=5)
e2 = ttk.Entry(root,show="*",width=50,bootstyle=PRIMARY)
e2.grid(row=10,column=1,sticky=W,padx=5)
b = ttk.Scrollbar()
e3 = ttk.Entry(root,textvariable=b)
e3.grid(row=15,column=1,sticky=W,padx=5)
def dhow():
    print(e1.get())
    print(e2.get())
    print(e3.get())

ttk.Button(root,text='a',command=dhow).grid(row=20,column=1,sticky=W)
root.mainloop()"""

"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
tely = ttk.Style()
them_names = tely.theme_names()
them_selec = ttk.Frame(root,padding=(10,10,10,10))
them_selec.pack(fill=X,expand=True)
lal = ttk.Label(them_selec,text='选择主题')
them_cob = ttk.Combobox(them_selec,text=tely.theme.name,values=them_names)
them_cob.pack(side=RIGHT,padx=10)
them_cob.current(them_names.index(tely.theme.name))
lal.pack(side=RIGHT)
them = ttk.Label(them_selec,text='litera',font=24)
them.pack(side=LEFT)
def show(event):
    them_cob_value = them_cob.get()
    them.configure(text=them_cob_value)
    tely.theme_use(them_cob_value)
    them_cob.select_clear()

them_cob.bind('<<ComboboxSelected>>',show)
root.mainloop()"""


"""ttk.Label(root,text='标签1',bootstyle=INFO).pack(side=LEFT)
ttk.Label(root,text='标签2',bootstyle=SUCCESS).pack(side=LEFT)
ttk.Label(root,text='标签3',bootstyle="inverse-danger").pack(side=LEFT)
ttk.Label(root,text='标签4',bootstyle=WARNING).pack(side=LEFT)
root.mainloop()"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
ttk.Button(root, text="Button 1", bootstyle=WARNING).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 2", bootstyle=PRIMARY).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 3", bootstyle=(INFO, OUTLINE)).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 4", bootstyle=SUCCESS).pack(side=LEFT, padx=5, pady=10)
ttk.Button(root, text="Button 5", bootstyle='disabled').pack(side=LEFT, padx=5, pady=10)
root.mainloop()
"""
"""def show():
    print('button1点击了一下')
def show1(event):
    a = event.widget["text"]
    print(a)
    print('button2点击了一下')
ttk.Button(root, text="Button 1", bootstyle=WARNING,command=show).pack(side=LEFT, padx=5, pady=10)
b = ttk.Button(root, text="Button 2", bootstyle=PRIMARY)
b.pack(side=LEFT, padx=5, pady=10)
b.bind("<Button-1>",show1)
root.mainloop()"""
"""e1 = ttk.Entry(root,show=None)
e1.insert('0',"默认插入内容")
e1.grid(row=5,column=1,sticky=ttk.W,padx=10,pady=10)
e2 = ttk.Entry(root,show='*',width=50,bootstyle=PRIMARY)

e2.grid(row=10,column=1,sticky=ttk.W,padx=10,pady=10)
a = ttk.StringVar()
e3 = ttk.Entry(root,textvariable=a)

e3.grid(row=15,column=1,sticky=ttk.W,padx=10,pady=10)
ttk.Button(root,bootstyle=SUCCESS,text="开动").grid(row=20,column=1,sticky=ttk.W,padx=10,pady=10)
root.mainloop()"""
#时间表
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
de1 = ttk.DateEntry()
de1.grid(row=5,column=1,stick=ttk.W,padx=10,pady=10)
de2 = ttk.DateEntry(bootstyle=SUCCESS,dateformat=r'%Y')
de2.grid(row=5,column=2,sticky=ttk.W,padx=1,pady=10)
def show():
    print(de1.entry.get())
ttk.Button(root,text='get',bootstyle=PRIMARY,command=show).grid(row=6,column=1,sticky=ttk.W,padx=1,pady=10)
root.mainloop()"""


"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
variable_value_dist = {
    "0":"男",
    "1":"女",
    "2":"未知"
}
a = ttk.IntVar()
ttk.Radiobutton(root,text='男',variable=a,value=0).pack(side=ttk.LEFT)
ttk.Radiobutton(root,text='女',variable=a,value=1).pack(side=ttk.LEFT)
ttk.Radiobutton(root,text='未知',variable=a,value=2).pack(side=ttk.LEFT)
def show():
    print(a.get())
b = ttk.IntVar()
c =ttk.IntVar()
d=ttk.IntVar()
ttk.Button(root,text='确实',command=show).pack(side=ttk.LEFT,padx=5)
ttk.Checkbutton(root,text='123',variable=b).pack(side=ttk.LEFT, padx=5)
ttk.Checkbutton(root,text='456',variable=c).pack(side=ttk.LEFT, padx=5)
ttk.Checkbutton(root,text='789',variable=d).pack(side=ttk.LEFT, padx=5)

root.mainloop()
"""
"""c = ttk.Combobox(root,bootstyle=SUCCESS,font=24,values=['1','2','3'])
c.current(1)
c.pack()
def  ensure(event):
    print(c.get())

c.bind('<<ComboboxSelected>>', ensure)"""
"""f = ttk.Frame(root,bootstyle=SUCCESS)
f.place(x=10,y=10,width=600,height=100)
ef = ttk.LabelFrame(root,text='提示',bootstyle=PRIMARY,width=100,height=60)
ef.place(x=10,y=210,width=300,height=100)
ttk.Label(ef,text='标签').pack()
ttk.Button(ef,text='按钮').pack()"""
"""import psutil,time,threading
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
ttk.Meter(
        master=root,
        bootstyle=DEFAULT,
        metertype="full",#将仪表显示为一个完整的圆形或半圆形（semi）
        wedgesize=5, #设置弧周围的指示器楔形长度,如果大于 0，则此楔形设置为以当前仪表值为中心的指示器
        amounttotal=50, #仪表的最大值，默认100
        amountused=10, #仪表的当前值
        metersize=200,#仪表大小
        showtext=True, #指示是否在仪表上显示左、中、右文本标签
        interactive=True, #是否可以手动调节数字的大小
        textleft='左边', #插入到中心文本左侧的短字符串
        textright='右边',
        textfont="-size 30", #中间数字大小
        subtext="文本",
        subtextstyle=DEFAULT,
        subtextfont="-size 20",#文本大小
        ).pack(side=ttk.LEFT, padx=5)
def _():
        meter = ttk.Meter(
                metersize=180,
                padding=50,
                amountused=0,
                metertype="semi",
                subtext="当前网速(kB/s)",
                subtextstyle="warning",
                interactive=False,
                bootstyle='primary',
                )
        meter.pack(side=ttk.LEFT, padx=5)
        while True:
                meter.configure(amountused=round(getNet(),2))
def getNet():
    recv_before = psutil.net_io_counters().bytes_recv
    time.sleep(1)
    recv_now = psutil.net_io_counters().bytes_recv
    recv = (recv_now - recv_before)/1024
    return recv

t = threading.Thread(target=_)
t.setDaemon(True)
t.start()
root.mainloop()"""

import time,threading
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window(size=(500,380))
f = ttk.Frame(root).pack(fill=BOTH,expand=True)
p1 = ttk.Progressbar(f,bootstyle=SUCCESS)
p1.place(x=20,y=20,width=380,height=48)
#p1.start()# 间隔默认为50毫秒（20步/秒）
#p1.step(10)
a = 0
while True:
    a += 1
    for i in (0,100,5):
        p1.step(i)
        root.update()
        time.sleep(0.1)
    if a == 20:
        break
p1.step(100)
root.update()
root.mainloop()
"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import time,threading
root = ttk.Window()"""
"""a = ttk.Scale(root,orient=HORIZONTAL,value=75,from_=0,to=100) #垂直（VERTICAL）或水平（HORIZONTAL）方向
a.pack(fill=X,padx=5,expand=True)
root.mainloop()"""

"""def scale():
    s2 = ttk.Scale(
        master=root,
        bootstyle=SUCCESS,
        orient=VERTICAL, #位置树值
        value=0,
        from_=100,
        to=0
    )
    s2.pack(fill=X, pady=5, expand=YES)
    for i in range(101):
        s2.configure(value=i)
        time.sleep(0.1)

t = threading.Thread(target=scale)
#t.setDaemon(True)
t.start()
root.mainloop()
#ttk.Progressbar进度条
#ttk.Scale
"""
"""p1 = ttk.Progressbar(root,bootstyle=INFO)
p1.pack(fill=X)
#p1.start()
p1.step(10)
for i in range(0,50,5):
    p1.step(i)
    time.sleep(1)
    root.update()#更新窗口"""
#ttk.Scale(root,orient=HORIZONTAL,value=75,from_=0,to=100).pack(fill=X,padx=5,pady=10,expand=True)
#水尺ttk.Floodgauge()
"""t2 = ttk.Floodgauge(root,bootstyle=SUCCESS,length=100,maximum=10,mode=INDETERMINATE,orient=VERTICAL,text='文本')
t2.pack(side=ttk.LEFT,padx=5)
t2.start()"""
"""import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import time,threading
root = ttk.Window()
t3 = ttk.Floodgauge(root, bootstyle=INFO, length=300, orient=HORIZONTAL, maximum=0, mask='loading...{}')
t3.pack(side=LEFT, padx=5)
for i in range(500):
    t3.configure(maximum=i)
    t3.start()
    root.update()
root.mainloop()"""
#滚动条ScrolledText()
"""f = ttk.Frame(root).pack(fill=ttk.BOTH,expand=True)
text_content = '''
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
'''
st = ScrolledText(f,padding=5,height=10,autohide=True)
st.pack(fill=BOTH,expand=True)
st.insert(END,text_content)"""
#查询框
#from ttkbootstrap.dialogs import Querybox
#子窗口ttk.Toplevel
"""root.wm_attributes('-topmost', 1)#让主窗口置顶
def my():
    ttk.Style("solar")
    #print(ttk.Style().theme_names())#可设置主题风格['cyborg', 'journal', 'darkly', 'flatly', 'solar', 'minty', 'litera', 'united', 'pulse', 'cosmo', 'lumen', 'yeti', 'superhero']
    mytoplevel = ttk.Toplevel(root,alpha=0.5)##里面的参数和Window()父窗口一致
ttk.Button(text="my_Toplevel ",command=my).pack()
root.mainloop()"""

#打开网页
"""import webbrowser
root = ttk.Window()
def open_url(event):
    webbrowser.open("http://www.baidu.com", new=0)  # 启动web浏览器访问给定的URL
label = ttk.Label(root, text="https://www.baidu.com/", bootstyle=PRIMARY)
label.pack(fill=BOTH)
label.bind("<Button-1>", open_url)
root.mainloop()"""


























