"""
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import re
import time
import requests
from tkinter import filedialog
import os

root = tk.Tk()
root.geometry('650x500+100+100')
root.resizable(0,0)
root.title('')
# side：指定组件在窗口中的放置位置，可选值为LEFT、RIGHT、TOP、BOTTOM。
# BOTTOM 由下到上排列 fill=tk.X沿水平方向填充，TOP从上到下
# fill：指定组件在窗口中的填充方式，可选值为X、Y、BOTH。
# both,上下左右都填充 ，
# expand：指定组件是否随窗口的大小改变而自动扩展，可选值为True或False。
# anchor：指定组件在窗口中的对齐方式，可选值为N、S、E、W、NE、NW、SE、SW。
# padx：指定组件的水平内边距（以像素为单位）。
# pady：指定组件的垂直内边距（以像素为单位）。
# Frame 　　框架，将几个组件组成一组
#布置左边
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)

#发送网络设置
fs_fram = tk.LabelFrame(left_frame,text='网络设置',padx=5,pady=5)
fs_fram.pack()
tk.Label(fs_fram,text='(1)协议类型').pack(anchor=tk.W)
#

#下拉框设置
xl = ttk.Combobox(fs_fram)
xl['values'] = ['TCP的服务器','TCP的客服端']
xl.pack()

tk.Label(fs_fram,text='(2)本地主机地址').pack(anchor=tk.W)
xl2 = ttk.Combobox(fs_fram)
ip = requests.get('https://myip.ipip.net', timeout=5).text  # 获取IP地址
fip = re.findall(r'当前 IP：(.*?) 来自于',ip)[0]
xl2['values'] = [f'{fip}']
xl2.pack()
tk.Label(fs_fram,text='(3)主机端口').pack(anchor=tk.W)
xl3 = ttk.Entry(fs_fram)
xl3.pack(fill=tk.X)
fl = ''
def opens():
    global fl,text
"""
import json

"""import json
import time
tkinter.filedialog.asksaveasfilename():选择以什么文件名保存，返回文件名
tkinter.filedialog.asksaveasfile():选择以什么文件保存，创建文件并返回文件流对象
tkinter.filedialog.askopenfilename():选择打开什么文件，返回文件名
tkinter.filedialog.askopenfile():选择打开什么文件，返回IO流对象
tkinter.filedialog.askdirectory():选择目录，返回目录名
tkinter.filedialog.askopenfilenames():选择打开多个文件，以元组形式返回多个文件名
tkinter.filedialog.askopenfiles():选择打开多个文件，以列表形式返回多个IO流对象"""
"""
    fle = filedialog.askopenfilename(defaultextension='txt')
    if fle == '':
        fl = None
    else:
        fl = fle
        fopen = open(fl,'r',encoding='utf-8')
        text.insert(1.0,fopen.read())
        fopen.close()
def close():
    global text
    text.delete(1.0,tk.END)
#按钮镶嵌
an = tk.Frame(fs_fram)
an.pack()
an1 = tk.Button(an,text='打开',font=5,padx=5,pady=2)
an1.pack(side=tk.LEFT,anchor=tk.CENTER)
an1.config(command=opens)
an2 = tk.Button(an,text='关闭',font=5,padx=5,pady=2)
an2.pack(side=tk.RIGHT,anchor=tk.CENTER)
an2.config(command=close)
#接收设置
js = tk.LabelFrame(left_frame,text='接收设置')
js.pack(side=tk.TOP,fill=tk.X)
#单选和多选设置
tk.Radiobutton(js,text='UTF-8').pack(anchor=tk.W)
tk.Radiobutton(js,text='GBK').pack(anchor=tk.W)
tk.Checkbutton(js,text='解析为JSON数据').pack(anchor=tk.W)
tk.Checkbutton(js,text='自动换行').pack(anchor=tk.W)

#发送设置
fs = tk.LabelFrame(left_frame,text='接收设置')
fs.pack(side=tk.TOP,fill=tk.X)
#单选和多选设置
tk.Radiobutton(fs,text='UTF-8').pack(anchor=tk.W)
tk.Radiobutton(fs,text='GBK').pack(anchor=tk.W)
tk.Checkbutton(fs,text='数据加密').pack(anchor=tk.W)
tk.Checkbutton(fs,text='信息接受').pack(anchor=tk.W)

#右边设置
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT,anchor=tk.N,padx=5,pady=5)

#数据日志
ino = tk.Frame(right_frame)
ino.pack()
tx = tk.StringVar()
a = time.time()
ad = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(a))





tx.set(f'数据日志--时间：{ad}')
rz = tk.Label(ino,textvariable=tx)
rz.pack(anchor=tk.W)
#文本框

text = tk.Text(ino,width=62)
text.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
#滚动栏
gdl = tk.Scrollbar(ino)
gdl.config(command=text.yview)
gdl.pack(side=tk.RIGHT,anchor=tk.N,fill=tk.Y)

xxfs = tk.Label(right_frame,text='信息发送')
xxfs.pack(anchor=tk.W)
te = tk.Frame(right_frame)
te.pack(side=tk.RIGHT,anchor=tk.N)

text1 = tk.Text(te,width=55,height=6,padx=5,pady=5)
text1.pack(side=tk.LEFT,anchor=tk.S,fill=tk.Y)
#滚动栏
gdl1 = tk.Scrollbar(te)
gdl1.config(command=text1.yview)
gdl1.pack(side=tk.RIGHT,anchor=tk.S,fill=tk.Y)

#发送按钮
fs = tk.Button(te,text='发送',height=6)
fs.pack(side=tk.RIGHT)
root.mainloop()"""


import re
import tkinter as tk
from tkinter import messagebox,ttk
import requests
import os
from tkinter import filedialog

"""class Wl:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('650x500+100+100')
        self.root.resizable(0,0)
        self.root.title('网络调试工具 v0.0.2')
        #左边布局
        self.l_frame = tk.Frame(self.root)
        self.l_frame.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)
        #左边布局网络设置
        #side：指定组件在窗口中的放置位置，可选值为LEFT、RIGHT、TOP、BOTTOM。
        #BOTTOM 由下到上排列 fill=tk.X沿水平方向填充，TOP从上到下
        #fill：指定组件在窗口中的填充方式，可选值为X、Y、BOTH。
        #both,上下左右都填充 ，
        #expand：指定组件是否随窗口的大小改变而自动扩展，可选值为True或False。
        #anchor：指定组件在窗口中的对齐方式，可选值为N、S、E、W、NE、NW、SE、SW。
        #padx：指定组件的水平内边距（以像素为单位）。
        #pady：指定组件的垂直内边距（以像素为单位）。
        #Frame 　　框架，将几个组件组成一组
        self.nate_frame = tk.LabelFrame(self.l_frame,text='网络设置',padx=5,pady=5)
        self.nate_frame.pack()
        tk.Label(self.nate_frame,text='(1)协议类型').pack(anchor=tk.W) #设置方向为西方
        self.socket_type = ttk.Combobox(self.nate_frame)      #下拉框设置
        self.socket_type['values'] = ['TCP的服务器','TCP的客服端']     #设置下拉框选项的值
        self.socket_type.pack()
        self.res = requests.get('https://myip.ipip.net', timeout=5).text #获取IP地址
        #print(self.res)
        self.b = re.findall(r'当前 IP：(.*?)来自于：',self.res)[0]
        tk.Label(self.nate_frame, text='(2)本主机的地址').pack(anchor=tk.W)
        self.diz = ttk.Combobox(self.nate_frame)
        self.diz['values'] = [f'{self.b}']
        self.diz.pack()
        tk.Label(self.nate_frame, text='(2)主机端口').pack(anchor=tk.W)
        self.duank = ttk.Entry(self.nate_frame)    #输入框
        #获取主机端口
        self.duank.pack(fill=tk.X)
        #按钮嵌套
        self.button = tk.Frame(self.nate_frame)
        self.button.pack()
        self.file = ''
        def opens():
            self.file = filedialog.askopenfilename(defaultextension='txt')
            if self.file == '':
                self.file = None
            else:
                self.f = open(self.file,'r',encoding='utf-8')
                self.text.delete(1.0,tk.END)
                self.text.insert(1.0,self.f.read())
                self.f.close()
        def close():
            self.text.delete(1.0,tk.END)   #清除文本框中的内容
        #打开
        self.open = tk.Button(self.button,text='打开',font=5,bd=1,pady=2)
        self.open.pack(side=tk.LEFT,anchor=tk.CENTER)
        self.open.config(command=opens)
        #关闭
        self.close = tk.Button(self.button, text='关闭', font=5,bd=1,pady=2)
        self.close.pack(side=tk.RIGHT, anchor=tk.CENTER)
        self.close.config(command=close)

        #接收设置
        self.js = tk.LabelFrame(self.l_frame,text='接收设置',padx=5,pady=0)
        self.js.pack(side=tk.TOP,anchor=tk.N,fill=tk.X)
        #单选框
        tk.Radiobutton(self.js,text='utf-8').pack(anchor=tk.W)
        tk.Radiobutton(self.js, text='gbk').pack(anchor=tk.W)
        #多选框
        tk.Checkbutton(self.js, text='解析为JSON数据').pack(anchor=tk.W)
        tk.Checkbutton(self.js, text='自动换行').pack(anchor=tk.W)
        #发送设置
        self.fs = tk.LabelFrame(self.l_frame, text='发送设置', padx=5, pady=0)
        self.fs.pack(side=tk.TOP, anchor=tk.N, fill=tk.X)
        # 单选框
        tk.Radiobutton(self.fs, text='utf-8').pack(anchor=tk.W)
        tk.Radiobutton(self.fs, text='gbk').pack(anchor=tk.W)
        # 多选框
        tk.Checkbutton(self.fs, text='数据加密').pack(anchor=tk.W)
        tk.Checkbutton(self.fs, text='信息接受').pack(anchor=tk.W)"""


"""import re
import tkinter as tk
from tkinter import messagebox,ttk
import requests
import os
from tkinter import filedialog
class Wl:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('650x500+100+100')
        self.root.resizable(0,0)
        self.root.title('网络调试工具 v0.0.2')

        #左边布局
        self.left_fram = tk.Frame(self.root)
        self.left_fram.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)

        #网络设置
        self.wlsz = tk.LabelFrame(self.left_fram,text='网络设置',padx=5,pady=5)
        self.wlsz.pack()
        tk.Label(self.wlsz,text='(1)协议类型',padx=5,pady=5).pack(anchor=tk.W)
        self.xlk = ttk.Combobox(self.wlsz)
        self.xlk['values'] = ['123','250']
        self.xlk.pack()
        tk.Label(self.wlsz, text='(2)地址', padx=5, pady=5).pack(anchor=tk.W)
        self.xlk1 = ttk.Combobox(self.wlsz)
        self.xlk1['values'] = ['123', '250']
        self.xlk1.pack()
        tk.Label(self.wlsz, text='(3)端口', padx=5, pady=5).pack(anchor=tk.W)
        self.xlk2 = tk.Entry(self.wlsz)
        self.xlk2.pack(fill=tk.X)
        self.fill = ''
        def opens():
            fillname = filedialog.askopenfilename(defaultextension='txt')
            self.fill = fillname
            if fillname == '':
                self.fill = None
            else:
                self.f = open(self.fill,'r',encoding='utf-8')
                self.wb.delete(1.0,tk.END)
                self.wb.insert(1.0,self.f.read())
                self.f.close()

        #按钮
        self.an = tk.Frame(self.wlsz)
        self.an.pack()
        tk.Button(self.an,text='打开',pady=5,padx=5,command=opens).pack(side=tk.LEFT,anchor=tk.CENTER)
        tk.Button(self.an, text='关闭', pady=5, padx=5).pack(side=tk.RIGHT, anchor=tk.CENTER)
        # 接受设置
        self.jssz = tk.LabelFrame(self.left_fram, text='网络设置', padx=5, pady=5)
        self.jssz.pack(side=tk.TOP,anchor=tk.W,fill=tk.X)
        ttk.Radiobutton(self.jssz,text='UTF-8').pack(anchor=tk.W)
        ttk.Radiobutton(self.jssz, text='GPEG').pack(anchor=tk.W)
        ttk.Checkbutton(self.jssz,text='123456').pack(anchor=tk.W)
        ttk.Checkbutton(self.jssz, text='741520').pack(anchor=tk.W)

        #右边设置
        self.r_frame = tk.Frame(self.root,padx=5,pady=5)
        self.r_frame.pack(side=tk.RIGHT,anchor=tk.N)
        #数据日志
        self.sjrz = tk.Frame(self.r_frame)
        self.sjrz.pack()
        tk.Label(self.sjrz,text='数据日志').pack(anchor=tk.W)
        #self.tetx = tk.StringVar()
        self.wb = tk.Text(self.sjrz,width=60)
        self.wb.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        self.gdl = tk.Scrollbar(self.sjrz)
        self.gdl.config(command=self.wb.yview)
        self.gdl.pack(side=tk.RIGHT,fill=tk.Y)

        #信息发送
        self.xx = tk.Frame(self.r_frame)
        self.xx.pack()
        tk.Label(self.xx,text='信息发送').pack(anchor=tk.W)
        self.wb1 =tk.Text(self.xx,width=55,height=7)
        self.wb1.pack(side=tk.LEFT)
        self.gdl1 = tk.Scrollbar(self.xx)
        self.gdl1.config(command=self.wb1.yview)
        self.gdl1.pack(side=tk.LEFT, fill=tk.Y)
        self.fs=tk.Button(self.xx,text='发\n送',height=5,width=2)
        self.fs.pack(side=tk.RIGHT)

        self.root.mainloop()

if __name__=="__main__":
    wl = Wl()"""

"""

from tkinter import *
from tkinter import ttk
魏大王学编程（www.weidawang.xyz）
tkinter 实用教程系列

Treeview 作为表格使用的简单案例
"""
"""main = Tk()
data = [(1, "小明", 23, '男', '2021-09-21'), (2, "小强", 23, '男', '2021-09-21'),
        (3, "小红", 23, '女', '2021-09-21'), (4, "铁头", 23, '男', '2021-09-21')]
tree = ttk.Treeview(main, columns=('id', 'name', 'age', 'sex', 'birth'), show="headings", displaycolumns="#all")
tree.heading('id', text="编号", anchor=W)
tree.heading('name', text="姓名", anchor=W)
tree.heading('age', text="年龄", anchor=W)
tree.heading('sex',text="性别",anchor=W)
tree.heading('birth', text="出生日期", anchor=W)
for itm in data:
    tree.insert("",END,values=itm)
tree.pack(expand=True, fill=BOTH)
main.mainloop()"""

#登陆页面
"""import tkinter as tk
from tkinter import messagebox,ttk

root = tk.Tk()
root.geometry('280x300')
root.resizable(0,0)
root.title('登陆页面')
tk.Label(root,text='登陆页面',font=24,pady=10,padx=5,height=3).grid(row=1,column=2)
tk.Label(root,text='账  号:',font=24,pady=5,padx=5,height=2).grid(row=2,column=1)
zh = tk.StringVar()
zh_xx = tk.Entry(root,textvariable=zh,width=22)
zh_xx.grid(row=2,column=2,columnspan=2,padx=20)
tk.Label(root,text='密  码:',font=24,pady=5,padx=5,height=3).grid(row=3,column=1)
mm = tk.StringVar()
mm_xx = tk.Entry(root,textvariable=mm,width=22)
mm_xx.grid(row=3,column=2,columnspan=3,padx=20)
#登陆
dl = tk.Button(root,text='登 陆',anchor=tk.CENTER,padx=5)
dl.grid(row=4,column=1)
#注册
zc = tk.Button(root,text='注 册',anchor=tk.CENTER,padx=5)
zc.grid(row=4,column=3)

root.mainloop()"""

#注册页面
"""import tkinter as tk
from tkinter import ttk,messagebox
root = tk.Tk()
root.geometry('400x450')
root.resizable(0,0)
root.title('注册页面')
tk.Label(root,text='注册页面:',font=24,height=4,padx=5,pady=10).grid(row=1,column=2)
tk.Label(root,text='账   号:',font=24,height=2,padx=5).grid(row=2,column=1)
zh = tk.StringVar()
zh_xx = tk.Entry(root,textvariable=zh,width=30)
zh_xx.grid(row=2,column=2,columnspan=2,padx=10)
tk.Label(root,text='密   码:',font=24,height=2,padx=5).grid(row=3,column=1)
mm = tk.StringVar()
mm_xx = tk.Entry(root,textvariable=mm,width=30)
mm_xx.grid(row=3,column=2,columnspan=2,padx=10)
tk.Label(root,text='重复密码:',font=24,height=2,padx=5).grid(row=4,column=1)
cmm = tk.StringVar()
cmm_xx = tk.Entry(root,textvariable=cmm,width=30)
cmm_xx.grid(row=4,column=2,columnspan=2,padx=10)
tk.Label(root,text='联系方式:',font=24,height=2,padx=5).grid(row=5,column=1)
lxfs = tk.StringVar()
lxfs_xx = tk.Entry(root,textvariable=lxfs,width=30)
lxfs_xx.grid(row=5,column=2,columnspan=2,padx=10)
tk.Label(root,text='验证码:',font=24,height=2,padx=5).grid(row=6,column=1)
yzm = tk.StringVar()
yzm_xx = tk.Entry(root,textvariable=yzm,width=20)
yzm_xx.grid(row=6,column=2,padx=10)
tk.Button(root,text='获取验证码').grid(row=6,column=3)
tk.Label(root,text='密  令:',font=24,height=2,padx=5).grid(row=7,column=1)
ml = tk.StringVar()
ml_xx = tk.Entry(root,textvariable=ml,width=30)
ml_xx.grid(row=7,column=2,columnspan=2,padx=10)
tk.Button(root,text='返回',pady=10,padx=10).grid(row=9,column=3)
root.mainloop()
"""
"""#主页面
import tkinter as tk
from tkinter import ttk,messagebox
root = tk.Tk()
root.geometry('900x650+100+100')
root.title('主页面')
mun = tk.Menu(root)
mun.add_cascade(label='录入')
mun.add_cascade(label='查询')
mun.add_cascade(label='删除')
mun.add_cascade(label='修改')
mun.add_cascade(label='关于')
#录入页面
lr_fram = tk.Frame(root)
tk.Label(lr_fram,text='录入信息',font=('黑体',30),height=4,anchor=tk.CENTER).grid(row=0,column=2)
tk.Label(lr_fram,text='姓  名：',font=25,height=3,anchor=tk.CENTER).grid(row=1,column=1)
xm = tk.StringVar()
xm1 = tk.Entry(lr_fram,textvariable=xm,width=30)
xm1.grid(row=1,column=2,columnspan=2,padx=5)
tk.Label(lr_fram,text='语  文：',font=25,height=3,anchor=tk.CENTER).grid(row=2,column=1)
yw = tk.StringVar()
yw1 = tk.Entry(lr_fram,textvariable=yw,width=30)
yw1.grid(row=2,column=2,columnspan=2,padx=5)
tk.Label(lr_fram,text='英  语：',font=25,height=3,anchor=tk.CENTER).grid(row=3,column=1)
yw = tk.StringVar()
yw1 = tk.Entry(lr_fram,textvariable=yw,width=30)
yw1.grid(row=3,column=2,columnspan=2,padx=5)
tk.Label(lr_fram,text='数  学：',font=25,height=3,anchor=tk.CENTER).grid(row=4,column=1)
sx = tk.StringVar()
sx1 = tk.Entry(lr_fram,textvariable=sx,width=30)
sx1.grid(row=4,column=2,columnspan=2,padx=5)
#lr_fram.pack()
#查询页面
cx = tk.Frame(root)
def information():
    with open('xssj.json','r',encoding='utf-8') as f:
        text = f.read()
    return json.loads(text)
def show():
    for _ in map(tre_view.delete, tre_view.get_children('')):  # 更新
        pass
    students = information()
    index = 0
    for stu in students:
        # print(stu)
        tre_view.insert('', index + 1, values=(
            stu['name'], stu['math'], stu['chinese'], stu['english']
        ))
colum = ("name","math","chinese","english")
tre_view = ttk.Treeview(cx,show='headings',columns=colum,height=20)
tre_view.column("name",anchor=tk.CENTER)
tre_view.column("math",anchor=tk.CENTER)
tre_view.column("chinese",anchor=tk.CENTER)
tre_view.column("english",anchor=tk.CENTER)
tre_view.heading("name",text='小米')
tre_view.heading("math",text='数学')
tre_view.heading("chinese",text='语文')
tre_view.heading("english",text='英语')
tre_view.pack(fill=tk.BOTH,expand=True)

show()
gdl = tk.Scrollbar(tre_view)
gdl.config(command=tre_view.yview)
gdl.pack(side=tk.RIGHT,fill=tk.Y)
an = tk.Frame(cx)
an.pack()
tk.Button(an,text='更新',font=15,padx=5,pady=10).pack(side=tk.BOTTOM,anchor=tk.CENTER)
#cx.pack(fill=tk.BOTH,side=tk.TOP,expand=True)
#删除
sc = tk.Frame(root)
tk.Label(sc,text='删除页面',font=('黑体',30),height=4,anchor=tk.CENTER).grid(row=1,column=2)
tk.Label(sc,text='根据名字删除',height=2,anchor=tk.CENTER).grid(row=2,column=1)
sc_xx = tk.StringVar()
sc_b = tk.Entry(sc,textvariable=sc_xx,width=40)
sc_b.grid(row=3,column=1,columnspan=2)
tk.Button(sc,text='删除',padx=5,pady=5).grid(row=3,column=3)
sc.pack()
root.config(menu=mun)
root.mainloop()
"""
"""import tkinter as tk
from tkinter import ttk,messagebox
import requests
import re
from tkinter import filedialog
import os
root = tk.Tk()
root.geometry('650x500+100+100')
root.resizable(0,0)
root.title('网络调试工具v1.02')
#设置左边
f1 = tk.Frame(root)
f1.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)
#网络设置
wl = tk.LabelFrame(f1,text='网络设置',pady=5,padx=5)
wl.pack()
tk.Label(wl,text='(1)协议类型',pady=5,padx=5).pack(anchor=tk.W)
a = ttk.Combobox(wl)
a['value'] = ["小茜总","小任总"]
a.pack()

def show_ip():
    aip = requests.get('https://myip.ipip.net', timeout=5).text  # 获取IP地址
    bip = re.findall(r'当前 IP：(.*?)来自于：',aip)
    return bip[0]
tk.Label(wl,text='(2)主机地址',pady=5,padx=5).pack(anchor=tk.W)
b = ttk.Combobox(wl)
b['value'] = [f"{show_ip()}"]
b.pack()
tk.Label(wl,text='(3)主机端口',pady=5,padx=5).pack(anchor=tk.W)
c = tk.StringVar()
d = tk.Entry(wl)
d.pack(fill=tk.X)
kaiguan = tk.Frame(wl)
kaiguan.pack()

fl = ''
def opens():
    global fl,text1
    fle = filedialog.askopenfilename(defaultextension='txt')
    if fle == '':
        fl = None
    else:
        fl = fle
        fopen = open(fl,'r',encoding='utf-8')
        text1.insert(1.0,fopen.read())
        fopen.close()
def closes():
    global text1
    #print(text1.get(1.0,tk.END))
    if text1.get(1.0,tk.END) != '':
        text1.delete(1.0,tk.END)
    else:
        messagebox.showwarning('提示','数据为空')
opena = tk.Button(kaiguan,text='打开',padx=5,pady=5,font=12)
opena.pack(side=tk.LEFT,anchor=tk.CENTER)
opena.config(command=opens)
close = tk.Button(kaiguan,text='关闭',padx=5,pady=5,font=12)
close.pack(side=tk.RIGHT,anchor=tk.CENTER)
close.config(command=closes)
js = tk.LabelFrame(f1,text="接受设置")
js.pack(fill=tk.X)
tk.Radiobutton(js,text='utf-8',value=1).pack(anchor=tk.W)
tk.Radiobutton(js,text='gbk',value=2).pack(anchor=tk.W)
tk.Checkbutton(js,text='解析为json数据').pack(anchor=tk.W)
tk.Checkbutton(js,text='自动换行').pack(anchor=tk.W)
fs = tk.LabelFrame(f1,text="发送设置")
fs.pack(fill=tk.X)
tk.Radiobutton(fs,text='utf-8',value=1).pack(anchor=tk.W)
tk.Radiobutton(fs,text='gbk',value=2).pack(anchor=tk.W)
tk.Checkbutton(fs,text='数据加密').pack(anchor=tk.W)
tk.Checkbutton(fs,text='消息接受').pack(anchor=tk.W)
#右边设置
f2 = tk.Frame(root)
f2.pack(side=tk.RIGHT,anchor=tk.N,padx=5,pady=5)
fsj = tk.Frame(f2)
fsj.pack()
tk.Label(fsj,text="数据日志").pack(anchor=tk.W)
text1 = tk.Text(fsj,width=60)
text1.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
es = tk.Scrollbar(fsj)
es.config(command=text1.yview)
es.pack(side=tk.RIGHT,fill=tk.Y,padx=5)
tk.Label(f2,text='信息发送').pack(anchor=tk.W,pady=5)
fxxfs = tk.Frame(f2)
fxxfs.pack()
te = tk.StringVar()
text2 = tk.Text(fxxfs,width=56,height=9)
text2.pack(side=tk.LEFT,pady=1)
rse = tk.Scrollbar(fxxfs)
rse.config(command=text2.yview)
def fsa():
    global text1,text2
    if text1.get(1.0,tk.END) !='':
        a = text1.get(1.0,tk.END)
        text2.insert(1.0,a)
        for i in range(len(text1.get(1.0,tk.END))):
            b = a[0:-i-1]
            text2.delete(1.0,tk.END)
            text2.insert(1.0,b)
            time.sleep(0.1)
            root.update()





tk.Button(fxxfs,text='发\n送',height=9,command=fsa).pack(side=tk.RIGHT)
rse.pack(side=tk.RIGHT,fill=tk.Y)
root.mainloop()"""

import tkinter as tk
from tkinter import ttk,messagebox
#登陆页
"""root = tk.Tk()
root.geometry('500x600+100+100')
root.resizable(0,0)
root.title('登陆页')
tk.Label(root,text='登陆页',font=("黑体",35),padx=5,pady=5,height=5).grid(row=1,column=3)
tk.Label(root,text='账  号:',font=("黑体",20),padx=5,pady=5,height=3).grid(row=2,column=1)
zh = tk.StringVar()
zh_sr = tk.Entry(root,textvariable=zh,width=40)
zh_sr.grid(row=2,column=3,columnspan=3,padx=10,pady=5)
tk.Label(root,text='密  码:',font=("黑体",20),padx=5,pady=5,height=3).grid(row=3,column=1,padx=5,pady=5)
mm = tk.StringVar()
mm_sr = tk.Entry(root,textvariable=mm,show='*',width=40)
mm_sr.grid(row=3,column=3,columnspan=3,padx=5,pady=5)
open = tk.Button(root,text='登陆',padx=5,pady=5,width=10)
open.grid(row=4,column=1,padx=5,pady=5)
close = tk.Button(root,text='注册',padx=5,pady=5,width=10)
close.grid(row=4,column=4,pady=5)
root.mainloop()"""
#注册页
"""root = tk.Tk()
root.geometry('600x850+100+100')
root.resizable(0,0)
root.title("注册页")

tk.Label(root,text='注册页',font=("黑体",35),height=4,anchor=tk.CENTER).grid(row=1,column=4,padx=5,pady=5)
tk.Label(root,text='账  号:',font=("黑体",20),height=3,anchor=tk.E).grid(row=2,column=2,padx=5,pady=5)
z = tk.StringVar()
z_sr = tk.Entry(root,textvariable=z,width=40,bd=5)
z_sr.grid(row=2,column=3,columnspan=3,padx=5,pady=5)
tk.Label(root,text='密  码:',font=("黑体",20),height=3,anchor=tk.E).grid(row=3,column=2,padx=5,pady=5)
m = tk.StringVar()
m_sr = tk.Entry(root,textvariable=m,width=40,bd=5)
m_sr.grid(row=3,column=3,columnspan=3,padx=5,pady=5)
tk.Label(root,text='重复密码:',font=("黑体",20),height=3,anchor=tk.E).grid(row=4,column=2,padx=5,pady=5)
cm = tk.StringVar()
cm_sr = tk.Entry(root,textvariable=cm,show='*',width=40,bd=5)
cm_sr.grid(row=4,column=3,columnspan=3,padx=5,pady=5)
tk.Label(root,text='联系方式:',font=("黑体",20),height=3,anchor=tk.E).grid(row=5,column=2,padx=5,pady=5)
l = tk.StringVar()
l_sr = tk.Entry(root,textvariable=l,width=40,bd=5)
l_sr.grid(row=5,column=3,columnspan=3,padx=5,pady=5)
tk.Label(root,text='验证码 :',font=("黑体",20),height=3,anchor=tk.E).grid(row=6,column=2,padx=5,pady=5)
y = tk.StringVar()
y_sr = tk.Entry(root,textvariable=y,width=30,bd=5)
y_sr.grid(row=6,column=3,columnspan=2,padx=5,pady=5)
yz_fs = tk.Button(root,text='发送验证码')
yz_fs.grid(row=6,column=5,padx=5,pady=5)
tk.Label(root,text='注册密令:',font=("黑体",20),height=3,anchor=tk.E).grid(row=7,column=2,padx=5,pady=5)
ml = tk.StringVar()
ml_sr = tk.Entry(root,textvariable=ml,show='#',width=40,bd=5)
ml_sr.grid(row=7,column=3,columnspan=3,padx=5,pady=5)
tk.Label(root,text='',font=15,height=2,width=5).grid(row=8,column=1)
zhce = tk.Button(root,text='注册',font=15,height=2,width=5)
zhce.grid(row=8,column=2)
fanh = tk.Button(root,text='返回',font=15,height=2,width=5)
fanh.grid(row=8,column=5)

root.mainloop()"""
#主页
"""root = tk.Tk()
root.title('主页')
root.geometry('600x560+100+100')
meun = tk.Menu(root,tearoff=False)
#fa = tk.Menu(meun,tearoff=False)
#fa.add_command(label='1')
#fa.add_command(label='2')
def ar(x):
    global f1,f2
    if x == 1:
        f1.pack()
        f2.pack_forget()
    else:
        f1.pack_forget()
        f2.pack(fill=tk.BOTH,expand=True)

meun.add_cascade(label='录入',command=lambda :ar(1))
meun.add_cascade(label='查询',command=lambda :ar(2))
meun.add_cascade(label='删除')
meun.add_cascade(label='修改')
meun.add_cascade(label='关于')
root.config(menu=meun)
f1 = tk.Frame(root)
tk.Label(f1,text='录入页面',font=("黑体",35),height=3,anchor=tk.CENTER).grid(row=1,column=2,padx=5,pady=5)
tk.Label(f1,text='姓 名:  ',font=("黑体",25),height=2,anchor=tk.CENTER).grid(row=2,column=1,padx=5,pady=5)
xm = tk.StringVar()
xm_cx = tk.Entry(f1,width=30,textvariable=xm)
xm_cx.grid(row=2,column=2,columnspan=2,padx=5,pady=5)
tk.Label(f1,text='数 学:  ',font=("黑体",25),height=2,anchor=tk.CENTER).grid(row=3,column=1,padx=5,pady=5)
sx = tk.StringVar()
sx_cx = tk.Entry(f1,width=30,textvariable=sx)
sx_cx.grid(row=3,column=2,columnspan=2,padx=5,pady=5)
tk.Label(f1,text='语 文:  ',font=("黑体",25),height=2,anchor=tk.CENTER).grid(row=4,column=1,padx=5,pady=5)
yw = tk.StringVar()
yw_cx = tk.Entry(f1,width=30,textvariable=yw)
yw_cx.grid(row=4,column=2,columnspan=2,padx=5,pady=5)
tk.Label(f1,text='英 语:  ',font=("黑体",25),height=2,anchor=tk.CENTER).grid(row=5,column=1,padx=5,pady=5)
yy = tk.StringVar()
yy_cx = tk.Entry(f1,width=30,textvariable=yy)
yy_cx.grid(row=5,column=2,columnspan=2,padx=5,pady=5)
lr = tk.Button(f1,text='录入',font=15)
lr.grid(row=6,column=3,padx=5,pady=5)

def show():
    global chax
    for _ in map(chax.delete,chax.get_children('')):
        pass
    with open('xssj.json', 'r', encoding='utf-8') as f:
        atetx = f.read()
    a = 0
    for stu in json.loads(atetx):
        chax.insert('',a+1,values=(stu['name'],stu['math'],stu['chinese'],stu['english']))
f2 = tk.Frame(root)
ae = ("name","math","chinese","english")
chax = ttk.Treeview(f2,show='headings',columns=ae,height=20)
chax.column("name",anchor=tk.CENTER)
chax.column("math",anchor=tk.CENTER)
chax.column("chinese",anchor=tk.CENTER)
chax.column("english",anchor=tk.CENTER)
chax.heading("name",text='姓名')
chax.heading("math",text='数学')
chax.heading("chinese",text='语文')
chax.heading("english",text='英语')
chax.pack(fill=tk.BOTH,expand=True)
show()

#f2.pack(fill=tk.BOTH,expand=True)
f3 = tk.Frame(root)
f4 = tk.Frame(root)
f5 = tk.Frame(root)
root.mainloop()"""
#登陆页
"""import tkinter as tk
from tkinter import ttk,messagebox

root = tk.Tk()
#登陆页
root.title('学生登陆系统')
root.geometry('400x500')
root.resizable(0,0)

font = ('黑体',25)
a = tk.Label(root,text='登陆页',font=('黑体',25),height=5,anchor=tk.CENTER)
a.grid(row=1,column=2,padx=5,pady=10)
tk.Label(root,text='账  号  ',font=('黑体',25),anchor=tk.CENTER).grid(row=2,column=1,padx=10)
i = tk.StringVar()
i.set('')
tk.Entry(root,textvariable=i,width=30).grid(row=2,column=2,columnspan=2,padx=5,pady=5)

tk.Label(root,text='密  码 ',font=('黑体',25),anchor=tk.CENTER,height=5).grid(row=3,column=1,padx=10)
j = tk.StringVar()
j.set('')
tk.Entry(root,textvariable=j,width=30).grid(row=3,column=2,columnspan=2,padx=15,pady=5)
tk.Button(root,text='登陆',width=10,height=2).grid(row=4,column=1,padx=5,pady=5)
tk.Button(root,text='注册',width=10,height=2).grid(row=4,column=3,padx=15,pady=5)
root.mainloop()"""
#注册页
"""import tkinter as tk
from tkinter import ttk,messagebox
root = tk.Tk()
root.title('注册页面')
root.geometry('600x850')
tk.Label(root,text='用户注册',font=('黑体',30),height=5).grid(row=1,column=2)
tk.Label(root,text='姓名',font=('黑体',30),height=2).grid(row=2,column=1,padx=10,pady=10)
tk.Entry(root,width=50).grid(row=2,column=2,columnspan=2,padx=10,pady=10)
tk.Label(root,text='密码',font=('黑体',30),height=2).grid(row=3,column=1,padx=10,pady=10)
tk.Entry(root,width=50).grid(row=3,column=2,columnspan=2,padx=10,pady=10)
tk.Label(root,text='财富密码',font=('黑体',30),height=2).grid(row=4,column=1,padx=10,pady=10)
tk.Entry(root,width=50).grid(row=4,column=2,columnspan=2,padx=10,pady=10)
tk.Label(root,text='联系方式',font=('黑体',30),height=2).grid(row=5,column=1,padx=10,pady=10)
tk.Entry(root,width=50).grid(row=5,column=2,columnspan=2,padx=10,pady=10)

root.mainloop()"""

#主页面
"""import tkinter as tk
from tkinter import ttk,messagebox
root = tk.Tk()
root.title('主页面')
root.geometry('600x560')
men = tk.Menu(root,takefocus=False)
def isy(x):
    global f1,f2
    if x == 1:
        f1.pack()
        f2.pack_forget()
    else:
        f1.pack_forget()
        f2.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

men.add_command(label='录入',command=lambda :isy(1))
men.add_command(label='查询',command=lambda :isy(2))
men.add_command(label='删除')
men.add_command(label='修改')
men.add_command(label='关于')
f1 = tk.Frame(root)
tk.Label(f1,text='录入信息',height=5,font=('黑体',30)).grid(row=1,column=2,padx=5,pady=5)
tk.Label(f1,text='姓名',height=2,font=('黑体',30)).grid(row=2,column=1,padx=5,pady=5)
tk.Entry(f1,width=40).grid(row=2,column=2,columnspan=2,padx=5,pady=5)
tk.Label(f1,text='数学',height=2,font=('黑体',30)).grid(row=3,column=1,padx=5,pady=5)
tk.Entry(f1,width=40).grid(row=3,column=2,columnspan=2,padx=5,pady=5)
tk.Label(f1,text='英语',height=2,font=('黑体',30)).grid(row=4,column=1,padx=5,pady=5)
tk.Entry(f1,width=40).grid(row=4,column=2,columnspan=2,padx=5,pady=5)
tk.Button(f1,text='录入',font=('黑体',20)).grid(row=6,column=3,padx=5)

f2 = tk.Frame(root)


def show():
    global tre_view
    for _ in map(tre_view.delete, tre_view.get_children('')):  # 更新
        pass
    with open('xssj.json','r',encoding='utf-8') as f:
        students = json.loads(f.read())
    index = 0
    for stu in students:
        #print(stu)
        tre_view.insert('', index + 1, values=(
            stu['name'], stu['math'], stu['chinese'], stu['english']
        ))

data_h = ['name','math','chinese','english']
tre_view = ttk.Treeview(f2,show='headings',columns=data_h,height=20)
show()
tre_view.column('name',anchor=tk.CENTER)
tre_view.column('math',anchor=tk.CENTER)
tre_view.column('chinese',anchor=tk.CENTER)
tre_view.column('english',anchor=tk.CENTER)
tre_view.heading('name',text='姓名')
tre_view.heading('math',text='数学')
tre_view.heading('chinese',text='语文')
tre_view.heading('english',text='英语')
tre_view.pack(fill=tk.BOTH,expand=True)
tk.Button(f2,text='更新',command=show).pack(side=tk.BOTTOM,anchor=tk.CENTER)
#f2.pack(expand=True,side=tk.TOP,fill=tk.BOTH)
root.config(menu=men)
root.mainloop()"""
import tkinter as tk
from tkinter import messagebox,ttk
root = tk.Tk()
root.title('153')
root.geometry('650x500+100+100')
root.resizable(0,0)
#左边设置
left = tk.Frame(root)
left.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)
#网络设置
wl = tk.LabelFrame(left,text='网络设置')
wl.pack()
tk.Label(wl,text='(1)协议类型').pack(anchor=tk.W)
se = ttk.Combobox(wl)
se['values'] = ['1','2']
se.pack()
tk.Label(wl,text='(2)主机地址').pack(anchor=tk.W)
se1 = ttk.Combobox(wl)
se1['values'] = ['1','2']
se1.pack()
tk.Label(wl,text='(3)主机端口').pack(anchor=tk.W)
se2 = tk.Entry(wl)
se2.pack(fill=tk.X)
ls = tk.Frame(wl)
ls.pack()
open = tk.Button(ls,text='打开')
open.pack(side=tk.LEFT,anchor=tk.CENTER)
close = tk.Button(ls,text='关闭')
close.pack(side=tk.RIGHT,anchor=tk.CENTER)

js = tk.LabelFrame(left,text='接受设置')
js.pack(side=tk.TOP,anchor=tk.N,fill=tk.X)
a1 = tk.IntVar()
a = tk.Radiobutton(js,text='utf-8',variable=a1,value=0)
a.pack(anchor=tk.W)
a2 = tk.IntVar()
a2 = tk.Radiobutton(js,text='utf-8',variable=a2,value=1)
a2.pack(anchor=tk.W)
a3 = tk.Checkbutton(js,text='1235')
a3.pack(anchor=tk.W)

#右边设置
rigt = tk.Frame(root)
rigt.pack(side=tk.RIGHT,anchor=tk.N,padx=5,pady=5)
#名称
rz = tk.Frame(rigt)
rz.pack()
tk.Label(rz,text='数据日志').pack(anchor=tk.W)
TEXT = tk.Text(rz,width=62)
TEXT.pack(fill=tk.BOTH,side=tk.LEFT,expand=True)
lsd = tk.Scrollbar(rz)
lsd.config(command=TEXT.yview)
lsd.pack(side=tk.RIGHT,fill=tk.Y)
TEXT.config(yscrollcommand=lsd.set)















root.mainloop()