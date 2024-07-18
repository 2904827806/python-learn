import re
import time
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
        self.l_frame = tk.Frame(self.root) #左边布局
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
        self.socket_type = ttk.Combobox(self.nate_frame)      #下拉框设置combobox
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
        a = tk.IntVar()
        b = tk.IntVar()
        tk.Radiobutton(self.js,text='utf-8',variable=a,value=0).pack(anchor=tk.W)
        tk.Radiobutton(self.js, text='gbk',variable=b,value=1).pack(anchor=tk.W)
        #多选框
        tk.Checkbutton(self.js, text='解析为JSON数据').pack(anchor=tk.W)
        tk.Checkbutton(self.js, text='自动换行').pack(anchor=tk.W)
        #发送设置
        self.fs = tk.LabelFrame(self.l_frame, text='发送设置', padx=5, pady=0)
        self.fs.pack(side=tk.TOP, anchor=tk.N, fill=tk.X)
        # 单选框
        tk.Radiobutton(self.fs, text='utf-8',value=1).pack(anchor=tk.W)
        tk.Radiobutton(self.fs, text='gbk',value=0).pack(anchor=tk.W)
        # 多选框
        tk.Checkbutton(self.fs, text='数据加密').pack(anchor=tk.W)
        tk.Checkbutton(self.fs, text='信息接受').pack(anchor=tk.W)
        #右边设置
        self.r_frame = tk.Frame(self.root)
        self.r_frame.pack(side=tk.RIGHT,anchor=tk.N,padx=5,pady=5)   #padx,pady加边距
        #数据日志
        self.info = tk.Frame(self.r_frame)
        self.info.pack()
        tk.Label(self.info,text='数据日志').pack(anchor=tk.W)
        #文本
        self.text = tk.Text(self.info,width=62)
        self.text.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)     #自上到下，自左道右填充
        #滚动栏
        self.gdl = tk.Scrollbar(self.info)
        # 滚动栏绑定
        self.gdl.config(command=self.text.yview)
        self.gdl.pack(side=tk.RIGHT,fill=tk.Y)
        #信息发送
        tk.Label(self.r_frame, text='信息发送').pack(anchor=tk.W)
        self.xxfs = tk.Frame(self.r_frame)
        self.xxfs.pack(side=tk.RIGHT,anchor =tk.N,padx=5,pady=5)
        #文本框
        self.text2 = tk.Text(self.xxfs,width=55,height=6)
        self.text2.pack(side=tk.LEFT)
        #滚动栏
        self.gdl2 = tk.Scrollbar(self.xxfs)
        self.gdl2.config(command=self.text2.yview)
        self.gdl2.pack(side=tk.LEFT,fill=tk.Y)
        #按钮
        self.button2 = tk.Button(self.xxfs,text='发送',width=1,padx=5,pady=5)
        self.button2.pack(side=tk.RIGHT,fill=tk.BOTH)
    def run(self):
        self.root.mainloop()

root = tk.Tk()
root.title('网络调试工具2.0版本')
root.geometry('600x500')
root.resizable(0,0)
#左边设置
lifet = tk.Frame(root)
lifet.pack(side=tk.LEFT,anchor=tk.N,padx=5,pady=5)

#网络设置
wl = tk.LabelFrame(lifet,text='网络设置')
wl.pack()
tk.Label(wl,text='(1)协议类型').pack(anchor=tk.W,padx=5,pady=5)
dx = ttk.Combobox(wl)
dx.pack(padx=5)
dx['values'] = ['1','2','3']
tk.Label(wl,text='(2)主机地址').pack(anchor=tk.W,padx=5,pady=5)
dx2 = ttk.Combobox(wl)
dx2.pack(padx=5)
dx2['values'] = ['1','2','3']
tk.Label(wl,text='(3)主机端口').pack(anchor=tk.W,padx=5,pady=5)
tesxt = tk.StringVar()
tesxt.set('000000000')
te = tk.Entry(wl,textvariable=tesxt)
te.pack(fill=tk.X,padx=5,pady=2)
anl = tk.Frame(wl)
anl.pack()
s = ''
def opens():
    global dat_text,s
    import os
    s1 = filedialog.askopenfilename(defaultextension='txt')
    if s1 == '':
        s = None
    else:
        s = s1
        fill = open(s,'r',encoding='utf-8')
        dat_text.delete(1.0,tk.END)
        dat_text.insert(1.0,fill.read())
        fill.close()
open1 = tk.Button(anl,text='打开')
open1.pack(side=tk.LEFT,anchor=tk.CENTER,pady=2)
open1.config(command=opens)
clos = tk.Button(anl,text='关闭')
clos.pack(side=tk.RIGHT,anchor=tk.CENTER,pady=2)

#js设置
js = tk.LabelFrame(lifet,text='接受设置',padx=5,pady=5)
js.pack(side=tk.TOP,anchor=tk.CENTER,fill=tk.X)
tk.Radiobutton(js,text='utf-8').pack(anchor=tk.W)
tk.Radiobutton(js,text='jbg').pack(anchor=tk.W)
tk.Checkbutton(js,text='解析为json数据').pack(anchor=tk.W)
tk.Checkbutton(js,text='自动换行').pack(anchor=tk.W)
fs = tk.LabelFrame(lifet,text='发送设置',padx=5,pady=5)
fs.pack(side=tk.TOP,anchor=tk.CENTER,fill=tk.X)
tk.Radiobutton(fs,text='utf-8').pack(anchor=tk.W)
tk.Radiobutton(fs,text='jbg').pack(anchor=tk.W)
tk.Checkbutton(fs,text='解析为json数据').pack(anchor=tk.W)
tk.Checkbutton(fs,text='自动换行').pack(anchor=tk.W)
#右边设置
rigt = tk.Frame()
rigt.pack(side=tk.RIGHT,anchor=tk.N,padx=5,pady=5)
rz = tk.Frame(rigt)
rz.pack()
tk.Label(rz,text='数据数据日志').pack(anchor=tk.W)

dat_text = tk.Text(rz,width=52)
dat_text.pack(fill=tk.BOTH,side=tk.LEFT)
gl = tk.Scrollbar(rz)
gl.config(command=dat_text.yview)
gl.pack(side=tk.RIGHT,fill=tk.Y)
dat_text.config(yscrollcommand=gl.set)
xs = tk.Frame(rigt)
xs.pack()
def cl():
    global dat_text,TA
    a = dat_text.get(1.0,tk.END)
    if len(a) == 1:
        TA.delete(1.0,tk.END)
        TA.insert(1.0, '没有导入数据')
    else:
        for i in range(len(a)):
            b = a[:-i-1]
            TA.delete(1.0, tk.END)
            TA.insert(1.0, a[:-i-1])
            root.update()
            if b =='':
                TA.insert(1.0, a[:-i - 1])
            else:
                continue


tk.Label(xs,text='信息发送').pack(anchor=tk.W)
TA = tk.Text(xs,width=48)
TA.pack(side=tk.LEFT,fill=tk.BOTH)
an2 = tk.Button(xs,text='发\n送',width=2,pady=5,command=cl)
an2.pack(side=tk.RIGHT,fill=tk.Y)
gdl2 = tk.Scrollbar(xs)
gdl2.config(command=TA.yview)
gdl2.pack(side=tk.LEFT,fill=tk.Y)
root.mainloop()



