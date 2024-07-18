"""import tkinter as tk
from tkinter import messagebox
from random import random
import time
import os
import re


#创建实列对象
root = tk.Tk()
root.geometry('295x260+100+100')
root.resizable(0,0)
root.title('简易计算器')
#计算器布局
#root.attributes('-alpha',1)   #设置不透明度
font = ('宋体',30)
font1 = ('宋体',16)
result = tk.StringVar()
result.set('0')

rla = tk.Label(root,
         textvariable=result,font=font1,height=3,width=25,justify=tk.LEFT,anchor=tk.SE
         )
rla.grid(row=1,column=1,columnspan=4)    #横跨四列

button_clear = tk.Button(root,text='c',font=30,bd=1,width=6,relief=tk.FLAT,bg='#b1b2b2')
button_back = tk.Button(root,text='<-',font=30,bd=1,width=6,relief=tk.FLAT,bg='#b1b2b2')
button_division = tk.Button(root,text='/',font=30,bd=1,width=6,relief=tk.FLAT,bg='#b1b2b2')
button_multiple = tk.Button(root,text='x',font=30,bd=1,width=5,relief=tk.FLAT,bg='#b1b2b2')

button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
button_back.grid(row=2,column=2,padx=4,pady=2)
button_division.grid(row=2,column=3,padx=4,pady=2)
button_multiple.grid(row=2,column=4,padx=4,pady=2)

button_7 = tk.Button(root,text='7',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_8 = tk.Button(root,text='8',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_9 = tk.Button(root,text='9',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_jian = tk.Button(root,text='-',font=30,bd=1,width=5,relief=tk.FLAT,bg='#b1b2b2')

button_7.grid(row=3,column=1,padx=4,pady=2)       #padx,pady 表示边距
button_8.grid(row=3,column=2,padx=4,pady=2)
button_9.grid(row=3,column=3,padx=4,pady=2)
button_jian.grid(row=3,column=4,padx=4,pady=2)

button_4 = tk.Button(root,text='4',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_5 = tk.Button(root,text='5',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_6 = tk.Button(root,text='6',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_j = tk.Button(root,text='+',font=30,bd=1,width=5,relief=tk.FLAT,bg='#b1b2b2')

button_4.grid(row=4,column=1,padx=4,pady=2)       #padx,pady 表示边距
button_5.grid(row=4,column=2,padx=4,pady=2)
button_6.grid(row=4,column=3,padx=4,pady=2)
button_j.grid(row=4,column=4,padx=4,pady=2)

button_1 = tk.Button(root,text='1',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_2 = tk.Button(root,text='2',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_3 = tk.Button(root,text='3',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
button_dy = tk.Button(root,text='=',font=30,bd=1,height=3,width=5,relief=tk.FLAT,bg='#b1b2b2')

button_1.grid(row=5,column=1,padx=4,pady=2)       #padx,pady 表示边距
button_2.grid(row=5,column=2,padx=4,pady=2)
button_3.grid(row=5,column=3,padx=4,pady=2)
button_dy.grid(row=5,column=4,padx=4,pady=2,rowspan=2) #横跨2行
button_0 = tk.Button(root,text='0',font=30,bd=1,width=14,relief=tk.FLAT,bg='#eacda1')
#button_01 = tk.Button(root,text='0',font=30,bd=1,height=2,width=6,relief=tk.FLAT,bg='#eacda1')
button_ = tk.Button(root,text='.',font=30,bd=1,width=6,relief=tk.FLAT,bg='#eacda1')
#button_dy1 = tk.Button(root,text='=',font=30,bd=1,height=2,width=5,relief=tk.FLAT,bg='#b1b2b2')

button_0.grid(row=6,column=1,padx=4,pady=2,columnspan=2)  #横跨列
#button_01.grid(row=5,column=2,padx=4,pady=2)
button_.grid(row=6,column=3,padx=4,pady=2)
#button_dy1.grid(row=5,column=4,padx=4,pady=2)

def click_button(x):


    if x == 'c':
        a = ''
        result.set('0')

    else:







#
button_0.config(command=lambda :click_button('0'))
button_1.config(command=lambda :click_button('1'))
button_2.config(command=lambda :click_button('2'))
button_3.config(command=lambda :click_button('3'))
button_4.config(command=lambda :click_button('4'))
button_5.config(command=lambda :click_button('5'))
button_6.config(command=lambda :click_button('6'))
button_7.config(command=lambda :click_button('7'))
button_8.config(command=lambda :click_button('8'))
button_9.config(command=lambda :click_button('9'))
button_j.config(command=lambda :click_button('+'))
button_jian.config(command=lambda :click_button('-'))
button_dy.config(command=lambda :click_button('='))
button_multiple.config(command=lambda :click_button('X'))
button_division.config(command=lambda :click_button('/'))
button_clear.config(command=lambda :click_button('c'))


def on_eixt():
   # messagebox.showwarning('')
    if messagebox.askyesno(title='提示',message='是否确定退出'):
        root.quit()
root.protocol('WM_DELETE_WINDOW',on_eixt)  # 重置事件，退出事件WM_DELETE_WINDOW 抓取退出的x键


root.mainloop()"""
from tkinter import messagebox
import tkinter as tk
from random import random
import time
class Jsuanq:
    #设置窗口的基础要素
    def __init__(self):
        self.root = tk.Tk()#设置窗口
        self.root.title('简易计算器') #窗口名称
        self.root.geometry('295x260+100+100') #设置窗口大小
        self.root.resizable(0,0) #设置窗口大小不可改变

    #设置窗口基本要素
        self.font = ('黑体',30)
        self.font2 = ('黑体',16)
        self.res = tk.StringVar()#设置可变文本
        self.res.set('0')
        self.label = tk.Label(self.root
                              ,height=3,width=25,font=self.font2,textvariable=self.res,justify=tk.LEFT,anchor=tk.SE
                              )
        self.label.grid(row=1,column=1,columnspan=4)#设置相对布局
        #button_clear = tk.Button(root, text='c', font=30, bd=1, width=6, relief=tk.FLAT, bg='#b1b2b2')
        self.c = tk.Button(self.root,text='c',font=30,bd=1,width=6,relief=tk.FLAT,bg='#b1b2b2')#bg设置颜色，relief=tk.FLAT表示控件没有可见的边框或浮雕效果
        self.dc = tk.Button(self.root, text='<-', font=30, bd=1, width=6, relief=tk.FLAT, bg='#b1b2b2')
        self.chu = tk.Button(self.root, text='/', font=30, bd=1, width=6, relief=tk.FLAT, bg='#b1b2b2')
        self.X = tk.Button(self.root, text='X', font=30, bd=1, width=5, relief=tk.FLAT, bg='#b1b2b2')
        #button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.c.grid(row=2, column=1, padx=4, pady=2)
        self.dc.grid(row=2, column=2, padx=4, pady=2)
        self.chu.grid(row=2, column=3, padx=4, pady=2)
        self.X.grid(row=2,column=4,padx=4,pady=2)

        self.a7 = tk.Button(self.root, text='7', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a8 = tk.Button(self.root, text='8', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a9 = tk.Button(self.root, text='9', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.aj = tk.Button(self.root, text='-', font=30, bd=1, width=5, relief=tk.FLAT, bg='#b1b2b2')
        # button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.a7.grid(row=3, column=1, padx=4, pady=2)
        self.a8.grid(row=3, column=2, padx=4, pady=2)
        self.a9.grid(row=3, column=3, padx=4, pady=2)
        self.aj.grid(row=3, column=4, padx=4, pady=2)
        self.a4 = tk.Button(self.root, text='4', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a5 = tk.Button(self.root, text='5', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a6 = tk.Button(self.root, text='6', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.aja = tk.Button(self.root, text='+', font=30, bd=1, width=5, relief=tk.FLAT, bg='#b1b2b2')
        # button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.a4.grid(row=4, column=1, padx=4, pady=2)
        self.a5.grid(row=4, column=2, padx=4, pady=2)
        self.a6.grid(row=4, column=3, padx=4, pady=2)
        self.aja.grid(row=4, column=4, padx=4, pady=2)
        self.a1 = tk.Button(self.root, text='1', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a2 = tk.Button(self.root, text='2', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.a3 = tk.Button(self.root, text='3', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        self.ady = tk.Button(self.root, text='=', font=30, bd=1,height=3, width=5, relief=tk.FLAT, bg='#eacda1')
        # button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.a1.grid(row=5, column=1, padx=4, pady=2)
        self.a2.grid(row=5, column=2, padx=4, pady=2)
        self.a3.grid(row=5, column=3, padx=4, pady=2)
        self.ady.grid(row=5, column=4, padx=4, pady=2,rowspan=2)
        self.a0 = tk.Button(self.root, text='0', font=30, bd=1, width=14, relief=tk.FLAT, bg='#eacda1')
        # button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.a0d = tk.Button(self.root, text='.', font=30, bd=1, width=6, relief=tk.FLAT, bg='#eacda1')
        # button_clear.grid(row=2,column=1,padx=4,pady=2)       #padx,pady 表示边距
        self.a0d.grid(row=6, column=3, padx=4, pady=2)
        self.a0.grid(row=6, column=1, padx=4, pady=2,columnspan=2)
        self.a = ''
        def click_button(x):
            #global self.a
            #print(x)
            if x == 'c':
                self.a = ''
                self.res.set('0')
            else:
                self.a += x
                #if self.a.count('.') >= 2:
                if self.a.count('0') == len(self.a) :
                    self.a = '0'
                    self.res.set(self.a)
                else:
                    if self.a[0] == '0':
                        self.res.set(self.a[1:len(self.a)])
                    else:
                        if self.a[0] == '.':
                            self.b = '0'+self.a
                            self.res.set(self.b)
                        else:
                            self.res.set(self.a)





        def equile(): #计算等于
            opt_str = self.res.get()
            dy = eval(opt_str)   #计算字符串中的正确表达式
            #self.a = ''
            self.res.set(dy)
        def huic():#删除数据
            s = self.res.get()
            b = s[0:len(s)-1]
            self.a = b
            if b == '':
                self.res.set('0')
            else:
                self.res.set(b)

        # 命令
        self.a0.config(command=lambda: click_button('0'))
        self.a1.config(command=lambda: click_button('1'))
        self.a2.config(command=lambda: click_button('2'))
        self.a3.config(command=lambda: click_button('3'))
        self.a4.config(command=lambda: click_button('4'))
        self.a5.config(command=lambda: click_button('5'))
        self.a6.config(command=lambda: click_button('6'))
        self.a7.config(command=lambda: click_button('7'))
        self.a8.config(command=lambda: click_button('8'))
        self.a9.config(command=lambda: click_button('9'))
        self.aja.config(command=lambda: click_button('+'))
        self.aj.config(command=lambda: click_button('-'))
        self.X.config(command=lambda: click_button('*'))
        self.chu.config(command=lambda: click_button('/'))
        self.c.config(command=lambda: click_button('c'))
        self.a0d.config(command=lambda: click_button('.'))
        self.ady.config(command=equile)
        self.dc.config(command=huic)
        self.root.mainloop()








a = Jsuanq()