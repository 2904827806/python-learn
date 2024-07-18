import tkinter as tk
from tkinter import messagebox,ttk
from db import db
from zjm import Zjm
from zhuce import Zhuce

class Dlu:
    def __init__(self,root):
        self.root = root
        self.root.title('学生登陆系统')
        self.root.geometry('280x300')
        self.root.resizable(0,0)
        #登陆页设置
        self.fram = tk.Frame(self.root)

        self.fram.pack()
        tk.Label(self.fram,text='登陆页',font=('黑体',25),height=3).grid(row=0,column=2)
        #账号

        tk.Label(self.fram,text='账户：',font=18,height=3,anchor=tk.CENTER).grid(row=1,column=1)
        self.zh = tk.Entry(self.fram)
        self.zh.grid(row=1,column=2,columnspan=2)
        #密码
        tk.Label(self.fram, text='密码：',font=18,height=3,anchor=tk.CENTER).grid(row=2,column=1)
        self.mm = tk.Entry(self.fram)
        self.mm.grid(row=2,column=2,columnspan=2)
        def zh(x):
            if x == -1:
                self.fram.destroy()#摧毁当前窗口
                self.root.resizable(1,1)
                Zhuce(self.root)
            else:
                a = self.zh.get()
                b = self.mm.get()
                name, messag = db.jc(a,b)
                if name:
                    messagebox.showwarning(title='提示', message='登陆成功')
                    self.fram.destroy() #销毁当前窗口
                    self.root.resizable(1,1)
                    Zjm(self.root)

                else:
                    messagebox.showwarning(title='警告', message=messag)

        #登陆，退出按钮
        self.open = tk.Button(self.fram,text='登陆',anchor=tk.W,padx=10)
        self.open.grid(row=3,column=1)
        self.close = tk.Button(self.fram, text='注册',anchor=tk.E,padx=10)
        self.close.grid(row=3,column=3)
        self.open.config(command=lambda: zh(1))
        self.close.config(command=lambda: zh(-1))
        self.root.mainloop()



if __name__=='__main__':
    root = tk.Tk()
    Dlu(root)


"""class Dl:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('280x300')
        self.root.resizable(0,0)
        self.root.title('登陆页面')

        self.dly = tk.Frame(self.root)
        self.dly.pack()
        tk.Label(self.dly,text='登陆页',font=('黑体',25),height=3).grid(row=0,column=2)
        tk.Label(self.dly, text='账号：', font=18,height=3,anchor=tk.CENTER).grid(row=1, column=1)
        tk.Entry(self.dly).grid(row=1, column=2,columnspan=2)
        tk.Label(self.dly, text='密码：', font=18,height=3,anchor=tk.CENTER).grid(row=2, column=1)
        tk.Entry(self.dly).grid(row=2, column=2, columnspan=2)
        self.open = tk.Button(self.dly,text='登陆',anchor=tk.W,padx=10)
        self.open.grid(row=3,column=1)
        self.close = tk.Button(self.dly, text='退出',anchor=tk.E,padx=10)
        self.close.grid(row=3,column=3)






        self.root.mainloop()"""



