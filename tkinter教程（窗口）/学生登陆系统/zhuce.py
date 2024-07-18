import tkinter as tk
from tkinter import ttk,messagebox
import random
import re
from db import db


class Zhuce:
    def __init__(self,root):
        self.root = root
        self.root.geometry('600x850+100+100')
        self.root.resizable(0,0)
        self.root.title('注册页面')
        self.zc = tk.Frame(self.root)
        self.zc.pack()
        self.font = ('黑体',30)
        self.font1 = ('黑体',20)
        tk.Label(self.zc,text='用户注册',padx=5,pady=5,font=self.font,anchor=tk.CENTER,height=3).grid(row=0,column=2)
        tk.Label(self.zc,text='姓名:   ',padx=5,pady=5,font=self.font1,anchor=tk.W,height=3).grid(row=1,column=1)
        self.xm = tk.Entry(self.zc,width=30,bd=5)
        self.xm.grid(row=1, column=2, columnspan=2)
        tk.Label(self.zc,text='密码:   ', padx=5, pady=5, font=self.font1, anchor=tk.W, height=3).grid(row=2,column=1)
        self.mm = tk.Entry(self.zc,width=30,bd=5,show='*')
        self.mm.grid(row=2, column=2, columnspan=2)
        tk.Label(self.zc,text='重复密码:', padx=5, pady=5, font=self.font1, anchor=tk.W,height=3).grid(row=3,column=1)
        self.cfmm = tk.Entry(self.zc,width=30,bd=5,show='*')
        self.cfmm.grid(row=3, column=2, columnspan=2)

        tk.Label(self.zc, text='联系方式:', padx=5, pady=5, font=self.font1, anchor=tk.W, height=3).grid(row=4,column=1)
        self.lxfs = tk.Entry(self.zc,width=30,bd=5)
        self.lxfs.grid(row=4, column=2, columnspan=2)
        tk.Label(self.zc, text='输入验证码:', padx=5, pady=5, font=self.font1, anchor=tk.W, height=3).grid(row=5,column=1)
        self.yzm_tk = tk.StringVar()
        self.yzm_tk.set('')
        self.yzm = tk.Entry(self.zc,bd=5,textvariable=self.yzm_tk,font=12,show='*',width=13)
        self.yzm.grid(row=5, column=2)
        tk.Button(self.zc,text='获取验证码',padx=5,pady=5,command=self.captcha,width=7).grid(row=5,column=3)
        tk.Label(self.zc, text='注册密令:', padx=5, pady=5, font=self.font1, anchor=tk.W, height=3).grid(row=6,column=1)
        self.ml = tk.Entry(self.zc, bd=5,width=30)
        self.ml.grid(row=6, column=2,columnspan=2)
        tk.Button(self.zc,text='注册',font=self.font1,pady=5,padx=5,command=self.zc_new).grid(row=7,column=1)

        def returns():#实现重新返回到登陆页面
            self.zc.destroy()  #销毁当前窗口
            self.root.resizable(0,0)
            from denglu import Dlu
            Dlu(self.root)

        tk.Button(self.zc, text='返回',font=self.font1, pady=5, padx=5,command=returns).grid(row=7, column=3)
        self.b = []

        self.root.mainloop()


    def captcha(self): #获取验证码
        self.yanzm = r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\*\#'
        self.a = []
        for i in range(5):
           self.a.append(random.choice(self.yanzm))
           if len(self.a) >= 5:
               break
        self.b = ''.join(self.a)
        self.yzm_tk.set(self.b)

        """def send_sms(text, mobile):
            import http.client
            import urllib
            # 用户名和密码分别为APIID和APIKEY，如图所示
            host = "106.ihuyi.com"
            sms_send_uri = "/webservice/sms.php?method=Submit"
            # 用户名是登录用户中心->验证码短信->产品总览->APIID
            account = "C35128220"
            # 密码 查看密码请登录用户中心->验证码短信->产品总览->APIKEY
            password = "1cf3f5b07b56211e8e9e2094158910c1"
            params = urllib.urlencode(
                {'account': account, 'password': password, 'content': text, 'mobile': mobile, 'format': 'json'})
            headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
            conn = http.client.HTTPConnection(host, port=80, timeout=30)
            conn.request("POST", sms_send_uri, params, headers)
            response = conn.getresponse()
            response_str = response.read()
            conn.close()
            return response_str

        if __name__ == '__main__':
            mobile = self.lxfs.get()
            text = f"您的验证码是:{self.b}。请不要把验证码泄露给其他人。"
            print(send_sms(text, mobile))"""


    def zc_new(self):
            pattern = r'(13[4-9]\d{8})$|(15[01289]\d{8})$'
            respons = re.match(pattern, self.lxfs.get())
            if self.xm.get() != '':
                if self.mm.get() != '':
                    if self.cfmm.get() == self.mm.get():
                        if self.lxfs.get() != '':
                            if respons != None:
                                if self.yzm.get() == self.b:
                                    if self.ml.get() == 'ynsfdx121cgxq':
                                        a = f"{self.xm.get()}"
                                        b = {"user": f"{self.xm.get()}", "password": f"{self.cfmm.get()}"}
                                        pd,jg =db.zc_newpeople(a,b)
                                        if pd:
                                            messagebox.showwarning('提示', jg)
                                        else:
                                            messagebox.showwarning('提示', jg)
                                    else:
                                        messagebox.showwarning('警告警告', '无效密令')
                                else:
                                    messagebox.showwarning('提示', '请输入正确的验证码')
                            else:
                                messagebox.showwarning('提示', '无效手机号码')
                    else:
                        messagebox.showwarning('提示', '两次输入的密码不一样')
            else:
                messagebox.showwarning('提示','请输入姓名')
















if __name__ == '__main__':
    root = tk.Tk()
    Zhuce(root)
