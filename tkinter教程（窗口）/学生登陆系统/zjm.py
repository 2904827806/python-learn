import json
import tkinter as tk
from tkinter import messagebox,ttk
from db import db
class Zjm:
    def __init__(self,root):
        self.root = root
        self.root.title('学生信息管理系统')
        self.root.geometry('600x650+100+100')
        #self.root.resizable(0, 0)
        #一级菜单
        self.menu = tk.Menu(self.root,tearoff=False)
        #tearoff=False去虚线
        # 二级菜单
        # side：指定组件在窗口中的放置位置，可选值为LEFT、RIGHT、TOP、BOTTOM。
        # BOTTOM 由下到上排列 fill=tk.X沿水平方向填充，TOP从上到下
        # fill：指定组件在窗口中的填充方式，可选值为X、Y、BOTH。
        # both,上下左右都填充 ，
        # expand：指定组件是否随窗口的大小改变而自动扩展，可选值为True或False。
        # anchor：指定组件在窗口中的对齐方式，可选值为N、S、E、W、NE、NW、SE、SW。
        # padx：指定组件的水平内边距（以像素为单位）。
        # pady：指定组件的垂直内边距（以像素为单位）。
        # Frame 　　框架，将几个组件组成一组
        self.font=('黑体',30)
        self.font1=('楷体',25)
        self.lr = tk.Frame(self.root)
        tk.Label(self.lr,text='录入信息',font=self.font,height=4,anchor=tk.CENTER).grid(row=0,column=2)
        tk.Label(self.lr, text='姓名:',font=self.font1,height=3,anchor=tk.CENTER).grid(row=1, column=1)
        self.name2 = tk.StringVar()
        self.math2 = tk.StringVar()
        self.yuw = tk.StringVar()
        self.english2 = tk.StringVar()
        self.xm = tk.Entry(self.lr,textvariable=self.name2,width=30)
        self.xm.grid(row=1,column=2,columnspan=2)
        tk.Label(self.lr, text='数学:',font=self.font1,height=3,anchor=tk.CENTER).grid(row=2, column=1)
        self.sx = tk.Entry(self.lr,width=30,textvariable=self.math2)
        self.sx.grid(row=2, column=2,columnspan=2)
        tk.Label(self.lr, text='语文:',font=self.font1,height=3,anchor=tk.CENTER).grid(row=3, column=1)
        self.yw = tk.Entry(self.lr,textvariable=self.yuw,width=30)
        self.yw.grid(row=3, column=2,columnspan=2)
        tk.Label(self.lr, text='英语:',font=self.font1,height=3,anchor=tk.CENTER).grid(row=4, column=1)
        self.yy = tk.Entry(self.lr,textvariable=self.english2,width=30)
        self.yy.grid(row=4, column=2,columnspan=2)
        def input():
            stu = {"name":self.xm.get(),"math":self.sx.get(),"chinese":self.yw.get(),"english":self.yy.get()}
            db.insert(stu)
        self.lr_button = tk.Button(self.lr,text='录入',padx=5,pady=5,command=input)
        self.lr_button.grid(row=5,column=3)
        self.menu.add_cascade(label='录入',command=lambda:self.show_frame(1))
        self.cx = tk.Frame(self.root)
        #tk.Label(self.cx,text='查询信息').pack()
        #创建表格
        def show():
            for _ in map(self.tre_view.delete,self.tre_view.get_children('')): #更新
                pass
            students = db.information()
            index = 0
            for stu in students:
                #print(stu)
                self.tre_view.insert('',index +1,values=(
                    stu['name'],stu['math'],stu['chinese'],stu['english']
                ))
        self.colum = ("name","math","chinese","english")
        #self.colums_view = ('姓名','数学','语文','英语')
        #tkinter 树形列表控件(Treeview)
        """
        columns	接收一个列表（tuple），列表中的每个元素都代表表格中的一列（可以理解为 ID），列表的长度就是表格的列数。
        displaycolumns	接收一个列表（tuple），列表每个元素都代表 columns 中列的编号，用于设置列表要显示的列，以及显示列的顺序（没有在列表中的列不会显示）。传入"#all"显示所有列。
        height	表格的高度，也就是能够显示的行数。
        padding	内容距离组件边缘的距离。
        selectmode	有 "extended"（默认选项）、"browse"、"none"三种选项，分别代表多选（Ctrl+鼠标左键），单选以及不能改变选项。
        show	有 "tree"、"headings"、"tree headings" 三种选项，分别代表显示图标列（编号为 "#0"）、不显示图标列（仅显示数值列）以及显示所有列（图标列和数值列）。 个人理解：用作树需要加上"tree"，用作表使用"headings"。"""
        self.tre_view = ttk.Treeview(self.cx,show='headings',columns=self.colum,height=0)
        self.tre_view.column("name",anchor=tk.CENTER)  #设置列表的位置
        self.tre_view.column("math", anchor=tk.CENTER)
        self.tre_view.column("chinese", anchor=tk.CENTER)
        self.tre_view.column("english",  anchor=tk.CENTER)
        self.tre_view.heading('name',text='姓名',) #给标题重命名
        self.tre_view.heading('math', text='数学')
        self.tre_view.heading('chinese', text='语文')
        self.tre_view.heading('english', text='英语')
        self.tre_view.pack(fill=tk.BOTH,expand=True)
        show()
        self.lg = tk.Scrollbar(self.tre_view)
        self.lg.config(command=self.tre_view.yview)
        self.tre_view.configure(yscrollcommand=self.lg.set) #能够使滚动栏正确运行
        self.lg.pack(side=tk.RIGHT,fill=tk.Y)
        self.genx = tk.Button(self.cx,text='更新',font=15,pady=5,padx=5,command=show)
        self.genx.pack(side=tk.BOTTOM,anchor=tk.CENTER)
        self.menu.add_cascade(label='查询',command=lambda:self.show_frame(2))
        self.sc = tk.Frame(self.root)
        tk.Label(self.sc, text='删除页面', font=('黑体',28), height=5,anchor=tk.CENTER).grid(row=1, column=2)
        tk.Label(self.sc,text='根据名字删除数据').grid(row=2,column=1)
        self.ex = tk.StringVar()
        self.ex.set('')
        self.sc_en = tk.Entry(self.sc,width=40,textvariable=self.ex)
        self.sc_en.grid(row=3,column=1,columnspan=2)
        self.scshuj = tk.Button(self.sc,text='删除',padx=5,pady=5,command=self.xgshuj)
        self.scshuj.grid(row=3,column=3)
        self.menu.add_cascade(label='删除',command=lambda:self.show_frame(3))
        self.xg = tk.Frame(self.root)
        self.name3 = tk.StringVar()
        self.name3.set('')
        self.math3 = tk.StringVar()
        self.math3.set('')
        self.yuw3 = tk.StringVar()
        self.yuw3.set('')
        self.english3 = tk.StringVar()
        self.english3.set('')
        tk.Label(self.xg, text='修改页面',font=self.font,height=4).grid(row=0, column=2)
        tk.Label(self.xg, text='姓 名:', font=self.font1, height=3).grid(row=1, column=1)
        self.xg_name = tk.Entry(self.xg,width=30,textvariable=self.name3)
        self.xg_name.grid(row=1,column=2,columnspan=2)
        tk.Label(self.xg, text='数 学:', font=self.font1, height=3).grid(row=2, column=1)
        self.xg_math = tk.Entry(self.xg, width=30,textvariable=self.math3)
        self.xg_math.grid(row=2, column=2, columnspan=2)
        tk.Label(self.xg, text='语 文:', font=self.font1, height=3).grid(row=3, column=1)
        self.xg_chinese = tk.Entry(self.xg, width=30,textvariable=self.yuw3)
        self.xg_chinese.grid(row=3, column=2, columnspan=2)
        tk.Label(self.xg, text='英 语:', font=self.font1, height=3,).grid(row=4, column=1)
        self.xg_english = tk.Entry(self.xg, width=30,textvariable=self.english3)
        self.xg_english.grid(row=4, column=2, columnspan=2)
        tk.Button(self.xg,text='查询',padx=5,pady=5,command=self.sg_cx).grid(row=5,column=1)
        tk.Button(self.xg, text='修改',padx=5,pady=5,command=self.sg_rew).grid(row=5, column=3)
        self.menu.add_cascade(label='修改',command=lambda:self.show_frame(4))
        self.gy = tk.Frame(self.root)
        tk.Label(self.gy, text='关于作品：简洁的信息录入系统',font=('黑体',28),height=2).pack(side=tk.TOP,anchor=tk.W,fill=tk.Y)
        tk.Label(self.gy, text='关于作者：太帅导致自闭',font=('黑体',28),height=2).pack(side=tk.TOP,anchor=tk.W,fill=tk.Y)
        tk.Label(self.gy, text='作者qq：2904827806',font=('黑体',28),height=2).pack(side=tk.TOP,anchor=tk.W,fill=tk.Y)
        self.menu.add_cascade(label='关于',command=lambda:self.show_frame(5))
        self.root.config(menu=self.menu)
        self.root.mainloop()


    def input(self):
        a = self.yy.get()
        print(a)
    def xgshuj(self):
        name = self.sc_en.get()
        pd,jg = db.shanchu(name)
        if pd:
            messagebox.showwarning(message=jg)
        else:
            messagebox.showwarning(message=jg)
            self.ex.set('')
    def sg_cx(self):
        name = self.xg_name.get()
        jg,a,b,c = db.geng_rew(name)
        if jg:
            self.math3.set(a)
            self.yuw3.set(b)
            self.english3.set(c)
        else:
            messagebox.showwarning(title='提示',message='查无此人')

    def sg_rew(self):
        xg_name =self.xg_name.get()
        xg_cj = {"name":self.xg_name.get(),"math":self.xg_math.get(),"chinese":self.xg_chinese.get(),"english":self.xg_english.get()}
        jg,message = db.geng_xg(xg_name,xg_cj)
        if jg:
            messagebox.showwarning('提示',message)
        else:
            messagebox.showwarning('提示', message)


    def show_frame(self,x):
        if x == 1:
            self.lr.pack()
            self.cx.pack_forget()
            self.sc.pack_forget()
            self.xg.pack_forget()
            self.gy.pack_forget()
        elif x == 2:
            self.cx.pack(side=tk.TOP,fill=tk.BOTH,expand=True) #设置表格的填充方式
            self.lr.pack_forget()
            self.sc.pack_forget()
            self.xg.pack_forget()
            self.gy.pack_forget()
        elif x == 3:
            self.sc.pack()
            self.lr.pack_forget()
            self.cx.pack_forget()
            self.xg.pack_forget()
            self.gy.pack_forget()
        elif x == 4:
            self.lr.pack_forget()
            self.cx.pack_forget()
            self.sc.pack_forget()
            self.xg.pack()
            self.gy.pack_forget()
        elif x == 5:
            self.lr.pack_forget()
            self.cx.pack_forget()
            self.sc.pack_forget()
            self.xg.pack_forget()
            self.gy.pack()




if __name__ == '__main__':
    root = tk.Tk()
    Zjm(root)

