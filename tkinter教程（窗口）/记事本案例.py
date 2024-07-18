import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os

root = tk.Tk() #实例化窗口
root.geometry('800x500+100+100') #窗口大小
root.resizable(0,0)#设置窗口是否可以扩展
root.title('记事本')#窗口名称
#菜单栏
mune = tk.Menu(root,tearoff=False)

"""
menu.add_cascade      添加子选项
menu.add_command      添加命令（label参数为显示内容）
menu.add_separator    添加分隔线
menu.add_checkbutton  添加确认按钮
delete                删除
"""
filename = ''       #设置文件名称
menu = tk.Menu(root,tearoff=False)  #一级菜单
def new_file(): #新建文件夹
    global root,filename,text_pad
    root.title('未命名文件')     #重新命名窗口的名称
    filename = None
    text_pad.delete(1.0,tk.END)
def open_file():  #打开文件
    global filename
    filename = filedialog.askopenfilename(defaultextension='txt')   #设置查找文本框的名称
    if filename =='':
        filename = None
    else:
        root.title("" + os.path.basename(filename))    #设置窗口名称=抓取文件的名称
        text_pad.delete(1.0,tk.END) #删除 Text 组件中的内容可以用 delete() 方法，下边代码用于删除所有内容
        file = open(filename,'r',encoding='utf8')        #打开文件
        text_pad.insert(1.0,file.read())  #将内容插入到光标处
        file.close() #关闭文件
def save(): #保存文件
    try:
        open_file = open(filename,'w',encoding='utf8')   #命名文件
        msg = text_pad.get(1.0,'end')   #获得 Text 组件的内容，可以使用 get() 方法（仅获取文本内容）
        open_file.write(msg)     #将获取的text数据，写入到文件中
        open_file.close()
    except:
        save_additionlly()
def save_additionlly():#另存为
    """tkinter.filedialog.asksaveasfilename():选择以什么文件名保存，返回文件名
tkinter.filedialog.asksaveasfile():选择以什么文件保存，创建文件并返回文件流对象
tkinter.filedialog.askopenfilename():选择打开什么文件，返回文件名
tkinter.filedialog.askopenfile():选择打开什么文件，返回IO流对象
tkinter.filedialog.askdirectory():选择目录，返回目录名
tkinter.filedialog.askopenfilenames():选择打开多个文件，以元组形式返回多个文件名
tkinter.filedialog.askopenfiles():选择打开多个文件，以列表形式返回多个IO流对象
"""
    try:
        new_file = filedialog.asksaveasfile(initialfile='未命名',filetypes=[('txt文件','.txt')], defaultextension='txt')
        #new_file1 = filedialog.askopenfilename(initialfile='未命名',defaultextension='txt')
        creat_file = open(new_file, 'w', encoding='utf-8')
        msg = text_pad.get(1.0,'end')
        creat_file.write(msg)
        creat_file.close()
        root.title(''+os.path.basename(new_file))
    except:
        pass



file_menu = tk.Menu(menu,tearoff=False) #创建2阶菜单
file_menu.add_command(label='新建',command=new_file) #添加命令
file_menu.add_command(label='麻花')
file_menu.add_command(label='打开',command=open_file)
file_menu.add_separator()#添加横线
file_menu.add_command(label='保存',command=save)
file_menu.add_command(label='另存为',command=save_additionlly)
menu.add_cascade(label='文件',menu=file_menu)  #添加子菜单

edt_menu = tk.Menu(menu,tearoff=False) #二阶菜单
edt_menu.add_command(label='撤销')
edt_menu.add_command(label='重做')     #添加命令
edt_menu.add_separator()      #添加分割线
edt_menu.add_command(label='复制')
edt_menu.add_command(label='粘贴')
edt_menu.add_command(label='剪贴')
menu.add_cascade(label='编辑',menu=edt_menu)     #添加子选项
fie_menu = tk.Menu(edt_menu,tearoff=False)  #三级菜单
fie_menu.add_command(label='s')
fie_menu.add_command(label='x')
fie_menu.add_command(label='z')
edt_menu.add_cascade(label='关于',menu=fie_menu)
about_menu = tk.Menu(menu,tearoff=False) #二阶菜单
about_menu.add_command(label='作者')
about_menu.add_command(label='版权')
menu.add_cascade(label='关于',menu=about_menu)

status_str_var = tk.StringVar() #主要用于存储字符串类型的数据。它与 Tkinter 中的组件（如 Entry、Label、Button 等）结合使用，以实现动态更新这些组件的内容
status_str_var.set('字符数: {}'.format(0))    # 使输入框内容可变
#布局下面左右
status_label = tk.Label(root,textvariable=status_str_var,bd=1,relief=tk.SUNKEN,anchor=tk.W)     #设置状态栏 textvariable动态显示
status_label.pack(side=tk.BOTTOM,fill=tk.X)     #BOTTOM 由下到上排列 fill=tk.X沿水平方向填充

var_line = tk.StringVar() #主要用于存储字符串类型的数据。它与 Tkinter 中的组件（如 Entry、Label、Button 等）结合使用，以实现动态更新这些组件的内容
line_label = tk.Label(root,textvariable=var_line,width=4,bg='#faebd7',anchor=tk.N,font=18)     #设置状态栏 textvariable动态显示
line_label.pack(side=tk.LEFT,fill=tk.Y)     #BOTTOM 由下到上排列 fill=tk.X沿水平方向填充


#设置文本框
text_pad = tk.Text(root,font=18)
text_pad.pack(fill=tk.BOTH,expand=True)           #both,上下左右都填充 ，expand允许进行拓展

#设置滚动栏
scroll = tk.Scrollbar(text_pad)
#text_pad.config(yscrollcommand=scroll.set)
scroll.config(command=text_pad.yview)
scroll.pack(side=tk.RIGHT,fill=tk.Y)   #fill 表示填充



root.config(menu=menu)      #布局


"""def on_eixt():
    messagebox.showwarning(title='提示',message='此路不通')
root.protocol('WM_DELETE_WINDOW',on_eixt)  # 重置事件，退出事件WM_DELETE_WINDOW"""




root.mainloop()