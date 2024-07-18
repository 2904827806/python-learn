import time
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

root = ttk.Window(size=(410,450))
root.resizable(0,0)
root.title("计算器")
font = ('宋体',30)
font1 = ('宋体',16)
es = ttk.Frame(root)
es.pack(fill=BOTH)
result = tk.StringVar()
result.set("0")
a1 = ttk.Frame(root)
a1.pack(pady=2)
ttk.Label(es,textvariable=result,font=font1,justify=LEFT,anchor=SE,width=20,padding=(18,25)).pack()
ttk.Button(a1,text=0,bootstyle=SUCCESS,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a1,text=1,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a1,text=2,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a1,text=3,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
a2 = ttk.Frame(root)
a2.pack(pady=5)
ttk.Button(a2,text=4,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a2,text=5,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a2,text=6,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a2,text='-',bootstyle=SUCCESS,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
a3 = ttk.Frame(root)
a3.pack(pady=5)
ttk.Button(a3,text=7,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a3,text=8,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a3,text=9,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a3,text='+',bootstyle=SUCCESS,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
a4 = ttk.Frame(root)
a4.pack(pady=5)
ttk.Button(a4,text=7,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a4,text=8,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a4,text=9,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
ttk.Button(a4,text='+',bootstyle=SUCCESS,width=5,padding=(18,25)).pack(side=LEFT,padx=5)
root.mainloop()