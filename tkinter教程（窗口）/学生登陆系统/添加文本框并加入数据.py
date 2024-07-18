import json
import tkinter as tk
from tkinter import ttk,messagebox
from db import db
def show():
    global la
    for _ in map(la.delete, la.get_children('')):
        pass
    ex = db.information()
    index = 0
    for stu in ex:
        la.insert('',index+1,values=(
            stu['name'],stu['math'],stu['chinese'],stu['english']
        ))

root = tk.Tk()
root.geometry('500x600+100+100')
root.title('现实')
ab = ("name","math","chinese","english")
la = ttk.Treeview(root,show='headings',columns=ab,height=0)
la.column("name",anchor=tk.CENTER)
la.column("math",anchor=tk.CENTER)
la.column("chinese",anchor=tk.CENTER)
la.column("english",anchor=tk.CENTER)
la.heading('name',text='姓名')
la.heading("math",text='数学')
la.heading("chinese",text='语文')
la.heading("english",text='英语')
la.pack(fill=tk.BOTH,expand=True)
a = tk.Scrollbar(la)
a.config(command=la.yview)
a.pack(side=tk.RIGHT,padx=5,fill=tk.Y)
show()

root.mainloop()

