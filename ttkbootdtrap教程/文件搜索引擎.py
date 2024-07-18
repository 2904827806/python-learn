import datetime
import pathlib
from queue import Queue
from threading import Thread
from tkinter.filedialog import askdirectory
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility


class FileSearchEngine(ttk.Frame):

    queue = Queue()
    searching = False

    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)

        # application variables
        _path = pathlib.Path().absolute().as_posix()
        self.path_var = ttk.StringVar(value=_path)
        self.term_var = ttk.StringVar(value='md')
        self.type_var = ttk.StringVar(value='endswidth')

        # header and labelframe option container
        option_text = "Complete the form to begin your search"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_path_row()     #调用create_path_row() 函数
        self.create_term_row()
        self.create_type_row()
        self.create_results_view()

        self.progressbar = ttk.Progressbar( #创建进度条
            master=self,
            mode=INDETERMINATE,       #水平
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progressbar.pack(fill=X, expand=YES)    #布局

    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)   #创建一个页面
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Path", width=8)  #标签
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.path_var)    #创建单行文本框
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        browse_btn = ttk.Button(     #创建一个按钮 并绑定self.on_browse方法
            master=path_row,
            text="Browse",
            command=self.on_browse,
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_term_row(self):
        """Add term row to labelframe"""
        term_row = ttk.Frame(self.option_lf)
        term_row.pack(fill=X, expand=YES, pady=15)
        term_lbl = ttk.Label(term_row, text="Term", width=8)
        term_lbl.pack(side=LEFT, padx=(15, 0))
        term_ent = ttk.Entry(term_row, textvariable=self.term_var)
        term_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        search_btn = ttk.Button(
            master=term_row,
            text="Search",
            command=self.on_search,
            bootstyle=OUTLINE,
            width=8
        )
        search_btn.pack(side=LEFT, padx=5)

    def create_type_row(self):
        """Add type row to labelframe"""
        type_row = ttk.Frame(self.option_lf)
        type_row.pack(fill=X, expand=YES)
        type_lbl = ttk.Label(type_row, text="Type", width=8)
        type_lbl.pack(side=LEFT, padx=(15, 0))

        contains_opt = ttk.Radiobutton( #创建单选框
            master=type_row,
            text="Contains",
            variable=self.type_var, #用于存储和跟踪单选按钮的值
            # 当用户选择不同的单选按钮时，self.type_var 的值会更新为所选按钮的value
            value="contains"#value 参数为这个特定的单选按钮指定了一个值，
            # 当用户选择这个按钮时，self.type_var 的值会被设置为“contains”。
        )
        contains_opt.pack(side=LEFT)

        startswith_opt = ttk.Radiobutton(
            master=type_row,
            text="StartsWith",
            variable=self.type_var,
            value="startswith"
        )
        startswith_opt.pack(side=LEFT, padx=15)

        endswith_opt = ttk.Radiobutton(
            master=type_row,
            text="EndsWith",
            variable=self.type_var,
            value="endswith"
        )
        endswith_opt.pack(side=LEFT)
        endswith_opt.invoke()

    def create_results_view(self):
        """Add result treeview to labelframe"""
        self.resultview = ttk.Treeview(     #用于在图形用户界面（GUI）中显示层次化的数据结构，
            # 如文件系统的目录结构或复杂的组织结构图
            master=self,
            bootstyle=INFO,
            columns=[0, 1, 2, 3, 4],
            show=HEADINGS
            #show=HEADINGS: 这个参数用于控制 Treeview 控件中哪些元素应该被显示。
            #HEADINGS 表示只显示列的标题，不显示数据行。
        )
        self.resultview.pack(fill=BOTH, expand=YES, pady=10)

        # setup columns and use `scale_size` to adjust for resolution
        self.resultview.heading(0, text='Name', anchor=W)
        self.resultview.heading(1, text='Modified', anchor=W)
        self.resultview.heading(2, text='Type', anchor=E)
        self.resultview.heading(3, text='Size', anchor=E)
        self.resultview.heading(4, text='Path', anchor=W)
        self.resultview.column(
            column=0,
            anchor=W,
            width=utility.scale_size(self, 125),
            stretch=False
        )
        self.resultview.column(
            column=1,
            anchor=W,
            width=utility.scale_size(self, 140),
            stretch=False
        )
        self.resultview.column(
            column=2,
            anchor=E,
            width=utility.scale_size(self, 50),
            stretch=False
        )
        self.resultview.column(
            column=3,
            anchor=E,
            width=utility.scale_size(self, 50),
            stretch=True
        )
        self.resultview.column(
            column=4,
            anchor=W,
            width=utility.scale_size(self, 300)
        )

    def on_browse(self):
        """Callback for directory browse"""
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var.set(path)

    def on_search(self):
        """Search for a term based on the search type"""
        search_term = self.term_var.get()
        search_path = self.path_var.get()
        search_type = self.type_var.get()

        if search_term == '':
            return

        # start search in another thread to prevent UI from locking
        Thread(#线程
            target=FileSearchEngine.file_search,
            args=(search_term, search_path, search_type),
            daemon=True
        ).start()
        self.progressbar.start(10)

        iid = self.resultview.insert(
            parent='',
            index=END,
        )
        self.resultview.item(iid, open=True)
        self.after(100, lambda: self.check_queue(iid))

    def check_queue(self, iid):
        """Check file queue and print results if not empty"""
        if all([
            FileSearchEngine.searching,
            not FileSearchEngine.queue.empty()
        ]):
            filename = FileSearchEngine.queue.get()
            self.insert_row(filename, iid)
            self.update_idletasks()#强制Tkinter处理所有待处理的“空闲任务”（idle tasks）
            self.after(100, lambda: self.check_queue(iid))
        elif all([
            not FileSearchEngine.searching,
            not FileSearchEngine.queue.empty()      #队列是否为空
        ]):
            while not FileSearchEngine.queue.empty():#while循环
                filename = FileSearchEngine.queue.get()    #获取队列中的数据
                self.insert_row(filename, iid)    #调用函数insert_row（）
            self.update_idletasks() #强制Tkinter处理所有待处理的“空闲任务”（idle tasks）
            self.progressbar.stop()#使用特定的stop方法来停止进度条
        elif all([
            FileSearchEngine.searching,
            FileSearchEngine.queue.empty()
        ]):
            self.after(100, lambda: self.check_queue(iid)) #elf.after(delay, function)是一个常用的方法，
            # 用于在指定的延迟（以毫秒为单位）后调用一个函数
        else:
            self.progressbar.stop()

    def insert_row(self, file, iid):
        """Insert new row in tree search results"""
        try:
            _stats = file.stat() #获取文件的基本信息
            _name = file.stem
            _timestamp = datetime.datetime.fromtimestamp(_stats.st_mtime) #最后一次修改时间
            _modified = _timestamp.strftime(r'%m/%d/%Y %I:%M:%S%p')
            _type = file.suffix.lower()  #对文件的后缀名进行小写转换
            _size = FileSearchEngine.convert_size(_stats.st_size)
            _path = file.as_posix()  #文件名称
            iid = self.resultview.insert(
                parent='',
                index=END,
                values=(_name, _modified, _type, _size, _path)
            )
            self.resultview.selection_set(iid)
            self.resultview.see(iid)
        except OSError:
            return


    """在Python中，@staticmethod是一个装饰器，用于将一个方法标记为静态方法。
静态方法不依赖于类的实例或类本身来执行，
它们就像普通的函数一样，只不过被封装在类的作用域内"""

    @staticmethod
    def file_search(term, search_path, search_type):
        """Recursively search directory for matching files"""
        FileSearchEngine.set_searching(1)    #调用类FileSearchEngine的set_searching函数
        if search_type == 'contains':
            FileSearchEngine.find_contains(term, search_path)      #调用类下的不同函数
        elif search_type == 'startswith':
            FileSearchEngine.find_startswith(term, search_path)
        elif search_type == 'endswith':
            FileSearchEngine.find_endswith(term, search_path)

    @staticmethod
    def find_contains(term, search_path):
        """Find all files that contain the search term"""
        for path, _, files in pathlib.os.walk(search_path):    #遍历文件夹
            if files:
                for file in files:
                    if term in file:
                        record = pathlib.Path(path) / file    #创建新的文件夹
                        FileSearchEngine.queue.put(record) #把文件加入到队列中
        FileSearchEngine.set_searching(False)

    @staticmethod
    def find_startswith(term, search_path):
        """Find all files that start with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.startswith(term): #检查字符串是否以指定的前缀开始
                        record = pathlib.Path(path) / file
                        FileSearchEngine.queue.put(record) #添加数据
        FileSearchEngine.set_searching(False)

    @staticmethod
    def find_endswith(term, search_path):
        """Find all files that end with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.endswith(term): #检查文件是否存在指定后缀
                        record = pathlib.Path(path) / file
                        FileSearchEngine.queue.put(record)
        FileSearchEngine.set_searching(False)

    @staticmethod
    def set_searching(state=False):
        """Set searching status"""
        FileSearchEngine.searching = state

    @staticmethod
    def convert_size(size):
        """Convert bytes to mb or kb depending on scale"""
        kb = size // 1000
        mb = round(kb / 1000, 1)
        if kb > 1000:
            return f'{mb:,.1f} MB'
        else:
            return f'{kb:,d} KB'


if __name__ == '__main__':

    app = ttk.Window("File Search Engine", "journal")
    FileSearchEngine(app)
    app.mainloop()
