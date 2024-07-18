from openpyxl import load_workbook

#读文件
def opens():
    #from openpyxl import load_workbook
    wb = load_workbook('z.xlsx')
    #获取工作表
    sh1 = wb.active #激活工作簿
    sh2 = wb['Sheet1']
    sh3=wb.get_sheet_by_name('Sheet1')
    print(sh1)
#获取工作簿名字
def show_sheet(f):
    #from openpyxl import load_workbook
    wb = load_workbook('电影数据.xlsx')
    print(wb.sheetnames)
    #通过遍历获取工作簿名称
    for sh in wb:
        print(sh.title)
#获取值
def get_one_values(f):
    #from openpyxl import load_workbook
    wb = load_workbook(f)
    #获取工作表
    sh1 = wb.active
    value1 = sh1.cell(2,3).vaule #获取第二行第三列的数据
    value2 = sh1['c2']#获取第二行第c列的数据

def get_many_values(f):
    #from openpyxl import load_workbook
    wb = load_workbook(f)
    # 获取工作表
    sh1 = wb['sheet1']
    #切片(一部分）
    cells1 = sh1['c2:d3'].value#获取数据
    #整行，整列
    cells2 = sh1[3].value#3行
    cells3 = sh1['c'].value#第c例
    #通过迭代获取数据
    #行
    for row in sh1.ite_row(min_row=2,max_row=5,max_coll = 3):
        for cell in row:
            print(cell.value)
#打印所有数据
def get_all(f):
    #from openpyxl import load_workbook
    wb = load_workbook(f)
    # 获取工作表
    sh1 = wb['sheet1']
    for row in sh1.rows:
        for cell in row:
            print(cell.value)
    for column in sh1.columns:
        for cell in column:
            print(cell.value)
#获取行数，列数
def get_num(f):
    #from openpyxl import load_woekbook
    wb = load_workbook(f)
    # 获取工作表
    sh1 = wb['sheet1']
    print(sh1.max_row)
    print(sh1.max_column)




