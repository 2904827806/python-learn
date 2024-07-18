import xlrd2
from xlutils.copy import copy
import xlutils
import xlwt

fill = r"C:\Users\29048\Desktop\电影数据.xlsx"
wb = xlrd2.open_workbook(fill)
sh = wb.sheet_by_index(0)
sj = {}
for i in range(1,sh.nrows):
    #print(sh.cell_value(i,0))
    #print(sh.cell_value(i, 1))
    a = {'name':sh.cell_value(i,0),'df':sh.cell_value(i,1),'dj':sh.cell_value(i,2)}
    #print(a)
    key = sh.cell_value(i,2)
    #print(key)
    if sj.get(key):
        sj[key].append(a)
    else:
        sj[key] = [a]

#print(sj)
wr = copy(wb)
#sh2 = wr.get_sheet(0)
r = ['电影名','得分','所属公司']
#print(sj.keys())
for i in sj.keys():
    sh1 = wr.add_sheet(f'{i}')
    sh1.write(0, 0, r[0])
    sh1.write(0, 1, r[1])
    sh1.write(0, 2, r[2])

    #print(i)
    #print(sj.get(i))
    for a,b in enumerate(sj.get(i)):
        #print(a)
        #print(b)
        for m,n in enumerate(b.keys()):
            #print(m)
            sh1.write(a+1,m,b[n])

wr.save('102.xlsx')
            #print(n)
            #print(m)
            #sh1.write()
