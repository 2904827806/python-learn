names= ('高一','高二','高三')
ages = (16,18,20)
jobs = ('程序员','公务员','老师')
d = zip(names,ages,jobs)
print(list(d))
for name,age,job in zip(names,ages,jobs):
    print('{2},{1},{0}'.format(name,age,job))

# 列表推到式
cells = [(row,col) for row in range(1,100) for col in range(1,100)]
for cell in cells:
    print(cell)


a = []
for i in  range(1,100):
    for j in range(1,100):
        d = (i,j)
        a.append(d)
print(a)
for k in range(0,99*99,1):
    f = a[k]
    print(f)
# 字典推导式
my = ' i love you,i love sxt,i love gaoqi'
ch = {c:my.count(c) for c in my}
print(ch)
print('******************************************')
m = list()
l = list()
for c in my:
    b = my.count(c)
    l.append(c)
    #print(l)
    m.append(b)
    #print(m)
d = dict(zip(l,m))
print(d)
print('\n''*********************************')
#集合推到式
a = {x for x in range(1,1000) if x %2==0}
print(a)
print('**********************************************')
#生成器推导式（生成元组）
gt = (x for x in range(1,1000) if x%2==0)
print(tuple(gt))
gt = (x for x in range(1,1000) if x%2==0)
b = list(gt)
print(b)
print('****************************************************************')
gt = (x for x in range(1,1000) if x%2==0)
for i in gt:
    print(i,end='\t')