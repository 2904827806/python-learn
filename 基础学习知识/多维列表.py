'''a = [
    ["高小一",20,30000,"上海"],
    ["高小二",18,40000,"北京"],
    ["高小三",17,30000,"深圳"],
    ["高小四",15,12000,"云南"],
    ["高小五",19,10000,"贵州"],
    ["高小六",17,10000,"天津"],
    ["高小七",18,30000,"深圳"],
    ["高小八",42,80005,"上海"],
    ["高小九",25,70000,"上海"]
]

for i in range(10):
    for j in range(4):
        print(a[i][j],end='\t')
    print()'''
            
r1 = {'name':"高小一",'age':20,'xz':50000,'city':"上海"}
r2 = {'name':"高小s",'age':20,'xz':70000,'city':"贵州"}
r3 = {'name':"高小g",'age':20,'xz':90000,'city':"上海"}
r4 = {'name':"高小e",'age':20,'xz':30000,'city':"天津"}
b = [r1,r2,r3,r4]
print(b)
#获得第二行人的薪资
print(b[1].get('xz'),'\n')
#打印表中所有人都薪资
for i in range (4):
    c = b[i].get('xz')
    print(c)
    print()
#打印表中的所有数据
for i in range(len(b)):
    print(b[i].get('name'),b[i].get('age'),b[i].get('xz'),b[i].get('city'))
for i in range(4):
    s = str(b[i].values())
    o=s.strip('dict_values')
    print(o)
    
