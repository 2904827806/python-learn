a = {'name':'gaoqi','age':18,'job':'programmer'}
#字典的访问
b = a.get('name')
print(b)

c = dict(name='gaoqi',age=18,job='programmer')
print(c)
#字典的创建

d = ['name','age','job']
e = ['gaoqi',18,'techer']
g = dict(zip(d,e))
print(g)

#a = dict.fromkeys(['name','age','job'])
#print(a)
b = a['age']
print(b)
print(a)
c = a.keys()
print(c)
#字典的遍历
for i,j in a.items():
    print(i,'的电话是',j)
# 字典的删除
del (a['name'])
print (a)
l = a.pop('age')
print(a)
print (l)
o = a.clear()
p = a.values()
print(p)
t=a.items()
print(t)
# 对字典进行序列解包
b = {'name':'gaoqi','age':18,'job':'programmer'}
n,a,z=b   #默认对建进行操作
print(n)
n,a,z = b.items()        #对建值对进行操作
print(a)
