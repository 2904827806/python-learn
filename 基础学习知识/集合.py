a = (1,2,3,4,5,6,'gggg')
c = set(a)
'''print(type(a))
b = str(a)
f = [1,2,3,'ff']
c = set(a)
c.add(53)
print(type(b))
print(b)
print(c)
print(f)
k = set(f)
k.add(46)
print(k)'''
print(c)
b = {1,5,'jh','gggg'}
print(b)

print(c|b)#并集
print(c&b)      #交集

print(c-b)            #差集

print(c.union(b))              #  并集
print(c.intersection(b))           #交集
print(c.difference(b))              # 差集
