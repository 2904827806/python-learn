'''a = 'abd_33'
b = 'abd_33'
print(a is b)
c = 'abd_##'
d = 'abd_##'
print(c is d)

print (a in a)
#去处首位信息
e = "*sx*fd*ffq*".strip("*")
print (e)
f = "*sx*fd*ffq*".lstrip("*")
print (f)
g = "*sx*fd*ffq*".rstrip("*")
print (g)


#大小写转换
t = input()
q = t.capitalize()
f = t.title()
r = t.upper()
y = r.lower()
u = t.swapcase()
print(q)
print(f)
print(r)
print(y)
print(u)'''


#海龟画图
'''import turtle
turtle.width(2)
a = 0
b = 59
for i in range(10000):
    a +=0.5
    print(a)
    if int(a) ==3600:
        break
if int(a) ==3600:
    for x in range(int(a)):
        for c in ('red','green','yellow'):
            turtle.color(c)
            turtle.forward(x)
        turtle.left(b)'''
        
        
                
