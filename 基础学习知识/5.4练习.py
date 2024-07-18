"""b = ('一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月')
for i in range(1024**1024):
   a = int(input('你想要输入的月份：'))

   ''' if a <=12:
        print(a,'月的中文名是',b[a-1])
    else:
        print('不合法')'''
   print(b[a-1])

'''for i in range(1021**1024):
    c = input('b')
    if c in b:
        print('拜拜')
    else:
        print('good')'''"""
   
import turtle
t = turtle.Pen()
for i in range(360):
   for j in ('blue','red','green'):
      t.forward(i)
      t.left(59)
      t.color(j)



turtle.done()
