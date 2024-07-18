import turtle

import math
t = turtle.pen()
turtle.width(1)
#定义多个坐标
x1,y1 = 100,100
x2,y2 = 100,-100
x3,y3 =-100,-100
x4,y4 = -100,100

#绘制折线
turtle.penup()
turtle.goto(x1,y1)
turtle.pendown()
turtle.goto(x2,y2)
turtle.goto(x3,y3)
turtle.goto(x4,y4)
turtle.goto(x1,y1)
turtle.penup()
turtle.goto(0,0)
turtle.pendown()

for x in range(360):
    for c in ('blue','red','yellow'):
        turtle.color(c)
        turtle.forward(x)
        turtle.left(59)
        
                
                
    
 

#计算起点和终点的距离
distance = math.sqrt((x1-x4)**2 + (y1-y4)**2)

turtle.write(distance)
