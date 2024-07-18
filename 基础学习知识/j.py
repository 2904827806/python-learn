
'''import turtle

import math
t = turtle.pen()
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
f=a = input('')
g=b = input('')
for x in range(f):
    for c in ('blue','red','yellow'):
        turtle.color(c)
        turtle.forward(x)
        turtle.left(g)'''
"""a = [
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
i = 0
for x in range(9):
    for n in range(4):
        print(a[x][n],end= "\t")
    print()
import claas.size
print(claas.size.height)
"""
import turtle
import time
def rose():
	window = turtle.Screen()
	window.bgcolor("white")
	window.title("draw")
	tt = turtle.Turtle()
	tt.penup()
	tt.left(90)
	tt.fd(200)
	tt.pendown()
	tt.right(90)
	tt.fillcolor("red")
	tt.begin_fill()
	tt.circle(10,180)
	tt.circle(25,110)
	tt.left(50)
	tt.circle(60,45)
	tt.circle(20,170)
	tt.right(24)
	tt.fd(30)
	tt.left(10)
	tt.circle(30,110)
	tt.fd(20)
	tt.left(40)
	tt.circle(90,70)
	tt.circle(30,150)
	tt.right(30)
	tt.fd(15)
	tt.circle(80,90)
	tt.left(15)
	tt.fd(45)
	tt.right(165)
	tt.fd(20)
	tt.left(155)
	tt.circle(150,80)
	tt.left(50)
	tt.circle(150,90)
	tt.end_fill()
	tt.left(150)
	tt.circle(-90,70)
	tt.left(20)
	tt.circle(75,105)
	tt.setheading(60)
	tt.circle(80,98)
	tt.circle(-90,40)
	tt.left(180)
	tt.circle(90,40)
	tt.circle(-80,98)
	tt.setheading(-83)
	tt.fd(30)
	tt.left(90)
	tt.fd(25)
	tt.left(45)
	tt.fillcolor("green")
	tt.begin_fill()
	tt.circle(-80,90)
	tt.right(90)
	tt.circle(-80,90)
	tt.end_fill()
	tt.right(135)
	tt.fd(60)
	tt.left(180)
	tt.fd(85)
	tt.left(90)
	tt.fd(80)
	tt.right(90)
	tt.right(45)
	tt.fillcolor("green")
	tt.begin_fill()
	tt.circle(80,90)
	tt.left(90)
	tt.circle(80,90)
	tt.end_fill()
	tt.left(135)
	tt.fd(60)
	tt.left(180)
	tt.fd(60)
	tt.right(90)
	tt.circle(200,60)
	tt.ht()

	time.sleep(1)


rose()
turtle.done()