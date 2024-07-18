import turtle
x = 0
y = 0
a = (x,y)
e = 400
r = 400
turtle.speed(50)
turtle.penup()
turtle.goto(0,0)
turtle.pendown()
for i in range(21):
    y += 20
    turtle.forward(e)
    turtle.penup()
    turtle.goto(x,y)
    if x == 0:
        turtle.pendown()
turtle.penup()
turtle.goto(0,0)
turtle.left(90)
turtle.pendown()
for j in range(21):
    x += 20
    turtle.forward(r)
    turtle.penup()
    turtle.goto(x,0)
    if r == 400:
        turtle.pendown()

#turtle.goto(0,0)
'''turtle.penup()
turtle.goto(0,200)
turtle.pendown()
turtle.goto(200,200)
turtle.penup()
turtle.goto(200,0)
turtle.pendown()
turtle.goto(200,200)'''

while True:
    turtle.width(4)
    turtle.color('red')
    c = eval(input('红方落子：'))
    turtle.penup()
    turtle.goto(c)
    turtle.pendown()
    turtle.circle(4)
    for i in range(1):
        turtle.color('green')
        c = eval(input('绿方落子：'))
        turtle.penup()
        turtle.goto(c)
        turtle.pendown()
        turtle.circle(4)

turtle.done()
