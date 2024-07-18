import turtle

for x in range(36):
    for c in ('blue', 'red', 'green'):
        turtle.color(c)
        turtle.forward(x)
        turtle.left(59)

turtle.penup()
turtle.goto(0,0)
turtle.pendown()
turtle.width(10)
turtle.left(25)
turtle.color("yellow")
turtle.forward(100)
turtle.penup()
turtle.goto(-35,0)
turtle.pendown()
turtle.width(3)
turtle.circle(36)
