#Turtle graphics
import turtle

#create window
loadWindow = turtle.Screen()
turtle.colormode(255)
#turn off draw mode
turtle.speed(100)

square = 5
for i in range(100):
    turtle.square(5*i)
    turtle.square(-5*i)
    turtle.left(i)
    turtle.color(2*i, 2*i, 2*i)

#wait for user to end
turtle.exitonclick()
