'''import io
s = "hello,sxt"
sio = io.StringIO(s)

a = sio.seek(7)

b = sio.write("g")

c = sio.getvalue()
print(c)'''
a = 25
print(a)


def s():
    global a
    a += 25
    print(a)
s()

print(a)
