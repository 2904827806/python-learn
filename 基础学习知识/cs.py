import time

time01 = time.time()
a = ""
for i in range(100):
    a += "sxt"
    print(a)
    if a == "sxt"*25:
        break
time02 = time.time()

print("运算时间："+str(time02-time01))

time03 = time.time()
li = []
for i in range(400):
    li.append("sxt")
    print(li)
    if len(li)==360:
        break
a = ''.join(li)
d = len(a)
print(a)
print(d)
time04 = time.time()

print("运算时间："+str(time04-time03))
