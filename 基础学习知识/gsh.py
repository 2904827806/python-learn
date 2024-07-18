def js(s):
    global a
    a.append(s)
    b = sum(a)
    yf = b
    if 500 <= b < 1000:
        yf = '{:.2f}'.format(b*0.9)
    elif 1000 <= b < 2000:
        yf = '{:.2f}'.format(b*0.8)
    elif 2000 <= b < 3000:
        yf = '{:.2f}'.format(b*0.7)
    elif 3000 <= b:
        yf = '{:.2f}'.format(b*0.6)
    return b, yf
a = list()
while True:
    s = float(input("请输入金额："))
    if s == 0:
        break
    else:
        c = js(s)
print('总金额{0},应付金额{1}'.format(c[0],c[1]))


