for i in range(360):
    a = float(input("年份为："))
    b = a%4
    c = a%100
    d = a%400
    if int(b) == 0 and (int(c) !=0) or (int(d) ==0):
        print("闰年")
    else:
        print("不是闰年")
