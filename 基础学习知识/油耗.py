'''for i in range(1000000000):
    a = float(input("加油总金额（元）："))
    b = float(input("公里数："))
    c = a/(6.27*b)
    print("汽车的真实油耗为："+str(c)+"升/公里")'''
def yh(s,m):
    c = s/(6.27*m)
    print("汽车的真实油耗为："+str(c)+"升/公里")
yh(25,7)

a = (1,2,3,4,5,6,7,8,9,10,11,12)
for i in range(100**100):
    b = int(input(''))
    if 0<b<=12:
        print(a[b-1])
    else:
        print('sb')