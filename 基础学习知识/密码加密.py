a = 5201314
print("原密码："+str(a))


for i in range(360):
    c = int(input("加密码："))

   
    
    b = a << c #定义a,c为整数才能进行移位
    
    f = hex(b)
    
    print("加密密码"+str(f))
   
