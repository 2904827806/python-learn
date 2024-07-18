#设计循环
for i in range(360):
#输入身高体重
    height = float(input("请输入你的身高(m):"))
    weight = float(input("请输入你的体重(kg):"))
#计算BIM指数
    bim = weight/(height**2)
    print("你的BIM指数是:"+str(bim))

#判断你的身材是否合理
    if bim < 18.5:
        print("您的体重过轻")
    if bim >=18.5 and bim < 24.9:
        print("正常范围，请注意保持")
    if bim >=24.9 and bim < 29.9:
        print("您的体重过重")
    if bim >= 29.9:
        print("肥胖")
    
