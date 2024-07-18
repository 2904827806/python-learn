print("请设计你的人生")
for i in range(1000000000):
    a = int(input("请输入你的年龄（岁）："))
    if a <=18:
        print("你的年龄尚小，请努力学习")
    elif 18 < a <=30:
        print("你的人生正处于努力奋斗的黄金阶段")
    elif 30 < a <=50:
        print("你现在的人生正处于黄金阶段")
    else:
        print("最美不过夕阳红")
