for i in range(100000):
    color = input("请输入需要查询的颜色:")

    match color:
        case "red"|"红"|"红色":
            r,g,b = 255,0,0
        case "green"|"绿"|"绿色":
            r,g,b = 0,255,0
        case "yellow"|"黄"|"黄色":
            r,g,b = 255,255,0
        case _:
            r,g,b = -1,-1,-1
    if r >=0:
        print(f"{color}的颜色代码:#{r:02x}{g:02x}{b:02x}")
    else:
        print(f"查询不到{color}的颜色代码！")

