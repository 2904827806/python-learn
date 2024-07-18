print('面包店正在打折，活动进行中......')
strweek = input('请输入中文星期（如星期一）:')
intTime = int(input('请输入时间中的小时（范围0--23):'))
if (strweek == "星期二" and (intTim >= 19 and intTim <=20)) or (strweek == "星期六" and  (intTim >= 17 and intTim <=18)):
    print("恭喜")
else:
    print('遗憾')
                                                                
