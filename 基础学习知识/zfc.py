import re
# 导入Python的re模块
pattern = r'([A-Za-z0-9_\-\.]+)(@)([A-Za-z0-9_\-\.]+)(.)([A-Za-z]{2,4})' #设置提取邮箱地址所需的模式字符串
b = '''2904827806@qq.com明日科技：mingrisoft@mingrisoft.com 琦琦：84978981@qq.com 无语：12345689@163.com 可可：987654321@192.168.1.66.com邮'''          #要进行匹配的字符串
match = re.findall(pattern, b)
for i in match:        # 得到的i属于元组
    if i != '':
        for j in i:                                         #需要从i 中获取主要的数据
            print(j,end='')
    print()
