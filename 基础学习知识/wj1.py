'''print('\n','='*10,'蚂蚁庄园','='*10)
file = open('mess.doc','w')      #只写模式每次都会重新写入
file.write('猕猴桃兑换黄金')
print('\n即将现身\n')
print('gbq',file.closed)               #closed判断是否关闭
file.close()
print('gbh',file.closed)
with open('mess.doc','r') as file:             #不用自己关闭文件

    print(file.read()) #读取文件内容


print('\n','='*10,'蚂蚁庄园','='*10)
file = open('mess.doc','a')                  #在原文件中加字要写追加数据
file.write('猕猴桃温热同样如图一兑换黄金')
print('\n即将现身\n')
file.flush()
with open('mess.doc','r')as file:
    print(file.read())



list1 = ['yaom','sfdgf','asfdsfbdg']
with open('mess.doc','w')as file:
   file.writelines([line+'\n'for line in list1])    #在文件中加入字符串


with open('mess.doc','r')as file:
    #file.seek(2)                                     #将指针移动到指定位置
    #print(file.readlines())
    meaaaa = file.readlines()
    for i in meaaaa:
        print(i)
'''
'''import os
import shutil

b = os.name
if b == 'nt':
    print('windows')
print(os.getcwd())       #获取当前工作目录
#os.mkdir(r'C:\Users\29048\Desktop\dome')
print(os.path.exists(r'C:\Users\29048\Desktop\dome'))
file = open(r'C:\Users\29048\Desktop\dome\tix.txt','w')
#os.makedirs(r'C:\Users\29048\Desktop\dome\name\mm\gg')
shutil.rmtree(r'C:\Users\29048\Desktop\dome\name')  #删除目录及下面所有内容'''
import os
def formTime(longtime):
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(longtime))


def formtByte(number):
    for (scale,label) in [(1024*1024*1024,'GB'),(1024*1024,'MB'),(1024,'KB')]:
        if number >= scale:
            return '%.2f%s' % (number*1.0/scale,label)
        elif number == 1:
            return '1 Byte'
        else:
            byte = '%.2f' % (number or 0)

    return (byte[:-3] if byte.endswith('.00')else byte) + 'Byte'
