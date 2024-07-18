#导入语音库
import pyttsx3
#初始化对象
talker = pyttsx3.init()
#创建一个用户名和密码
userName = '鹏宇'
userPass = '202311'
print('欢迎使用AI聊天机器人：')
#把演讲稿给talker对象
talker.say('欢迎使用AI聊天机器人：')
#启动
talker.runAndWait()
#input 输入函数，提示用户输入账号密码
talker.say('请输入你的账号和密码')
talker.runAndWait()
inputName = input('请输入你的账号：')
inputPass = input('请输入你的密码：')
print(inputName,inputPass)

#判断输入的密码和账号是否一致
if inputName ==userName and inputPass == userPass:
    print('登陆成功')
    talker.say('恭喜你登陆成功'+userName)
    talker.say('想和我聊点什么吗？')
    talker.runAndWait()
#循环
    while True:
        # 接受用户输入问题
        question = input(inputName + '请输入你的问题：')

        # 针对问题作出简答.strip（）函数：对字符串进行处理
        answer = question.strip('吗？？' + '!')
        if question == '不聊了拜拜':
            talker.say('好的，拜拜就拜拜！')
            talker.runAndWait()
            print('机器人回答：好的，拜拜就拜拜！')
            break
        print('机器人的回答：' + answer)
        talker.say(answer)
        talker.runAndWait()

else:
    print('登陆失败！')
    talker.say('账号或密码错误，很遗憾登陆失败')
    talker.runAndWait()
