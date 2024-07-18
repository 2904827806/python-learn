import json

class Mss:
    def __init__(self):
        with open('zhmm.json', mode='r', encoding='utf-8') as f:
            text = f.read() #读取数据
        self.users = json.loads(text)#获取数据，将数据转化为Python格式
        #print(self.users)
    #登陆页面按钮设置
    def jc(self,username,password): #形参：账户名和密码
        for user in self.users:
            if username == user['user']: #如果用户名在数据库中
                if password == user['password']: #如果密码在数据库中
                    return True,'登陆成功'
                else:
                    return False,'密码错误'
            else:
                return False, '登陆失败·，账号不存在'

    #查询页面的数据
    def information(self):
        with open('xssj.json','r',encoding='utf-8') as f:
            test1 = f.read()
        self.xx = json.loads(test1)
        return self.xx  #获取数据

    #查询页面数据更新
    def insert(self,x): #往json中添加数据
        with open('xssj.json', 'r', encoding='utf-8') as f:
            self.crea = json.loads(f.read())
        self.crea.append(x)     #数据库中加入数据
        with open('xssj.json', 'w', encoding='utf-8') as f:
            json.dump(self.crea, f) #重新更新数据
    #删除数据
    def shanchu(self,x):
        with open('xssj.json','r',encoding='utf-8') as f:
            self.name = json.loads(f.read())
        #print(self.name)
        for student in self.name:
            if x == student['name']:
                #print(student['name'])
                self.name.remove(student)
                with open('xssj.json', 'w', encoding='utf-8') as f:
                    json.dump(self.name, f)
                return True, f'{x}数据已经删除'
            else:
                continue
        else:
            return False, f'{x}数据不存在'

    #修改页面的查询按钮设置
    def geng_rew(self,name):
        with open('xssj.json','r',encoding='utf-8') as f:
            self.studeent_information = json.loads(f.read())
        for student in self.studeent_information:
            #print(student)
            if name == student['name']:
                return True,student["math"],student["chinese"],student["english"]
    #修改页面的修改按钮设置
    def geng_xg(self,name,messes):
        with open('xssj.json', 'r', encoding='utf-8') as f:
            self.studeent = json.loads(f.read())
        for student in self.studeent:
            if name == student['name']:
                self.studeent.remove(student)
                self.studeent.append(messes)
                with open('xssj.json', 'w', encoding='utf-8') as f:
                    json.dump(self.studeent, f)
                return True,'数据更新成功'
        else:
            with open('xssj.json', 'w', encoding='utf-8') as f:
                json.dump(self.studeent, f)
            return False, '数据已经存在'

    def zc_newpeople(self,user,xx):
        for abc in self.users:
            #print(abc)
            if user == abc['user']:
                return False,'用户已存在'
            else:
                self.users.append(xx)
                with open('zhmm.json', mode='w', encoding='utf-8') as f:
                    json.dump(self.users, f)
                return True, '注册成功'







































db = Mss()
if __name__ == '__main__':
    a = {"user": "123456789", "password": "52013"}
    print(db.zc_newpeople("123456789", a))







"""class Msss:
    def __init__(self):
        with open('zhmm.json','r',encoding='utf-8') as f:
            text = f.read()
            print(text)
        self.users = json.loads(text)

    def jsx(self,username,password):
        for use in self.users:
            #print(use['user'])
            if username == use['user']:
                if password == use['password']:
                    return True,'登陆成功'
                else:
                    return False ,'密码错误'
        else:
            return False ,'登陆失败，账号不存在'
            #print(use)"""





