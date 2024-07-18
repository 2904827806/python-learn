'''class TVshow:   # 定义电视节目类
    lsi = ['战狼2','红海行动','西游记女儿国','熊出没变形计']

    def __init__(self,show):
        self._show = show

    @property
    def show(self):
        return self._show            #返回私有值

    @show.setter                      #转换为让属性可以修改
    def show(self,value):
        if value in TVshow.lsi:
            self._show = '你选择了'+ '《'+ value + '》，稍后将播放'
        else:
            self._show = '你选择的电影不存在'


tv = TVshow('正在播放，满江红')            #创建类的实列
print(tv.show)                        #获取属性值
tv.show = ('红海行动')
print(tv.show)'''

