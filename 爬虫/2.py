# ui
import tkinter
import tkinter.font
# 爬虫
import requests
import re

# 其他
import random
import time
from tkinter import messagebox

"""class Logic_BaiduSpider(object):
    def __init__(self, input_data):
        # 翻译结果存在的url
        self.form_url = 'https://fanyi.baidu.com/v2transapi?from=en&to=zh'
        #user池
        user_agents = [
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko)',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188']
        self.header = {
            'cookie': '__yjs_duid=1_674911807aeb99a64bd42a9c8b981be81635202045938; BDUSS=jVOUVJ5UzVHYm5RLTZOcXQwMTlnYUh6R2xkOXpuNnVGSDZ5cVJDOHJsamJJNkpoRVFBQUFBJCQAAAAAAAAAAAEAAAAxky3Otqu3vbrsvrTAz9S6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANuWemHblnphU; BDUSS_BFESS=jVOUVJ5UzVHYm5RLTZOcXQwMTlnYUh6R2xkOXpuNnVGSDZ5cVJDOHJsamJJNkpoRVFBQUFBJCQAAAAAAAAAAAEAAAAxky3Otqu3vbrsvrTAz9S6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANuWemHblnphU; BIDUPSID=20A7A480A559295EE6B79E9177E7CB49; PSTM=1636242361; REALTIME_TRANS_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; FANYI_WORD_SWITCH=1; SOUND_PREFER_SWITCH=1; ZFY=1:A:AGMUxBbGCY3losh7kUruzIglNWlYdh1Xp2l07QsbU:C; __bid_n=184357c5337c82193f4207; BAIDU_WISE_UID=wapp_1678495157915_916; FPTOKEN=AprdIvLEU9bU4XXMVm2db8fvIxpW/caHG5gIQcOmzzSKJ2tkUQOhukAMEaXTSw8i1pIsmbcHSXd+KuM6Fvln3K6S1ulFUeneuAd2BOW49c0e/ovKelxFgmeZfzUK90OIQlcuCtSO2HgEVNsbq+0kn9Ip0aUswiejuNsMHjaMTNnb93IuDF6APFdYbOCIbxKkDLWGjf3XcpWhpnAsDK3+j5eb+MUuhFeQk4TMrUfoaLReaTyDkcLefpCFpOUc+/G698uU6/tRq9aG/noenrxIubFfwj1xCT3qXoyfPrI8TTpF/uWelM6uchrSjp5hPd3SdsKcjcz4m04cCdBoPhbA15m/pXArbqf8fhMZAw1PRmaH7ZIm/fvNzFTUwSyQW9rleFsvv5skEugEB4AfsDVdww==|/ZKheBQe/vVpFZKqAJ0btWcOxxJw/SXD/xQ0UtxXePs=|10|4c88b6a1deb83d4e7a67b690c46f7f76; BAIDUID=F8A3226E1FE8708108AE17228963E1E5:FG=1; BAIDUID_BFESS=F8A3226E1FE8708108AE17228963E1E5:FG=1; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1691377121,1691457865,1691481204; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1691481213; ab_sr=1.0.1_MTdhOTRlN2Y4ZGQzYWE3ZDUxNjliMjA5OTBkYmMyNWQwODQ4ODk1MmY0ZjIzN2M1ODJhZWNkMTYyYWY5NTdhNmJmYzcxNTVhMzdkZGE4YzUzMzgxNzA2MDk4NDNjZDQ5YThlMTkxYmQyYjFmNmNkYTU1M2NlYzkyZTQwOWJmZjdlMWQzNGE4OTUwMmY3MzcxNGNjOTJkMmM0ZGEzYjczMjFiYzE1OTk0NWZlZmQxYmExNmI1YmI5MGMxNmUwZjg1',
            'user-agent': random.choice(user_agents),
            'Referer': 'https://fanyi.baidu.com/',
            'Host': 'fanyi.baidu.com',
            'Sec-Ch-Ua': '"Not/A)Brand";v="99", "Microsoft Edge";v="115", "Chromium";v="115"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Acs-Token': '1691481214316_1691481256005_/DBI7uAea2WMqgnHYTjOamrbCNblUgG8hE7zhBGZWrYBBSksm7aPbhvjrehrIgt0EhBIDDM9SCxqnM11X4Ik4g4Ha1Etk/DvZmp897QRXb0XVD2jAClg2VlrkKLoFysvXtRsEcbicLwgGKdyOcKzIFf6EXnEV5zjNXtUmQ+BUvV9WyRUS7c5Zdq0XwSl6QN6vIfeJchPqxWhj7GrUcE7xKlDZJeULVtoax4SipcmMYRcLGl3QfIce9SUxtuAptlr0wvPp/4/K7KecJUL2AXnIjKlji4G/9SkJorsOGd0HR3CzT1394o9OP+dUeDzWEUDvgob2xCmJjJjYSjwIP+vwm2Bvj83wIdF4Low+yTTc/snIOhKoUmbkF3bdZ5cm9ht+duquTm2dLZpdAiUWaj+FzIh20eC2nKYcKb4scH1+QhL7g8WtJTjKoq8LmCknVECYVmV5xaL7Qbdt/LmlGL6/lvy/j658dQO5cU6vMfHGmvgoQge2u9Meuc59kHwxWPAXocKoaYSE92RgOHH6sTwoA==',
            'Cache-Control': 'no-cache',
        }
        self.proxies = {

        }
        self.input_data = input_data

    def get_WindowGtk_and_Token(self):
        # 获取token和window_gtk用的url
        uri = 'https://fanyi.baidu.com/translate?aldtype=16047&query=&keyfrom=baidu&smartresult=dict&lang=auto2zh&ext_channel=Aldtype'
        headers2 = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0',
            'Referer': 'https://www.baidu.com/',
            'Cookie': 'BIDUPSID=D42C189A7B0D4D15A73A11D5A30D255A; PSTM=1686744192; BAIDUID=067D49DEFA72D908A5F46D368B157E07:FG=1; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BDUSS=hBazNvdGk4R2lDeENkUHhUN01qSTc4ejR5MjN1MWtaQms1RHdXMlRifmREYlZsRVFBQUFBJCQAAAAAAAAAAAEAAAAj-9Zc0-7AxzAwNwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN2AjWXdgI1ldW; BDUSS_BFESS=hBazNvdGk4R2lDeENkUHhUN01qSTc4ejR5MjN1MWtaQms1RHdXMlRifmREYlZsRVFBQUFBJCQAAAAAAAAAAAEAAAAj-9Zc0-7AxzAwNwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN2AjWXdgI1ldW; MCITY=-336%3A; APPGUIDE_10_6_9=1; H_WISE_SIDS_BFESS=39997_40124_40156_40171_40200_39661_40210_40207_40216_40223; H_PS_PSSID=40171_39661_40210_40207_40216_40223_40295_40291_40288_40317_40079; H_WISE_SIDS=40171_39661_40210_40207_40216_40223_40295_40291_40288_40317_40079; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; BA_HECTOR=0g8k8g8005008h0g8h050g0gdjl6sa1iu3agd1t; BAIDUID_BFESS=067D49DEFA72D908A5F46D368B157E07:FG=1; delPer=0; PSINO=2; ZFY=LAqK8PXy2H74ozXiAdP:AGnmMUdFWxIoxuU:BStZ:AlTCM:C; BDRCVFR[VO33ettmAk_]=mk3SLVN4HKm; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1709290813; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1709290830'
        }
        res2 = requests.get(url=uri, headers=headers2)
        # print(res2.content.decode())
        gtk = re.findall(r';window.gtk = "(.*?)";', res2.content.decode())
        # get_vd = re.findall(r'"id":64,"baseUrl":"(.*?)","base_url"', res2.content.decode())
        # html2 = json.loads(res2.text)
        # gtk = jsonpath(html2,'$..window.gtk')
        return gtk

    # 调用js文件
    def get_Sign(self):
        with open(r'E:\Pythonproject\爬虫\baidu-trans.js', 'r', encoding="utf-8") as fp:
            js_code = fp.read()
        js_obj = execjs.compile(js_code)

        gtk = self.get_WindowGtk_and_Token()
        res = 'getSign("{0}", "{1}")'.format(self.input_data, gtk)
        sign = js_obj.eval(res)
        return sign

    #翻译
    def translate(self):
        gtk = self.get_WindowGtk_and_Token()
        print(gtk)
        data = {
            'from': 'zh',
            'to': 'en',
            'query': self.input_data,
            'transtype': 'realtime',
            'simple_means_flag': '3',
            'sign': self.get_Sign(),
            'token': '506d1e9ab95c7f4eb153b61cc96a8dc5',
            'domain': 'common',
            'ts': int(time.time()*1000)

        }

        html_json = requests.post(url=self.form_url, headers=self.header, data=data, proxies=self.proxies).json()
        print(html_json)
        if ' ' in self.input_data:
            res_singleMean = html_json['trans_result']['data'][0]['dst']  # 这是单个翻译结果
            return res_singleMean, 'sentence'
        else:
            res_list = html_json['dict_result']['simple_means']['word_means']
            return res_list, 'word'"""

class Logic_BaiduSpider:
    def __init__(self,input_data):
        self.input_data = input_data
    def  translate(self):
        if __name__ == '__main__':
            import requests
            import json
            url = 'https://ifanyi.iciba.com/index.php?c=trans'
            header = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"}
            # 构建data参数字典
            key = self.input_data
            post_data = {
                'from': 'zh',
                'to': 'en',
                'q': key
            }
            # 发送请求
            respons = requests.post(url, headers=header, data=post_data)
            # 获取响应
            htlm = respons.text
            # print(htlm)
            # 解析数据
            # 将json数据转换为python字典
            dict = json.loads(htlm)
            return dict['out'], 'word'


class Ui_BaiduSpider(object):
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('线上翻译')  #窗口名称
        self.root.resizable(0, 0)  # 设置窗口大小不可变
        # root.geometry("440x400+600+300") # width * height + 水平偏移 + 垂直偏移

        for i in range(2):
            self.root.columnconfigure(i, minsize=50, pad=100)      #当调整主窗口大小时，该框架也随之改变；
            if i == 0:
                self.root.rowconfigure(i, minsize=30, pad=40)
            else:
                self.root.rowconfigure(i, minsize=30, pad=50)

        """grid组件使用行列的方法放置组件的位置，参数有：
        row:               组件所在的行起始位置；
        column:            组件所在的列起始位置；
        rowspam:           组件的行宽；
        columnspam:        组件的列宽；
        ipadx、ipady:      组件的内部间隔距离
        padx、pady:        组件的外部间隔距离
        sticky:            对齐方式，东南西北中九方向 +
        in_:               指定父插件的子插件
        """
        # label#设置标签
        self.label1 = tkinter.Label(self.root, text="输入：", font=("黑体", 24), fg="black")   #设置输入窗口
        self.label1.grid(row=0, column=0, sticky=tkinter.W + tkinter.E + tkinter.N + tkinter.S)  #执行设置输入窗口
        '''
         self.label2 = tkinter.Label(self.root, text="输出：", font=("黑体", 24), fg="black")  # 设置输入窗口
        self.label2.grid(row=0, column=0, sticky=tkinter.W + tkinter.E + tkinter.N + tkinter.S)  # 执行设置输入窗口
        '''

        # entry  #输入框
        self.input_data = tkinter.StringVar()  # 使输入框内容可变
        self.entry1 = tkinter.Entry(self.root, font=("宋体", 24),textvariable=self.input_data)   #Entry(tk,textvariable = username)单行文本框；
        # entry1.bind("<KeyRelease>", ButtonAction) # ButtonAction()绑定键盘
        self.entry1.grid(row=0, column=1, sticky=tkinter.W)      #设置输入框
        print(self.entry1.get())

        # text输出成果
        self.text1 = tkinter.Text(self.root)           #多行文本框；
        self.text1.grid(row=1, columnspan=2)             #设置文本框的位置

        # button 按钮
        self.button = tkinter.Button(self.root, text="翻译", font=("宋体", 24), command=self.ButtonAction)
        self.button.grid(row=0, column=2, pady=10, padx=10)      #设置按钮的位置

        def tc():
            if messagebox.askyesno('退出', '确定要退出吗？'):
                self.root.destroy()
        self.root.protocol('WM_DELETE_WINDOW', tc)


    def ButtonAction(self):  # 使用entry的key release时，参数要用event，即def function(event)
        data = self.entry1.get().strip()  # 获取输入框内容

        #print(data)
        if len(data) == 0:
            return

        spider = Logic_BaiduSpider(data)
        trans_results, trans_type = spider.translate()

        if trans_type == 'sentence':
            self.text1.delete(1.0, tkinter.END)
            self.text1.insert(1.0, trans_results)
        elif trans_type == 'word':
            self.text1.delete(1.0, tkinter.END)
            #for i in range(len(trans_results)):
            self.text1.insert( 1.0,trans_results + "\n")
            # text行从1开始，所以在 i + 1.0 行插入

    def run(self):
        self.root.mainloop()  # 运行窗口

    #





if __name__ == "__main__":
    ui = Ui_BaiduSpider()
    ui.run()



