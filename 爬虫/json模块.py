#json格式和Python格式相互转换
"""
import json #直接导入
dic = {'name':'后悔'}
print(type(dic))

#python数据转换为json数据
dic2 = json.dumps(dic,ensure_ascii=False)
print(dic2)
print(type(dic2))
#json数据转换为Python数据
dic3 = json.loads(dic2)
print(type(dic3))
print(dic3)
"""
"""
import json
from jsonpath import jsonpath
#re = jsonpath(a,"jsonpath"语法规则字符串")
import requests
if __name__ == '__main__':
    url = 'https://www.lagou.com/lbs/getAllCitySearchLabels.json'

    # 头文件header
    header = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
    }
    # 代理ip

    # 发送请求
    res = requests.get(url=url, headers=header)
    # 获取响应
    html = res.text
    # 打印响应

    #把json格式转换为pyton
    res2 = json.loads(html)

    #获取所有的name\code
    city_name = jsonpath(res2,'$..name')
    city_code = jsonpath(res2,'$..code')
    for i in range(len(city_name)):
        print('{0}:{1}:{2}'.format(i+1,city_name[i],city_code[i]))

    #获取节点
    city_a = jsonpath(res2,'$..A')
    print(city_a)"""

# 提取豆瓣单页数据
import json
from jsonpath import jsonpath
import requests

url = 'https://m.douban.com/rexxar/api/v2/movie/recommend?refresh=0&start=0&count=20&selected_categories=%7B%22%E7%B1%BB%E5%9E%8B%22:%22%E5%96%9C%E5%89%A7%22%7D&uncollect=false&tags=%E5%96%9C%E5%89%A7&ck=AVT9'
b = 'https://m.douban.com/rexxar/api/v2/movie/recommend?refresh=0&start=20&count=20&selected_categories=%7B%22%E7%B1%BB%E5%9E%8B%22:%22%E5%96%9C%E5%89%A7%22%7D&uncollect=false&tags=%E5%96%9C%E5%89%A7&ck=AVT9'
# 获取头信息
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"}
# 发送请求
"""res = requests.get(url=url, headers=header)

if __name__ == '__main__':
    import requests
    import json
    import re
    import warnings

    warnings.filterwarnings("ignore", category=Warning)  # 关闭弃用报错

    url = 'https://fanyi.baidu.com/v2/transapi'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"}
    # 构建data参数字典
    key = input('输入你要翻译的内容：')
    post_data = {
        'from': 'zh',
        'to': 'en',
        'query': key,
        'transtype': 'translang',
        'simple_means_flag': '3',
        'token': '506d1e9ab95c7f4eb153b61cc96a8dc5'

    }
    # 发送请求
    respons = requests.post(url, headers=header, data=post_data)
    #respons.encoding = 'utf-8'
    # 获取响应
    htlm = respons.content
    respons.encoding = 'utf-8'
    print(respons.status_code)  # 获取状态码
    print(re.search("[\\u4e00-\\u9fa5]+", respons.content.decode('unicode_escape'), flags=re.S)[0])  # 正则表达式查找汉字
"""
    # 解析数据
    # 将json数据转换为python字典



"""global b,d
    import requests
    import json
    import re
    import warnings

    warnings.filterwarnings("ignore", category=Warning)  # 关闭弃用报错

    url = 'https://fanyi.baidu.com/v2/transapi'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"}
    # 构建data参数字典
    key = b.get()
    post_data = {
        'from': 'zh',
        'to': 'en',
        'query': '中国',
        'transtype': 'translang',
        'simple_means_flag': '3',
        'token': '506d1e9ab95c7f4eb153b61cc96a8dc5'

    }
    # 发送请求
    respons = requests.post(url, headers=header, data=post_data)
    #respons.encoding = 'utf-8'
    # 获取响应
    htlm = respons.content
    respons.encoding = 'utf-8'
    #print(respons.status_code)  # 获取状态码
    print(re.search("[\\u4e00-\\u9fa5]+", respons.content.decode('unicode_escape'), flags=re.S)[0])  # 正则表达式查找汉字"""











"""import tkinter as tk

def trans(x):
    import requests
    import json
    url = 'https://ifanyi.iciba.com/index.php?c=trans'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"}
    # 构建data参数字典

    post_data = {
        'from': 'zh',
        'to': 'en',
        'q': str(x)
    }
    # 发送请求
    respons = requests.post(url, headers=header, data=post_data)
    # 获取响应
    htlm = respons.text
    #print(htlm)
    fyd = json.loads(htlm)
    return fyd['out']



def fy():
    date = b.get().strip()
    date1 = trans(x=date)
    d.delete(1.0,tk.END)
    d.insert(1.0,date1)
root = tk.Tk()
root.geometry('550x300')
root.resizable(0,0)
root.title('翻译')

tk.Label(root,text='输入：',font=20).grid(row=1,column=0,sticky=tk.W)
tet = tk.StringVar()
b = tk.Entry(root,font=24,width=35,textvariable=tet)
b.grid(row=1,column=1,sticky=tk.W)
c = tk.Button(root,text='翻译',font=12,pady=5,padx=5,command=fy)
c.grid(row=1,column=3,sticky=tk.W)

gr = tk.StringVar()
d = tk.Text(root)
d.grid(row=2,columnspan=4)

root.mainloop()
"""




