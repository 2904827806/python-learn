import requests
from openpyxl import Workbook
from lxml import etree
wb = Workbook()
sh = wb.create_sheet('数据',0)
del wb['Sheet']
a = []
for i in range(0,25):
    url = r'https://movie.douban.com/top250?start=' + str(f'{i*25}&filter=')
    #print(url)
    header = {
        'Cookie': 'll="118350"; bid=EcQxwZCEAAk; _pk_id.100001.4cf6=40dc9b188a39e9c3.1687090946.; __yadk_uid=kQXZsiii9quUJW5VYAkgQwPvt87M8sLV; _vwo_uuid_v2=D22F7663ECE419685B0F7342B336FC4B4|8fdb600eddb5b2ca426843350ff987cb; douban-fav-remind=1; _ga=GA1.1.1049362381.1687090947; _ga_RXNMP372GL=GS1.1.1706428395.1.0.1706428400.55.0.0; __utmv=30149280.27782; viewed="31544727_26655959"; dbcl2="277824043:tUeqQ5gvSqY"; push_noty_num=0; push_doumail_num=0; ck=Llyk; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1712505884%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DCdj-Q1UbKd3YS0tX3_MolcE14Iou_uzjwYoZAegKYfa0kvMm4Z8yccNUgz1nEdei%26wd%3D%26eqid%3Dbe742405007a0cc0000000066612c414%22%5D; _pk_ses.100001.4cf6=1; __utma=30149280.1049362381.1687090947.1711908787.1712505884.23; __utmb=30149280.0.10.1712505884; __utmc=30149280; __utmz=30149280.1712505884.23.15.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.39358061.1687090947.1711908787.1712505884.23; __utmb=223695111.0.10.1712505884; __utmc=223695111; __utmz=223695111.1712505884.23.14.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; ap_v=0,6.0; __gads=ID=81664f961e26089b-2293da42afe100d8:T=1687090947:RT=1712505886:S=ALNI_MYWaaB0g0Zgm203GYkyW4Roh6bvWQ; __gpi=UID=00000c511d7f531b:T=1687090947:RT=1712505886:S=ALNI_Mb81GTCy6fJSU0IjazWw_FOMoSF5w; __eoi=ID=a02e201de769e0e0:T=1711908788:RT=1712505886:S=AA-AfjaJu0_YVmI03jz34-Pco4_x',
        'Referer': url,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'
    }
    re = requests.get(url=url, headers=header)
    html = etree.HTML(re.text)
    titles = html.xpath('//*[@id="content"]/div/div[1]/ol/li[*]/div/div[2]/div[1]/a/span[1]/text()')
    pfs = html.xpath('//*[@id="content"]/div/div[1]/ol/li[*]/div/div[2]/div[2]/div/span[2]/text()')
    pjrss = html.xpath('//*[@id="content"]/div/div[1]/ol/li[*]/div/div[2]/div[2]/div/span[4]/text()')
    if titles != []:
        for j in range(len(titles)):
            l = []
            title = titles[j]
            pf = float(pfs[j])
            pjrs = int(pjrss[j][:-3]) /100000
            pj =round(pjrs,2)
            l.append(title)
            l.append(pf)
            l.append(pj)
            if pf >= 9.0 or pjrs >= 9.00:
                c = 1
                l.append(c)
            else:
                c = 0
                l.append(c)
            #print(l)
            if l != []:
                a.append(l)
    else:
        break
sh.append(['电影','评分','评价人数','是否选择'])
for k in a:
    sh.append(k)
wb.save(r"C:\Users\29048\Desktop\逻辑回归1.xls")
