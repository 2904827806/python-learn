import time

url = r"https://sh.fang.anjuke.com/?from=HomePage_TopBar"
#https://sh.fang.anjuke.com/loupan/all/p2/
#https://sh.fang.anjuke.com/loupan/all/p3/
import requests
from lxml import etree
import xlrd2
from openpyxl import Workbook
from threading import Thread
wb = Workbook()
sh = wb.create_sheet('数据', 0)
a = []
for i in range(1, 37):
    url = f'https://sh.fang.anjuke.com/loupan/all/p{i}/'
    header = {
            'Referer': url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
            'Cookie': 'aQQ_ajkguid=4020D1E4-923D-4C2A-8A51-E9D9D1D8FE8D; sessid=2A45307B-85A2-4295-9CD4-6F355E274D45; id58=CrIezWVt99yaExMfECFhAg==; isp=true; wmda_uuid=79a5c412ad370f3d7697a6c555c68b95; wmda_new_uuid=1; wmda_visited_projects=%3B8788302075828; 58tj_uuid=7c318164-5684-4aad-80cb-e60aaf9faf29; als=0; xxzlclientid=56dacaa2-7cb7-4ead-8a20-1706102771794; xxzlxxid=pfmxmLw+UYAqpgAKklzeCI7bsf7t7OyEnjshxJK/yaTgr8b9svaKN8jI/1klT3F76Abt; twe=2; ajk-appVersion=; seo_source_type=1; fzq_h=3d241b0961b97dda40b0c15a90646be8_1711892793230_33a31eea7abc437e8f0bde81e2115eb9_3084929099; isp=true; xxzlcid=b8fa7ae8f096445686922d371aa85ab0; xxzl-cid=b8fa7ae8f096445686922d371aa85ab0; obtain_by=2; ctid=11; ved_loupans=509324%3A444720%3A525369%3A413237%3A519276; lp_lt_ut=235ff800df1cb9656f26f2e3df493100; wmda_session_id_8788302075828=1711935256041-092e6857-da98-632b; init_refer=; new_uv=7; ajk_member_verify=5eEsukXROo93x58mdK9N8XcUh1dNClw8bhEv5DE1pJs%3D; ajk_member_verify2=Mjg1NTQxMTI3fHJsZlZwd1F8MQ%3D%3D; ajk_member_id=285541127; new_session=0; xxzl_cid=3abfabf1ea914fdebdbe527ff33c8f3e; xxzl_deviceid=hF5+LKMbYw7cRZJX+uJb5CjHH/o1ovC+TWycZrRAKSUUbssW7DsnM0A3iEHTI9Nz; xxzlbbid=pfmbM3wxMDMyMnwxLjkuMXwxNzExOTM1NDI3NTYxMzEyNTE1fCtHRkNmV2xUbU5VcEErMXRCTXEvcUF1Vjh2YXprWUhlNHB4Nkl3RFhQcnM9fDU1MjgzZDI1OTBmNGRmNTA5YWFjMzRkZmE2NDY2NWY2XzE3MTE5MzU0MjY0NDFfMDQzYzUwNWNlMjI4NDM0NjkyNGM1MDQ0MDlkOTI3MjFfMzA4NDkyOTA5OXw5YWRjODdhNzM5MjY4ZDIyYzBiNDU1ZmVmOTQxOTE5N18xNzExOTM1NDI3MDg1XzI1NQ==; ajkAuthTicket=TT=483693379766b3cfa18c29641063a688&TS=1711935432161&PBODY=gjrR4d5yQZxebKJKxPt-85HUvswECPnQt1ENUWEd9bvc-bNJrrwmeZmOWLo4JTclGdtqAhUu2tXYdO_dpwwfRt4flVdqELZ7F1B7A5-ACSCi16iRxaokJg3I6HCn2ESQcGlH4ykObb1d7UkXAH_2wwSH7lSsY4aH7Z_hKDD76Fk&VER=2&CUID=YKnCIR4vfaSwYUOHpeViEaS2r3T3qS61'    }
    reas = requests.get(url=url, headers=header)
    html = etree.HTML(reas.text)
    name = html.xpath('//*[@id="container"]/div[2]/div[1]/div[3]/div[*]/a[1]/@href')

    for i in name:
        b = []
        url1 = i
        # print(url)
        header1 = {
                'Referer': url,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
                'Cookie': 'isp=true; SECKEY_ABVK=eZYngQH4fp6sIEtIizLAUZg4M0oNMwgO4VQZD4axRLU%3D; BMAP_SECKEY=cRFSpvHPH1Lz9FviLe5xstD51IJpZzCE_WHum_mhCeaF4LkcdYfnhI2X1e5BUcd2ch2oC7oxwB6-uZE-ursWaFgRkeXrFsSMkO2YPvb8SXo19XF25hv3qE1CSnXwLW7rsD6CMsqFnUA-sJmxhLGs6G1DGrB1MaRPiY4E-xT5pxyGm9C004JZUlojcTwXnlym; aQQ_ajkguid=4020D1E4-923D-4C2A-8A51-E9D9D1D8FE8D; sessid=2A45307B-85A2-4295-9CD4-6F355E274D45; id58=CrIezWVt99yaExMfECFhAg==; ctid=11; isp=true; wmda_uuid=79a5c412ad370f3d7697a6c555c68b95; wmda_new_uuid=1; wmda_visited_projects=%3B8788302075828; 58tj_uuid=7c318164-5684-4aad-80cb-e60aaf9faf29; als=0; xxzlclientid=56dacaa2-7cb7-4ead-8a20-1706102771794; xxzlxxid=pfmxmLw+UYAqpgAKklzeCI7bsf7t7OyEnjshxJK/yaTgr8b9svaKN8jI/1klT3F76Abt; obtain_by=2; twe=2; ajk-appVersion=; seo_source_type=1; fzq_h=3d241b0961b97dda40b0c15a90646be8_1711892793230_33a31eea7abc437e8f0bde81e2115eb9_3084929099; isp=true; wmda_session_id_8788302075828=1711892800356-42968c59-96f2-30c8; init_refer=https%253A%252F%252Fshanghai.anjuke.com%252F; new_uv=3; new_session=0; xxzlcid=b8fa7ae8f096445686922d371aa85ab0; xxzl-cid=b8fa7ae8f096445686922d371aa85ab0; lp_lt_ut=ae289874d7dc623684cf17c0e4940f29; ved_loupans=509324%3A444720; xxzl_cid=3abfabf1ea914fdebdbe527ff33c8f3e; xxzl_deviceid=hF5+LKMbYw7cRZJX+uJb5CjHH/o1ovC+TWycZrRAKSUUbssW7DsnM0A3iEHTI9Nz; xxzlbbid=pfmbM3wxMDMyMnwxLjkuMXwxNzExODkzNjc1NjMzMTc4NDk5fGlTVTVlSktmaU9rM2tsbjdmdndvZHk2dHZBSlhGYlhLWmtmLy90cDNERk09fGJiN2Q1Y2FmODU1MDYxY2ExMjdjYjljMDZjMDMyNzI0XzE3MTE4OTM2NzQxMzZfZTRmNDYwZmQwNTU1NGVmMTljYWExNTkxNTRjYjRlMGVfMzA4NDkyOTA5OXwwMjYzZGZlNGQ1NTU0ZTE2MzdjMjM1ODU0ZmZiMzkwN18xNzExODkzNjc0NjEyXzI1NQ=='
        }
        requests1 = requests.get(url=url1, headers=header1)
        html1 = etree.HTML(requests1.text)
        name = html1.xpath('//*[@id="container"]/div[1]/div[2]/div[1]/div/div/div[1]/h1/text()')
        meny = html1.xpath('//*[@id="container"]/div[1]/div[2]/div[1]/dl/dd[1]/p/em/text()')

        # print(name)
        fx1 = html1.xpath('//*[@id="container"]/div[1]/div[2]/div[1]/dl/dd[4]/div/a[1]/text()')
        fx2 = html1.xpath('//*[@id="container"]/div[1]/div[2]/div[1]/dl/dd[4]/div/a[2]/text()')
        fx = fx1 + fx2
        b.append(len(a))
        if name != []:
            b.append(name[0])
            if meny != []:
                menys = str(meny[0]) + '元/平方米'
                b.append(menys)
                if fx != []:
                    b.append(fx[0])
        else:
            b.append('')
        print(b)
        sh.append(b)
        a.append(b)
wb.save('数据205.xlsx')
