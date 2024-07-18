import os
import re
import requests
from lxml import etree
import json
from jsonpath import jsonpath
from openpyxl import Workbook
import xlrd2
wb = Workbook()
sh = wb.create_sheet('数据获取',0)
del wb['Sheet']
#['Nurer Chala, Badda, Dhaka', 'Apartment', '1,050 sqft', '70 Lakh', '31 March 2024', 'https://www.bproperty.com/en/property/details-1961859.html']
sh.append(['地址','房型','面积','价格','更新时间','网址'])
d = 0
for i in range(1,100):
    if i == 1:
        url = r'https://www.bproperty.com/en/dhaka/properties-for-sale/'
    else:
        url = r'https://www.bproperty.com/en/bangladesh/properties-for-sale/page-'+str(i)+'/'
    hs = {
        'Cookie': 'anonymous_session_id=688b36e2-dd1f-4860-9e5f-9c542427674f; device_id=lufpx5tn31qkbfy7e; os_name=N%2FA; os_version=N%2FA; browser_name=N%2FA; browser_version=N%2FA; settings=%7B%22area%22%3Anull%2C%22currency%22%3A%22BDT%22%2C%22installBanner%22%3Afalse%2C%22searchHitsLayout%22%3A%22LIST%22%7D; abTests=%7B%22WhatsappButton%22%3A%22original%22%7D; banners=%7B%7D; userGeoLocation=%7B%22coordinates%22%3Anull%2C%22closestLocation%22%3Anull%2C%22loading%22%3Afalse%2C%22error%22%3Anull%7D; _ga=GA1.2.2083860482.1711901453; _gid=GA1.2.1966502629.1711901453; _gcl_au=1.1.1796649570.1711901464; crisp-client%2Fsession%2Ff186d25b-60af-4d4d-917e-1e4eaf8cc0c1=session_e114b23d-abfa-4898-b4fe-25e25a936a6d; PHPSESSID=3ctdg6oiu85ir31v70gg0ovmi0; scale=14.113; _ga_EXNWR0XMHG=GS1.2.1711901453.1.1.1711911640.60.0.0; AWSALB=6ipZCY0/GMwK6/i6MKQYLf5Cob5YqlLcrDUmnR0eo+g70P+EAdzGDbQlJK9S6qdHeJrvRuzJJ/1xZwpcWGj/J2FrokTch4UH4DeM88Mv4vukz91VuuTDAgDhRFSu; AWSALBCORS=6ipZCY0/GMwK6/i6MKQYLf5Cob5YqlLcrDUmnR0eo+g70P+EAdzGDbQlJK9S6qdHeJrvRuzJJ/1xZwpcWGj/J2FrokTch4UH4DeM88Mv4vukz91VuuTDAgDhRFSu; XSRF-TOKEN=eyJpdiI6IktFSmEyTGtielNkckdHMkhmU05MXC9nPT0iLCJ2YWx1ZSI6IjQzVjMwQ1wvR0RScWJBbjE1dHZ3Vm9jUFUxYWpCTXRLendDVFY3OU9xbG1lbXRNVWlZaU55YlB5b2ljajJ5eXoxdFwvSTZKM1RpT21MSVdFXC80Z29EN1N3PT0iLCJtYWMiOiI3NGE5NDFjMTc2MjIyYWFkNTEyYzRkZjAzYmZhNDg2MWZjNTRiYjJhOTEwYTg0MzU5YTI0ZDY2ZjZiZjJhN2M5In0%3D; bprop_session=eyJpdiI6Im92RFBMbDNkRkp3U3NKXC90VHZJZ1BBPT0iLCJ2YWx1ZSI6IjR3NDRzNW1sUzNvYXBKWU1XY09kN05RMDJXRlhlVUFpRnpBNkl4cUR2Z0wwb0Z2U3JQTUVSOGtjbTlwVkladU1aR1h0STR5bXo1R2doN1Jyb01jR3hBPT0iLCJtYWMiOiI4MjI3NGM4MjdhMTIxMjFmMjViZTY0MjZjM2U1MGExMGM5MjU2ZDliMWYxZWE5MmZiNTIwZjAzMjEyMmYzNzI4In0%3D; __gads=ID=9746c1d2fa4c084b:T=1711901498:RT=1711911693:S=ALNI_MZPLLa3q79-n3X6sCAzR00-WCtThg; __gpi=UID=00000d7b7a7d0d04:T=1711901498:RT=1711911693:S=ALNI_Mb2YZQFb9URfjfIx3Gbqae3KMc1vg; __eoi=ID=1080c2050b873fc3:T=1711901498:RT=1711911693:S=AA-AfjagmmlDYMfd3Mvk-xoGx4G0; referrer=%2Fbangladesh%2Fproperties-for-sale%2Fpage-3%2F; original_referrer=https%3A%2F%2Fwww.bproperty.com%2Fen%2Fbangladesh%2Fproperties-for-sale%2F%3Fmap_active%3Dtrue; landing_url=%2Fproperty%2Fdetails-1961860.html',
        'Referer': url,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'
    }
    rs = requests.get(url=url, headers=hs)
    html = etree.HTML(rs.text)
    es = html.xpath('//*[@id="body-wrapper"]/main/div[3]/div[3]/div[2]/div[1]/ul/li[*]/article/div[1]/a/@href')
    a = re.findall('{"@type":"ItemPage","url":"(.*?)",', rs.content.decode())
    if a != []:
        for j in a:
            # print(j)
            url1 = j
            header = {
                'Cookie': 'anonymous_session_id=688b36e2-dd1f-4860-9e5f-9c542427674f; device_id=lufpx5tn31qkbfy7e; os_name=N%2FA; os_version=N%2FA; browser_name=N%2FA; browser_version=N%2FA; settings=%7B%22area%22%3Anull%2C%22currency%22%3A%22BDT%22%2C%22installBanner%22%3Afalse%2C%22searchHitsLayout%22%3A%22LIST%22%7D; abTests=%7B%22WhatsappButton%22%3A%22original%22%7D; banners=%7B%7D; userGeoLocation=%7B%22coordinates%22%3Anull%2C%22closestLocation%22%3Anull%2C%22loading%22%3Afalse%2C%22error%22%3Anull%7D; _ga=GA1.2.2083860482.1711901453; _gid=GA1.2.1966502629.1711901453; _gcl_au=1.1.1796649570.1711901464; PHPSESSID=3ctdg6oiu85ir31v70gg0ovmi0; scale=14.113; crisp-client%2Fsession%2Ff186d25b-60af-4d4d-917e-1e4eaf8cc0c1=session_06426b08-6bd8-4faf-85c9-cb67461e793c; _dc_gtm_UA-201547-25=1; _ga_EXNWR0XMHG=GS1.2.1711967253.3.0.1711967253.60.0.0; __gads=ID=9746c1d2fa4c084b:T=1711901498:RT=1711967253:S=ALNI_MZPLLa3q79-n3X6sCAzR00-WCtThg; __gpi=UID=00000d7b7a7d0d04:T=1711901498:RT=1711967253:S=ALNI_Mb2YZQFb9URfjfIx3Gbqae3KMc1vg; __eoi=ID=1080c2050b873fc3:T=1711901498:RT=1711967253:S=AA-AfjagmmlDYMfd3Mvk-xoGx4G0; AWSALB=bEK+I801ktTpmfz9RlCd+0gL9sKbHh0nIuPoRtvHi8X9ko93xQmFDoHns63pj4NDB0fQ9RxgBpLzmUGIwc3qRCcZ8W2Wkep4WJo0m40LDj4U4Dr4V6EbSHYUev9o; AWSALBCORS=bEK+I801ktTpmfz9RlCd+0gL9sKbHh0nIuPoRtvHi8X9ko93xQmFDoHns63pj4NDB0fQ9RxgBpLzmUGIwc3qRCcZ8W2Wkep4WJo0m40LDj4U4Dr4V6EbSHYUev9o; XSRF-TOKEN=eyJpdiI6InJHODBmUndNUjhlME4ycjQ3OWFBSHc9PSIsInZhbHVlIjoibk9WXC9lWVhKaHB0N0RBTlJnZVp1M3RpVEdzMGlRbGc0U3RzQ3BaelVTNkJqMG5IQllNTVQ1cHhwNERwcHZZK1dWSm1hSkVqSXVcL3dtWk5aSVlXNUFUZz09IiwibWFjIjoiOGU1ZDBlNGUxNWFkOGE5MTY4MWYxZDFhYWQzN2M5ZmMxOTgzY2RmMjk5OGRiMDk1YTBiMTRjNzI0MTEzZWFiZiJ9; bprop_session=eyJpdiI6Ilord3ZBcnlyVE5yWXJDVXIreGxcL1dRPT0iLCJ2YWx1ZSI6InVPU1pvbEo5eUNUNmpcL3luQVpnUWgyQ0VtYWlJNFFRd1FWVDZnRVZWRk1nZUNuK25maFRcL1h3OGx5ZE9zUXlBUVwvUGpcL3NWVkdlVnlidnJraUVFa0VGUT09IiwibWFjIjoiN2ZiNjc1YmE4NDAxMWU0YjAzNzRjYzQ5ZTY5OTIxOTlhMzQzODA3MGE4YmQ2Y2IyOWJlYWI2ZTNmYzFiMzdlYSJ9; referrer=%2Fdhaka%2Fproperties-for-sale%2Fpage-3%2F; landing_url=%2Fdhaka%2Fproperties-for-sale%2F%3Fmap_active%3Dtrue',
                'Referer': j,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'
            }
            shj = []
            # print(len(j))
            if len(j) != 0:
                d += 1
                res = requests.get(url=url1, headers=header)
                #print(res.text)
                #价格：//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[1]/div[1]/div/span[3]/text()
                #地址：//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[2]/text()
                #面积：//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[4]/div[3]/span[3]/text()
                #跟新时间：//*[@id="body-wrapper"]/main/div[2]/div[4]/div[3]/div[1]/div/div[2]/ul/li[5]/span[4]/text()
                #类型
                html1 = etree.HTML(res.text)
                jg = html1.xpath('//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[1]/div[1]/div/span[3]/text()')
                dz = html1.xpath('//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[2]/text()')
                mj = html1.xpath('//*[@id="body-wrapper"]/main/div[2]/div[4]/div[1]/div[3]/div[3]/span[2]/span/span/text()')
                sj = html1.xpath('//*[@id="body-wrapper"]/main/div[2]/div[4]/div[3]/div[1]/div/div[2]/ul/li[5]/span[2]/text()')
                lx = html1.xpath('//*[@id="body-wrapper"]/main/div[2]/div[4]/div[3]/div[1]/div/div[2]/ul/li[1]/span[2]/text()')
                if dz !=[]:
                    shj.append(dz[0])
                    if lx !=[]:
                        shj.append(lx[0])
                        if mj !=[]:
                            shj.append(mj[0])
                            if jg !=[]:
                                shj.append(jg[0])
                                if sj !=[]:
                                    shj.append(sj[0])
                                    shj.append(url1)
                                else:
                                    shj.append('')
                            else:
                                shj.append('')
                        else:
                            shj.append('')
                    else:
                        shj.append('')
                else:
                    shj.append('')
            sh.append(shj)
            with open('达卡.json','r',encoding='utf-8')as f:
                huq = json.loads(f.read())
                print(huq)
                huq.append(shj)
            with open('达卡.json','w') as f1:
                json.dump(huq,f1,encoding='utf-8')
            print(shj)
wb.save('达卡房屋.xlsx')



