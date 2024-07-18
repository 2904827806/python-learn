import time

from openpyxl import load_workbook
from openpyxl import Workbook
from tqdm import tqdm


def new():
    wb = Workbook() #创建一个表格
    sh = wb.active#激活
    sh1 = wb.create_sheet('数据1')  # 创建工作簿
    sh2 = wb.create_sheet('数据',0)#创建工作簿
    del wb['Sheet']
    wb.save('120.xlsx')


def data():
    import requests
    from lxml import etree
    a = []
    for i in range(65):
        # 获取url
        # https://movie.douban.com/top250
        # https://movie.douban.com/top250?start=25&filter=
        # https://movie.douban.com/top250
        url = f"https://movie.douban.com/top250?start={i * 25}&filter="
        # print(url)
        # 获取头文件
        # 最好使用cookie池
        header = {
            'Cookie': 'll="118350"; bid=EcQxwZCEAAk; _pk_id.100001.4cf6=40dc9b188a39e9c3.1687090946.; __yadk_uid=kQXZsiii9quUJW5VYAkgQwPvt87M8sLV; __gads=ID=81664f961e26089b-2293da42afe100d8:T=1687090947:RT=1687090947:S=ALNI_MYWaaB0g0Zgm203GYkyW4Roh6bvWQ; __gpi=UID=00000c511d7f531b:T=1687090947:RT=1687090947:S=ALNI_Mb81GTCy6fJSU0IjazWw_FOMoSF5w; _vwo_uuid_v2=D22F7663ECE419685B0F7342B336FC4B4|8fdb600eddb5b2ca426843350ff987cb; douban-fav-remind=1; _ga=GA1.1.1049362381.1687090947; _ga_RXNMP372GL=GS1.1.1706428395.1.0.1706428400.55.0.0; __utmv=30149280.27782; viewed="31544727_26655959"; ap_v=0,6.0; __utmc=30149280; __utmc=223695111; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1711116014%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DiXGkzUPfaEVB4V_2C5c1xBuG0x-LCZZF00QQuAuTxFkyfBAlg8kGGHCZJUGim7Ob%26wd%3D%26eqid%3D98fc9dac006ecdfd0000000665fd7e40%26tn%3D02003390_114_hao_pg%22%5D; _pk_ses.100001.4cf6=1; __utma=30149280.1049362381.1687090947.1711111689.1711116014.18; __utmb=30149280.0.10.1711116014; __utmz=30149280.1711116014.18.14.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.39358061.1687090947.1711111748.1711116014.18; __utmb=223695111.0.10.1711116014; __utmz=223695111.1711116014.18.13.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; dbcl2="277824043:tUeqQ5gvSqY"; ck=Llyk; push_noty_num=0; push_doumail_num=0',
            'Referer': 'https://www.baidu.com/link?url=iXGkzUPfaEVB4V_2C5c1xBuG0x-LCZZF00QQuAuTxFkyfBAlg8kGGHCZJUGim7Ob&wd=&eqid=98fc9dac006ecdfd0000000665fd7e40',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
        }
        # 发送请求
        response = requests.get(url=url, headers=header)
        html = etree.HTML(response.text)
        names = html.xpath('//*[@id="content"]/div/div[1]/ol/li[*]/div/div[2]/div[1]/a/span[1]/text()')
        pf = html.xpath('//*[@id="content"]/div/div[1]/ol/li[*]/div/div[2]/div[2]/div/span[2]/text()')

        if len(names) != 0:
            # print(names)
            for j in range(len(names)):
                dicts = {f'电影名：{names[j]}': f'评分{pf[j]}'}
                a.append(dicts)
        else:
            break
    return a


def set_value():
    from openpyxl.styles import Font, Border, Alignment, colors  # 引入设置样式的模块
    #添加数据
    wb = Workbook()
    sh = wb.create_sheet('数据',0)
    j = data()
    bold_italic_30_font = Font(name='黑体',italic=True,size=10,
                               bold=True,color=colors.BLUE) #设置格式
    #sh['A[1]'] = 'hello'
    sh.font = bold_italic_30_font
    for i in range(len(j)):
        # print(i)
        #print(j[i].items())
        for a,b in j[i].items():
            sh.cell(i+1,2).value = float(b[2:])
            sh[f'B{i+1}'].font =bold_italic_30_font
            sh[f'A{i+1}'] = a[4:]

            time.sleep(0.1)

            #print(b)
    from openpyxl.chart import Reference,LineChart
    c1 = LineChart()
    c1.title = '25'
    c1.x_axis.title='5'
    c1.y_axis.title='6'

    date = Reference(sh,min_row=1,max_row=251,min_col=2)
    c1.add_data(date,titles_from_data=True)
    sh.add_chart(c1,'e5')

    #sh['A1'] = 'hello!'
    wb.save('520.xlsx')
    for i in tqdm(range(len(j)),desc='输出'):
        time.sleep(0.1)
def set_value2(): #将多行数据加入到Excel表中
    wb = Workbook()
    sh = wb.active
    data = ['1','25','225']
    for i,d in enumerate(data):
        sh.cell(i+1,1).value = d
    wb.save('25.xls')
def set_style():#修改行和列的感受
    from openpyxl.styles import colors,Border,Alignment,Font
    wb = Workbook()
    sh = wb.active
    #设置数据
    sh.row_dimensions[1].height = 30  #设置行高
    sh.column_dimensions['A'].width = 30  #设置列宽
    data = ['1', '25', '225']
    for i,d in enumerate(data):
        sh.cell(i+1,1).value = d
        sh.cell(i+1,1).alignment = Alignment(horizontal='center',vertical='center')

    wb.save('25.xls')

def set_merge():#合并单元格
    from openpyxl.styles import colors, Border, Alignment, Font
    wb = Workbook()
    sh = wb.active
    # 合并多个单元格
    sh.merge_cells('A2:C2') #单元格顺序一定是从小到大，从前往后，从上到下
    sh.merge_cells('d2:e5')
    sh['a2'] = '横向和' #单元格设置一定是左上角的单元格
    #设置单元格居中对齐
    sh['a2'].alignment = Alignment(horizontal='center',vertical='center')
    sh['d2'] = '多合并'
    wb.save('25.xls')

def set_image():
    from openpyxl.styles import colors, Border, Alignment, Font
    from datetime import date
    from openpyxl.chart import LineChart,Reference #依赖
    wb = Workbook()
    sh = wb.active
    rows = [
        ['data','batch','batch2','batch3'],
        [date(2020,12, 1), 40, 30, 25],
        [date(2020, 12, 2), 40, 20, 35],
        [date(2020, 12, 3), 80, 60, 24],
        [date(2020, 12, 4), 0, 70, 15],
        [date(2020, 12, 5), 40, 10, 5],
        [date(2020, 12, 6), 40, 30, 25],
    ]
    for row in rows:
        sh.append(row) #当数据是列表时使用
    #加图表
    c1 = LineChart()
    c1.title = 'Line Chart'
    c1.x_axis.title = 'test'
    c1.y_axis.title = 'DAT'
    #往图标中加数据
    data =Reference(sh,min_col=2,min_row=1,max_col=4,max_row=7) #可以设置依赖的数据，第列到几列，几行到几行
    c1.add_data(data,titles_from_data=True)   #设置启用标题
    c1.shape = 4
    sh.add_chart(c1,'A9')  #工作博上添加图表，所添加到图标和位置

    wb.save('25.xls')



if __name__ == '__main__':
    set_value()
    #set_style()
    #set_merge()
    #set_image()