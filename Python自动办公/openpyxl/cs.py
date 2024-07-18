import time
from openpyxl import load_workbook
from openpyxl import Workbook
from tqdm import tqdm
from openpyxl.styles import Font,Alignment,colors
from openpyxl.chart import LineChart,Reference
from datetime import date
wb = Workbook()
sh = wb.create_sheet('电影',0)
sh1 = wb.create_sheet('自媒体')
font1 = Font(name='黑体',bold=True,size=10,italic=True,color=colors.BLUE)
sh.cell(1,1).value = '自媒体'

sh['a1'].font = font1
sh.cell(1,1).alignment = Alignment(horizontal='center',vertical='center')
sh.row_dimensions[1].height = 30
sh.column_dimensions['a'].width = 20
#sh.merge_cells('e4:f5')
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
    sh.append(row)
c1 = LineChart()
c1.title = '25'
c1.x_axis.title = '258'
c1.y_axis.title = '265'
data = Reference(sh,min_row=2,max_row=8,min_col=2,max_col=4)
c1.add_data(data,titles_from_data=True)
sh.add_chart(c1,'k5')


wb.save('102.xlsx')