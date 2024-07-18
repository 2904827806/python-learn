# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/31 15:35
@Auth ： RS迷途小书童
@File ：Batch download of Sentinel data.py
@IDE ：PyCharm
@Purpose ：批量下载哨兵数据
"""
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
# 导入用户登录，兴趣区识别模块
from subprocess import call
# 用来唤醒IDM下载数据
from datetime import date
import time
import xlwt
import xlrd2
# excel的读取和写入模块
from tqdm import tqdm


def Search_data(login, key, path_geojson, start_date, end_date, name, product_type, cloud, filepath):
    """
    :param login: 欧空局账号，字符串类型
    :param key: 欧空局密码，字符串类型
    :param path_geojson: 兴趣区路径及文件名
    :param start_date: 开始时间，字符串
    :param end_date: 结束时间，字符串
    :param name: 卫星名称
    :param product_type: 卫星类型
    :param cloud: 云量筛选，格式：（0，15）
    :param filepath: Url保存路径及文件名
    :return: 返回存有下载链接的excel路径
    """
    api = SentinelAPI(login, key, 'https://scihub.copernicus.eu/dhus')
    # 登陆账号https://scihub.copernicus.eu/apihub/
    footprint = geojson_to_wkt(read_geojson(path_geojson))
    # 读取兴趣区，兴趣区由http://geojson.io导出
    products = api.query(footprint,
                         date=(start_date, end_date),  # 搜索的日期范围
                         platformname=name,  # 卫星平台名，Sentinel-2
                         producttype=product_type,  # 产品数据等级，Sentinel-2： S2MSI2A，S2MSI1C，S2MS2Ap/Sentinel-1：SLC，GRD，OCN
                         cloudcoverpercentage=cloud)  # 云量百分比
    # 搜索A、B双星的数据
    row = 0
    workbook_write = xlwt.Workbook(encoding='utf-8')
    worksheet_write = workbook_write.add_sheet('Url_image')
    for product in products:
        # 通过for循环遍历并打印、下载出搜索到的产品文件名
        info_product = api.get_product_odata(product)
        # 通过OData API获取单一产品数据的主要元数据信息
        worksheet_write.write(row, 0, info_product['url'])
        worksheet_write.write(row, 1, info_product['title'])
        print(info_product['title'])
        # print(product_info['url'])
        # 打印下载的产品数据文件名，id/uuid代码编号，size数据大小，title标题，url链接，md5，date时间
        # api.download(product)
        row += 1
    workbook_write.save(filepath)
    return filepath, api
    # 循环结束后，保存表格


def Download_image(filepath, Path_Download, Path_IDM, api):
    workbook_read = xlrd2.open_workbook(filepath)
    # 打开表格，创建工作空间
    sheet1 = workbook_read.sheet_by_name('Url_image')
    # 选择需要读取的sheet
    link_list = sheet1.col_values(0)
    # 获取第一列的数据
    print('所有链接下载完成，现在开始下载对应数据')
    num = 0
    while link_list:
        print('---------------------------------------------------')
        num += 1
        print('\n')
        print('第' + str(num) + '次循环' + '\n')
        id = link_list[0].split('\'')[1]
        link = link_list[0]
        info_product = api.get_product_odata(id)
        print('查询当前列表里的第一个数据的状态')
        if info_product['Online']:
            print(info_product['title'] + '为：online产品')
            print('加入IDM的下载列表中: ')
            print('\n')
            call([Path_IDM, '/d', link, '/p', Path_Download, '/n', '/a'])
            link_list.remove(link)
            call([Path_IDM, '/s'])
        else:
            print(info_product['title'] + '为：offline产品')
            print('\n')
            print('激活offline产品')
            code_id = link_list[0].split('\'')[1]
            api.trigger_offline_retrieval(code_id)
            # 激活offline产品
            print('检查任务列表里是否存在online产品: .........')
            # 等待激活成功的时候，检查现在的列表里还有没有online产品
            # 如果有online的产品那就下载
            # 首先检查列表中是否需要下载的数据
            if len(link_list) > 1:
                # 记录列表里可以下载的链接，并在最后把它们删除
                link_list_1 = []
                # 开始寻找列表剩下的元素是否有online产品
                for i in range(1, len(link_list)):
                    id2 = link_list[i].split('\'')[1]
                    link_1 = link_list[i]
                    info_product2 = api.get_product_odata(id2)
                    if info_product2['Online']:
                        print(info_product2['title'] + '为Online产品')
                        print('加入IDM的下载列表中: ')
                        print('--------------------------------------------')
                        call([Path_IDM, '/d', link_1, '/p', Path_Download, '/n', '/a'])
                        # 在列表中加入需要删除产品的HTTP链接信息
                        # 直接在link_list中删除会link_list的长度会发生改变，最终造成i的值超过link_list的长度
                        link_list_1.append(link_1)
                    else:
                        continue
                # 把已经下载的数据的链接给删除掉
                if len(link_list_1) > 0:
                    call([Path_IDM, '/s'])
                    for link_2 in link_list_1:
                        link_list.remove(link_2)
            print('本轮次检查结束，开始等到40分钟')
            # 将该激活的产品删除，再加入到最后
            link_list.remove(link)
            link_list.append(link)
            # 两次激活offline数据的间隔要大于30分钟
            for i in tqdm(range(int(1200)), ncols=100):
                time.sleep(2)


if __name__ == "__main__":
    """说明文档：https://sentinelsat.readthedocs.io/en/latest/api_overview.html,
    https://scihub.copernicus.eu/userguide/AdvancedSearch"""
    login = '**********'
    key = '********'
    path_geojson = "G:/map.geojson"
    start_date = "20230101"
    end_date = "20230301"
    name = 'Sentinel-2'
    product_type = 'S2MSI2A'
    cloud = (0, 15)
    filepath = 'G:/url.xls'
    # 存储下载链接的表格
    filepath, api = Search_data(login, key, path_geojson, start_date, end_date, name, product_type, cloud, filepath)
    Download_Path = 'G:/try_download/'
    # 数据要下载的地址,IDM的下载地址
    Path_IDM = "D:/IDM/IDMan.exe"
    Download_image(filepath, Download_Path, Path_IDM, api)
