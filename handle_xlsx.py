# coding=utf-8
from xlrd import open_workbook
from xlutils.copy import copy

xlsx_path = '/root/captcha/tda.xls'
rb = open_workbook(xlsx_path)

# 通过sheet_by_index()获取的sheet没有write()方法
rs = rb.sheet_by_index(0)

wb = copy(rb)

# 通过get_sheet()获取的sheet有write()方法
ws = wb.get_sheet(0)
for rownum in range(1, rs.nrows):
    # print rs.cell(rownum, 3).value
    description = rs.cell(rownum, 3).value

    http_index = description.lower().find('http request')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的HTTP请求'.decode('utf-8'))

    http_index = description.lower().find('http response')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的HTTP响应'.decode('utf-8'))

    http_index = description.lower().find('tcp request')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的TCP请求'.decode('utf-8'))

    http_index = description.lower().find('tcp response')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的TCP响应'.decode('utf-8'))

    http_index = description.lower().find('ftp request')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的FTP请求'.decode('utf-8'))

    http_index = description.lower().find('ftp response')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的FTP响应'.decode('utf-8'))

    http_index = description.lower().find('irc request')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的IRC请求'.decode('utf-8'))

    http_index = description.lower().find('irc response')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的IRC响应'.decode('utf-8'))

    http_index = description.lower().find('irc nickname')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, 'IRC僵尸网络'.decode('utf-8'))

    http_index = description.lower().find('smb request')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '检测到带'.decode('utf-8') + virus_name + '的SMB请求'.decode('utf-8'))

    http_index = description.lower().find('tcp connection')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '感染'.decode('utf-8') + virus_name + '病毒的TCP连接'.decode('utf-8'))

    http_index = description.lower().find('udp connection')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '感染'.decode('utf-8') + virus_name + '病毒的UDP连接'.decode('utf-8'))

    http_index = description.lower().find('dns connection')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '感染'.decode('utf-8') + virus_name + '病毒的DNS连接'.decode('utf-8'))

    http_index = description.lower().find('smb connection')
    if http_index != -1:
        virus_name = description[:http_index - 1]
        ws.write(rownum, 2, '感染'.decode('utf-8') + virus_name + '病毒的SMB连接'.decode('utf-8'))

wb.save('changed.xls')

