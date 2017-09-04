# coding=utf-8
import csv


def get_items(traffic_txt):
    items = []
    item = []
    for line in open(traffic_txt):
        if line.strip() == '':
            items.append(item)
            item = []
        else:
            item.append(line.strip())
    return items


def get_data_column(item):
    columns = []
    for i in item[:8]:
        columns.append(i.split(':')[0])
    return columns


def data_uniform(item):
    data = []
    for i in item[:8]:
        data.append(i.split(':')[1].strip())
    return data


if __name__ == "__main__":
    traffic_txt = '/root/captcha/traffic99.txt'
    csv_file = '/root/captcha/traffic99.csv'
    data_items = get_items(traffic_txt)
    column = get_data_column(data_items[0])
    data = map(data_uniform, data_items)
    print column
    print data[0]
    print data[1]

    csvfile = file(csv_file, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(column)
    writer.writerows(data)
    csvfile.close()
