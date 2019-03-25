import csv
import os
import random

# file_list=os.listdir('images')
# file_list=random.shuffle(file_list)
# arr = []
# # with open('filename.txt', 'r')as f:
# #     fs = f.readlines()
# #     for f in fs:
# #         f = f.strip('\n')
# #         arr.append()
# datas = [['image_url' for i in range(100)], ]
#
# with open('example.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for row in datas:
#         writer.writerow(row)

import xlrd
import xlwt

data = xlrd.open_workbook('data.xls')
table = data.sheets()[0]  # 通过索引顺序获取
arr = []
nrows = table.nrows  # 获取该sheet中的有效行数
for rowx in range(1, nrows):
    table.row(rowx)  # 返回由该行中所有的单元格对象组成的列表
    d = table.row_values(rowx)  # 返回由该列中所有的单元格对象组成的列表
    for i in d:
        arr.append('http://123.56.19.49:8888/' + str(i))

print(arr)
data = xlwt.Workbook()
table = data.add_sheet('data')  # 通过索引顺序获取
for index, i in enumerate(arr):
    table.write(index, 0, i)
data.save('newdata.xls')
