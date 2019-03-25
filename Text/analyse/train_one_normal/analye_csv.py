import csv
import numpy
import re

data=csv.reader(open('/home/kaiyuan_xu/Downloads/Batch_3580068_batch_results.csv','r'))

c=0
result = [0 for i in range(50)]
sum=[0 for i in range(100)]
for i in data:
    if not c:
        c+=1
        continue
    true_label=i[-2][-5]
    person_label=i[-1]
    print(i[-2])
    level = re.search('_(.+)_', i[-2]).group(1)
    if true_label==person_label:
        result[int(level)]+=1
    sum[int(level)]+=1
print(sum)
a=(numpy.divide(result,38))
print(a)
for i in a :
    if i !=0:
        print(i)