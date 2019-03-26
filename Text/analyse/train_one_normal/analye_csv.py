import csv
import numpy
import re

data = csv.reader(open('/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/train_one_normal/Batch_3580068_batch_results.csv', 'r'))
zero=[0,'0','O','o']
c = 0
result = [0 for i in range(100)]
sum = [0 for i in range(100)]
for i in data:
    if not c:
        c += 1
        continue
    true_label = i[-2][-5]
    person_label = i[-1]
    level = re.search('_(.+)_', i[-2]).group(1)
    if true_label in zero:
        if person_label.upper() in zero:
            result[int(level)] += 1
    elif true_label == person_label.upper():
        result[int(level)] += 1

    else:
        print(true_label, person_label)
    sum[int(level)] += 1
print(sum)

a = (numpy.true_divide(result,sum))
print(a)
for i in a:
    if not numpy.isnan(i):
        print(i)

