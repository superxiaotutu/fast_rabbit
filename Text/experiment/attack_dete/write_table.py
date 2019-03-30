with open('cnn_vs_ocr_sample_head_result.txt','r')as f:
    acc_arr = [i for i in range(4)]
    ls = f.readlines()
    for l in ls:
        l = l.strip('\n')
        arr = l.split(' ')
        type, true, pred = arr[0], arr[1], arr[2]
        if true == pred:
            acc_arr[int(type)] += 1
print(acc_arr)

# [117, 29, 74, 12]