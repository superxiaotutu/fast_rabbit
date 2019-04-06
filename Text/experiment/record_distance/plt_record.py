import os
import glob
import re

def plt_attack_char():
    filenames = glob.glob('log/attack*.log')
    type_arr = [i for i in range(4)]
    for i in range(1, 5):
        acc = 0
        all_time = 0
        all_dis = 0
        filename = 'log/attack_char=%s_step=0.01.log' % i
        with open(filename, 'r')as f_r:
            ls = f_r.readlines()[:500]
            for l in ls:
                l = l.strip('\n')
                dis = float(re.search('dis:(.+) cost', l).group(1))
                cost_time = float(re.search('cost_time:(.+) c', l).group(1))
                c = re.search('c:(.+) at', l).group(1)
                attack = (re.search('attack_scc:(.+)', l).group(1))

                # print(dis,cost_time,c,attack)
                # print(attack)
                if attack=='True':
                    acc += 1
                all_time += cost_time
                all_dis += dis
        print(all_time / 500, all_dis / 500, acc/500)
#1 0.01805528736114502 189.05559467008464 0.806
#2 0.01768256664276123 189.6734356151257 0.804
#3 0.01786677026748657 183.2765199841146 0.802
#4 0.018107698917388917 189.44957020440327 0.814


def plt_c():
    for i in range(10, 31):
        acc = 0
        all_time = 0
        all_dis = 0
        filename = 'log/comfort_c=%s_step=0.01.log' % i
        with open(filename, 'r')as f_r:
            ls = f_r.readlines()[:500]
            for l in ls:
                l = l.strip('\n')
                dis = float(re.search('dis:(.+) cost', l).group(1))
                cost_time = float(re.search('cost_time:(.+) c', l).group(1))
                c = re.search('c:(.+) at', l).group(1)
                attack = (re.search('attack_scc:(.+)', l).group(1))

                # print(dis,cost_time,c,attack)
                # print(attack)
                if attack == 'True':
                    acc += 1
                all_time += cost_time
                all_dis += dis
        print(i,all_time / 500, all_dis / 500, acc / 500)
# 0.028236234664916992 86.86268291079767 0.394
# 0.025534380912780763 97.28711024675435 0.452
# 0.024474246978759766 108.13924590948804 0.504
# 0.024429670810699463 111.71023935755183 0.552
# 0.028302732467651366 121.51963728442472 0.62
# 0.020907005310058593 127.35370083176724 0.65
# 0.02088157844543457 149.19585605813083 0.658
# 0.020227845668792724 150.52478172554476 0.692
# 0.019161662101745604 154.35021255228554 0.74
# 0.018470641613006593 177.31918332240247 0.758
# 0.018464792728424072 172.78407460206932 0.78
# 0.018734838485717772 203.6267392139672 0.784
# 0.018241076946258546 195.05806350839455 0.812
# 0.01791336536407471 216.9779632426201 0.83
# 0.01729722309112549 230.6707436135392 0.846
# 0.017017686367034913 238.8606524041582 0.866
# 0.016900686264038085 244.90302582295362 0.878
# 0.016689040184020995 250.6387627560657 0.898
# 0.01645245361328125 241.2249749357576 0.918
# 0.016515748023986816 263.3454962786786 0.916
# 0.018515748023986816 283.3454962786786 0.920
plt_attack_char()