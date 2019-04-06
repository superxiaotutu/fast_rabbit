import os
import random
import shutil

from config import LABEL_CHOICES_LIST
from gen_type_codes import *
import shutil
def remo(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)
# a=np.array([])
# b=np.array([])
# for i in range(10000):
#     a=np.append(a,np.random.normal(50, 30, 1))
#     b=np.append(b,np.random.normal(10, 8, 1))
# print(a)
# print(np.min(a),np.max(a))
# print(np.min(b),np.max(b))

def gen_1000():
    for i in range(1000):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        im = gene_code_clean(captcha).save("%s_%s.png"%(i,captcha))
def gen_1000_type_3():
#     if os.path.isdir('ori_type_1'):
#         shutil.rmtree('ori_type_1')
#         os.mkdir('ori_type_1')
#     if os.path.isdir('ori_type_2'):
#         shutil.rmtree('ori_type_2')
#         os.mkdir('ori_type_2')
    if os.path.isdir('ori_type_3'):
        shutil.rmtree('ori_type_3')
        os.mkdir('ori_type_3')
    for i in range(1000):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        # gene_code_clean(captcha).save("ori_type_1/%s_%s.png" % (i, captcha))
        # gen_type_2(captcha).save("ori_type_2/%s_%s.png" % (i, captcha))
        gen_type_3(captcha).save("ori_type_3/%s_%s.png" % (i, captcha))

gen_1000_type_3()