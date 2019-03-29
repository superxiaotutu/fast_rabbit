from config import LABEL_CHOICES_LIST
from gen_type_codes import *

for i in range(1000):
    slice = random.sample(LABEL_CHOICES_LIST, 4)
    captcha = ''.join(slice)
    im = gene_code_clean_one(captcha).save("%s_%s.png"%(i,captcha))