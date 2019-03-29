from config import LABEL_CHOICES
from gen_type_codes import *
import numpy as np
import csv
import matplotlib.pyplot as plt
from config import *


def gen_oppose():
    with open('data.csv', 'w')as f:
        f = csv.writer(f)
        f.writerow(['image_url'])
        arr = []
        for num in range(50):
            for i in range(0, 46, 5):
                slice = random.sample(LABEL_CHOICES, 1)
                captcha = ''.join(slice)
                img = add_gauss(captcha, i)
                plt.axis('off')
                filname = "images/%s_%s_%s.png" % (num, i, captcha)
                plt.imsave(filname, img)
                f.writerow([filname])


def add_gauss(captcha, level):
    img = gene_code_clean_one(captcha)
    img = np.asarray(img).astype(np.float32) / 255.

    img.flags.writeable = True
    for j in range(level):
        img = random_noise(img)
    # np.clip(img, 0, 1)
    # img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


# arr=[]
# for i in LABEL_CHOICES:
#     arr.append(i)
# print(arr)
# print(LABEL_CHOICES.split(''))
gen_oppose()
