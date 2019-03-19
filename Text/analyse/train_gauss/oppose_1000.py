import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise

from test_LSTM_model import LABEL_CHOICES_LIST
from anaylse import gen_gauss_code, gene_code_clean
import numpy as np
def gen_oppose():
    for num in range(1000):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        img = gene_code_clean(captcha)
        img.save("images/%s.png" % (captcha))

def add_gauss(img_name,level,show=False):
    img = Image.open(img_name)
    img = np.asarray(img).astype(np.float32) / 255.
    img.flags.writeable = True
    for j in range(level):
        img = random_noise(img)
    np.clip(img, 0, 1)
    if show:
        plt.imshow(img)
        plt.show()
    return img

def add_gauss(img_name,level,show=False):
    img = np.asarray(img_name).astype(np.float32) / 255.
    img.flags.writeable = True
    for j in range(level):
        img = random_noise(img)
        
    return img
#
# for i in range(30):
#
