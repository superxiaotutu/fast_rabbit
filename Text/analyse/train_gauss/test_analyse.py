import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS
from skimage.util import random_noise
import matplotlib.pyplot as plt

from Text.analyse.anaylse import gene_code_clean

SALT_LEVEL = []
NOISE_NUM = [i for i in range(0, 41, 7)]



def gen_gauss_code(captcha):
    for j in NOISE_NUM:
        img = gene_code_clean(captcha)
        img = np.asarray(img).astype(np.float32) / 255.
        img.flags.writeable = True
        for l in range(j):
            img = random_noise(img)
        np.clip(img, 0, 1)

        plt.imshow(img)
        plt.show()
        plt.imsave('test.png', img)

    # img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


img = gen_gauss_code("SVJ2")
