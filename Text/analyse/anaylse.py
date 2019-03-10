import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS

SALT_LEVEL = [0.3, 0.4, 0.5, 0.6, 0.7]
image_height = 60
image_width = 180


def gene_code_clean(chars):
    font = ImageFont.truetype(DEFAULT_FONTS[0], size=random.choice([42, 50, 56]))
    font_width, font_height = font.getsize(chars)
    im = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    per_width = (image_width - font_width) / 4
    per_height = (image_height - font_height) / 4
    draw.text((per_width, per_height), chars,
              font=font, fill=(0, 0, 0))
    im = im.filter(ImageFilter.SMOOTH)
    return im


# 0.3-0.7 alpha个像素点保留原值
def addsalt_pepper(img, alpha=0.7):
    img = np.asarray(img)
    h, w, c = img.shape
    img.flags.writeable = True
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[alpha, (1 - alpha) / 2., (1 - alpha) / 2.])
    mask = np.repeat(mask, 3, axis=2)
    img[mask == 1] = 255
    img[mask == 2] = 0
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def gen_salt_code(captcha):
    img = gene_code_clean(captcha)
    salt_level = random.choice(SALT_LEVEL)
    img = addsalt_pepper(img, alpha=salt_level)
    return img
