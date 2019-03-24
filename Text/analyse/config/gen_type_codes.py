import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS, ImageCaptcha
from skimage.util import random_noise
from config import image_height, image_width, plt

image_width = image_width // 4
image = ImageCaptcha(width=image_width, height=image_height)

SALT_LEVEL = []
NOISE_NUM = [i for i in range(0, 46, 5)]
print(NOISE_NUM)


def gene_code_all(chars):
    flag = random.randint(0, 99)
    im = gene_code_normal(chars) if flag else gene_code_clean_one(chars)
    im = preprocess(im)
    return im


def gene_code_clean(chars):
    font = ImageFont.truetype(DEFAULT_FONTS[0], size=random.choice([42, 50, 56]))
    font_width, font_height = font.getsize(chars)
    im = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    per_width = (image_width - font_width) / 4
    per_height = (image_height - font_height) / 4
    draw.text((per_width, per_height), chars,
              font=font, fill=(100, 149, 237))
    im = im.filter(ImageFilter.SMOOTH)
    return im


def gene_code_clean_one(chars):
    font = ImageFont.truetype(DEFAULT_FONTS[0], size=random.choice([42, 50, 56]))
    font_width, font_height = font.getsize(chars)
    im = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    per_width = (image_width - font_width)
    per_height = (image_height - font_height)
    draw.text((per_width - 10, per_height - 10), chars,
              font=font, fill=(100, 149, 237))
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


def gen_gauss_code(captcha):
    img = gene_code_clean_one(captcha)
    img = np.asarray(img).astype(np.float32) / 255.
    img.flags.writeable = True
    level = random.choice(NOISE_NUM)
    for j in range(level):
        img = random_noise(img)
    np.clip(img, 0, 1)
    # img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def preprocess(image):
    flag = random.randint(0, 5)
    if flag == 0:
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > np.mean(image) else 0)
        image = image.convert('RGB')
    elif flag == 1:
        image.filter(ImageFilter.GaussianBlur)
    elif flag == 2:
        image.filter(ImageFilter.GaussianBlur)
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > np.mean(image) else 0)
        image = image.convert('RGB')
    elif flag == 3:
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > np.mean(image) else 0)
        image.filter(ImageFilter.GaussianBlur)
        image = image.convert('RGB')
    elif flag == 4:
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > 255 // 2 else 0)
        image = image.convert('RGB')
    else:
        image = np.asarray(image).astype(np.float32) / 255.
        return image
    image = np.asarray(image).astype(np.float32) / 255.
    return image


def normal_choice():
    dot, line = -1, -1
    while True:
        if dot < 0 or line < 0:
            line = np.random.normal(50, 30, 1)
            dot = np.random.normal(10, 8, 1)
        else:
            return int(line), int(dot)


def gene_code_normal(chars):
    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))
    im = image.create_captcha_image(chars, color, background)
    dot, line = normal_choice()
    image.create_noise_dots(im, color, number=dot)
    for i in range(line):
        image.create_noise_curve(im, color)
    im = im.filter(ImageFilter.SMOOTH)
    return im


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


def gen_code_analy(chars):
    flag = random.randint(0, 10)
    if flag == 0:
        im = gene_code_clean_one(chars)
        im = np.asarray(im).astype(np.float32) / 255.
    elif 1<= flag <7:
        im = gen_gauss_code(chars)
    else:
        image = gene_code_clean_one(chars)
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > np.mean(image) else 0)
        im = image.convert('RGB')
        im = np.asarray(im).astype(np.float32) / 255.

    return im


# a=gen_code_analy('S')
# plt.imshow(a)
# plt.imsave('a.png',a)
# m=gen_gauss_code('A')
# plt.imshow(m)
# plt.show()
# gene_code('A').show()