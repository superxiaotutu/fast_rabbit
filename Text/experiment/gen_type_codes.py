import cv2
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS, ImageCaptcha
from skimage.util import random_noise
from config import image_height, image_width, plt

image = ImageCaptcha(width=image_width, height=image_height)

SALT_LEVEL = []
NOISE_NUM = [i for i in range(0, 46, 5)]

radius=1
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)


def gene_code_all(chars):
    flag = random.randint(0, 50)
    im = gene_code_clean(chars) if flag else gene_code_normal(chars)
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
    return img


def add_gauss(image, radius=2):
    image = image.filter(MyGaussianBlur(radius))
    return image


def binary(image):
    image = image.convert('L')
    image = image.point(lambda x: 255 if x > np.mean(image) else 0)
    image = image.convert('RGB')
    return image

def throsh_binary(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval, image = cv2.threshold(c, 0, 255, cv2.THRESH_OTSU)
    image = Image.fromarray(image).convert('RGB')
    return image

def preprocess(image):
    flag = random.randint(0, 5)
    # flag=4
    if flag == 0:
        image = binary(image)
    elif flag == 1:
        image = add_gauss(image, radius)
    elif flag == 2:
        image=throsh_binary(image)
    elif flag == 3:
        image=throsh_binary(image)
        image = add_gauss(image, radius)
        image = image.convert('RGB')
    elif flag == 4:
        image = add_gauss(image, radius)
        image=throsh_binary(image)
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
    elif 1 <= flag < 7:
        im = gen_gauss_code(chars)
    else:
        image = gene_code_clean_one(chars)
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > np.mean(image) else 0)
        im = image.convert('RGB')
        im = np.asarray(im).astype(np.float32) / 255.

    return im


def gen_type_1(chars):
    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))
    im = image.create_captcha_image(chars, color, background)
    im = im.filter(ImageFilter.SMOOTH)
    return im


def gen_type_2(chars):
    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))
    im = image.create_captcha_image(chars, color, background)
    dot, line = 10, 1
    image.create_noise_dots(im, color, number=dot)
    for i in range(line):
        image.create_noise_curve(im, color)
    im = im.filter(ImageFilter.SMOOTH)
    return im


def gen_type_3(chars):
    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))
    im = image.create_captcha_image(chars, color, background)
    dot, line = 60, 20
    image.create_noise_dots(im, color, number=dot)
    for i in range(line):
        image.create_noise_curve(im, color)
    im = im.filter(ImageFilter.SMOOTH)
    return im

#
# gen_type_2("BASD").show()
# gen_type_3("BASD").show()

# a=gen_code_analy('S')
# plt.imshow(a)
# plt.imsave('a.png',a)
# m=gen_gauss_code('A')
# plt.imshow(m)
# plt.show()
# gene_code('A').show()
