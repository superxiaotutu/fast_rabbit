import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
from random import randint
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from captcha.image import DEFAULT_FONTS
import random
import numpy as np

image_height = 60
image_width = 180
image = ImageCaptcha(width=image_width, height=image_height)


def gene_code(chars):
    flag = random.randint(0, 10)
    im = gene_code_normal(chars) if flag else gene_code_clean(chars)
    im = preprocess(im)
    return im


def preprocess(image):
    flag = random.randint(0, 4)
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
    else:
        return image
    return image


def normal_choice():
    dot, line = -1, -1
    while True:
        if dot < 0 or line < 0:
            line = np.random.normal(50, 30, 1)
            dot = np.random.normal(10, 8, 1)
        else:
            return int(line), int(dot)


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


# a = (gene_code("asd"))
# print(np.asanyarray(a))
