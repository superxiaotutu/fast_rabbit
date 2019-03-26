from config import *
from gen_type_codes import *
image_width = image_width // 4
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
for i in range(50):
    slice = random.sample(LABEL_CHOICES, 1)
    captcha = ''.join(slice)
    gene_code_clean_one(captcha).save('ori/%s_%s.png'%(i,captcha))