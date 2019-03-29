import shutil
import sys
import tensorflow as tf
from one_char_model import LSTMOCR
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# plt.switch_backend('agg')
sys.path.append('../config')
import random
from gen_type_codes import *
from config import *
import tensorflow.contrib.slim as slim

image_width = 48


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


model = LSTMOCR("test")
model.build_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

target_label = tf.placeholder(tf.int32, [None, 1, 38])

ADV_LOSS = slim.losses.softmax_cross_entropy(model.logits, target_label)
grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
LL_targeet = tf.arg_min(model.logits, dimension=2)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
Var_restore = tf.global_variables()
saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
# ckpt = tf.train.latest_checkpoint('../train_one_normal/train_one_char/model')
ckpt = tf.train.latest_checkpoint('/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/train_one_normal/train_one_char/model')
if ckpt:
    saver.restore(sess, ckpt)
    print('restore from ckpt{}'.format(ckpt))
else:
    print('cannot restore')

imgs=os.listdir('ori')
shutil.rmtree('simple_adv')
os.mkdir('simple_adv')
acc = 0
levels = [i for i in range(50)]
result = [0 for i in range(50)]

for index1,i in enumerate(imgs):
    label_arr = [0 for j in range(38)]
    imgs_input = []
    label_inputs = []
    labels_arr = []
    imgs_input = [np.asarray(Image.open('ori/' + i)) / 255]

    code= sess.run(LL_targeet, feed_dict={model.inputs: imgs_input})
    label_arr[code[0][0]] = 1
    labels_arr.append([label_arr])
    label_inputs.append(i[-5])
    feed = {model.inputs: imgs_input, target_label: labels_arr}
    for level in [k for k in range(0,42,5)]:
        for l in range(level):
            g = sess.run(grad_y2x, feed_dict=feed)
            imgs_input = imgs_input - 0.001 * g
            imgs_input=np.clip(imgs_input,0,1)
            feed = {model.inputs: imgs_input, target_label: labels_arr}
        plt.imsave("simple_adv/%s_%s_%s.png" % (index1, level, label_inputs[0]), imgs_input[0])

        dense_decoded_code = sess.run(model.dense_decoded, feed)
        for index, j in enumerate(dense_decoded_code):
            expression = ''
            for i in j:
                if i == -1:
                    expression += ''
                else:
                    expression += decode_maps[i]
            if expression == label_inputs[index]:
                acc += 1
                result[level] += 1
        print(acc,result,expression)
print(result)
    # cv2.imshow("tu", imgs_input[0])
    # cv2.waitKey(0)
n=[]
result=[50, 0, 0, 0, 0, 50, 0, 0, 0, 0, 50, 0, 0, 0, 0, 47, 0, 0, 0, 0, 40, 0, 0, 0, 0, 28, 0, 0, 0, 0, 9, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in result:
    if i!=0:
        n.append(i)
print(n)
