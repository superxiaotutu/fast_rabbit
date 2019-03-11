# -*- coding:utf-8 -*-
import argparse
import datetime
import json
import random

import numpy
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS
from skimage.util import random_noise
from datasets.load_data import *
import os
import tensorflow.contrib.slim as slim
from datasets.constant import *

LABEL_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LABEL_CHOICES_LIST = [str(i) for i in LABEL_CHOICES]
encode_maps = {}
decode_maps = {}
for i, char in enumerate(LABEL_CHOICES, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))
FLAG = 0
x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
y_ = tf.placeholder(tf.int32, [None, NUM_PER_IMAGE * LABEL_SIZE])
keep_prob = tf.placeholder(tf.float32)


def CNN(input_data, keep_prob):
    end_point = {}
    resized = end_point['resized'] = tf.reshape(input_data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    tf.summary.image('input', resized, max_outputs=LABEL_SIZE)
    conv1 = end_point['conv1'] = slim.conv2d(resized, 32, 3, padding='SAME', activation_fn=tf.nn.relu)
    pooling1 = end_point['pool1'] = slim.max_pool2d(conv1, 2)
    conv2 = end_point['conv2'] = slim.conv2d(pooling1, 64, 3, padding='SAME', activation_fn=tf.nn.relu)
    pooling2 = end_point['pool2'] = slim.max_pool2d(conv2, 2)
    flatten1 = end_point['flatten1'] = slim.flatten(pooling2)
    full1 = end_point['full1'] = slim.fully_connected(flatten1, 1024, activation_fn=tf.nn.relu)
    drop_out = end_point['drop_out'] = slim.dropout(full1, keep_prob)
    full2 = end_point['full2'] = slim.fully_connected(drop_out, NUM_PER_IMAGE * LABEL_SIZE, activation_fn=None)
    logits = end_point['logits'] = tf.reshape(full2, [-1, NUM_PER_IMAGE, LABEL_SIZE])
    predict = end_point['predict'] = tf.nn.softmax(logits)
    return end_point, logits, predict


y_expect_reshaped = tf.reshape(y_, [-1, NUM_PER_IMAGE, LABEL_SIZE])
end, log, pre = CNN(x, keep_prob)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=log))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# forword prop
predict = tf.argmax(pre, axis=2)
expect = tf.argmax(y_expect_reshaped, axis=2)

correct_prediction = tf.equal(predict, expect)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


def save():
    saver.save(sess, "model/model.ckpt")


def restore():
    saver.restore(sess, "model/model.ckpt")
def gene_code_clean(chars):
    font = ImageFont.truetype(DEFAULT_FONTS[0], size=random.choice([42, 50, 56]))
    font_width, font_height = font.getsize(chars)
    im = Image.new('RGB', (110, 55), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    per_width = (110 - font_width) / 4
    per_height = (55 - font_height) / 4
    draw.text((per_width, per_height), chars,
              font=font, fill=(100, 149, 237))
    return im


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    captcha = "ASD1"
    x1 = gene_code_clean(captcha)
    x1 = numpy.asarray(x1)
    x1 = [x1]
    code = read_labels([captcha])
    y1 = code
    for i in range(10000):
            # Test trained model
        test_accuracy,predict1,expect1 = sess.run([accuracy,predict,expect],feed_dict={x: x1, y_: y1, keep_prob: 0.8})
        print((predict1))
        print((expect1))

        print('step %s, testing accuracy = %.2f%%' % (
            i,  test_accuracy * 100))