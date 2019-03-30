import sys
import tensorflow as tf
from tensorflow.contrib import slim

import one_char_model as LSTM
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


# train_feeder = LSTM.DataIterator()
# val_feeder = LSTM.DataIterator()

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


def fgsm(Checkpoint_PATH):
    imgs_input = []
    labels_input = []
    labels_arr = []
    for i in range(batch_size):
        slice = random.sample(LABEL_CHOICES, 1)
        captcha = ''.join(slice)
        img = np.asarray(gene_code_clean_one(captcha)).astype(np.float32) / 255.
        imgs_input.append(img)
        labels_input.append(captcha)
        code = [0 for i in range(num_classes)]
        code [encode_maps[captcha]]=1

        labels_arr.append([code])
        break
    imgs_input_before = imgs_input
    model = LSTM.LSTMOCR("test")
    model.build_graph()

    labels=tf.placeholder(tf.int32, [None, 1, num_classes])
    ADV_LOSS = -slim.losses.softmax_cross_entropy(model.logits, labels)
    grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    Var_restore = tf.global_variables()
    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from ckpt{}'.format(ckpt))
    else:
        print('cannot restore')
        return
    adv_step = 0.01
    for level in [i for i in range(45, 46, 5)]:
        acc = 0
        imgs_input = imgs_input_before
        feed={model.inputs: imgs_input,labels:labels_arr}
        for i in range(level):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed_dict=feed)
            imgs_input = imgs_input - grad * adv_step
            imgs_input = np.clip(imgs_input, 0, 1)
            feed = {model.inputs: imgs_input,labels:labels_arr}
        print(loss_now)

        for z in range(batch_size):
            plt.imsave("adv/%s_%s_%s.png" % (z,level,labels_input[z]), imgs_input[z])
        dense_decoded_code = sess.run(model.dense_decoded, {model.inputs: imgs_input,labels:labels_arr})
        attack_arr = []
        for j in dense_decoded_code:
            expression = ''
            for i in j:
                if i == -1:
                    expression += ''
                else:
                    expression += LSTM.decode_maps[i]
            attack_arr.append(expression)
        for true_l, adv_l in zip(labels_input, attack_arr):
            if (true_l == adv_l):
                acc += 1
            print(adv_l, true_l)
        with open('res.txt', 'a')as f:
            f.write(str(acc) + "\n")
        # print(acc)


def adv_many(Checkpoint_PATH, img_PATH):
    shuff_dir = {}
    lst = [i for i in range(36)]
    random.shuffle(lst)

    def creat_onehot(num):
        re = np.zeros([38])
        re[num - 1] = 1
        return re

    for i in range(36):
        shuff_dir.update({i + 1: creat_onehot(lst[i])})
    shuff_dir.update({37: creat_onehot(37)})

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR("test", "lenet")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # adv_node
    target = tf.placeholder(tf.float32, [12, batch_size, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    predict = tf.nn.softmax(model.logits)
    # ADV_LOSS = tf.reduce_sum(tf.square(predict - target)) + tf.reduce_mean(tf.square(origin_inputs - model.inputs))
    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])

    # ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + tf.reduce_mean(
    #     tf.reduce_sum(predict * current_mengban) - tf.reduce_sum(predict * target))

    # FGSM
    ADV_LOSS = -tf.losses.softmax_cross_entropy(model.inputs, model.logits)

    grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    Var_restore = tf.global_variables()

    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from ckpt{}'.format(ckpt))
    else:
        print('cannot restore')
        return
    imgs_input = []
    labels_input = []
    for i in img_PATH:
        labels_input.append(i[1])
        im = np.asarray(i[0]).astype(np.float32) / 255.
        imgs_input.append(im)
    imgs_input = np.asarray(imgs_input)
    feed = {model.inputs: imgs_input}
    # dense_decoded_code = sess.run(model.dense_decoded, feed)
    #
    # pred_arr = []
    # for j in dense_decoded_code:
    #     expression = ''
    #     for i in j:
    #         if i == -1:
    #             expression += ''
    #         else:
    #             expression += LSTM.decode_maps[i]
    #     pred_arr.append(expression)

    # adv
    log = sess.run(model.logits, feed)
    fir = log[:, 0, :]
    ex = np.argmax(fir, axis=1)
    imgs_input_before = imgs_input
    target_creat = []
    for i in range(12):
        target_creat.append(shuff_dir[ex[i]])
    target_creat = np.asarray(target_creat)
    target_creat = target_creat[:, np.newaxis, :]
    target_creat = np.repeat(target_creat, batch_size, axis=1)

    adv_step = 0.02
    for level in [i for i in range(0, 49, 10)]:
        acc = 0
        imgs_input = imgs_input_before
        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(level):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            imgs_input = imgs_input - grad * adv_step
            feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        plt.imshow(imgs_input[0])
        plt.imsave("adv/%s_.png" % level, imgs_input[0])
        imgs_input_after = imgs_input
        feed = {model.inputs: imgs_input_after, target: target_creat, origin_inputs: imgs_input_before}
        dense_decoded_code = sess.run(model.dense_decoded, feed)
        attack_arr = []
        for j in dense_decoded_code:
            expression = ''
            for i in j:
                if i == -1:
                    expression += ''
                else:
                    expression += LSTM.decode_maps[i]
            attack_arr.append(expression)
        for true_l, adv_l in zip(labels_input, attack_arr):
            if (true_l == adv_l):
                acc += 1
            print(adv_l, true_l)
        with open('res.txt', 'a')as f:
            f.write(str(acc) + "\n")
        print(acc)
        del imgs_input, img_PATH



def main():
    fgsm('../train_gauss/train_gauss/model')


if __name__ == '__main__':
    main()
