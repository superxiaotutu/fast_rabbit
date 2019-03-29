import tensorflow as tf
import model as LSTM
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib.pyplot as plt
import random
from config import *
from gen_type_codes import *


def cnn_generate(Checkpoint_PATH, img_PATH, model_name, gauss=False, bnf=False, all=False):
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
    model = LSTM.LSTMOCR(model_name, "infer", gauss, bnf, all)
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

    ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + tf.reduce_mean(
        tf.reduce_sum(predict * current_mengban) - tf.reduce_sum(predict * target))

    # grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
    grad_y2x = tf.gradients(ADV_LOSS, model.inputs)[0]

    Var_restore = tf.global_variables()
    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)

    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from ckpt{}'.format(ckpt))
    else:
        print('cannot restore')
        return
    acc=0
    for i in range(50):
        imgs_input = []
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        true_label=captcha
        im = gene_code_clean(captcha)
        if gauss:
            im = add_gauss(im)
        if bnf:
            im = binary(im)
        if all:
            im = binary(im)
            im = add_gauss(im)

        im = np.asarray(im).astype(np.float32) / 255.
        imgs_input.append(im)
        imgs_input_before = imgs_input
        feed = {model.inputs: imgs_input}
        log = sess.run(model.logits, feed)
        fir = log[:, 0, :]
        ex = np.argmax(fir, axis=1)
        print(ex)

        target_creat = []
        for i in range(12):
            target_creat.append(shuff_dir[ex[i]])

        target_creat = np.asarray(target_creat)
        target_creat = target_creat[:, np.newaxis, :]
        target_creat = np.repeat(target_creat, batch_size, axis=1)

        # dense_decoded_code = sess.run(model.dense_decoded, feed)
        # expression = ''
        # for i in dense_decoded_code[0]:
        #     if i == -1:
        #         expression += '-'
        #     else:
        #         expression += LSTM.decode_maps[i]
        # print("BEFORE:{}".format(expression))
        adv_step = 0.1
        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(100):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            # if (i + 1) % 10 == 0:
            #     print("LOSS:{}".format(np.max(grad)))
            imgs_input = imgs_input - grad * adv_step
            feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}

        imgs_input_after = imgs_input

        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        dense_decoded_code = sess.run(model.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        if true_label==expression:
            acc+=1
        print("BEFORE:{} ,AFTER:{}".format(true_label,expression))
        print(acc)
    plt.imshow(imgs_input_before[0])
    plt.show()
    plt.imshow(imgs_input_after[0])
    plt.show()
    return


def test_model(Checkpoint_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR('cnn', "infer")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    Var_restore = tf.global_variables()
    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)

    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from ckpt{}'.format(ckpt))
    else:
        print('cannot restore')
        return
    imgs_input = []

    im = gene_code_clean("ABCD")
    im = np.asarray(im).astype(np.float32) / 255.
    label = "ABCD"
    imgs_input.append(im)
    feed = {model.inputs: imgs_input}

    dense_decoded_code = sess.run(model.dense_decoded, feed)
    expression = ''
    for i in dense_decoded_code[0]:
        if i == -1:
            expression += '-'
        else:
            expression += LSTM.decode_maps[i]
    print("BEFORE:{}".format(expression))
    plt.imshow(im)
    plt.show()
    return


if __name__ == '__main__':
    cnn_generate('../train_lenet/model', 0, 'lenet', bnf=True)
    # test_model('../train_cnn/model', )
