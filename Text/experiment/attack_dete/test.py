import glob
import shutil
import time

import tensorflow as tf
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import random
import sys
from tensorflow.contrib import slim
from attack_dete.ocr_line import update_matrix, get_acc, get_process

sys.path.append('../')
import model as LSTM
import model_gauss

from config import *
from gen_type_codes import *

Checkpoint_PATH = '/home/kirin/Python_Code/fast_rabbit/train_model/train_lenet_fine/model'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

img_files = glob.glob("../images/ori/*.png")


def update_matrix(matrix, labels, dense_decoded_code):
    for index, j in enumerate(dense_decoded_code):
        expression = ''
        for i in j:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        if labels[index] != expression:
            matrix[index] = 0
    return matrix


def attack(model, sess, imgs_input, imgs_label, type, ):
    # adv_node
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
    OCR_target = tf.placeholder(tf.float32, [12, batch_size, 38])
    CNN_target = tf.placeholder(tf.float32, [batch_size, 4, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    # predict = tf.nn.softmax(model.logits)
    predict = model.logits
    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])

    if type == 'ocr':
        node_num = 12
        axis = 1
        ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + c * tf.reduce_mean(
            tf.reduce_sum(predict * OCR_target) - tf.reduce_sum(predict * current_mengban))
    else:
        node_num = 4
        axis = 0
        ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + c * tf.reduce_mean(
            tf.reduce_sum(predict * CNN_target) - tf.reduce_sum(predict * current_mengban))

    is_attacked_matrix = np.ones((batch_size, image_height, image_width, image_channel))
    # grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
    grad_y2x = tf.gradients(ADV_LOSS, model.inputs)[0]
    imgs_input_before = imgs_input
    feed = {model.inputs: imgs_input}
    log = sess.run(model.logits, feed)
    fir = log[:, 0, :] if type == 'ocr' else log[0, :, :]
    ex = np.argmax(fir, axis=1)
    target_creat = []
    start = time.time()
    for j in range(node_num):
        target_creat.append(shuff_dir[ex[j]])
    target_creat = np.asarray(target_creat)
    if type == 'ocr':
        target_creat = target_creat[:, np.newaxis, :]
        target_creat = np.repeat(target_creat, batch_size, axis=axis)
        feed = {model.inputs: imgs_input, OCR_target: target_creat, origin_inputs: imgs_input_before}
        for i in range(adv_count):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            if (i + 1) % 2 == 0:
                print("LOSS:{}".format(loss_now))
            imgs_input = imgs_input - grad * adv_step * is_attacked_matrix
            # imgs_input = np.clip(imgs_input, 0, 1)
            dense_decoded_code = sess.run(model.dense_decoded, {model.inputs: imgs_input})
            is_attacked_matrix = update_matrix(is_attacked_matrix, imgs_label, dense_decoded_code)
            feed = {model.inputs: imgs_input, OCR_target: target_creat, origin_inputs: imgs_input_before}
    else:
        target_creat = target_creat[np.newaxis, :, :]
        target_creat = np.repeat(target_creat, batch_size, axis=axis)
        feed = {model.inputs: imgs_input, CNN_target: target_creat, origin_inputs: imgs_input_before}
        for i in range(adv_count):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            if (i + 1) % 2 == 0:
                print("LOSS:{}".format(loss_now))
            imgs_input = imgs_input - grad * adv_step * is_attacked_matrix
            dense_decoded_code = sess.run(model.dense_decoded, {model.inputs: imgs_input})
            is_attacked_matrix = update_matrix(is_attacked_matrix, imgs_label, dense_decoded_code)
            imgs_input = np.clip(imgs_input, 0, 1)
            feed = {model.inputs: imgs_input, CNN_target: target_creat, origin_inputs: imgs_input_before}
    imgs_input_after = imgs_input
    end = time.time()
    distance = np.linalg.norm(imgs_input_after - imgs_input_before)
    cost_time = end - start
    for i in range(batch_size):
        plt.imshow(imgs_input_after[i])
        plt.show()
        break

    return distance, cost_time, imgs_input_after


g1 = tf.Graph()
g2 = tf.Graph()

sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)
with sess1.as_default():
    with g1.as_default():
        model = LSTM.LSTMOCR('lenet', "infer")
        model.build_graph()
        Var_restore = tf.global_variables()
        saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
        sess1 = tf.Session(config=config)
        sess1.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
        if ckpt:
            saver.restore(sess1, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
with sess2.as_default():
    with g2.as_default():
        model2 = model_gauss.LSTMOCR_GAUSS('lenet', "infer")
        model2.build_graph()
        Var_restore = tf.global_variables()
        saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
        sess2 = tf.Session(config=config)
        sess2.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
        if ckpt:
            saver.restore(sess2, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
file_count = 10
with open('ocr_result.txt', 'w')as f:
    # -,G,B,gb
    for type in range(0, 4):
        adv_acc = 0
        prec_acc = 0
        img_files = glob.glob("../images/ori/*.png")
        for epoch in range(file_count // batch_size):
            ori_imgs_input = [img_files.pop() for i in range(batch_size)]
            print(len(ori_imgs_input))
            imgs_label = [i[-8:-4] for i in ori_imgs_input]
            imgs_input = get_process(ori_imgs_input, 0)
            imgs_input_before = imgs_input
            feed = {model.inputs: imgs_input_before}
            dense_decoded_code = sess1.run(model.dense_decoded, feed)
            prec_acc += get_acc(imgs_label, dense_decoded_code)

            with sess2.as_default():
                with g2.as_default():
                    distance, cost_time, imgs_input_after = attack(model2, sess2, imgs_input, imgs_label, 'ocr')
            feed = {model.inputs: imgs_input_after}
            dense_decoded_code = sess1.run(model.dense_decoded, feed)
            adv_acc += get_acc(imgs_label, dense_decoded_code)

            print("%s %s %s\n" % (
                type, prec_acc, adv_acc,))
