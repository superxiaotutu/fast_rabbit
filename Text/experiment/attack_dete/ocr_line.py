import glob
import shutil
import time

import tensorflow as tf
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import random
import sys

from tensorflow.contrib import slim

sys.path.append('../')
import model as LSTM
from config import *
from gen_type_codes import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
RELEASE = False
adv_step = 0.05
adv_count = 10
c = 100
radius = 0.8
log_file = open("log/%s.log" % datetime.datetime.now(), 'a')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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
    predict = tf.nn.softmax(model.logits)
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
    grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
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
            imgs_input = np.clip(imgs_input, 0, 1)
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


def cnn_generate(Checkpoint_PATH, model_name='cnn'):
    model = LSTM.LSTMOCR(model_name, "infer")
    model.build_graph()
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
    file_count = 1000
    with open('cnn_result.txt', 'w')as f:
        # ocr识别图片三种
        for type in range(4):
            adv_acc = 0
            prec_acc = 0
            img_files = glob.glob("../images/ori/*.png")
            for epoch in range(file_count // batch_size):
                ori_imgs_input = [img_files.pop() for i in range(batch_size)]
                print(len(ori_imgs_input))
                imgs_label = [i[-8:-4] for i in ori_imgs_input]
                imgs_input = get_process(ori_imgs_input, type)

                # attack
                distance, cost_time, imgs_input_after = attack(model, sess, imgs_input, imgs_label, 'cnn')
                if RELEASE:
                    for i, v in enumerate(imgs_input_after):
                        plt.imsave("../images/cnn_adv/%s_%s_%s_%s.png" % (type, epoch, i, imgs_label[i]), v)
                feed = {model.inputs: imgs_input_after}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                adv_acc += get_acc(imgs_label, dense_decoded_code)
                if RELEASE:
                    f.write(
                        "%s  %s %s\n" % (
                            type, prec_acc, adv_acc,))


def ocr_generate(Checkpoint_PATH, model_name='lenet'):
    adv_sample_dir='../images/ocr_adv'
    if os.path.isdir(adv_sample_dir):
        shutil.rmtree(adv_sample_dir)
        os.mkdir(adv_sample_dir)
    model = LSTM.LSTMOCR(model_name, "infer")
    model.build_graph()
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
    file_count = 1000
    with open('ocr_result.txt', 'w')as f:
        # -,G,B,gb
        for type in range(4):
            adv_acc = 0
            prec_acc = 0
            img_files = glob.glob("../images/ori/*.png")
            for epoch in range(file_count // batch_size):
                ori_imgs_input = [img_files.pop() for i in range(batch_size)]
                print(len(ori_imgs_input))
                imgs_label = [i[-8:-4] for i in ori_imgs_input]
                imgs_input = get_process(ori_imgs_input, type)
                imgs_input_before = imgs_input
                feed = {model.inputs: imgs_input_before}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                prec_acc += get_acc(imgs_label, dense_decoded_code)
                distance, cost_time, imgs_input_after = attack(model, sess, imgs_input, imgs_label, 'ocr')
                if RELEASE:
                    for i, v in enumerate(imgs_input_after):
                        plt.imsave("%s/%s_%s_%s_%s.png" % (adv_sample_dir,type, epoch, i, imgs_label[i]), v)

                log_file.write("distance: %s time: %ss" % (distance, cost_time))

                feed = {model.inputs: imgs_input_after}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                adv_acc += get_acc(imgs_label, dense_decoded_code)

                print("%s %s %s\n" % (
                    type, prec_acc, adv_acc,))
                if RELEASE:
                    f.write(
                        "%s %s %s\n" % (
                            type, prec_acc, adv_acc,))


def test_model(Checkpoint_PATH, model_name, head=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR(model_name, "infer")
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

    if model_name == 'lenet':
        model_name = 'ocr'
        taget_name = 'cnn'
    else:
        taget_name = 'ocr'
    for loop in range(0, 1):
        if loop == 0:
            img_files = glob.glob("../images/%s_adv/*.png" % model_name)
            filename = '%s_vs_%s_sample_%s_result.txt' % (model_name, model_name, 'head' if head else '')
        else:
            img_files = glob.glob("../images/%s_adv/*.png" % taget_name)
            filename = '%s_vs_%s_sample_%s_result.txt' % (model_name, taget_name, 'head' if head else '')
        acc = 0
        acc_0 = 0
        type_acc_arr = [0 for i in range(4)]
        with open(filename, 'w')as f:
            for index in range(4000 // batch_size):
                imgs_input = []
                imgs_label = []
                type_arr = []
                for im in img_files[index * batch_size:(index + 1) * batch_size]:
                    str_file = im[len("../images/%s_adv/" % model_name):-4]
                    arr_file = str_file.split('_')
                    type, label = arr_file[0], arr_file[3]
                    im = Image.open(im).convert("RGB")
                    if head:
                        # im = add_gauss(im,radius=0.5)
                        im = binary(im)

                    im = np.asarray(im).astype(np.float32) / 255.
                    type_arr.append(type)
                    imgs_label.append(label)
                    imgs_input.append(im)

                feed = {model.inputs: imgs_input}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                acc += get_acc(imgs_label, dense_decoded_code)
                # plt.imshow(imgs_input[index])
                # plt.show()

                print("True:{} BEFORE:{} ".format(imgs_label[index], acc))
                # print(acc)
                if RELEASE:
                    f.write(
                        "%s %s\n" % (acc, type_acc_arr))
            print(type_acc_arr)
            # break
            # plt.imshow(im)
            # plt.show()


def get_acc(imgs_label, dense_decoded_code):
    acc = 0
    for index, j in enumerate(dense_decoded_code):
        expression = ''
        for i in j:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        if imgs_label[index] == expression:
            # print(imgs_label[index], expression)
            acc += 1
        # print("True:{} ; AFTER:{}".format(imgs_label[index], expression))

    return acc


def get_process(ori_imgs_input, type):
    imgs_input = []
    if type == 0:
        for index, i in enumerate(ori_imgs_input):
            im = Image.open(i)
            im = np.asarray(im).astype(np.float32) / 255.
            imgs_input.append(im)
    if type == 1:
        for index, i in enumerate(ori_imgs_input):
            im = add_gauss(Image.open(i), radius=0.8)
            im = np.asarray(im).astype(np.float32) / 255.
            imgs_input.append(im)
    elif type == 2:
        for index, i in enumerate(ori_imgs_input):
            im = binary(Image.open(i))
            im = np.asarray(im).astype(np.float32) / 255.
            imgs_input.append(im)
    elif type == 3:
        for index, i in enumerate(ori_imgs_input):
            im = add_gauss(Image.open(i), radius=0.8)
            im = binary(im)
            im = np.asarray(im).astype(np.float32) / 255.
            imgs_input.append(im)
    return imgs_input


if __name__ == '__main__':
    ocr_generate('../train_lenet/model')
    # cnn_generate('../train_cnn/model')
    # test_model('../train_lenet/model', 'lenet')
    # test_model('../train_cnn/model', 'cnn')

    # test_model('../train_cnn/model', 'cnn', head=True)
    # test_model('../train_cnn/model', 'cnn', head=False)
