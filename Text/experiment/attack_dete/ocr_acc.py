import glob
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
model = LSTM.LSTMOCR('lenet', "infer", "all")
model.build_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
RELEASE = False


def attack(sess, imgs_input,type='ocr'):
    # adv_node
    shuff_dir = {}
    lst = [i for i in range(36)]
    random.shuffle(lst)

    def creat_onehot(num):
        re = np.zeros([38])
        re[num - 1] = 1
        return re

    for i in range(batch_size):
        plt.imshow(imgs_input[i])
        plt.show()
        break
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
            feed = {model.inputs: imgs_input, OCR_target: target_creat, origin_inputs: imgs_input_before}
    else:
        target_creat = target_creat[np.newaxis, :, :]
        target_creat = np.repeat(target_creat, batch_size, axis=axis)
        feed = {model.inputs: imgs_input, CNN_target: target_creat, origin_inputs: imgs_input_before}
        for i in range(adv_count):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            if (i + 1) % 2 == 0:
                print("LOSS:{}".format(loss_now))
            imgs_input = imgs_input - grad * adv_step
            imgs_input = np.clip(imgs_input, 0, 1)
            dense_decoded_code = sess.run(model.dense_decoded, {model.inputs: imgs_input})
            feed = {model.inputs: imgs_input, CNN_target: target_creat, origin_inputs: imgs_input_before}
    imgs_input_after = imgs_input
    for i in range(batch_size):
        plt.imshow(imgs_input_after[i])
        plt.show()
        break
    return imgs_input_after

def test_model(Checkpoint_PATH):
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
    type_acc_arr = [0 for i in range(4)]
    adv_acc = 0
    adv_arr = [0 for i in range(4)]
    for level in range(2, 4):
        if level == 0:
            dir_name = "../images/ori/*.png"
        else:
            dir_name = "../images/ori_type_%s/*.png" % level
        img_files = glob.glob(dir_name)
        filename = 'ocr_accuracy.txt'
        print(img_files, dir_name)
        acc = 0
        with open(filename, 'w')as f:
            for index in range(file_count // batch_size):
                imgs_input = []
                imgs_label = []
                type_arr = []
                for im in img_files[index * batch_size:(index + 1) * batch_size]:
                    label = im[-8:-4]
                    im = Image.open(im).convert("RGB")
                    im = np.asarray(im).astype(np.float32) / 255.
                    imgs_label.append(label)
                    imgs_input.append(im)
                feed = {model.inputs: imgs_input}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                for index, j in enumerate(dense_decoded_code):
                    expression = ''
                    for i in j:
                        if i == -1:
                            expression += ''
                        else:
                            expression += LSTM.decode_maps[i]
                    if expression == imgs_label[index]:
                        type_acc_arr[level] += 1
                        acc += 1
                imgs_input_after = attack(sess, imgs_input)
                feed = {model.inputs: imgs_input_after}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                for index, j in enumerate(dense_decoded_code):
                    expression = ''
                    for i in j:
                        if i == -1:
                            expression += ''
                        else:
                            expression += LSTM.decode_maps[i]
                    if expression == imgs_label[index]:
                        adv_arr[level] += 1
                        print("True:{} BEFORE:{} ".format(imgs_label[index], expression))
                f.write(
                    "%s %s \n" % (
                        adv_acc, type_acc_arr,))
            print(adv_arr)
            print(type_acc_arr)
            # break
            # plt.imshow(im)
            # plt.show()


# test_model('/home/kirin/Python_Code/fast_rabbit _xky/train_model/train_cnn/model')
test_model('/home/kirin/Python_Code/fast_rabbit _xky/train_model/train_lenet_fine/model')
