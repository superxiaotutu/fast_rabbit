import Model as M
import tensorflow as tf
import os
import numpy as np
import random
from config import *
import matplotlib.pylab as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

CNN4 = M.CNN4_OCR('train')
RES = M.RESNET_OCR('train')
INCE = M.INCEPTIONNET_OCR('train')
DENSE = M.Dense_OCR('train')

data_train = M.DataIterator()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
var = tf.global_variables()


saver = tf.train.Saver(var)


def train():
    saver.restore(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/4ens_model/4ens.ckpt")

    for i in range(100):
        for j in range(100):
            data_train.modify_data()
            batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

            feed_CNN4 = {CNN4.inputs: batch_inputs, CNN4.labels: batch_labels}
            feed_RES = {RES.inputs: batch_inputs, RES.labels: batch_labels}
            feed_INCE = {INCE.inputs: batch_inputs, INCE.labels: batch_labels}
            feed_DENSE = {DENSE.inputs: batch_inputs, DENSE.labels: batch_labels}

            l1, _ = sess.run([CNN4.cost, CNN4.train_op], feed_dict=feed_CNN4)
            l2, _ = sess.run([RES.cost, RES.train_op], feed_dict=feed_RES)
            l3, _ = sess.run([INCE.cost, INCE.train_op], feed_dict=feed_INCE)
            l4, _ = sess.run([DENSE.cost, DENSE.train_op], feed_dict=feed_DENSE)

        print("i:{}: LOSS_CNN4:{}, LOSS_RES:{}, LOSS_INCEP:{}, LOSS_DENSE:{}".format(i, l1, l2, l3, l4))
        if i % 100 == 0:
            saver.save(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/4ens_model/4ens.ckpt")
    saver.save(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/4ens_model/4ens.ckpt")


def test():
    saver.restore(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/4ens_model/4ens.ckpt")
    batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

    feed_CNN4 = {CNN4.inputs: batch_inputs, CNN4.labels: batch_labels}
    feed_RES = {RES.inputs: batch_inputs, RES.labels: batch_labels}
    feed_INCE = {INCE.inputs: batch_inputs, INCE.labels: batch_labels}
    feed_DENSE = {DENSE.inputs: batch_inputs, DENSE.labels: batch_labels}
    l1 = sess.run(CNN4.cost, feed_dict=feed_CNN4)
    l2 = sess.run(RES.cost, feed_dict=feed_RES)
    l3 = sess.run(INCE.cost, feed_dict=feed_INCE)
    l4 = sess.run(DENSE.cost, feed_dict=feed_DENSE)

    print("LOSS_CNN4:{}, LOSS_RES:{}, LOSS_INCEP:{}, LOSS_DENSE:{}".format(l1, l2, l3, l4))

    model = DENSE

    batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

    plt.imshow(batch_inputs[0])
    plt.show()
    imgs_input = []
    imgs_input.append(batch_inputs[0])
    imgs_input = np.asarray(imgs_input)
    imgs_input = np.repeat(imgs_input, batch_size, axis=0)
    feed = {model.inputs: imgs_input}

    dense_decoded_code = sess.run(model.dense_decoded, feed)
    expression = ''
    for i in dense_decoded_code[0]:
        if i == -1:
            expression += '-'
        else:
            expression += M.decode_maps[i]
    print("INFER:{}".format(expression))
    return 0


def S2T(source_model, target_model, num):
    log_file=open('%s_%s_%s.log'%(source_model,target_model,num),'w')
    saver.restore(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/4ens_model/4ens.ckpt")

    target = tf.placeholder(tf.float32, [12, batch_size, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])

    if source_model != "ENS3":
        predict = tf.nn.softmax(source_model.logits)
        current_mengban = tf.transpose(tf.one_hot(tf.argmax(predict, axis=-1), 38, axis=0), [1, 2, 0])
        ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - source_model.inputs)) + tf.reduce_mean(
            tf.reduce_sum(predict * current_mengban) - tf.reduce_sum(predict * target))
        grad_y2x = tf.sign(tf.gradients(ADV_LOSS, source_model.inputs)[0])
    else:
        print("Source From ENS3!!!")
        predict_CNN = tf.nn.softmax(CNN4.logits)
        current_mengban_CNN = tf.transpose(tf.one_hot(tf.argmax(predict_CNN, axis=-1), 38, axis=0), [1, 2, 0])
        ADV_LOSS_CNN = tf.reduce_mean(tf.square(origin_inputs - CNN4.inputs)) + tf.reduce_mean(
            tf.reduce_sum(predict_CNN * current_mengban_CNN) - tf.reduce_sum(predict_CNN * target))
        grad_y2x_CNN = tf.sign(tf.gradients(ADV_LOSS_CNN, CNN4.inputs)[0])

        predict_RES = tf.nn.softmax(RES.logits)
        current_mengban_RES = tf.transpose(tf.one_hot(tf.argmax(predict_RES, axis=-1), 38, axis=0), [1, 2, 0])
        ADV_LOSS_RES = tf.reduce_mean(tf.square(origin_inputs - RES.inputs)) + tf.reduce_mean(
            tf.reduce_sum(predict_RES * current_mengban_RES) - tf.reduce_sum(predict_RES * target))
        grad_y2x_RES = tf.sign(tf.gradients(ADV_LOSS_RES, RES.inputs)[0])

        predict_INCE = tf.nn.softmax(INCE.logits)
        current_mengban_INCE = tf.transpose(tf.one_hot(tf.argmax(predict_INCE, axis=-1), 38, axis=0), [1, 2, 0])
        ADV_LOSS_INCE = tf.reduce_mean(tf.square(origin_inputs - INCE.inputs)) + tf.reduce_mean(
            tf.reduce_sum(predict_INCE * current_mengban_INCE) - tf.reduce_sum(predict_INCE * target))
        grad_y2x_INCE = tf.sign(tf.gradients(ADV_LOSS_INCE, INCE.inputs)[0])

        ADV_LOSS = (ADV_LOSS_CNN + ADV_LOSS_RES + ADV_LOSS_INCE)/3
        grad_y2x = (grad_y2x_CNN + grad_y2x_RES + grad_y2x_INCE)/3


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
    shuff_dir.update({0: creat_onehot(0)})

    def creat_one(model, img, sess):
        imgs_input = []
        imgs_input.append(img)
        imgs_input = np.asarray(imgs_input)
        imgs_input = np.repeat(imgs_input, batch_size, axis=0)

        imgs_input_before = imgs_input
        feed = {model.inputs: imgs_input}

        log = sess.run(model.logits, feed)
        fir = log[:, 0, :]
        ex = np.argmax(fir, axis=1)
        print('\n'+str(ex))

        target_creat = []
        for i in range(12):
            target_creat.append(shuff_dir[ex[i]])

        target_creat = np.asarray(target_creat)
        target_creat = target_creat[:, np.newaxis, :]
        target_creat = np.repeat(target_creat, batch_size, axis=1)

        dense_decoded_code = sess.run(model.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += M.decode_maps[i]
        print("BEFORE:{}".format(expression))
        bexpression = expression

        adv_step = 0.01
        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(50):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            # if (i + 1) % 10 == 0:
            #     print("LOSS:{}".format(loss_now))
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
                expression += M.decode_maps[i]
        print("AFTER:{}".format(expression))
        return imgs_input_after[0], bexpression

    def creat_one_from_ENS3(img, sess):
        imgs_input = []
        imgs_input.append(img)
        imgs_input = np.asarray(imgs_input)
        imgs_input = np.repeat(imgs_input, batch_size, axis=0)

        imgs_input_before = imgs_input
        feed = {CNN4.inputs: imgs_input}

        log = sess.run(CNN4.logits, feed)
        fir = log[:, 0, :]
        ex = np.argmax(fir, axis=1)
        print('\n' + str(ex))

        target_creat = []
        for i in range(12):
            target_creat.append(shuff_dir[ex[i]])

        target_creat = np.asarray(target_creat)
        target_creat = target_creat[:, np.newaxis, :]
        target_creat = np.repeat(target_creat, batch_size, axis=1)

        dense_decoded_code = sess.run(CNN4.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += M.decode_maps[i]
        print("BEFORE:{}".format(expression))
        bexpression = expression

        adv_step = 0.01
        feed = {CNN4.inputs: imgs_input, RES.inputs: imgs_input, INCE.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(50):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            if (i + 1) % 25 == 0:
                print("LOSS:{}".format(loss_now))
            imgs_input = imgs_input - grad * adv_step
            feed = {CNN4.inputs: imgs_input, RES.inputs: imgs_input, INCE.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}

        imgs_input_after = imgs_input

        feed = {CNN4.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        dense_decoded_code = sess.run(CNN4.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += M.decode_maps[i]
        print("AFTER:{}".format(expression))
        return imgs_input_after[0], bexpression

    def infer(model, img, G_label, sess):
        imgs_input = []
        imgs_input.append(img)
        imgs_input = np.asarray(imgs_input)
        imgs_input = np.repeat(imgs_input, batch_size, axis=0)
        feed = {model.inputs: imgs_input}

        dense_decoded_code = sess.run(model.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += M.decode_maps[i]
        print("INFER:{}".format(expression))
        if G_label == expression:
            return 1
        return 0

    if source_model != "ENS3":
        count = {}
        for T in target_model:
            count[str(T)] = 0
        for i in range(num):
            example_img, example_label = data_train.get_example()
            adv_img, ori_label = creat_one(source_model, example_img, sess)
            for T in target_model:
                count[str(T)] += infer(T, adv_img, ori_label, sess)
        for T in target_model:
            print("\nSource:{}, Target:{}, Num_Success:{}/{}".format(str(source_model), str(T), count[str(T)], num))
        log_file.write("\nSource:{}, Target:{}, Num_Success:{}/{}".format(str(source_model), str(T), count[str(T)], num))
    else:
        count = {}
        for T in target_model:
            count[str(T)] = 0
        for i in range(num):
            example_img, example_label = data_train.get_example()
            adv_img, ori_label = creat_one_from_ENS3(example_img, sess)
            for T in target_model:
                count[str(T)] += infer(T, adv_img, ori_label, sess)
        for T in target_model:
            print("\nSource:ENS3, Target:{}, Num_Success:{}/{}".format(str(T), count[str(T)], num))
        log_file.write("\nSource:{}, Target:{}, Num_Success:{}/{}".format(str(source_model), str(T), count[str(T)], num))

if __name__ == '__main__':
#     # train()
#     # test()
#     # source: CNN4, RES, INCE, ENS3, (DENSE)
#     # target: CNN4, RES, INCE, DENSE
    S2T(CNN4, [CNN4, RES, INCE, DENSE], 50)
