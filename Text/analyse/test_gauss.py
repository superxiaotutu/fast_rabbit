import tensorflow as tf
import test_LSTM_model as LSTM
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import random

from PIL import Image

from oppose_1000 import add_gauss

num_epochs = 2500
num_batches_per_epoch = 100
save_steps = 5000
validation_steps = 1000
batch_size = 1000
image_height = 60
image_width = 180
image_channel = 3

train_feeder = LSTM.DataIterator()
val_feeder = LSTM.DataIterator()


def infer(Checkpoint_PATH, img_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR("infer")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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
    im = cv2.imread(img_PATH).astype(np.float32) / 255.
    imgs_input = []
    imgs_input.append(im)

    imgs_input = np.asarray(imgs_input)
    imgs_input = np.repeat(imgs_input, batch_size, axis=0)

    feed = {model.inputs: imgs_input}
    dense_decoded_code = sess.run(model.dense_decoded, feed)
    expression = ''
    for i in dense_decoded_code[0]:
        if i == -1:
            expression += ''
        else:
            expression += LSTM.decode_maps[i]
    print(expression)


def infer_many(Checkpoint_PATH, dir_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = LSTM.LSTMOCR("infer")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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
    imgs = os.listdir(dir_PATH)


    for level in range(50):
        print(level)
        imgs_input = []
        imgs_name = []
        for index,img_name in enumerate(imgs):
            img = add_gauss(dir_PATH + img_name, level,show=False)
            # img = np.asarray(img)[:, :, :3].astype(np.float32) / 255.
            imgs_input.append(img)
            imgs_name.append(img_name[:-4])
            plt.imsave("images/adv/%s_%s.png"%(level,index),img)


        feed = {model.inputs: imgs_input}
        dense_decoded_code = sess.run(model.dense_decoded, feed)
        result = []
        acc = 0
        for index, j in enumerate(dense_decoded_code):
            expression = ''
            for i in j:
                if i == -1:
                    expression += ''
                else:
                    expression += LSTM.decode_maps[i]
            if expression == imgs_name[index]:
                acc += 1
            result.append(expression + ' ' + imgs_name[index])
        with open("%s_%s.txt",'a')as f:
            f.write("%s %s \n"%(acc,acc/len(imgs_input)))


def main():
    infer_many("train_gauss/model", "images/ori/")


if __name__ == '__main__':
    main()
