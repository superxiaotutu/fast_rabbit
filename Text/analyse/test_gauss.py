import tensorflow as tf
import LSTM_model as LSTM
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib.pyplot as plt
import random

num_epochs = 2500
batch_size = 128
num_batches_per_epoch = 100
save_steps = 5000
validation_steps = 1000

image_height = 60
image_width = 180
image_channel = 3

train_feeder = LSTM.DataIterator()
val_feeder = LSTM.DataIterator()


def infer(Checkpoint_PATH, img_PATH):
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



def infer_many(Checkpoint_PATH, img_PATH):
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
    p = [('adv_example/%s.png' % i) for i in range(66,76)]
    imgs_input = []
    for img_PATH in p:
        im = cv2.imread(img_PATH).astype(np.float32) / 255.
        imgs_input.append(im)

    imgs_input = np.asarray(imgs_input)
    # imgs_input = np.repeat(imgs_input, batch_size, axis=0)

    feed = {model.inputs: imgs_input}
    dense_decoded_code = sess.run(model.dense_decoded, feed)
    expression = ''
    print(dense_decoded_code)
    for j in dense_decoded_code:
        for i in j:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        expression+=','
    print(expression)


def main():

    infer("train_salt/model", "test.png")


if __name__ == '__main__':
    main()
