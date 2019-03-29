import re
import tensorflow as tf
import gauss_model as LSTM
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from config import *
import random
import re
from PIL import Image


train_feeder = LSTM.DataIterator()
val_feeder = LSTM.DataIterator()



def infer_many(Checkpoint_PATH, dir_PATH):
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
    imgs = os.listdir(dir_PATH)
    imgs_input = []
    true_labels = []
    levels=[]

    for index,img_name in enumerate(imgs[:batch_size]):
        if img_name.endswith('png'):
            img = Image.open(dir_PATH+img_name).convert('RGB')
            img = np.asarray(img)[:, :, :3].astype(np.float32) / 255.
            assert  img.shape == (64, 48, 3)
            imgs_input.append(img)
            true_labels.append(img_name[-5:-4])
            level = re.search('_(.+)_', img_name).group(1)
            levels.append(level)

    feed = {model.inputs: imgs_input}
    dense_decoded_code = sess.run(model.dense_decoded, feed)
    result = [0 for i in range(50)]
    acc = 0
    for index, j in enumerate(dense_decoded_code):
        expression = ''
        for i in j:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        if expression == true_labels[index]:
            acc += 1
            result[int(levels[index])]+=1
    print(acc,result)

    with open("rees.txt",'a')as f:
        f.write("%s\n"%(str(result)))


def main():
    infer_many("/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/train_one_normal/train_one_char/model", "images/")


if __name__ == '__main__':
    main()
