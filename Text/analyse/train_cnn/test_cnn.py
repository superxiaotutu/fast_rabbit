# -*- coding:utf-8 -*-
import argparse
import datetime
import json
import numpy
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets.load_data import *
import os
import tensorflow.contrib.slim as slim
from datasets.constant import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))
FLAG = 0
x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
y_ = tf.placeholder(tf.int32, [None, NUM_PER_IMAGE * LABEL_SIZE])
keep_prob = tf.placeholder(tf.float32)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


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

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=log))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    variable_summaries(cross_entropy)

# forword prop
with tf.name_scope('forword-prop'):
    predict = tf.argmax(log, axis=2)
    expect = tf.argmax(y_expect_reshaped, axis=2)

# evaluate accuracy

with tf.name_scope('evaluate_accuracy'):
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_summaries(accuracy)

test_data = test_read_and_decode()
saver = tf.train.Saver()


def save():
    saver.save(sess, "model/model.ckpt")


def restore():
    saver.restore(sess, "model/model.ckpt")


with tf.Session(config=config) as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)
    # restore()
    print('restore')
    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 启动队列填充才可以是用batch
    threads = tf.train.start_queue_runners(sess, coord)
    sess.graph.finalize()

    for i in range(TEST_MAX_STEPS):
            # Test trained model
            test_batch = sess.run(test_data)
            test_batch[1] = read_labels(test_batch[1])
            print(test_batch[1])

            test_summary, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 0.8})
            test_writer.add_summary(test_summary, i)
            print('step %s, testing accuracy = %.2f%%' % (
                i,  test_accuracy * 100))

    train_writer.close()
    test_writer.close()

