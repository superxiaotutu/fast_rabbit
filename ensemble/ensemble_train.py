import ensemble.Model as M
import tensorflow as tf

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

CNN4 = M.CNN4_OCR('train')
RES = M.RESNET_OCR('train')
INCE = M.INCEPTIONNET_OCR('train')

data_train = M.DataIterator()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


def train():
    f = open('log', 'w')
    for i in range(1000):
        for j in range(100):
            data_train.modify_data()
            batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

            feed_CNN4 = {CNN4.inputs: batch_inputs, CNN4.labels: batch_labels}
            feed_RES = {RES.inputs: batch_inputs, RES.labels: batch_labels}
            feed_INCE = {INCE.inputs: batch_inputs, INCE.labels: batch_labels}

            l1, _ = sess.run([CNN4.cost, CNN4.train_op], feed_dict=feed_CNN4)
            l2, _ = sess.run([RES.cost, RES.train_op], feed_dict=feed_RES)
            l3, _ = sess.run([INCE.cost, INCE.train_op], feed_dict=feed_INCE)
        f.writelines("i:{}: LOSS_CNN4:{}, LOSS_RES:{}, LOSS_INCEP:{}\n".format(i, l1, l2, l3))
        print("i:{}: LOSS_CNN4:{}, LOSS_RES:{}, LOSS_INCEP:{}".format(i, l1, l2, l3))
    saver.save(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/ens_model/")


def test():
    saver.restore(sess, "/home/kirin/Python_Code/Ensambel/fast_rabbit/ensemble/ens_model/ens")
    batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

    feed_CNN4 = {INCE.inputs: batch_inputs, INCE.labels: batch_labels}
    print(sess.run(INCE.cost, feed_CNN4))


if __name__ == '__main__':
    train()
    # test()
