import glob
import tensorflow as tf

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import sys
sys.path.append('../')
import model as LSTM
from config import *
from gen_type_codes import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def cnn_generate(Checkpoint_PATH, model_name):
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

    model = LSTM.LSTMOCR(model_name, "infer")
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
    adv_step = 0.1
    adv_count = 100
    file_count = 1000
    with open('result.txt', 'w')as f:
        for type in range(4):
            acc = 0
            img_files = glob.glob("../images/ori/*.png")
            for epoch in range(file_count  // batch_size):
                ori_imgs_input = [img_files.pop() for i in range(batch_size)]
                print(len(ori_imgs_input))
                imgs_label = [i[-8:-4] for i in ori_imgs_input]
                imgs_input = []
                if type == 0:
                    for i in ori_imgs_input:
                        im = Image.open(i)
                        im = np.asarray(im).astype(np.float32) / 255.
                        imgs_input.append(im)
                if type == 1:
                    for i in ori_imgs_input:
                        im = add_gauss(Image.open(i))
                        im = np.asarray(im).astype(np.float32) / 255.
                        imgs_input.append(im)
                elif type == 2:
                    for i in ori_imgs_input:
                        im = binary(Image.open(i))
                        im = np.asarray(im).astype(np.float32) / 255.
                        imgs_input.append(im)
                elif type == 3:
                    for i in ori_imgs_input:
                        im = add_gauss(Image.open(i))
                        im = binary(im)
                        im = np.asarray(im).astype(np.float32) / 255.
                        imgs_input.append(im)
                imgs_input_before = imgs_input
                feed = {model.inputs: imgs_input}
                log = sess.run(model.logits, feed)
                fir = log[:, 0, :]
                ex = np.argmax(fir, axis=1)
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
                feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
                for i in range(adv_count):
                    loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
                    # if (i + 1) % 10 == 0:
                    #     print("LOSS:{}".format(np.max(grad)))
                    imgs_input = imgs_input - grad * adv_step
                    feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
                imgs_input_after = imgs_input
                feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
                for index, j in enumerate(dense_decoded_code):
                    expression = ''
                    for i in j:
                        if i == -1:
                            expression += ''
                        else:
                            expression += LSTM.decode_maps[i]
                    if expression == imgs_label[index]:
                        acc += 1

                    print("BEFORE:{} ,AFTER:{}".format(imgs_label[index], expression))
                    print(acc)

                    f.write("%s %s %s %s \n" % (type, imgs_label[index], expression, acc,))
                plt.subplot(1, 2, 1)
                plt.imshow(imgs_input_before[0])
                plt.subplot(1, 2, 2)
                plt.imshow(imgs_input_after[0])
                plt.show()
                # plt.show()
                # plt.imshow(imgs_input_after[0])
                # plt.show()
                # return


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
    cnn_generate('../train_lenet/model', 'lenet')
    # test_model('../train_cnn/model', )
