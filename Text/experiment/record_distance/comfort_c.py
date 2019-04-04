import glob
import time
import tensorflow as tf
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import random
import sys
from tensorflow.contrib import slim
from attack_dete.ocr_line import update_matrix

sys.path.append('../')
import model as LSTM
from config import *
from gen_type_codes import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = LSTM.LSTMOCR('lenet', "infer", 'ori')
model.build_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
Checkpoint_PATH = '/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/experiment/train_lenet/model'
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


def attack(imgs_input, imgs_label, c=20):
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
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    # predict = tf.nn.softmax(model.logits)
    predict = model.logits
    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])
    node_num = 12
    axis = 1
    ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + c * tf.reduce_mean(
        tf.reduce_sum(predict * OCR_target) - tf.reduce_sum(predict * current_mengban))
    is_attacked_matrix = np.ones((batch_size, image_height, image_width, image_channel))
    # grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
    grad_y2x = tf.gradients(ADV_LOSS, model.inputs)[0]
    imgs_input_before = imgs_input
    feed = {model.inputs: imgs_input}
    log = sess.run(model.logits, feed)
    fir = log[:, 0, :]
    ex = np.argmax(fir, axis=1)
    target_creat = []
    start = time.time()
    for j in range(node_num):
        target_creat.append(shuff_dir[ex[j]])
    for k in range(6):
        target_creat[k] = target_creat[k] - 1
    target_creat=np.clip(target_creat,0,1)
    target_creat = np.asarray(target_creat)
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
        print(dense_decoded_code[0])
        is_attacked_matrix = update_matrix(is_attacked_matrix, imgs_label, dense_decoded_code)
        if is_attacked_matrix.any() == 0:
            break
        feed = {model.inputs: imgs_input, OCR_target: target_creat, origin_inputs: imgs_input_before}
    imgs_input_after = imgs_input
    end = time.time()
    distance = np.linalg.norm(imgs_input_after - imgs_input_before)
    cost_time = end - start
    for i in range(batch_size):
        plt.imshow(imgs_input_after[i])
        plt.show()
        break
    return distance, cost_time, imgs_input_after


def multiply_c(c):
    log_file = open("log/comfort_c=%s.log" % c, 'w')
    img_files = glob.glob('/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/experiment/images/ori/*.png')
    print(img_files)
    for im in img_files:
        label = im[-8:-4]
        im = Image.open(im).convert("RGB")
        im = np.asarray(im).astype(np.float32) / 255.
        feed = {model.inputs: [im]}
        dense_decoded_code = sess.run(model.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        print(expression, label)
        if expression == label:
            distance, cost_time, imgs_input_after = attack([im], [label], c)
            log_file.write("dis:%s cost_time:%s c:%s" % (distance, cost_time, c))
            break
    log_file.close()


if __name__ == '__main__':
    for i in range(5, 35):
        multiply_c(i)
        break
