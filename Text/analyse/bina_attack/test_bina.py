import sys,os
sys.path.append('../')
from config import *
from LSTM_clean_model import DataIterator
from LSTM_clean_model import LSTMOCR
from gen_type_codes import *
import tensorflow as tf
plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = LSTMOCR("test", "lenet")
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

grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


def get_adv_target(feed):
    log = sess.run(model.logits, feed)
    fir = log[:, 0, :]
    ex = np.argmax(fir, axis=1)
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
    target_creat = []

    for i in range(12):
        target_creat.append(shuff_dir[ex[i]])
    target_creat = np.asarray(target_creat)
    target_creat = target_creat[:, np.newaxis, :]
    target_creat = np.repeat(target_creat, batch_size, axis=1)
    return target_creat


def adv_many(Checkpoint_PATH, img_PATH):
    Var_restore = tf.global_variables()
    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from ckpt{}'.format(ckpt))
    else:
        print('cannot restore')
        return
    imgs_input = []
    labels_input = []
    for i in img_PATH:
        labels_input.append(i[1])
        im = np.asarray(i[0]).astype(np.float32) / 255.
        imgs_input.append(im)
    imgs_input = np.asarray(imgs_input)
    feed = {model.inputs: imgs_input}
    # dense_decoded_code = sess.run(model.dense_decoded, feed)
    #
    # pred_arr = []
    # for j in dense_decoded_code:
    #     expression = ''
    #     for i in j:
    #         if i == -1:
    #             expression += ''
    #         else:
    #             expression += decode_maps[i]
    #     pred_arr.append(expression)

    imgs_input_before = imgs_input
    target_creat = get_adv_target(feed)

    adv_step = 0.02
    for level in [i for i in range(0, 50, 49)]:
        acc = 0
        imgs_input = imgs_input_before
        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(level):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            imgs_input = imgs_input - grad * adv_step
            feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}

        # imgs_input_after = binary_arr(imgs_input)
        imgs_input_after = (imgs_input)
        plt.imshow(imgs_input[0])
        plt.imsave("images/bin_adv_%s.png" % level, np.clip(imgs_input[0], 0, 1))

        feed = {model.inputs: imgs_input_after, target: target_creat, origin_inputs: imgs_input_before}



        dense_decoded_code = sess.run(model.dense_decoded, feed)
        attack_arr = []
        for j in dense_decoded_code:
            expression = ''
            for i in j:
                if i == -1:
                    expression += ''
                else:
                    expression += decode_maps[i]
            attack_arr.append(expression)
        for true_l, adv_l in zip(labels_input, attack_arr):
            if (true_l == adv_l):
                acc += 1
            print(adv_l, true_l)
        with open('res.txt', 'a')as f:
            f.write(str(acc) + "\n")
        print(acc)


def main():
    arr = []

    adv_many('/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/train_one_normal/train_one_char/model')


if __name__ == '__main__':
    main()
