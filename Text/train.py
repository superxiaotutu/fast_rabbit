import glob
import shutil

import tensorflow as tf
from PIL import Image

import LSTM_model as LSTM
import time
import numpy as np
import os
import datetime
import cv2
import matplotlib.pyplot as plt
import random
from skimage.transform import resize

from gen_type_codes import gene_code_clean, add_gauss, add_gauss_code

num_epochs = 500
batch_size = 1
num_batches_per_epoch = 100
save_steps = 5000
validation_steps = 1000

image_height = 64
image_width = 192
image_channel = 3
# [[34  6 10 24]]
train_feeder = LSTM.DataIterator()
val_feeder = LSTM.DataIterator()
c = 20
adv_step = 0.015
adv_count = 20
gauss_level = 1


def train(restore=False, checkpoint_dir="train/model"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR('train')
    model.build_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    Var_restore = tf.global_variables()

    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)

    train_writer = tf.summary.FileWriter(checkpoint_dir.replace('model', 'log'), sess.graph)
    acc_sum = tf.Summary()

    if restore:
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            # the global_step will restore sa well
            saver.restore(sess, ckpt)
            print('restore from checkpoint{0}'.format(ckpt))

    print('=============================begin training=============================')
    # sess.graph.finalize()

    for cur_epoch in range(num_epochs):
        train_cost = 0
        start_time = time.time()
        batch_time = time.time()

        # the training part
        for cur_batch in range(num_batches_per_epoch):
            if (cur_batch + 1) % 100 == 0:
                print('batch', cur_batch, ': time', time.time() - batch_time)
            if (cur_batch + 1) % 2 == 0:
                train_feeder.modify_data()

            batch_time = time.time()

            batch_inputs, _, batch_labels, _ = train_feeder.input_index_generate_batch()

            feed = {model.inputs: batch_inputs, model.labels: batch_labels}

            # if summary is needed
            summary_str, batch_cost, step, _ = sess.run(
                [model.merged_summay, model.cost, model.global_step, model.train_op], feed)
            # calculate the cost
            train_cost += batch_cost * batch_size

            train_writer.add_summary(summary_str, step)

            # save the checkpoint
            if step % save_steps == 1:
                saver.save(sess, checkpoint_dir + '/ocr-model', global_step=step // 1000)

            # do validation
            if step % validation_steps == 0:
                acc_batch_total = 0
                lastbatch_err = 0
                lr = 0
                for j in range(2):
                    val_feeder.refresh_data()
                    val_inputs, _, val_labels, val_rar_label = val_feeder.input_index_generate_batch()
                    val_feed = {model.inputs: val_inputs, model.labels: val_labels}

                    dense_decoded, err, lr = sess.run([model.dense_decoded, model.cost, model.lrn_rate], val_feed)
                    # print the decode result
                    acc = LSTM.accuracy_calculation(val_rar_label, dense_decoded, ignore_value=-1, isPrint=False)
                    acc_batch_total += acc
                    lastbatch_err += err

                LSTM.accuracy_calculation(val_rar_label, dense_decoded, ignore_value=-1, isPrint=True)

                accuracy = acc_batch_total / 2
                acc_sum.value.add(tag='acc', simple_value=accuracy)
                train_writer.add_summary(acc_sum, global_step=step)

                avg_train_cost = err / 2

                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "accuracy = {:.3f}, avg_train_cost = {:.3f}, " \
                      "lastbatch_err = {:.3f}, time = {:.3f}, lr={:.8f}"
                print(log.format(now.month, now.day, now.hour, now.minute, now.second, cur_epoch + 1, num_epochs,
                                 accuracy, avg_train_cost, err, time.time() - start_time, lr))


def infer(Checkpoint_PATH, img_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR("train")
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

    batch_inputs, _, batch_labels, _ = train_feeder.input_index_generate_batch()

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


def creat_adv(Checkpoint_PATH, img_PATH, type):
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

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = LSTM.LSTMOCR("train", type)

    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # adv_node
    target = tf.placeholder(tf.float32, [12, batch_size, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    predict = tf.nn.softmax(model.logits)

    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])

    ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + tf.reduce_mean(
        tf.reduce_sum(predict * current_mengban) - tf.reduce_sum(predict * target))

    grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])

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

    for img in img_PATH:
        im = cv2.imread(img).astype(np.float32) / 255.
        im = cv2.resize(im, (192, 64))

        imgs_input = []
        imgs_input.append(im)
        imgs_input = np.asarray(imgs_input)
        imgs_input = np.repeat(imgs_input, batch_size, axis=0)

        imgs_input_before = imgs_input
        feed = {model.inputs: imgs_input}

        log = sess.run(model.logits, feed)
        fir = log[:, 0, :]
        ex = np.argmax(fir, axis=1)
        print(ex)

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
                expression += '-'
            else:
                expression += LSTM.decode_maps[i]
        print("BEFORE:{}".format(expression))

        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
        for i in range(adv_count):
            loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
            if (i + 1) % 10 == 0:
                print("LOSS:{}".format(loss_now))
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
                expression += LSTM.decode_maps[i]
        print("AFTER:{}".format(expression))

        test = (sess.run([predict, current_status, current_mengban], feed_dict=feed))
        plt.imshow(imgs_input_after[0])
        onehot_out = tf.one_hot(tf.argmax(model.logits, axis=2), 38)
        tar = np.ones([12, batch_size, 38])
        tar[:, :, 0] = 0
        tar[:, :, 37] = 0
        tar_tensor = tf.convert_to_tensor(tar, dtype=tf.float32)
        onehot_out = tar_tensor * onehot_out
        gradcam = grad_cam(model.attention_pool, model.logits, onehot_out)
        feed = {model.inputs: imgs_input_after}
        g_img = sess.run(gradcam, feed_dict=feed)
        ori_img = imgs_input_after[0]

        g_img[0] /= np.abs(np.max(g_img[0]))
        prehot_img = resize(g_img[0], (64, 192))

        prehot_img /= prehot_img.max()
        heatmap = cv2.applyColorMap(np.uint8(255 * prehot_img), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        alpha = 0.0072
        heatmap = ori_img + alpha * heatmap
        heatmap /= heatmap.max()
        plt.imshow(heatmap)
        plt.show()
    return


def grad_cam(pool, init, Y):
    conv_layer = pool
    signal = tf.multiply(init, Y)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    weights = tf.reduce_mean(norm_grads, axis=(1, 2))
    weights = tf.expand_dims(weights, 1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, 4, 12, 1])

    pre_cam = tf.multiply(weights, conv_layer)
    cam = tf.reduce_sum(pre_cam, 3)
    cam = tf.nn.relu(cam)
    # effect
    cam = tf.sqrt(cam)
    # cam = tf.sqrt(cam)
    return cam


def GRADCAM_infer(Checkpoint_PATH, img_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR("train")
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
    for i, img in enumerate(img_PATH):
        im = cv2.imread(img).astype(np.float32) / 255.
        imgs_input = []
        imgs_input.append(im)

        imgs_input = np.asarray(imgs_input)
        imgs_input = np.repeat(imgs_input, batch_size, axis=0)

        onehot_out = tf.one_hot(tf.argmax(model.logits, axis=2), 38)
        tar = np.ones([12, batch_size, 38])
        tar[:, :, 0] = 0
        tar[:, :, 37] = 0
        tar_tensor = tf.convert_to_tensor(tar, dtype=tf.float32)
        onehot_out = tar_tensor * onehot_out
        gradcam = grad_cam(model.attention_pool, model.logits, onehot_out)
        feed = {model.inputs: imgs_input}
        g_img = sess.run(gradcam, feed_dict=feed)
        ori_img = imgs_input[0]
        g_img[0] /= np.abs(np.max(g_img[0]))
        prehot_img = resize(g_img[0], (64, 192))
        prehot_img /= prehot_img.max()
        heatmap = cv2.applyColorMap(np.uint8(255 * prehot_img), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        alpha = 0.0072
        heatmap = ori_img + alpha * heatmap
        heatmap /= heatmap.max()
        plt.axis('off')
        plt.imshow(heatmap)
        # plt.imsave("%s/%s.png",i)
        plt.show()


LABEL_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LABEL_CHOICES_LIST = [str(i) for i in LABEL_CHOICES]


def attack(sess, model, imgs_input):
    # adv_node
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
    target = tf.placeholder(tf.float32, [12, batch_size, 38])
    CNN_target = tf.placeholder(tf.float32, [batch_size, 4, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    predict = tf.nn.softmax(model.logits)
    # ADV_LOSS = tf.reduce_sum(tf.square(predict - target)) + tf.reduce_mean(tf.square(origin_inputs - model.inputs))
    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])
    ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + c * tf.reduce_mean(
        tf.reduce_sum(predict * target) - tf.reduce_sum(predict * current_mengban))

    # grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])
    grad_y2x = tf.gradients(ADV_LOSS, model.inputs)[0]

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
    feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
    for i in range(adv_count):
        loss_now, grad = sess.run([ADV_LOSS, grad_y2x], feed)
        if (i + 1) % 10 == 0:
            print("LOSS:{},{}".format(np.max(grad), sess1))
        imgs_input = imgs_input - grad * adv_step
        imgs_input = np.clip(imgs_input, 0, 1)
        feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
    imgs_input_after = imgs_input
    print('ok')
    return imgs_input_after


def get_cam(img_input, grad_img):
    ori_img = img_input
    grad_img[0] /= np.abs(np.max(grad_img[0]))
    prehot_img = resize(grad_img[0], (64, 192))
    prehot_img /= prehot_img.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * prehot_img), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    heatmap = ori_img + alpha * heatmap
    heatmap /= heatmap.max()
    return heatmap


def throsh_binary(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval, image = cv2.threshold(c, 0, 255, cv2.THRESH_OTSU)
    image = Image.fromarray(image).convert('RGB')
    return image


def get_process(ori_imgs_input):
    im = throsh_binary(ori_imgs_input)
    im = add_gauss(im, radius=2)
    im = np.asarray(im).astype(np.float32) / 255.
    return im


def get_ori_grads(Checkpoint_PATH, img_PATH):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with sess1.as_default():
        with g1.as_default():
            imgs_input_ori = np.asarray(cv2.imread(img_PATH[0])).astype(np.float32) / 255.
            onehot_out = tf.one_hot(tf.argmax(model1.logits, axis=2), 38)
            tar = np.ones([12, batch_size, 38])
            tar[:, :, 0] = 0
            tar[:, :, 37] = 0
            tar_tensor = tf.convert_to_tensor(tar, dtype=tf.float32)
            onehot_out = tar_tensor * onehot_out
            gradcam = grad_cam(model1.attention_pool, model1.logits, onehot_out)
            feed = {model1.inputs: [imgs_input_ori]}
            # ori_att
            g_img = sess1.run(gradcam, feed_dict=feed)
            heatmap_ori = get_cam(imgs_input_ori, g_img)

            plt.axis('off')
            plt.imshow(heatmap_ori)
            plt.imsave("%s/%s" % ('images/ori_att',
                                  img_PATH[0][len('images/ori/'):]), heatmap_ori)

            # gauss
            img = add_gauss_code(Image.open(img_PATH[0]), level=gauss_level)
            gauss_file = "%s/%s" % ('images/gauss',
                                    img_PATH[0][len('images/ori/'):])
            plt.imsave(gauss_file, img)

            imgs_input = np.asarray(cv2.imread(gauss_file)).astype(np.float32) / 255.
            feed = {model1.inputs: [imgs_input]}
            g_img = sess1.run(gradcam, feed_dict=feed)
            heatmap_gauss = get_cam(imgs_input_ori, g_img)
            plt.axis('off')
            plt.imshow(heatmap_gauss)
            plt.imsave("%s/%s" % ('images/gauss_att',
                                  img_PATH[0][len('images/ori/'):]), heatmap_gauss)

            # they
            processed = get_process(Image.open(img_PATH[0]))
            they_attack_img = attack(sess1, model1, [processed])
            feed = {model1.inputs: they_attack_img}
            g_img = sess1.run(gradcam, feed_dict=feed)
            heatmap_they = get_cam(they_attack_img[0], g_img)
            plt.axis('off')
            plt.imshow(heatmap_they)

    # they

    # our
    with sess2.as_default():
        with g2.as_default():
            onehot_out = tf.one_hot(tf.argmax(model2.logits, axis=2), 38)
            tar = np.ones([12, batch_size, 38])
            tar[:, :, 0] = 0
            tar[:, :, 37] = 0
            tar_tensor = tf.convert_to_tensor(tar, dtype=tf.float32)
            onehot_out = tar_tensor * onehot_out
            gradcam = grad_cam(model2.attention_pool, model2.logits, onehot_out)
            our_attack_img = attack(sess2, model2, [processed])
            feed = {model2.inputs: our_attack_img}
            g_img = sess2.run(gradcam, feed_dict=feed)
            heatmap_our = get_cam(our_attack_img[0], g_img)
            plt.axis('off')
            plt.imshow(heatmap_our)
    plt.subplot(2, 4, 1)
    plt.axis('off')
    plt.imsave("ori.png", Image.open(img_PATH[0]))
    plt.imshow(Image.open(img_PATH[0]))

    plt.subplot(2, 4, 2)
    plt.axis('off')
    plt.imsave("gauss.png", Image.open(gauss_file))
    plt.imshow(Image.open(gauss_file))

    plt.subplot(2, 4, 3)
    plt.axis('off')
    our_attack_img = cv2.cvtColor(our_attack_img[0], cv2.COLOR_BGR2RGB)
    plt.imsave("our_adv.png", our_attack_img)
    plt.imshow(our_attack_img)

    plt.subplot(2, 4, 4)
    plt.axis('off')
    they_attack_img = cv2.cvtColor(they_attack_img[0], cv2.COLOR_BGR2RGB)
    plt.imsave("they_adv.png", they_attack_img)
    plt.imshow(they_attack_img)

    plt.subplot(2, 4, 5)
    plt.axis('off')
    plt.imsave("ori_att.png", heatmap_ori)
    plt.imshow(heatmap_ori)

    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.imsave("gauss_att.png", heatmap_gauss)
    plt.imshow(heatmap_gauss)

    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.imsave("our_att.png", heatmap_our)
    plt.imshow(heatmap_our)

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.imsave("they_att.png", heatmap_they)

    plt.imshow(heatmap_they)

    plt.show()


def gen_clean():
    dirname = 'images/ori/'
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    # 生成干净样本
    for i in range(batch_size):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        gene_code_clean(captcha).save('%s/%s_%s.png' % (dirname, i, captcha))


def gen_gauss_clean():
    dirname = 'images/gauss/'
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    # 生成高斯样本
    for i in range(batch_size):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        img = gene_code_clean(captcha)
        img = add_gauss_code(img, level=gauss_level)
        plt.imsave('%s/%s_%s.png' % (dirname, i, captcha), img)


# check_model = '/home/kirin/Python_Code/Ensambel/fast_rabbit/Text/train/model'
check_model = '/home/kirin/Python_Code/fast_rabbit _xky/train_model/train_lenet_fine/model'

g1 = tf.Graph()
g2 = tf.Graph()

sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)
with sess1.as_default():
    with g1.as_default():
        model1 = LSTM.LSTMOCR('train', "ori")
        model1.build_graph()
        Var_restore = tf.global_variables()
        saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
        sess1 = tf.Session()
        sess1.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(check_model)
        if ckpt:
            saver.restore(sess1, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
with sess2.as_default():
    with g2.as_default():
        model2 = LSTM.LSTMOCR('train', "all")
        model2.build_graph()
        Var_restore = tf.global_variables()
        saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
        sess2 = tf.Session()
        sess2.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(check_model)
        if ckpt:
            saver.restore(sess2, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
gen_clean()
gauss_flienames = glob.glob('images/gauss/*.png')
ori_flienames = glob.glob('images/ori/*.png')
our_ori_flienames = glob.glob('images/ori_adv/*.png')
common_ori_flienames = glob.glob('images/common_adv/*.png')

if __name__ == '__main__':
    # 原图的attention
    get_ori_grads(check_model, ori_flienames)
    #
    # # 高斯的attention
    # GRADCAM_infer(check_model, gauss_flienames)
    #
    # # 我們考慮預處理的方法的對抗樣本attention
    # creat_adv(check_model, ori_flienames,'all')
    #
    # # 普通的方法的對抗樣本attention
    # creat_adv(check_model, ori_flienames,'ori')
