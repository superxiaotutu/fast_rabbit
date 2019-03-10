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

image_height = 64
image_width = 192
image_channel = 3

train_feeder = LSTM.DataIterator()
val_feeder = LSTM.DataIterator()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def train(restore=False, cnn_name="lenet"):
    checkpoint_dir = "train_%s/model"%cnn_name
    model = LSTM.LSTMOCR('train','inception')
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
                with open(checkpoint_dir.replace('model', 'log') + '/test_acc.txt', 'a')as f:
                    f.write(str(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                           cur_epoch + 1, num_epochs, accuracy, avg_train_cost,
                                           err, time.time() - start_time, lr)) + "\n")
                print(log.format(now.month, now.day, now.hour, now.minute, now.second, cur_epoch + 1, num_epochs,
                                 accuracy, avg_train_cost, err, time.time() - start_time, lr))


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


def creat_adv(Checkpoint_PATH, img_PATH):
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
    model = LSTM.LSTMOCR("infer")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # adv_node
    target = tf.placeholder(tf.float32, [12, 128, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    predict = tf.nn.softmax(model.logits)
    ADV_LOSS = tf.reduce_sum(tf.square(predict - target)) + tf.reduce_mean(tf.square(origin_inputs - model.inputs))
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

    im = cv2.imread(img_PATH).astype(np.float32) / 255.

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

    adv_step = 0.01
    feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
    for i in range(30):
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

    plt.imshow(imgs_input_after[0])
    plt.show()
    return


def test():
    num_test = 128
    t = 0
    train_feeder.refresh_data()
    for e in range(num_test // batch_size):
        for i in range(batch_size):
            plt.imsave("temp.png", train_feeder.image[i])
            expression = creat_adv("train_1/model", "temp.png")
            if expression != train_feeder.labels[i]:
                t += 1
            print()
    print(t / num_test)


def darw_table(Checkpoint_PATH):
    img_table = np.zeros([10, 10])

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTM.LSTMOCR("infer")
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

    for i in range(10):
        for j in range(10):
            count = 0
            for k in range(100):
                im, la = train_feeder.get_test_img(i * 2, j * 10)

                imgs_input = []
                imgs_input.append(im)
                imgs_input = np.asarray(imgs_input)
                imgs_input = np.repeat(imgs_input, batch_size, axis=0)

                feed = {model.inputs: imgs_input}

                dense_decoded_code = sess.run(model.dense_decoded, feed)
                expression = ''
                for c in dense_decoded_code[0]:
                    if c == -1:
                        expression += ''
                    else:
                        expression += LSTM.decode_maps[c]
                if expression == la:
                    count += 1
                    # print(expression, la)
            img_table[i, j] = count / 100
            print("i:{}, j:{}, p={}".format(i, j, img_table[i, j]))
    np.save("table2.npy", img_table)


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
    train(restore=False,cnn_name="resnet")
    # infer("train_all/model", "example/1.png")
    # creat_adv("train_3/model", "example/2.png")
    # test()
    # darw_table("train_2/model")


if __name__ == '__main__':
    main()
