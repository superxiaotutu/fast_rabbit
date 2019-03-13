import tensorflow as tf
import LSTM_model as LSTM
import time
import os
import datetime

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


def train(restore=False, checkpoint_dir="train_3/model"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
                with open(checkpoint_dir.replace('model', 'log') + '/test_acc.txt', 'a')as f:
                    f.write(str(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                           cur_epoch + 1, num_epochs, accuracy, avg_train_cost,
                                           err, time.time() - start_time, lr)) + "\n")
                print(log.format(now.month, now.day, now.hour, now.minute, now.second, cur_epoch + 1, num_epochs,
                                 accuracy, avg_train_cost, err, time.time() - start_time, lr))


if __name__ == '__main__':
    checkpoint_dir = "train_gauss/model"
    train(False, checkpoint_dir)
