import tensorflow.contrib.slim as slim
import tensorflow as tf

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 180
image_channel = 3
out_channels = 64
LABEL_SIZE = 36
NUM_PER_IMAGE = 4
output_keep_prob = 0.8
initial_learning_rate = 1e-3


class CNN(object):
    def __init__(self, mode):
        self.mode = mode
        self.inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, image_channel])
        self.labels = tf.placeholder(tf.int32, [None, NUM_PER_IMAGE * LABEL_SIZE])

        self._extra_train_ops = []

    def build_graph(self):
        self.merged_summay = tf.summary.merge_all()
        self._build_model()
        self._build_train_op()

    def _build_model(self):
        end_point = {}
        # resized = end_point['resized'] = tf.reshape(self.inputs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        # tf.summary.image('input', resized, max_outputs=LABEL_SIZE)
        conv1 = end_point['conv1'] = slim.conv2d(self.inputs, 32, 3, padding='SAME', activation_fn=tf.nn.relu)
        pooling1 = end_point['pool1'] = slim.max_pool2d(conv1, 2)
        conv2 = end_point['conv2'] = slim.conv2d(pooling1, 64, 3, padding='SAME', activation_fn=tf.nn.relu)
        pooling2 = end_point['pool2'] = slim.max_pool2d(conv2, 2)
        flatten1 = end_point['flatten1'] = slim.flatten(pooling2)
        full1 = end_point['full1'] = slim.fully_connected(flatten1, 1024, activation_fn=tf.nn.relu)
        drop_out = end_point['drop_out'] = slim.dropout(full1, output_keep_prob)
        full2 = end_point['full2'] = slim.fully_connected(drop_out, NUM_PER_IMAGE * LABEL_SIZE, activation_fn=None)
        self.logits = end_point['logits'] = tf.reshape(full2, [-1, NUM_PER_IMAGE, LABEL_SIZE])
        predict = end_point['predict'] = tf.nn.softmax(self.logits )

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        # self.labels = tf.reshape(y_, [-1, NUM_PER_IMAGE, LABEL_SIZE])
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

        self.cost = tf.reduce_mean(self.loss)
        self.lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   self.global_step,
                                                   decay_steps,
                                                   decay_rate,
                                                   staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate).minimize(self.loss ,
                                                                                      global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
        #
        # # forword prop
        # with tf.name_scope('forword-prop'):
        #     predict = tf.argmax(log, axis=2)
        #     expect = tf.argmax(y_expect_reshaped, axis=2)
        #
        # # evaluate accuracy
        #
        # with tf.name_scope('evaluate_accuracy'):
        #     correct_prediction = tf.equal(predict, expect)
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     variable_summaries(accuracy)
        #
        # end, log, pre = CNN(x, output_keep_prob)