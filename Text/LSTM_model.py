import tensorflow as tf
import numpy as np
import random

from Text.image_process import gene_code

image_height = 60
image_width = 180
image_channel = 3
out_channels = 64
cnn_count = 4
leakiness = 0.01
num_hidden = 128
initial_learning_rate = 1e-3
num_classes = 36 + 2

decay_steps = 8000
decay_rate = 0.97
output_keep_prob = 0.8

batch_size = 1
LABEL_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LABEL_CHOICES_LIST = [str(i) for i in LABEL_CHOICES]
encode_maps = {}
decode_maps = {}
for i, char in enumerate(LABEL_CHOICES, 1):
    encode_maps[char] = i
    decode_maps[i] = char


SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape



class DataIterator:
    def __init__(self):
        self.image = []
        self.labels = []
        self.refresh_data()

    def refresh_data(self):
        self.image = []
        self.labels = []
        for num in range(batch_size):
            slice = random.sample(LABEL_CHOICES_LIST, 4)
            captcha = ''.join(slice)
            img = gene_code(captcha)
            img = np.asarray(img).astype(np.float32) / 255.
            code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
            self.labels.append(code)
            self.image.append(img)

    def modify_data(self):
        target = random.randint(0, batch_size - 1)
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        img = gene_code(captcha)
        img = np.asarray(img).astype(np.float32) / 255.
        code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
        self.image[target], self.labels[target] = img, code


    def get_test_img(self, num_line, num_point):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        img = gene_code(captcha)
        img = np.asarray(img).astype(np.float32) / 255.
        code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
        return img, captcha


    @property
    def size(self):
        return len(self.labels)

    def input_index_generate_batch(self):
        image_batch = self.image
        rar_label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([out_channels for _ in sequences], dtype=np.int64)
            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(rar_label_batch)
        return batch_inputs, batch_seq_len, batch_labels, rar_label_batch


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])

        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        # self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [3, 64, 128, 128, out_channels]
        strides = [1, 2]

        feature_h = image_height
        feature_w = image_width

        count_ = 0
        min_size = min(image_height, image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count_ += 1
        assert (cnn_count <= count_, "FLAGS.cnn_count should be <= {}!".format(count_))

        # CNN part
        with tf.variable_scope('cnn'):
            x = self.inputs
            for i in range(cnn_count):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
                    x = self._batch_norm('bn%d' % (i + 1), x)
                    x = self._leaky_relu(x, leakiness)
                    x = self._max_pool(x, 2, strides[1])

                    # print('----x.get_shape().as_list(): {}'.format(x.get_shape().as_list()))
                    _, feature_h, feature_w, _ = x.get_shape().as_list()
            print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))

        # LSTM part
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in lstm.
            x = tf.reshape(x, [batch_size, feature_w, feature_h * out_channels])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            # print('self.seq_len.shape: {}'.format(self.seq_len.shape.as_list()))

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)

            cell1 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, FLAGS.num_hidden]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]

            W = tf.get_variable(name='W_out',
                                shape=[num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        # self.global_step = tf.Variable(0, trainable=False)
        self.global_step = tf.train.get_or_create_global_step()

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   self.global_step,
                                                   decay_steps,
                                                   decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate).minimize(self.loss,
                                                                                      global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated=False)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=0.9,
                    center=True,
                    scale=True,
                    epsilon=1e-5,
                    updates_collections=None,
                    is_training=self.mode == 'train',
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=True,
                    scope='BatchNorm'
                )
        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        if isPrint:
            print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if origin_label == decoded_label:
            count += 1
    if isPrint:
        print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))
    return count * 1.0 / len(original_seq)

