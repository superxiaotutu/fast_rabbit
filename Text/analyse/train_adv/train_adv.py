import tensorflow as tf
import tensorflow.contrib.image
import tensorflow.contrib.slim as slim
import time
from ...config import *
# from image_process import gene_code
image_channel = 3
out_channels = 64
cnn_count = 4
leakiness = 0.01
num_hidden = 128
initial_learning_rate = 0.001
num_classes = 36 + 2

decay_steps = 4000
decay_rate = 0.96
output_keep_prob = 0.8

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
            img = gen(captcha)
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
        # if LSTM:
        # self._build_model()
        # self._build_model_with_resnet()
        self._build_model_with_inception()
        self._build_train_op()
        # else:
        # self._bulid_CNN_with_4_FC()

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
            print(outputs.shape)

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
            print(self.logits)

            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            print(self.logits)

            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))
            print(self.logits)

    def _build_model_with_resnet(self):
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
        with tf.variable_scope('mini-resnet'):
            x = self.inputs
            x = self._conv2d(x, 'first', 3, 3, 4, 1)
            conv1 = self._residual_block(x, 4, if_first=True, name='block1')
            conv2 = self._residual_block(conv1, 8, name='block2')
            conv3 = self._residual_block(conv2, 16, name='block3')
            conv4 = self._residual_block(conv3, 32, name='block4')
            conv5 = self._residual_block(conv4, 64, name='block5')

            x = conv5
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
            print(outputs.shape)

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
            print(self.logits)

            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            print(self.logits)

            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))
            print(self.logits)

    def _build_model_with_inception(self):
        count_ = 0
        min_size = min(image_height, image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count_ += 1
        assert (cnn_count <= count_, "FLAGS.cnn_count should be <= {}!".format(count_))

        # CNN part
        with tf.variable_scope('mini-inception'):
            x = self.inputs
            x = self._conv2d(x, 'cnn1', 3, 3, 4, 1)
            x = self._batch_norm('bn1', x)
            x = self._leaky_relu(x, leakiness)
            x = self._max_pool(x, 2, 2)

            x = self._conv2d(x, 'cnn2', 3, 4, 8, 1)
            x = self._batch_norm('bn2', x)
            x = self._leaky_relu(x, leakiness)
            x = self._max_pool(x, 2, 2)

            block_1 = self._inception_block(x, 8, 'block1')
            block_1 = self._max_pool(block_1, 2, 2)

            block_2 = self._inception_block(block_1, 120, 'block2a')
            block_2 = self._max_pool(block_2, 2, 2)

            x = self._conv2d(block_2, 'cnn3', 3, 120, 64, 1)
            x = self._batch_norm('bn3', x)
            x = self._leaky_relu(x, leakiness)

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
            print(outputs.shape)

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
            print(self.logits)

            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            print(self.logits)

            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))
            print(self.logits)

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

    def _bulid_CNN_with_4_FC(self):
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

        with tf.variable_scope('4fc'):
            x = tf.reshape(x, [batch_size, feature_w * feature_h * out_channels])

            _, feature_l = x.get_shape().as_list()
            print('\nfeature_l: {}'.format(feature_l))

            y1, sy1 = self._bulid_fc(x, num_classes, '1')
            y2, sy2 = self._bulid_fc(x, num_classes, '2')
            y3, sy3 = self._bulid_fc(x, num_classes, '3')
            y4, sy4 = self._bulid_fc(x, num_classes, '4')

            self.logits = tf.concat(axis=1, values=[y1, y2, y3, y4])
            self.log_prob = tf.concat(axis=1, values=[sy1, sy2, sy3, sy4])

            self.logits = tf.reshape(self.logits, [batch_size, 4, num_classes])
            self.log_prob = tf.reshape(self.log_prob, [batch_size, 4, num_classes])

            self.dense_decoded = tf.arg_max(self.logits, dimension=2)

        self.global_step = tf.train.get_or_create_global_step()

        true_label = tf.sparse_tensor_to_dense(self.labels, default_value=0)
        one_hot_true_label = tf.one_hot(true_label, depth=num_classes, axis=1)
        one_hot_true_label = tf.transpose(one_hot_true_label, [0, 2, 1])

        self.loss = slim.losses.softmax_cross_entropy(y1, one_hot_true_label[:, 0, :]) + \
                    slim.losses.softmax_cross_entropy(y2, one_hot_true_label[:, 1, :]) + \
                    slim.losses.softmax_cross_entropy(y3, one_hot_true_label[:, 2, :]) + \
                    slim.losses.softmax_cross_entropy(y4, one_hot_true_label[:, 3, :])

        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   self.global_step,
                                                   decay_steps,
                                                   decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate).minimize(self.loss,
                                                                                      global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

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

    def _avg_pool(self, x, ksize, strides):
        return tf.nn.avg_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='avg_pool')

    def _residual_block(self, input_layer, output_channel, if_first=False, name=None):
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        with tf.variable_scope('conv1_in_block'):
            if if_first:
                x = self._conv2d(input_layer, 'cnn1' + name, 3, input_channel, output_channel, 1)
            else:
                x = self._conv2d(input_layer, 'cnn1' + name, 3, input_channel, output_channel, stride)
                x = self._batch_norm('bn1' + name, x)
                x = self._leaky_relu(x, leakiness)

        with tf.variable_scope('conv2_in_block'):
            x = self._conv2d(x, 'cnn2' + name, 3, output_channel, output_channel, 1)
            x = self._batch_norm('bn2' + name, x)
            conv2 = self._leaky_relu(x, leakiness)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def _inception_block(self, input, input_channel, name=None):
        branch_0 = self._conv2d(input, name + 'Conv2d_0a_1x1', 1, input_channel, 32, 1)

        branch_1 = self._conv2d(input, name + 'Conv2d_1a_1x1', 1, input_channel, 32, 1)
        branch_1 = self._batch_norm(name + "branch_1_bn", branch_1)
        branch_1 = self._leaky_relu(branch_1, leakiness)
        branch_1 = self._conv2d(branch_1, name + 'Conv2d_1b_3x3', 3, 32, 48, 1)

        branch_2 = self._conv2d(input, name + 'Conv2d_2a_1x1', 1, input_channel, 8, 1)
        branch_2 = self._batch_norm(name + "branch_2_bn", branch_2)
        branch_2 = self._leaky_relu(branch_2, leakiness)
        branch_2 = self._conv2d(branch_2, name + 'Conv2d_2b_3x3', 3, 8, 24, 1)

        branch_3 = self._max_pool(input, 3, 1)
        branch_3 = self._conv2d(branch_3, name + 'Conv2d_3b_1x1', 1, input_channel, 16, 1)

        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        net = self._batch_norm(name + "net_bn", net)
        net = self._leaky_relu(net, leakiness)
        return net

    def _bulid_fc(self, x, out_num, name):
        with tf.variable_scope('fc' + name):
            x = slim.layers.fully_connected(x, 512)
            x = slim.layers.fully_connected(x, 256)
            x = slim.layers.fully_connected(x, out_num, None)
            after_softmax_x = slim.softmax(x)
        return x, after_softmax_x

    def head_B(self, input):
        phi = 0.5
        S = 1 / (1 + tf.exp(-(input - phi)))
        return S

    def head_Guss(self, input):
        def getGuessValue(kerStd, posX, posY):
            return 1. / (2. * np.pi * (np.power(kerStd, 2))) * np.exp(
                -(np.power(posX, 2) + np.power(posY, 2)) / (2. * (np.power(kerStd, 2))))

        def getGuessKernel(kerStd):
            K11 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K12 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K13 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K21 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K22 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K23 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K31 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K32 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K33 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            kernel = tf.constant(np.array([[K11, K12, K13], [K21, K22, K23], [K31, K32, K33]]),
                                 dtype=tf.float32)  # 3*3*4*4
            return kernel

        kernel = getGuessKernel(0.8)
        Guss = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return Guss


train_feeder = DataIterator()
val_feeder = DataIterator()


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
    model = LSTMOCR("infer")
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    target = tf.placeholder(tf.float32, [12, batch_size, 38])
    origin_inputs = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
    predict = tf.nn.softmax(model.logits)
    # ADV_LOSS = tf.reduce_sum(tf.square(predict - target)) + tf.reduce_mean(tf.square(origin_inputs - model.inputs))
    current_status = tf.argmax(predict, axis=-1)
    current_mengban = tf.one_hot(current_status, 38, axis=0)
    current_mengban = tf.transpose(current_mengban, [1, 2, 0])

    ADV_LOSS = tf.reduce_mean(tf.square(origin_inputs - model.inputs)) + tf.reduce_mean(
        tf.square(tf.reduce_sum(predict * current_mengban) - tf.reduce_sum(predict * target)))

    grad_y2x = tf.sign(tf.gradients(ADV_LOSS, model.inputs)[0])

    Var_restore = tf.global_variables()
    saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)

    im = cv2.imread(img_PATH).astype(np.float32) / 255.
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
            expression += decode_maps[i]
    print("BEFORE:{}".format(expression))

    adv_step = 0.01
    feed = {model.inputs: imgs_input, target: target_creat, origin_inputs: imgs_input_before}
    for i in range(50):
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
            expression += decode_maps[i]
    print("AFTER:{}".format(expression))

    test = (sess.run([predict, current_status, current_mengban], feed_dict=feed))

    plt.imshow(imgs_input_after[0])
    plt.show()
    return


def train(restore=False, checkpoint_dir="train_/model"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = LSTMOCR('train')
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
                    acc = accuracy_calculation(val_rar_label, dense_decoded, ignore_value=-1, isPrint=False)
                    acc_batch_total += acc
                    lastbatch_err += err

                accuracy_calculation(val_rar_label, dense_decoded, ignore_value=-1, isPrint=True)

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
