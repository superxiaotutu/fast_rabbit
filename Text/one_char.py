import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.image
import tensorflow.contrib.slim as slim

# from image_process import gene_codeµ

image_height = 64
image_width = 192
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

batch_size = 32
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

from PIL import Image, ImageFont, ImageDraw, ImageFilter
from captcha.image import ImageCaptcha

image = ImageCaptcha(width=image_width, height=image_height)


def gene_code(chars):
    def random_color(start, end, opacity=None):
        red = random.randint(start, end)
        green = random.randint(start, end)
        blue = random.randint(start, end)
        if opacity is None:
            return (red, green, blue)
        return (red, green, blue, opacity)

    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))
    im = image.create_captcha_image(chars, color, background)
    image.create_noise_dots(im, color, number=30)
    image.create_noise_curve(im, color)
    im = im.filter(ImageFilter.SMOOTH)
    return im


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
            slice = random.sample(LABEL_CHOICES_LIST, 1)
            captcha = ''.join(slice)
            img = gene_code(captcha)
            img = np.asarray(img).astype(np.float32) / 255.
            code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
            self.labels.append(code)
            self.image.append(img)

    def modify_data(self):
        target = random.randint(0, batch_size - 1)
        slice = random.sample(LABEL_CHOICES_LIST, 1)
        captcha = ''.join(slice)
        img = gene_code(captcha)
        img = np.asarray(img).astype(np.float32) / 255.
        code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
        self.image[target], self.labels[target] = img, code

    def get_test_img(self, num_line, num_point):
        slice = random.sample(LABEL_CHOICES_LIST, 1)
        captcha = ''.join(slice)
        img = gene_code(captcha)
        img = np.asarray(img).astype(np.float32) / 255.
        code = [SPACE_INDEX if captcha == SPACE_TOKEN else encode_maps[c] for c in list(captcha)]
        return img, captcha

    @property
    def size(self):
        return len(self.labels)

    def input_original_batch(self):
        return self.image, self.labels

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
        self.labels = tf.sparse_placeholder(tf.int32)
        self._extra_train_ops = []

    def build_graph(self):
        self._bulid_CNN_with_FC()
        self.merged_summay = tf.summary.merge_all()


    def _bulid_CNN_with_FC(self):
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

            self.logits = tf.reshape(y1, [batch_size, 1, num_classes])
            self.log_prob = tf.reshape(sy1, [batch_size, 1, num_classes])

            self.dense_decoded = tf.arg_max(self.logits, dimension=2)

        self.global_step = tf.train.get_or_create_global_step()

        true_label = tf.sparse_tensor_to_dense(self.labels, default_value=0)
        one_hot_true_label = tf.one_hot(true_label, depth=num_classes, axis=1)
        one_hot_true_label = tf.transpose(one_hot_true_label, [0, 2, 1])

        self.loss = slim.losses.softmax_cross_entropy(y1, one_hot_true_label[:, 0, :])
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
