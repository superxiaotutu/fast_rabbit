import re
import sys,os
sys.path.append('../')
from config import *
from one_char_model import *
from gen_type_codes import *
import tensorflow as tf
import one_char_model as LSTM
from gen_type_codes import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = LSTMOCR("test")
model.build_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
Var_restore = tf.global_variables()
saver = tf.train.Saver(Var_restore, max_to_keep=5, allow_empty=True)
acc = 0
levels = [i for i in range(50)]
result = [0 for i in range(50)]
Checkpoint_PATH='/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/train_one_normal/train_one_char/model'
ckpt = tf.train.latest_checkpoint(Checkpoint_PATH)
if ckpt:
    saver.restore(sess, ckpt)
    print('restore from ckpt{}'.format(ckpt))
else:
    print('cannot restore')
imgs_input = []
labels_arr=[]
img_dir='/home/kaiyuan_xu/PycharmProjects/fast_rabbit/Text/analyse/adv_attack/simple_adv/'
print(len(os.listdir(img_dir)))

for i in os.listdir(img_dir):
    label_inputs = []
    label_inputs.append(i[-5])
    img=Image.open(img_dir+i)
    imgs_input = np.asarray(img)[:,:,:3]
    b_bin = imgs_input.copy()
    # b_bin[b_bin >=255//2] = 255
    # b_bin[b_bin < 255//2] = 0
    b_bin[b_bin >=np.mean(b_bin)] = 255
    b_bin[b_bin < np.mean(b_bin)] = 0
    b_bin=b_bin.astype(np.float32) / 255
    feed = {model.inputs: [b_bin]}
    plt.imshow(b_bin)
    plt.show()
    level = re.search('_(.+)_',i).group(1)
    print(i[-5])
    # print(level)
    # plt.imsave("adv/%s_%s_%s.png" % (index1, level, label_inputs[0]), imgs_input[0])

    dense_decoded_code = sess.run(model.dense_decoded, feed)
    for index, j in enumerate(dense_decoded_code):
        expression = ''
        for i in j:
            if i == -1:
                expression += ''
            else:
                expression += LSTM.decode_maps[i]
        if expression == label_inputs[0]:
            acc += 1
            result[int(level)] += 1
    print(acc, label_inputs[0] , expression)
print(result)

[50, 0, 0, 0, 0, 50, 0, 0, 0, 0, 50, 0, 0, 0, 0, 50, 0, 0, 0, 0, 38, 0, 0, 0, 0, 24, 0, 0, 0, 0, 8, 0, 0, 0, 0, 7, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
