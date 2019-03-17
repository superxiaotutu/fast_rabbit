
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime

import cv2
num_epochs = 2500
num_batches_per_epoch = 100
save_steps = 5000
validation_steps = 1000
image_channel = 3
image_height = 64
image_width = 192
batch_size = 1
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

