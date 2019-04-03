import ensemble.Model as M

CNN4 = M.CNN4_OCR()
RES = M.RESNET_OCR()
INCE = M.INCEPTIONNET_OCR()
data_train = M.DataIterator()

if __name__ == '__main__':
    for i in range(100):
        data_batch = 0
        batch_inputs, _, batch_labels, _ = data_train.input_index_generate_batch()

        feed_CNN4 = {CNN4.inputs: batch_inputs, CNN4.labels: batch_labels}
        l1, _ = CNN4.sess.run([CNN4.loss, CNN4.train_op], feed_dict=feed_CNN4)

        feed_RES = {RES.inputs: batch_inputs, RES.labels: batch_labels}
        l2, _ = RES.sess.run([RES.loss, RES.train_op], feed_dict=feed_RES)

        feed_INCE = {INCE.inputs: batch_inputs, INCE.labels: batch_labels}
        l3, _ = INCE.sess.run([INCE.loss, INCE.train_op], feed_dict=feed_INCE)
        print(l1, l2, l3)
