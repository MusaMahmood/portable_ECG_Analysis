import os
import glob
import numpy as np
import ecg_data as ecg
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tf_backend
import keras as k

from keras.layers import Dense, Dropout  # Activation
from keras.layers import Bidirectional, CuDNNLSTM  # , LSTM, CuDNNGRU
# Try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(0)


def qtdb_load_dat(dat_files):
    # xx = []
    # yy = []
    for dat_file in dat_files:
        print(dat_file)
        if os.path.basename(dat_file).split(".", 1)[0] in exclude:
            continue
        qf = os.path.splitext(dat_file)[0] + '.q1c'
        if os.path.isfile(qf):
            # print("yes",qf,datfile)
            x, y = ecg.get_ecg_data(dat_file)
            x, y = ecg.remove_seq_gaps(x, y)
            x, y = ecg.splitseq(x, 1000, 150), ecg.splitseq(y, 1000, 150)
            # create equal sized numpy arrays of n size and overlap of o
            x = ecg.normalize_signal_array(x)
            # todo; add noise, shuffle leads etc. ?
            try:  # concat
                xx = np.vstack((xx, x))  # n, 1300, 2
                yy = np.vstack((yy, y))
            except NameError:  # if xx does not exist yet (on init)
                xx = x
                yy = y
    return xx, yy


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_session(gpu_fraction=0.9):
    # allocate % of gpu memory.
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def get_model_v2():
    model = Sequential()
    model.add(k.layers.Conv2D(32, (2, 2), strides=(2, 1), padding='same', activation='relu',
                              input_shape=(seqlength, features, 1)))
    return model


def get_model():
    model = Sequential()
    model.add(Dense(32, kernel_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
    model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
    # , input_shape=(seqlength, features)) ) # bidirectional ---><---
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(dimout, activation='softmax'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model


##################################################################
##################################################################
qtdbpath = "qtdb/"  # first argument = qtdb database from physionet.
perct = 0.81  # percentage training
percv = 0.19  # percentage validation

exclude = set()
exclude.update(
    ["sel35", "sel36", "sel37", "sel50", "sel102", "sel104", "sel221", "sel232", "sel310"])  # no P annotated:

# load data
datfiles = glob.glob(qtdbpath + "*.dat")
xxt, yyt = qtdb_load_dat(datfiles[:round(len(datfiles) * perct)])  # training data.
xxt, yyt = unison_shuffled_copies(xxt, yyt)  # shuffle
xxv, yyv = qtdb_load_dat(datfiles[-round(len(datfiles) * percv):])  # validation data.
seqlength = xxt.shape[1]
features = xxt.shape[2]
dimout = yyt.shape[2]
print("xxv/validation shape: {}, Seqlength: {}, Features: {}".format(xxv.shape[0], seqlength, features))

# call keras/tensorflow and build lstm model
tf_backend.set_session(get_session())
with tf.device('/gpu:0'):  # switch to /cpu:0 to use cpu
    # if not os.path.isfile('model.h5'):
    model = get_model()  # build model
    model.fit(xxt, yyt, batch_size=512, epochs=500, verbose=1)  # train the model
    model.save('model.h5')

    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
        score, acc = model.evaluate(xxv, yyv, batch_size=16, verbose=1)
        print('Test score: {} , Test accuracy: {}'.format(score, acc))

    # predict
    yy_predicted = model.predict(xxv)

    # maximize probabilities of prediction.
    for i in range(yyv.shape[0]):
        b = np.zeros_like(yy_predicted[i, :, :])
        b[np.arange(len(yy_predicted[i, :, :])), yy_predicted[i, :, :].argmax(1)] = 1
        yy_predicted[i, :, :] = b

    # plot:
    with PdfPages('ecg.pdf') as pdf:
        for i in range(xxv.shape[0]):
            print(i)
            ecg.plotecg_validation(xxv[i, :, :], yy_predicted[i, :, :], yyv[i, :, :], 0,
                                   yy_predicted.shape[1])  # top = predicted, bottom=true
            pdf.savefig()
            plt.close()
