import os
import glob
import numpy as np
import ecg_data as ecg
import tf_shared as tfs
import tensorflow as tf
import keras.backend.tensorflow_backend as tf_backend
import keras as k

# Try TimeDistributed(Dense(...))
from scipy.io import savemat
from keras.layers import Dense, Dropout  # Activation
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, CuDNNLSTM, CuDNNGRU  # , LSTM, CuDNNGRU
from keras.layers.normalization import BatchNormalization

np.random.seed(0)

output_folder = 'data_out/'
version_num = 0
LSTM_UNITS = 32
description = 'seq2seq_prescal_' + 'lstmU' + str(LSTM_UNITS)
file_name = description
use_orig_model = False
prescale_data = True
ch1_only = True
gru = False

train_batch_size = 588  # for more complex models use 256


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


def get_model_seq2seq():
    model = Sequential()
    if ch1_only:
        model.add(k.layers.Reshape((seq_length, features, 1), input_shape=(input_shape,)))
    else:
        model.add(k.layers.Reshape((seq_length, features, 1), input_shape=input_shape))
    model.add(k.layers.Conv2D(128, (2, 2), strides=(2, 1), padding='same', activation='relu'))
    model.add(k.layers.Conv2D(256, (8, 1), strides=(2, 1), padding='same', activation='relu'))
    model.add(k.layers.Reshape(target_shape=(seq_length, 1 * 64)))
    model.add(Dense(LSTM_UNITS, kernel_regularizer=regularizers.l2(l=0.01), input_shape=(seq_length, 1 * 128)))
    if gru:
        model.add(Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True)))
    else:
        model.add(Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(6, activation='softmax'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model


def get_model():
    model = Sequential()
    if ch1_only:
        model.add(k.layers.Reshape((seq_length, features), input_shape=(input_shape,)))
        model.add(Dense(32, kernel_regularizer=regularizers.l2(l=0.01)))
    else:
        model.add(Dense(32, kernel_regularizer=regularizers.l2(l=0.01), input_shape=input_shape))
    model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
    # , input_shape=(seq_length, features)) ) # bidirectional ---><---
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(dim_out, activation='softmax'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model


qtdbpath = "qtdb/"  # first argument = qtdb database from physionet.
perct = 0.81  # percentage training
percv = 0.19  # percentage validation

exclude = set()
exclude.update(["sel35", "sel36", "sel37", "sel50", "sel102", "sel104", "sel221", "sel232", "sel310"])  # no P annotated

# load data
datfiles = glob.glob(qtdbpath + "*.dat")
xxt, yyt = ecg.qtdb_load_dat(datfiles[:round(len(datfiles) * perct)], exclude, prescale_data)  # training data.
xxt, yyt = unison_shuffled_copies(xxt, yyt)  # shuffle
xxv, yyv = ecg.qtdb_load_dat(datfiles[-round(len(datfiles) * percv):], exclude, prescale_data)  # validation data.

seq_length = xxt.shape[1]
dim_out = yyt.shape[2]

if ch1_only:
    xxt = xxt[:, :, 0]
    xxv = xxv[:, :, 0]
    features = 1
    input_shape = seq_length
else:
    features = xxt.shape[2]
    input_shape = (seq_length, features)

if ch1_only:
    xx_flex, yy_flex = tfs.load_data('flexEcg_bilstm_test2', [1300], key_x='relevant_data', key_y='Y')
else:
    xx_flex = np.zeros([1, 1300, 2])

print("xxv/validation shape: {}, Seqlength: {}, Features: {}".format(xxv.shape[0], seq_length, features))

# call keras/tensorflow and build lstm model
tf_backend.set_session(get_session())
with tf.device('/gpu:0'):  # switch to /cpu:0 to use cpu
    start_time_ms = tfs.current_time_ms()
    if use_orig_model:
        model = get_model()
    else:
        model = get_model_seq2seq()

    model.fit(xxt, yyt, batch_size=train_batch_size, epochs=500, verbose=1)  # train the model
    model.save('model.h5')

    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
        score, acc = model.evaluate(xxv, yyv, batch_size=16, verbose=1)
        print('Test score: {} , Test accuracy: {}'.format(score, acc))

    # predict
    yy_probabilities = model.predict(xxv)
    # TODO: Change this and keep the probability:
    yy_predicted = tfs.maximize_output_probabilities(yy_probabilities)  # Maximize probabilities of prediction.

    # Evaluate other dataset:
    yy_probabilities_f = model.predict(xx_flex)
    yy_predicted_f = tfs.maximize_output_probabilities(yy_probabilities_f)  # Maximize probabilities of prediction.

    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)

    if ch1_only:
        data_dict = {'x_val': xxv, 'y_val': yyv, 'y_out': yy_predicted, 'y_flex': yy_predicted_f, 'x_flex': xx_flex,
                     'y_prob': yy_probabilities, 'y_flex_prob': yy_probabilities_f}
    else:
        data_dict = {'x_val': xxv, 'y_val': yyv, 'y_out': yy_predicted}
    savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)

    # plot:
    # with PdfPages(output_folder + 'ecg' + file_name + '.pdf') as pdf:
    #     for i in range(xxv.shape[0]):
    #         print(i)
    #         # TODO: Save as .mat file, (as well as PDF).
    #         # ecg.plotecg_validation(xxv[i, :, :], yy_predicted[i, :, :], yyv[i, :, :], 0, yy_predicted.shape[1])
    #         ecg.plotecg_validation(xxv[i, :], yy_predicted[i, :], yyv[i, :], 0, yy_predicted.shape[1])
    #         # top = predicted, bottom=true
    #         pdf.savefig()
    #         plt.close()
