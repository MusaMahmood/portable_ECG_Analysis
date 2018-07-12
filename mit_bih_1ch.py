# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os
import keras as k
import numpy as np
import tf_shared as tfs
import tensorflow as tf

from keras.backend import tensorflow_backend as tf_backend
from scipy.io import savemat
from keras.layers import Dense, Dropout, Reshape
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, CuDNNLSTM, LeakyReLU, PReLU
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization

# Setup:
num_channels = 1
num_classes = 5

output_folder = 'classify_data_out/n' + str(num_channels) + 'ch/'
version_num = 0
LSTM_UNITS = 32
learn_rate = 0.01
description = 'flex.conv1d_seq2seq_prescal_' + 'prelu_lstmU' + str(LSTM_UNITS) + 'lr' + str(learn_rate) + 'v0'
# description = 'seq2seq_only_prescal_' + 'lstmU' + str(LSTM_UNITS) + 'v3'
# description = 'cnn2layer' + 'U' + str(1024) + 'v0'
file_name = description
seq_length = 1000
if num_channels < 2:
    x_shape = [seq_length, 1]
    input_shape = seq_length
else:
    x_shape = [seq_length, 2]
    input_shape = (seq_length, num_channels)
y_shape = [1000, num_classes]

# Import Data:
x_tt, y_tt = tfs.load_data_v2('data/mit_bih_tlabeled_2ch_fixed', [seq_length, 2], y_shape, 'relevant_data', 'Y')
if num_channels < 2:
    x_tt = np.reshape(x_tt[:, :, 0], [-1, seq_length, 1])
xx_flex, y_flex = tfs.load_data_v2('data/flexEcg_1ch_invert', [seq_length, 1], [1], 'relevant_data', 'Y')
x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)  # 0.66


def get_model_cnn():
    k_model = Sequential()
    if num_channels < 2:
        k_model.add(Reshape((seq_length, num_channels, 1), input_shape=(input_shape, 1)))
    else:
        k_model.add(Reshape((seq_length, num_channels, 1), input_shape=input_shape))
    k_model.add(k.layers.Conv2D(128, (2, 2), strides=(2, 1), padding='same'))
    k_model.add(BatchNormalization())
    k_model.add(LeakyReLU(alpha=0.2))
    k_model.add(Dropout(0.2))
    k_model.add(k.layers.Conv2D(256, (8, 1), strides=(2, 1), padding='same'))
    k_model.add(BatchNormalization())
    k_model.add(LeakyReLU(alpha=0.2))
    k_model.add(Dropout(0.2))
    if num_channels < 2:
        k_model.add(Reshape(target_shape=(seq_length, 64)))
    else:
        k_model.add(Reshape(target_shape=(seq_length, num_channels * 64)))
    k_model.add(Dense(32, kernel_regularizer=regularizers.l2(l=0.01), input_shape=(seq_length, 1 * 128)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(k_model.summary())
    return k_model


def get_model_seq2seq():
    k_model = Sequential()
    if num_channels < 2:
        k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    else:
        k_model.add(Reshape((seq_length, num_channels), input_shape=input_shape))
    k_model.add(Dense(LSTM_UNITS, kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Reshape(target_shape=(seq_length, LSTM_UNITS)))
    k_model.add(Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(k_model.summary())
    return k_model


# Model:
def get_model_conv_seq2seq():
    k_model = Sequential()
    if num_channels < 2:
        k_model.add(Reshape((seq_length, num_channels, 1), input_shape=(input_shape, 1)))
    else:
        k_model.add(Reshape((seq_length, num_channels, 1), input_shape=input_shape))
    # TODO: Replace with Conv1D (?)
    k_model.add(k.layers.Conv2D(128, (8, 8), strides=(2, 1), padding='same', activation='relu'))
    k_model.add(k.layers.Conv2D(256, (8, 8), strides=(2, 1), padding='same', activation='relu'))
    k_model.add(k.layers.Conv2D(512, (2, 1), strides=(2, 1), padding='same', activation='relu'))
    if num_channels < 2:
        k_model.add(Reshape(target_shape=(seq_length, 64)))
    else:
        k_model.add(Reshape(target_shape=(seq_length, 128)))
    k_model.add(Dense(LSTM_UNITS, kernel_regularizer=regularizers.l2(l=0.01), input_shape=(seq_length, 1 * 128)))
    k_model.add(Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(k_model.summary())
    return k_model


def get_model_conv1d_seq2seq():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    k_model.add(k.layers.Conv1D(128, 8, strides=2, padding='same', activation='relu'))
    k_model.add(k.layers.Conv1D(256, 8, strides=2, padding='same', activation='relu'))
    k_model.add(k.layers.Conv1D(512, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Reshape(target_shape=(seq_length, 64)))
    k_model.add(Dense(LSTM_UNITS, kernel_regularizer=regularizers.l2(l=0.01), input_shape=(seq_length, 1 * 128)))
    k_model.add(Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(k_model.summary())
    return k_model


# Train:
batch_size = 512
epochs = 20

tf_backend.set_session(tfs.get_session(0.9))
with tf.device('/gpu:0'):
    start_time_ms = tfs.current_time_ms()
    # model = get_model_cnn()
    # model = get_model_seq2seq()
    # model = get_model_conv_seq2seq()
    model = get_model_conv1d_seq2seq()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)  # train the model
    model.save('model.h5')

    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
        score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        print('Test score: {} , Test accuracy: {}'.format(score, acc))

    # predict
    yy_probabilities = model.predict(x_test)
    # TODO: Change this and keep the probability:
    yy_predicted = tfs.maximize_output_probabilities(yy_probabilities)  # Maximize probabilities of prediction.

    # Evaluate other dataset:
    yy_probabilities_f = model.predict(xx_flex)
    yy_predicted_f = tfs.maximize_output_probabilities(yy_probabilities_f)  # Maximize probabilities of prediction.

    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)

    # data_dict = {'x_val': x_test, 'y_val': y_test, 'y_out': yy_predicted, 'y_prob': yy_probabilities}
    data_dict = {'x_val': x_test, 'y_val': y_test, 'y_out': yy_predicted, 'y_prob': yy_probabilities, 'x_flex': xx_flex,
                 'y_prob_flex': yy_probabilities_f, 'y_out_flex': yy_predicted_f}
    savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)

# Results: