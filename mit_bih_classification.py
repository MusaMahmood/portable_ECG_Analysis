# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os

import keras as k
import numpy as np
import tensorflow as tf
from keras import optimizers, regularizers
from keras.backend import tensorflow_backend as tf_backend
from keras.layers import Bidirectional, CuDNNLSTM, LSTM
from keras.layers import Dense, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from scipy.io import savemat
from sklearn.model_selection import train_test_split

import tf_shared as tfs

# Setup:
TRAIN = True
TEST = True
SAVE_HIDDEN = False
SAVE_PREDICTIONS = False
CUSTOM_EVALUATION = False
SAVE_ALT_MODEL_CPU = True
EXPORT_OPT_BINARY = True
VERSION_NUMBER = 2

epochs = 40
num_channels = 1
num_classes = 5
model_dir = "model_exports"
output_folder = 'classify_data_out/n' + str(num_channels) + 'ch/'
version_num = 0
LSTM_UNITS = 64
learn_rate = 0.01
description = 'normal_2cnn_fixed.conv1d_seq2seq_' + str(LSTM_UNITS) + 'lr' + str(learn_rate) + 'ep' + str(
    epochs) + '_v' + str(VERSION_NUMBER)
keras_model_name = description + '.h5'
if SAVE_ALT_MODEL_CPU:
    keras_model_name = 'alt_' + keras_model_name
file_name = description
seq_length = 2000
if num_channels < 2:
    x_shape = [seq_length, 1]
    input_shape = seq_length
else:
    x_shape = [seq_length, 2]
    input_shape = (seq_length, num_channels)
y_shape = [seq_length, num_classes]

# Start Timer:
start_time_ms = tfs.current_time_ms()

# Import Data:
x_tt, y_tt = tfs.load_data_v2('data/extended_5_class/mit_bih_tlabeled_w8s_fixed_all', [seq_length, 2], y_shape,
                              'relevant_data', 'Y')
if num_channels < 2:
    x_tt = np.reshape(x_tt[:, :, 0], [-1, seq_length, 1])
xx_flex, y_flex = tfs.load_data_v2('data/flexEcg_8s_normal', [seq_length, 1], [1], 'relevant_data', 'Y')
x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)  # 0.66

ptb_data_lead_ii = tfs.load_mat('_adversarial/data/lead_ii_all/all_y.mat', key='Y', shape=[seq_length, 1])


def get_model_alt():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    k_model.add(k.layers.Conv1D(128, 8, strides=2, padding='same', activation='relu'))
    k_model.add(k.layers.Conv1D(256, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Reshape(target_shape=(seq_length, 64)))
    k_model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)))
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


def get_model_conv1d_bilstm():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    k_model.add(k.layers.Conv1D(128, 8, strides=2, padding='same', activation='relu'))
    k_model.add(k.layers.Conv1D(256, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Reshape(target_shape=(seq_length, 64)))
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
batch_size = 256

tf_backend.set_session(tfs.get_session(0.9))
with tf.device('/gpu:0'):
    if TRAIN:
        if SAVE_ALT_MODEL_CPU:
            model = get_model_alt()
            print('Model Summary: \n', model.summary())
            adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        else:
            model = get_model_conv1d_bilstm()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)  # train the model
        model.save(keras_model_name)

    if os.path.isfile(keras_model_name):
        if not TRAIN:
            model = load_model(keras_model_name)
            print(model.summary())
            if TEST:
                score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
                print('Test score: {} , Test accuracy: {}'.format(score, acc))
        else:
            if TEST:
                score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
                print('Test score: {} , Test accuracy: {}'.format(score, acc))
    else:
        print("Model Not Found!")
        if not TRAIN:
            exit(-1)

    if SAVE_PREDICTIONS:
        # predict
        yy_probabilities = model.predict(x_test)
        yy_predicted = tfs.maximize_output_probabilities(yy_probabilities)  # Maximize probabilities of prediction.

        # Evaluate other dataset:
        yy_probabilities_f = model.predict(xx_flex)
        yy_predicted_f = tfs.maximize_output_probabilities(yy_probabilities_f)  # Maximize probabilities of prediction.

        data_dict = {'x_val': x_test, 'y_val': y_test, 'y_out': yy_predicted, 'y_prob': yy_probabilities,
                     'x_flex': xx_flex, 'y_prob_flex': yy_probabilities_f, 'y_out_flex': yy_predicted_f}
        savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)

    if SAVE_HIDDEN:
        # Evaluate hidden layers: # 'conv1d_3'
        # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
        layers_of_interest = ['conv1d_1', 'conv1d_2', 'reshape_2', 'bidirectional_1', 'dense_1', 'dense_2']
        np.random.seed(0)
        rand_indices = np.random.randint(0, x_test.shape[0], 250)
        print('Saving hidden layers: ', layers_of_interest)
        tfs.get_keras_layers(model, layers_of_interest, x_test[rand_indices], y_test[rand_indices],
                             output_dir='I:/_gan_data_backup/hidden_layers/', fname='hidden_all_' + file_name + '.mat')

    if CUSTOM_EVALUATION:
        rand_indices = np.random.randint(0, ptb_data_lead_ii.shape[0], 3500)
        ptb_sampling = ptb_data_lead_ii[rand_indices]
        yy_probabilities = model.predict(ptb_sampling)
        yy_predicted = tfs.maximize_output_probabilities(yy_probabilities)
        data_dict = {'x_val': ptb_sampling, 'y_prob': yy_probabilities, 'y_out': yy_predicted}
        savemat(tfs.prep_dir(output_folder + 'ptb/') + 'ptb_sample_data_probs.mat', mdict=data_dict)
    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)


# TODO: FIX:
if EXPORT_OPT_BINARY:
    tfs.export_model_keras(keras_model_name, export_dir=tfs.prep_dir("graph"), model_name=description, sequential=False)
