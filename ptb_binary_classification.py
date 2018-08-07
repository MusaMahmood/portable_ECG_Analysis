# Rat ECG Classification using Deep Neural Networks:
# Copied from mit_bih_classification - for 2 classes
# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os

import numpy as np
import tensorflow as tf
from keras import optimizers, regularizers
from keras.backend import tensorflow_backend as tf_backend
from keras.layers import Dense, Dropout, Reshape, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tf_shared as tfs

# Setup:
TRAIN = True
SAVE_HIDDEN = False
IND_TO_VEC = False
SAVE_FROZEN_MODEL = False
SAVE_TEST_DATA = False
VERSION_NUMBER = 0
epochs = 50
num_channels = 1
num_classes = 2
model_dir = "model_exports"
output_folder = 'classify_data_out/rat_n' + str(num_channels) + 'ch/'
learn_rate = 0.001
description = 'ecg_ptb_2cnn.lr' + str(learn_rate) + 'ep' + str(epochs) + '_v' + str(VERSION_NUMBER)
keras_model_name = description + '.h5'
file_name = description
seq_length = 2000

if num_channels < 2:
    x_shape = [seq_length, 1]
    input_shape = (seq_length, num_channels)
else:
    x_shape = [seq_length, 2]
    input_shape = (seq_length, num_channels)

if IND_TO_VEC:
    y_shape = [1]
else:
    y_shape = [num_classes]

# Import Data:
x_tt, y_tt = tfs.load_data_v2('data/ptb_ecg_1ch_half_overlap_labeled/lead_v2_all', x_shape, y_shape, 'X', 'Y',
                              ind2vec=IND_TO_VEC)

if num_channels < 2:
    x_tt = np.reshape(x_tt[:, :, 0], [-1, seq_length, 1])

x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)


def get_model():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=input_shape))
    k_model.add(Conv1D(128, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Conv1D(256, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Conv1D(512, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Reshape(target_shape=(1, seq_length * 64)))
    k_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Dropout(0.25))
    k_model.add(BatchNormalization())
    k_model.add(Dense(num_classes, activation='softmax'))
    k_model.add(Reshape(target_shape=(num_classes,)))
    adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(k_model.summary())
    return k_model


# Train:
batch_size = 256

tf_backend.set_session(tfs.get_session(0.9))
with tf.device('/gpu:0'):
    start_time_ms = tfs.current_time_ms()
    if os.path.isfile(keras_model_name):
        model = load_model(keras_model_name)
        print(model.summary())
        if TRAIN:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            model.save(keras_model_name)
    else:
        if TRAIN:
            model = get_model()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            model.save(keras_model_name)
        else:
            exit(-1)

    score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('Test score: {} , Test accuracy: {}'.format(score, acc))

    # Predict & Print Confmat
    yy_probabilities = model.predict(x_test)
    yy_predicted = tfs.max_out_probs(yy_probabilities, sample_axis=0, dimensions=2)
    confmat = confusion_matrix(y_test, yy_predicted)
    print("Confusion Matrix: \n", confmat)
    # Maximize probabilities of prediction.

    # Evaluate hidden layers: # 'conv1d_3'
    # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
    if SAVE_HIDDEN:
        layers_of_interest = ['conv1d_1', 'conv1d_2', 'reshape_2', 'dense_1', 'dense_2']
        np.random.seed(0)
        rand_indices = np.random.randint(0, x_test.shape[0], 250)
        print('Saving hidden layers: ', layers_of_interest)
        tfs.get_keras_layers(model, layers_of_interest, x_test[rand_indices], y_test[rand_indices],
                             output_dir=tfs.prep_dir('I:/_ecg_data_backup/classification/hidden_layers'),
                             fname='rat_hidden_all_' + file_name + '.mat')
    if SAVE_TEST_DATA:
        data_dict = {'x_val': x_test, 'y_val': y_test, 'y_prob': yy_probabilities, 'y_pred': yy_predicted}
        savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)

    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)

if SAVE_FROZEN_MODEL:
    tf_backend.set_learning_phase(0)
    tfs.export_model_keras(keras_model_name, tfs.prep_dir("graph_rat"), model_name=description)
