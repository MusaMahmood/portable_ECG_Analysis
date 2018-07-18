# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os
import numpy as np
import tf_shared as tfs
import tensorflow as tf

from scipy.io import savemat
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Reshape
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as tf_backend
from keras.layers import Bidirectional, CuDNNLSTM, Conv1D, LeakyReLU, Flatten, Activation, Input

# Setup: TODO: CONVERT TO LSGAN
epochs = 200
num_channels = 1
num_classes = 1
model_dir = "model_exports"
output_folder = 'outputs/'
version_num = 0
LSTM_UNITS = 64
learn_rate = 0.01
description = 'ecg_conversion_' + str(LSTM_UNITS) + 'lr' + str(learn_rate) + 'ep' + str(epochs) + '_v1'
keras_model_name = description + '.h5'
file_name = description
seq_length = 2000
if num_channels < 2:
    x_shape = [seq_length, 1]
    input_shape = seq_length
else:
    x_shape = [seq_length, 2]
    input_shape = (seq_length, num_channels)
y_shape = [seq_length, num_classes]

# Import Data: XDAT
x_tt, y_tt = tfs.load_data_v2('data/flex_overlap', [seq_length, 1], [1], 'relevant_data', 'Y')  # Ignore Y.
if num_channels < 2:
    x_tt = np.reshape(x_tt[:, :, 0], [-1, seq_length, 1])
xx_flex, y_flex = tfs.load_data_v2('data/br_overlap', [seq_length, 1], [1], 'relevant_data', 'Y')
x_train, x_test, y_train, y_test = train_test_split(x_tt, xx_flex, train_size=0.75, random_state=1)  # 0.66


def model_conv1d_bilstm():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    k_model.add(Conv1D(128, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Conv1D(256, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Conv1D(512, 8, strides=2, padding='same', activation='relu'))
    k_model.add(Reshape(target_shape=(seq_length, 64)))
    k_model.add(Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    k_model.add(Dropout(0.2))
    k_model.add(BatchNormalization())
    k_model.add(Dense(1))
    k_model.add(Activation('sigmoid'))
    # adam = optimizers.adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # k_model.compile(loss='mse', optimizer=adam, metrics=None)
    print('model_generator:')
    print(k_model.summary())
    return k_model


def model_discriminator():
    k_model = Sequential()
    k_model.add(Reshape((seq_length, num_channels), input_shape=(input_shape, 1)))
    k_model.add(Conv1D(64, 5, strides=2, padding='same'))
    k_model.add(BatchNormalization())
    k_model.add(LeakyReLU(alpha=0.1))
    k_model.add(Conv1D(128, 5, strides=2, padding='same'))
    k_model.add(BatchNormalization())
    k_model.add(LeakyReLU(alpha=0.1))
    k_model.add(Flatten())
    k_model.add(Dense(256))
    k_model.add(BatchNormalization())
    k_model.add(LeakyReLU(alpha=0.1))
    k_model.add(Dense(1, activation='linear'))
    inputs = Input(shape=(input_shape, 1))
    score = k_model(inputs)
    model = Model(inputs, score)
    print('model_discriminator:')
    print(k_model.summary())
    return model


# Train:
batch_size = 32

tf_backend.set_session(tfs.get_session(0.9))
with tf.device('/gpu:0'):
    start_time_ms = tfs.current_time_ms()
    disc = model_discriminator()
    disc.compile(optimizer=Adam(lr=learn_rate), loss='mse')
    generator = model_conv1d_bilstm()
    g_inputs = Input(shape=(input_shape, 1))
    fake = generator(g_inputs)
    disc.trainable = False
    fake = disc(fake)
    combined_model = Model(inputs=g_inputs, outputs=fake)
    combined_model.compile(optimizer=Adam(lr=learn_rate), loss='mse')
    print('model_combined:')
    print(combined_model.summary())

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))
        nb_batches = int(x_train.shape[0] / batch_size) - 1
        progress_bar = Progbar(target=nb_batches)
        epoch_disc_loss = []
        epoch_gen_loss = []
        index = 0
        while index < nb_batches:
            progress_bar.update(index)
            index += 1
            input_batch = x_train[index * batch_size:(index + 1) * batch_size]
            target_batch = y_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(input_batch, verbose=0)

            X = np.concatenate((target_batch, generated_images))
            y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
            epoch_disc_loss.append(disc.train_on_batch(X, y))
            target = np.ones(batch_size)  # c == b == 1, cf. Eq. (9)
            epoch_gen_loss.append(combined_model.train_on_batch(input_batch, target))

        print('\n[Loss_D: {:.3f}, Loss_G: {:.3f}]'.format(np.mean(epoch_disc_loss), np.mean(epoch_gen_loss)))
        # TODO: Save samples
        if epoch % 10 == 0:
            gen_imgs = generator.predict(x_train)
            md = {'x_val': x_train, 'y_true': y_train, 'y_pred': gen_imgs}
            savemat(tfs.prep_dir(output_folder) + "ecg_test_%d.mat" % epoch, mdict=md)
    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)
    yy_probabilities = generator.predict(x_test)
    data_dict = {'x_val': x_test, 'y_true': y_test, 'y_pred': yy_probabilities}
    savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)
    combined_model.save(keras_model_name)

    # if os.path.isfile(keras_model_name):
    #     model = load_model(keras_model_name)
    #     score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    #     # predict
    #     yy_probabilities = model.predict(x_test)
    #
    #     print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    #     print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)
    #
    #     data_dict = {'x_val': x_test, 'y_true': y_test, 'y_pred': yy_probabilities}
    #     savemat(tfs.prep_dir(output_folder) + file_name + '.mat', mdict=data_dict)
    #     print('Test score: {}'.format(score))

# model = tfs.export_model_keras(keras_model_name, export_dir=tfs.prep_dir("graph"), model_name=description)
