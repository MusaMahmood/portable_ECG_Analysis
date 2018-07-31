# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os
import datetime
import numpy as np
import tf_shared_k as tfs

from keras.layers import Dropout
from keras.optimizers import Adam
from scipy.io import savemat, loadmat
from keras.models import Model, load_model
from keras.utils.generic_utils import Progbar
from keras.layers.convolutional import UpSampling1D
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LeakyReLU, Input, Concatenate
from keras_contrib.layers.normalization import InstanceNormalization

# Sources: (Ctrl-LMB in Pycharm)
# Instance Norm: https://arxiv.org/abs/1701.02096
# Cycle GAN: https://arxiv.org/abs/1703.10593
# GAN (orig paper): http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

# Setup:
TRAIN = True  # TRAIN ANYWAY FOR # epochs, or just evaluate
batch_size = 128
epochs = 100
num_channels = 1
num_classes = 1
learn_rate = 0.0002
lambda_cycle = 10.0  # Cycle-consistency loss
lambda_id = 0.1 * lambda_cycle  # Identity loss
label = 'ptb_ecg_cycle_gan_leadv2_lr' + str(learn_rate) + '_r0'
model_dir = "model_exports/" + label + '/'
output_folder = 'outputs/' + label + '/'
description = label
seq_length = 2000
input_length = seq_length
x_shape = [seq_length, 1]
y_shape = [seq_length, num_classes]

x_lead_v2 = tfs.load_mat('data/lead_v2_all/all_x.mat', key='X', shape=[seq_length, 1])
x_lead_ii = tfs.load_mat('data/lead_ii_all/all_y.mat', key='Y', shape=[seq_length, 1])
x_train, x_test, y_train, y_test = train_test_split(x_lead_v2, x_lead_ii, train_size=0.75, random_state=1)


def build_generator():
    def conv_layer(layer_input, filters, kernel_size=5, strides=2):
        d = Conv1D(filters, kernel_size, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.20)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv_layer(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling1D(size=2)(layer_input)
        u = Conv1D(filters, f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Input samples
    input_samples = Input(shape=(input_length, 1))

    # Downsampling:
    d1 = conv_layer(input_samples, 32, 5, 2)
    d2 = conv_layer(d1, 64, 5, 2)
    d3 = conv_layer(d2, 128, 5, 2)
    d4 = conv_layer(d3, 256, 5, 2)

    # Now Upsample:
    u1 = deconv_layer(d4, d3, 128, f_size=5)
    u2 = deconv_layer(u1, d2, 64, f_size=5)
    u3 = deconv_layer(u2, d1, 32, f_size=5)
    u4 = UpSampling1D(size=2)(u3)
    output_samples = Conv1D(1, kernel_size=5, strides=1, padding='same', activation='tanh')(u4)  #
    # output_samples = Activation('tanh')(cu4)
    return Model(input_samples, output_samples)


def build_discriminator():
    def discriminator_layer(layer_input, filters, f_size=5, strides=2, normalization=True):
        d = Conv1D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    input_samples = Input(shape=(input_length, 1))
    d1 = discriminator_layer(input_samples, 64, 5, 2, normalization=False)
    d2 = discriminator_layer(d1, 128, 5, 2)
    d3 = discriminator_layer(d2, 256, 5, 2)
    d4 = discriminator_layer(d3, 512, 5, 2)
    validity = Conv1D(1, kernel_size=5, strides=1, padding='same')(d4)
    return Model(input_samples, validity)


# Restore Model if Present:
keras_training_file = tfs.prep_dir(model_dir) + description + 'training.mat'
keras_training_epochs_key = 'training_epochs'
keras_training_batch_size_key = 'training_batch_size'
keras_combined_model_location = tfs.prep_dir(model_dir) + description + 'combined_model.h5'
keras_d_A_location = tfs.prep_dir(model_dir) + description + 'd_A.h5'
keras_d_B_location = tfs.prep_dir(model_dir) + description + 'd_B.h5'
keras_g_BA_location = tfs.prep_dir(model_dir) + description + 'g_BA.h5'
keras_g_AB_location = tfs.prep_dir(model_dir) + description + 'g_AB.h5'

keras_g_AB_opt_location = tfs.prep_dir(model_dir) + '/opt_ptb_cycle_gan'
# Load if it exists:
if os.path.isfile(keras_d_A_location) and os.path.isfile(keras_d_B_location) and os.path.isfile(
        keras_g_AB_location) and os.path.isfile(keras_g_BA_location) and os.path.isfile(keras_combined_model_location):
    print('Loading existing models from directory: ', model_dir, description)
    # Load discriminators
    d_A = load_model(keras_d_A_location)
    d_B = load_model(keras_d_B_location)
    print('Discriminator: ')
    print(d_A.summary())
    optimizer = Adam(learn_rate, beta_1=0.50)
    d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    # Load Generators:
    g_AB = load_model(keras_g_AB_location)
    g_BA = load_model(keras_g_BA_location)
    print('Generator: ')
    print(g_AB.summary())
    input_A = Input(shape=(input_length, 1))
    input_B = Input(shape=(input_length, 1))
    # Translate images to other domain
    fake_B = g_AB(input_A)
    fake_A = g_BA(input_B)
    # Translate Images back to original domain
    reconstr_A = g_BA(input_B)
    reconstr_B = g_AB(input_A)
    # Identity mapping of images
    input_A_id = g_BA(input_A)
    input_B_id = g_AB(input_B)
    # For the combined model we only train the generators:
    d_A.trainable = False
    d_B.trainable = False
    # Discriminators determines validity of translated data
    valid_A = d_A(fake_A)
    valid_B = d_B(fake_B)

    combined_model = load_model(keras_combined_model_location)
    print(combined_model.summary())
    combined_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                           loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id], optimizer=optimizer)
else:
    # Set train = 1
    print('Existing model of description [', description, '] not found!')
    TRAIN = True
    print('Training automatically enabled! - Running ', epochs, ' epochs!')
    # Manually create models:
    optimizer = Adam(learn_rate, beta_1=0.50)
    # Build and compile the discriminators
    d_A = build_discriminator()
    d_B = build_discriminator()
    print('Discriminator: ')
    print(d_A.summary())

    d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Build and compile the generators
    g_AB = build_generator()
    g_BA = build_generator()
    print('Generator: ')
    print(g_AB.summary())

    input_A = Input(shape=(input_length, 1))
    input_B = Input(shape=(input_length, 1))

    # Translate images to other domain
    fake_B = g_AB(input_A)
    fake_A = g_BA(input_B)

    # Translate Images back to original domain
    reconstr_A = g_BA(input_B)
    reconstr_B = g_AB(input_A)

    # Identity mapping of images
    input_A_id = g_BA(input_A)
    input_B_id = g_AB(input_B)

    # For the combined model we only train the generators:
    d_A.trainable = False
    d_B.trainable = False
    # Discriminators determines validity of translated data
    valid_A = d_A(fake_A)
    valid_B = d_B(fake_B)

    combined_model = Model(inputs=[input_A, input_B],
                           outputs=[valid_A, valid_B, reconstr_A, reconstr_B, input_A_id, input_B_id])
    print(combined_model.summary())
    combined_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                           loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id], optimizer=optimizer)

# If mat file doesn't exist, make one and set to zero:
if os.path.isfile(keras_training_file):
    prev_training_epochs = loadmat(keras_training_file).get(keras_training_epochs_key)
    prev_batch_size = loadmat(keras_training_file).get(keras_training_batch_size_key)
else:
    prev_training_epochs = 0
    prev_batch_size = 0
    savemat(keras_training_file, mdict={keras_training_epochs_key: 0, keras_training_batch_size_key: batch_size})

# Train:
if TRAIN:
    print('This model has been trained to ', prev_training_epochs, 'epochs, with batch size:', prev_batch_size)
    print('Starting Training! - batch_size:', batch_size, ' epochs: ', epochs)
    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + (125, 1))
    fake = np.zeros((batch_size,) + (125, 1))
    # Training Routine:
    start_time = datetime.datetime.now()
    last_epoch_update = 0
    for epoch in range(epochs + 1):
        print('Epoch {} of {}'.format(epoch, epochs))
        number_batches = int(x_train.shape[0] / batch_size) - 1
        progress_bar = Progbar(target=number_batches)
        index = 0
        while index < number_batches:
            progress_bar.update(index)
            index += 1
            inputs_A = x_train[index * batch_size:(index + 1) * batch_size]
            inputs_B = y_train[index * batch_size:(index + 1) * batch_size]

            # Convert inputs using generator.
            fake_B = g_AB.predict(inputs_A)
            fake_A = g_BA.predict(inputs_B)

            # # # Train Discriminators:
            dA_loss_real = d_A.train_on_batch(inputs_A, valid)
            dA_loss_fake = d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = d_B.train_on_batch(inputs_B, valid)
            dB_loss_fake = d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total Disc Loss:
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

            # # # Train Generators:
            g_loss = combined_model.train_on_batch([inputs_A, inputs_B],
                                                   [valid, valid, inputs_A, inputs_B, inputs_A, inputs_B])
            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] "
                "[G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "
                % (epoch, epochs,
                   index, number_batches,
                   d_loss[0], 100 * d_loss[1],
                   g_loss[0],
                   float(np.mean(g_loss[1:3])),
                   float(np.mean(g_loss[3:5])),
                   float(np.mean(g_loss[5:6])),
                   elapsed_time))
        if epoch % 10 == 0 and epoch != 0:
            # Translate inputs to other domain:
            fake_B = g_AB.predict(x_train)
            fake_A = g_BA.predict(y_train)
            # translate back to original domain:
            reconstr_A = g_BA.predict(fake_B)
            reconstr_B = g_AB.predict(fake_A)

            # Save Model & Update Stats
            combined_model.save(keras_combined_model_location)
            g_AB.save(keras_g_AB_location)
            g_BA.save(keras_g_BA_location)
            d_A.save(keras_d_A_location)
            d_B.save(keras_d_B_location)
            # Update Number of Epochs saved
            prev_training_epochs = loadmat(keras_training_file).get(keras_training_epochs_key)
            savemat(keras_training_file, mdict={keras_training_epochs_key: prev_training_epochs + 10,
                                                keras_training_batch_size_key: batch_size})
            # gen_imgs = generator.predict(x_train)
            mdict = {'x_val': x_train, 'y_true': y_train, 'fake_A': fake_A, 'fake_B': fake_B, 'reconstr_A': reconstr_A,
                     'reconstr_B': reconstr_B}
            total_epochs = prev_training_epochs + 10
            savemat(tfs.prep_dir(output_folder) + description + "_%d.mat" % total_epochs, mdict=mdict)
            last_epoch_update = epoch

    # Save Models After Training Completed
    combined_model.save(keras_combined_model_location)
    g_AB.save(keras_g_AB_location)
    g_BA.save(keras_g_BA_location)
    d_A.save(keras_d_A_location)
    d_B.save(keras_d_B_location)
    update = epochs - last_epoch_update
    prev_training_epochs = loadmat(keras_training_file).get(keras_training_epochs_key)
    savemat(keras_training_file, mdict={keras_training_epochs_key: prev_training_epochs + update,
                                        keras_training_batch_size_key: batch_size})

# print('Evaluating & Saving Test Set:')
# Generate Fake Images:
fake_B = g_AB.predict(x_test)
fake_A = g_BA.predict(y_test)

# Translate back to original domain:
reconstr_A = g_BA.predict(fake_B)
reconstr_B = g_AB.predict(fake_A)
total_epochs = loadmat(keras_training_file).get(keras_training_epochs_key)
mdict = {'x_val': x_test, 'y_true': y_test, 'fake_A': fake_A, 'fake_B': fake_B, 'reconstr_A': reconstr_A,
         'reconstr_B': reconstr_B}
savemat(tfs.prep_dir(output_folder) + 'test_' + description + '_' + str(total_epochs) + 'epochs.mat', mdict=mdict)
print('Test Data Saved: ', output_folder + 'test_' + description + '_' + str(total_epochs) + 'epochs.mat')

# Export Generator g_AB:
model = tfs.export_model_keras(keras_g_AB_location, export_dir=tfs.prep_dir(keras_g_AB_opt_location),
                               model_name=description + 'g_AB', sequential=False)
