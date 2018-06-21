# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.5.0

# IMPORTS:
import tensorflow as tf
import tf_shared as tfs
import numpy as np
import os

from scipy.io import savemat

EXPORT_DIRECTORY = 'model_exports/'
win_len = 500
num_channels = 1
TRAINING_FOLDER = r'' + 'flexEcg'  # 'mit_db_ds_small'
TRAINING_FOLDER2 = r'' + 'mit_db_ds_small'
input_shape = [win_len, num_channels]  # for single sample reshape(x, [1, *input_shape])
NUMBER_CLASSES = 2
# LOAD DATA:
x_flex, y_flex = tfs.load_data(TRAINING_FOLDER, [win_len], key_x='relevant_data', key_y='Y', shuffle=True)
# TODO: Change to normal ECG Only.
x_train, y_train = tfs.load_data(TRAINING_FOLDER2, [win_len, 2], key_x='relevant_data', key_y='Y', shuffle=True)

output_folder_name = "matout/" + TRAINING_FOLDER2 + "_gan_new/"

g_conv_params = [[[3, 3], [1, 1]], [[3, 3], [1, 1]], [[3, 3], [1, 1]]]  # [ [c1k, c1s] [c2k, c1s] [c3k, c3s] ]

d_conv_params = [[[3, 3], [2, 1]], [[2, 2], [2, 1]]]

d_outputs = [32, 64, 256]

# Batch, LR, Training Iterations:
batch_size = 512
learning_rate = 1e-3
train_its = 10000
latent_space_size = 100
g_units = 128

# Input vars:
# x is 'fake' fECG data
x = tf.placeholder(tf.float32, shape=[None, win_len, num_channels])
x_image = tf.reshape(x, [-1, win_len, num_channels, 1])
# TODO y = 'real' data
# y = tf.placeholder(tf.float32, shape=[None, win_len, num_channels])
# y_image = tf.reshape(x, [-1, win_len, num_channels, 1])
# Latent Space:
z_in = tf.placeholder(tf.float32, shape=[batch_size, latent_space_size])  # Latent space (of size 512 * 100)

wt_init = tf.truncated_normal_initializer(stddev=0.02)

# Setup GAN Graph:
# Generators:
g_out = tfs.generator_contrib(z_in, input_shape, g_units, wt_init, g_conv_params)
# TODO # g_out = tfs.generator(x, )
# Discriminators:
d_out_fake = tfs.discriminator_contrib(g_out, d_outputs, d_conv_params, wt_init)
d_out_real = tfs.discriminator_contrib(x_image, d_outputs, d_conv_params, wt_init, reuse=True)

# loss and optimization:
disc_loss = tf.reduce_sum(tf.square(d_out_real - 1) + tf.square(d_out_fake)) / 2
gen_loss = tf.reduce_sum(tf.square(d_out_fake - 1)) / 2

tvars = tf.trainable_variables()

gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

d_grads = d_optimizer.compute_gradients(disc_loss,
                                        dis_variables)  # Only update the weights for the discriminator network.
g_grads = g_optimizer.compute_gradients(gen_loss, gen_variables)  # Only update the weights for the generator network.

update_D = d_optimizer.apply_gradients(d_grads)
update_G = g_optimizer.apply_gradients(g_grads)

saver, init_op, config = tfs.tf_initialize()
with tf.Session(config=config) as sess:
    sess.run(init_op)
    start_time_ms = tfs.current_time_ms()
    # Print Model Information:
    for i in range(train_its):
        offset = (i * batch_size) % (x_train.shape[0] - batch_size)
        if len(x_train.shape) > 2:
            batch_x_train = x_train[offset:(offset + batch_size), :, 0]
        else:
            batch_x_train = x_train[offset:(offset + batch_size)]
        batch_x_train = np.reshape(batch_x_train, [batch_size, win_len, 1])
        z_input = np.random.uniform(0, 1.0, size=[batch_size, latent_space_size])

        _, d_loss = sess.run([update_D, disc_loss], feed_dict={x: batch_x_train, z_in: z_input})

        for j in range(4):
            _, g_loss = sess.run([update_G, gen_loss], feed_dict={z_in: z_input})

        print("i: {} / d_loss: {} / g_loss: {}".format(i, np.sum(d_loss) / batch_size, np.sum(g_loss) / batch_size))

        if i % 10 == 0:  # PERIODICALLY SAVE MATLAB
            gen_o = sess.run(g_out, feed_dict={z_in: z_input})
            fn = output_folder_name + "{}.mat".format(i)
            if not os.path.exists(output_folder_name):
                os.makedirs(output_folder_name)
            savemat(fn, mdict={'gen0': gen_o[0][:, :, 0]})
        #     # result = plt.imshow(gen_o[0][:, :, 0], cmap="gray")
        #     plt.imsave("{}.png".format(i), gen_o[0][:, :, 0], cmap="gray")
elapsed_time_ms = (tfs.current_time_ms() - start_time_ms)
print('Elapsed Time (ms): ', elapsed_time_ms)
