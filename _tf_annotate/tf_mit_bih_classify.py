# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn
# from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import tf_shared as tfs

# Setup:
TRAIN = True
TEST = True
SAVE_HIDDEN = False
SAVE_PREDICTIONS = False
CUSTOM_EVALUATION = True
VERSION_NUMBER = 2
epochs = 10
num_channels = 1
num_classes = 5
model_dir = "model_exports"
output_folder = 'classify_data_out/n' + str(num_channels) + 'ch/'
version_num = 0
LSTM_UNITS = 64
learn_rate = 0.01
description = 'normal_2cnn_fixed.conv1d_seq2seq_' + str(LSTM_UNITS) + 'lr' + str(learn_rate) + '_v' + str(
    VERSION_NUMBER)
keras_model_name = description + '.h5'

output_folder_name = tfs.prep_dir('model_out/')
checkpoint_filename = output_folder_name + description + '.ckpt'

batch_size = 256
eval_batch_size = 512

file_name = description
seq_length = 2000
if num_channels < 2:
    x_shape = [seq_length, 1]
    input_shape = seq_length
else:
    x_shape = [seq_length, 2]
    input_shape = (seq_length, num_channels)
y_shape = [seq_length, num_classes]

# Import Data:
x_tt, y_tt = tfs.load_data_v2('../data/extended_5_class/mit_bih_tlabeled_w8s_fixed_all', [seq_length, 2], y_shape,
                              'relevant_data', 'Y')
# Keep Only Lead II
if num_channels < 2:
    x_tt = np.reshape(x_tt[:, :, 0], [-1, seq_length, 1])

x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)  # 0.66
xx_flex, y_flex = tfs.load_data_v2('../data/flexEcg_8s_normal', [seq_length, 1], [1], 'relevant_data', 'Y')


def get_graph_tf(inputs, keep_prob_, name_output_node='output'):
    r1 = tf.reshape(inputs, [-1, seq_length, num_channels])
    d1 = tf.layers.conv1d(inputs=r1, filters=128, kernel_size=8, strides=2, padding='same', activation='relu')
    d2 = tf.layers.conv1d(inputs=d1, filters=256, kernel_size=8, strides=2, padding='same', activation='relu')
    # 64 x (?, 2000), x2
    r2 = tf.reshape(d2, [-1, seq_length, 64])
    r2_unstacked = tf.unstack(r2, axis=2)
    lstm_fw_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)
    # Get LSTM Output:
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, r2_unstacked, dtype=tf.float32)
    lstm_out = tf.stack(outputs, axis=2)
    lstm_out_reshape = tf.reshape(lstm_out, [-1, 64*128])
    # No activation ???
    drop1 = tf.nn.dropout(lstm_out_reshape, keep_prob=keep_prob_)
    bn1 = tf.layers.batch_normalization(drop1)
    # dense1 = tf.nn.relu(tfs.fully_connect_3d(bn1, [2 * LSTM_UNITS, LSTM_UNITS], [LSTM_UNITS]))
    dense1 = tf.nn.relu(tfs.fully_connect(bn1, [8192, 10000], [10000]))
    # dense1 = tf.layers.dense(inputs=bn1, units=64, activation='relu', kernel_regularizer=l2_regularizer(scale=0.01))
    drop2 = tf.nn.dropout(dense1, keep_prob=keep_prob_)
    bn2 = tf.layers.batch_normalization(drop2)
    dense2 = tf.reshape(bn2, [-1, 2000, 5])
    # dense2 = tf.contrib.layers.fully_connected(bn2, num_classes)
    softmax_out = tf.nn.softmax(dense2, name=name_output_node)
    return softmax_out


# Create placeholders for our model:
tf.reset_default_graph()

input_node_name = 'input'
output_node_name = 'output'
keep_prob_node_name = 'keep_prob'

keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
X = tf.placeholder(tf.float32, [None, *x_shape], name=input_node_name)
Y = tf.placeholder(tf.float32, [None, *y_shape])

pred_y = get_graph_tf(X, keep_prob, output_node_name)

# to calculate loss, we flatten the output (n, 2000x5)
loss = tf.reduce_mean(categorical_crossentropy(y_true=Y, y_pred=pred_y))

optimizer = tf.train.AdamOptimizer(learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred_y, 2), tf.argmax(Y, 2))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

saver, init_op, config = tfs.tf_initialize()

with tf.Session(config=config) as sess:
    sess.run(init_op)
    tf.train.write_graph(sess.graph_def, output_folder_name, description + '.pbtxt', True)
    start_time_ms = tfs.current_time_ms()
    # Load Checkpoint if Available:
    if os.path.isfile(checkpoint_filename):
        saver.restore(sess, checkpoint_filename)

    if TRAIN:
        # Load model and train
        batches = x_train.shape[0] // batch_size - 1
        for i in range(0, epochs):
            for b in range(0, batches):
                start_idx = b * 256
                end_idx = start_idx + batch_size
                batch_x = x_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                _, acc_train, loss_train = sess.run([optimizer, accuracy, loss], feed_dict={X: batch_x, Y: batch_y,
                                                                                            keep_prob: 0.8})

                print("Epoch [%d/%d], batch [%d/%d], Idx [%d:%d/%d], Elapsed: %2.3f min, Accuracy: %3.3f, Loss: %2.4f" %
                      (i + 1, epochs, b + 1, batches, start_idx, end_idx, x_train.shape[0],
                       (tfs.current_time_ms() - start_time_ms) / 60000, acc_train, float(np.sum(loss_train))))
            elapsed_min = (tfs.current_time_ms() - start_time_ms) / 60000
            eta = (epochs - (i + 1)) / ((i + 1) / elapsed_min)  # (remaining epochs - avg epoch train time).
            print("ETA: %3.3f min @ %3.3f epoch/min" % (eta, ((i + 1) / elapsed_min)))
            # Test Accuracy Evaluation
            if i % 10 == 0 and i > 0:
                tfs.test_v3(sess, X, Y, accuracy, x_test, y_test, keep_prob=keep_prob)

    if TEST:
        test_accuracy = tfs.test_v3(sess, X, Y, accuracy, x_test, y_test, keep_prob=keep_prob)

    elapsed_time_ms = (tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
    print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)

    saver.save(sess, checkpoint_filename)
    print("Input layer name: ", input_node_name)
    print("Output layer name: ", output_node_name)
    output_graph_path = output_folder_name + '/frozen_' + description + '.pb'

    # for v in sess.graph.get_operations():
    #     print(v.name)

    freeze_graph.freeze_graph(output_folder_name + description + '.pbtxt',
                              "",
                              False,
                              checkpoint_filename,
                              output_node_name,
                              "save/restore_all",
                              "save/Const:0",
                              output_graph_path,
                              True,
                              "")

    input_graph_def = tf.GraphDef()
    # Read frozen pb file to Graphdef:
    with tf.gfile.Open(name=output_graph_path, mode='rb') as f:
        input_graph_def.ParseFromString(f.read())

    # Convert to Android-optimized Model:
    output_graph_def = \
        optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def,
                                                          input_node_names=[input_node_name],
                                                          output_node_names=[output_node_name],
                                                          placeholder_type_enum=tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(output_folder_name + '/opt_' + description + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
