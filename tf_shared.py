# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.5.0

import glob
import numpy as np
import pandas as pd
import winsound as ws
import tensorflow as tf
import time
import warnings
from scipy.io import loadmat, savemat
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from sklearn import metrics as skmet


# GAN:
# def generator_a(inputs, latent_space):
# input_shape = np.shape(inputs)
# tf.nn.conv2d(inputs, latent_space, strides=[1,1,1,1], padding='SAME')
#     return 0


def lrelu(x, a=0.2, name='lrelu'):
    return tf.nn.leaky_relu(x, a, name=name)


def generator_b(latent_space, output_shape, fc_units, wt_init, c_params):
    batch_size = get_tensor_shape(latent_space)[0]
    fc_output_units = output_shape[0] // 4 * output_shape[1] * fc_units
    with tf.variable_scope("Generator"):
        norm_input = tf.layers.batch_normalization(latent_space)
        fc1 = fc_layer(norm_input, fc_output_units, wt_init, name='g_fc1')
        fc1 = reshape(fc1, batch_size, output_shape[0] // 4, output_shape[1], fc_units)
        c1 = conv2d_layer(fc1, fc_units, c_params[0][0], c_params[0][1], wt_init, name='g_c1')
        c1 = reshape(c1, batch_size, output_shape[0] // 2, output_shape[1], fc_units // 2)
        c2 = conv2d_layer(c1, fc_units // 2, c_params[1][0], c_params[1][1], wt_init, name='g_c2')
        c2 = reshape(c2, batch_size, output_shape[0], output_shape[1], fc_units // 4)
        return conv2d_layer(c2, 1, c_params[2][0], c_params[2][1], wt_init, name='g_c3', norm=False, init_weights=False,
                            activation=tf.nn.tanh)  # ALSO tf.nn.elu WORKS


# TODO: Rewrite using only builtin funcs.
def generator_contrib(latent_space, output_shape, fc_units, wt_init, c_params):
    batch_size = get_tensor_shape(latent_space)[0]
    fc_output_units = output_shape[0] // 4 * output_shape[1] * fc_units
    with tf.variable_scope("Generator"):
        fc1 = fully_connect_contrib(latent_space, fc_output_units, wt_init, scope="g_fc1")
        fc1 = reshape(fc1, batch_size, output_shape[0] // 4, output_shape[1], fc_units)
        c1 = conv2d_contrib(fc1, fc_units, c_params[0][0], c_params[0][1], wt_init, scope="g_c1")
        c1 = reshape(c1, batch_size, output_shape[0] // 2, output_shape[1], fc_units // 2)
        c2 = conv2d_contrib(c1, fc_units // 2, c_params[1][0], c_params[1][1], wt_init, scope="g_c2")
        c2 = reshape(c2, batch_size, output_shape[0], output_shape[1], fc_units // 4)
        c3 = conv2d_contrib_b(c2, 1, c_params[2][0], c_params[2][1], activation=tf.nn.tanh, scope="g_c3", norm=None)
        return c3


def discriminator_contrib(inputs, units, c_params, wt_init, reuse=False):
    if get_tensor_shape(inputs).shape[0] > 3:
        s_offset = 1
    else:
        s_offset = 0
    with tf.variable_scope("Discriminator"):
        c1 = conv2d_contrib(inputs, units[0], c_params[0][0], c_params[0][1], wt_init, "d_c1", reuse,
                            activation=lrelu, norm=None)
        c2 = conv2d_contrib(c1, units[1], c_params[1][0], c_params[1][1], wt_init, "d_c2", reuse,
                            activation=lrelu, norm=None)
        c2_shape = get_tensor_shape(c2)
        fc1 = tf.reshape(c2, [-1, c2_shape[0 + s_offset] * c2_shape[1 + s_offset] * c2_shape[2 + s_offset]])
        fc1 = fully_connect_contrib(fc1, units[2], wt_init, scope="d_fc1", activation=lrelu, reuse=reuse)
        fc2 = fully_connect_contrib(fc1, 1, wt_init, scope="d_fc2", activation=tf.nn.sigmoid, norm=None, reuse=reuse)
        return fc2


def conv2d_layer(inputs, filters, kernel_size, strides, wt_init, name, padding="SAME", activation=tf.nn.relu, reuse=None,
                 norm=True, init_weights=True):
    if init_weights:
        kernel_initializer = wt_init
    else:
        kernel_initializer = None
    if norm:
        input_norm = tf.layers.batch_normalization(inputs)
        return tf.layers.conv2d(input_norm, filters, kernel_size, strides, padding=padding, activation=activation,
                                kernel_initializer=kernel_initializer, name=name, reuse=reuse)
    else:
        return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding, activation=activation,
                                kernel_initializer=kernel_initializer, name=name, reuse=reuse)


def fc_layer(inputs, units, init, name, activation=tf.nn.relu):
    return tf.layers.dense(inputs, units, kernel_initializer=init, activation=activation, name=name)


def fully_connect_contrib(inputs, num_outputs, init, scope, activation=tf.nn.relu,
                          norm=tf.contrib.layers.batch_norm,
                          reuse=None):
    return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=activation,
                                             normalizer_fn=norm, weights_initializer=init, scope=scope, reuse=reuse)


def conv2d_contrib(inputs, num_outputs, kernel, stride, wt_init, scope, reuse=None, activation=tf.nn.relu,
                   padding="SAME", norm=tf.contrib.layers.batch_norm):
    return tf.contrib.layers.conv2d(inputs, num_outputs, kernel, stride, padding, reuse=reuse,
                                    activation_fn=activation,
                                    weights_initializer=wt_init, scope=scope, normalizer_fn=norm)


def conv2d_contrib_b(inputs, num_outputs, kernel, stride, scope, reuse=None, activation=tf.nn.relu,
                     padding="SAME", norm=tf.contrib.layers.batch_norm):
    return tf.contrib.layers.conv2d(inputs, num_outputs, kernel, stride, padding, reuse=reuse,
                                    activation_fn=activation,
                                    scope=scope, normalizer_fn=norm)


def tf_initialize():
    # Merges all summaries collected in the default graph.
    tf.summary.merge_all()
    saver = tf.train.Saver()  # Initialize tf Saver for model export
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return saver, init_op, config


def confusion_matrix_test(sess, x, y, keep_prob, prediction, input_shape, x_val_data, y_val_data, number_classes=5):
    y_val_tf = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    predictions = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    for i in range(0, x_val_data.shape[0]):
        predictions[i] = sess.run(prediction,
                                  feed_dict={x: x_val_data[i].reshape(input_shape),
                                             y: y_val_data[i].reshape([1, number_classes]), keep_prob: 1.0})
        for c in range(0, number_classes):
            if y_val_data[i][c]:
                y_val_tf[i] = c

    tf_confusion_matrix = tf.confusion_matrix(labels=y_val_tf, predictions=predictions, num_classes=number_classes)
    confusion_matrix_nd = tf.Tensor.eval(tf_confusion_matrix, feed_dict=None, session=None)
    print(confusion_matrix_nd)
    print("Ground Truth: ", np.sum(np.asarray(pd.get_dummies(y_val_tf).values).astype(np.float32), axis=0))
    acc = skmet.accuracy_score(y_val_tf, predictions)
    print(acc)
    return confusion_matrix_nd, acc


def current_time_ms():
    return int(round(time.time() * 1000))


def placeholder(shape, name, dtype=tf.float32):
    return tf.placeholder(dtype, shape=shape, name=name)


def reshape(x, batch, l_x, l_y, out_ch):
    return tf.reshape(x, [batch, l_x, l_y, out_ch])


# Model Building Macros: #
def get_tensor_shape_tuple(x_):
    shape_as_list = x_.get_shape().as_list()
    shape_as_list = list(filter(None.__ne__, shape_as_list))
    return tuple([1, *shape_as_list])


def get_tensor_shape(x_):
    shape_as_list = x_.get_shape().as_list()
    # filter out  'None' type:
    shape_as_list = list(filter(None.__ne__, shape_as_list))
    return np.asarray(shape_as_list)


def flatten(x):
    shape = get_tensor_shape(x)
    dense_shape = 1
    # check if dimension is valid before multiplying out
    for i in range(0, shape.shape[0]):
        if shape[i] is not None:
            dense_shape = dense_shape * shape[i]
    return tf.reshape(x, [-1, dense_shape]), dense_shape


# Zero Padded Max Pooling
def max_pool(x_, ksize, stride, padding='SAME'):
    return tf.nn.max_pool(x_, ksize=ksize, strides=stride, padding=padding)


# Simple matmul + bias add. Used for output layer
def connect(x, w, b):
    return tf.add(tf.matmul(x, w), b)


def connect_v2(x, w, b):
    return tf.nn.bias_add(tf.matmul(x, w), b)


def loss_layer(y, y_conv):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))


def loss_layer_v2(y, y_conv):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))


def get_accuracy(correct_prediction):
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_optimize(learning_rate, cross_entropy):
    return tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def train_loss(y, y_conv, learning_rate):
    # Compute Cross entropy:
    cross_entropy = loss_layer_v2(y, y_conv)
    # Optimizing using Adam Optimizer
    return train_optimize(learning_rate, cross_entropy)


def get_outputs(y, y_conv, output_node_name):
    outputs = tf.nn.softmax(y_conv, name=output_node_name)
    prediction_check, prediction = check_prediction(y, outputs)
    accuracy = get_accuracy(prediction_check)
    return outputs, prediction, accuracy


def check_prediction(y, outputs):
    prediction = tf.argmax(outputs, 1)
    correct_class = tf.argmax(y, 1)
    return tf.equal(prediction, correct_class), prediction


def load_data(data_directory, image_shape, key_x, key_y, shuffle=False):
    x_train_data = np.empty([0, *image_shape], np.float32)
    y_train_data = np.empty([0], np.float32)
    training_files = glob.glob(data_directory + "/*.mat")
    for f in training_files:
        x_array = loadmat(f).get(key_x)
        y_array = loadmat(f).get(key_y)
        y_array = y_array.reshape([np.amax(y_array.shape)])
        x_train_data = np.concatenate((x_train_data, x_array), axis=0)
        y_train_data = np.concatenate((y_train_data, y_array), axis=0)
    if shuffle:
        np.random.shuffle(x_train_data)
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)
    # return data_array
    print("Loaded Data Shape: X:", x_train_data.shape, " Y: ", y_train_data.shape)
    return x_train_data, y_train_data


# Save graph/model:
def export_model(input_node_names, output_node_name_internal, export_dir, model_name):
    freeze_graph.freeze_graph(export_dir + model_name + '.pbtxt', None, False,
                              export_dir + model_name + '.ckpt', output_node_name_internal, "save/restore_all",
                              "save/Const:0", export_dir + '/frozen_' + model_name + '.pb', True, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(export_dir + '/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name_internal], tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(export_dir + '/opt_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", export_dir + '/frozen_' + model_name + '.pb')
    print("2 - Android Optimized Model:", export_dir + '/opt_' + model_name + '.pb')


def get_activations_mat(x, keep_prob, sess, layer, input_sample, input_shape):
    units = sess.run(layer, feed_dict={x: np.reshape(input_sample, input_shape, order='F'), keep_prob: 1.0})
    return units


def get_trained_vars(sess, filename):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    d = {'ignore': 1.0}
    for i in range(0, len(local_vars)):
        key = local_vars[i]._shared_name
        var = sess.run(local_vars[i])
        # add to dict
        d[key] = var
    filename += 'trained_variables.mat'
    savemat(file_name=filename, mdict=d)


def beep(freq=900, length_ms=1000):
    # A 900Hz Beep:
    ws.Beep(freq, length_ms)
