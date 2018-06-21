# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.5.0

# IMPORTS:
import tensorflow as tf
import os as os
import tf_shared as tfs
import numpy as np
from sklearn.model_selection import train_test_split

EXPORT_DIRECTORY = 'model_exports/'
win_len = 500
TRAINING_FOLDER = r'' + 'mit_db_ds'  # + '/S' + str(subject_number) + '/' + 'w' + str(win_len)
input_shape = [win_len, 1]  # for single sample reshape(x, [1, *input_shape])
NUMBER_CLASSES = 2
# LOAD DATA:
x_tt, y_tt = tfs.load_data(TRAINING_FOLDER, [win_len, 2], key_x='relevant_data', key_y='Y')
x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)
x_val, y_val = tfs.load_data(TRAINING_FOLDER + '/v', [win_len, 2], key_x='relevant_data', key_y='Y')
x_val2, y_val2 = tfs.load_data(TRAINING_FOLDER + '/v2', [win_len, 2], key_x='relevant_data', key_y='Y')
# Initialize CNN Components
NUM_LAYERS = 2  # Default
N_FILTERS = [50, 100, 35, 60, 5, 5, 5, 5]  # Number of filters for 8 layers
Str_X = [2, 2, 2, 2, 1, 1, 1, 1]
Str_Y = [1, 2, 1, 2, 1, 1, 1, 1]
W_x = [8, 4, 2, 2, 2, 2, 2, 2]  # Weights in x-axis
W_y = [8, 4, 2, 2, 2, 2, 2, 2]  # Weights in y-axis, kernel size = [x, y]
Alphas = [0.1, 0.25, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]  # Parametric ReLU values for Conv Layers
Alpha_fc = 0.01  # If using parametric ReLU in fully connected layer.
do = 'dropout'
keep_prob_feed = 0.5
train_steps = 5000
LR_EXP = 3
LR_COEFF = 1
learn_rate = float(LR_COEFF) * float(10.0 ** (-float(LR_EXP)))
activation_fc = 'relu'
activation = 'parametricrelu'
UNITS_FC = 1024
Model_description = tfs.get_model_description(NUM_LAYERS, activation, do, keep_prob_feed, UNITS_FC,
                                              activation_fc, LR_COEFF, LR_EXP, N_FILTERS, Alphas[0:NUM_LAYERS],
                                              Alpha_fc)
Model_description += 'filt[' + str(W_x[0]) + ',' + str(W_y[0]) + ']'

if not os.path.exists(EXPORT_DIRECTORY):
    os.mkdir(EXPORT_DIRECTORY)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'

# 1. Load Model Placeholders:
x, y, keep_prob = tfs.placeholders(input_shape, NUMBER_CLASSES, input_node_name, keep_prob_node_name)
# 2. Reshape x input tensor:
x_input = tfs.reshape_input(x, input_shape)
# 3. Add First Convolution Layers: (Collect, layer, weights and biases)
hwb_list = [tfs.conv_layer_named(x_input, [W_x[0], W_y[0]], 1, N_FILTERS[0], [Str_X[0], Str_Y[0]], 'hc1', 'bc1',
                                 activation, Alphas[0])]
# 4. Add Additional Conv Layers as Needed:
for i in range(1, NUM_LAYERS):
    hwb_list.append(tfs.conv_layer(hwb_list[i - 1][0], [W_x[i], W_y[i]], N_FILTERS[i - 1], N_FILTERS[i],
                                   [Str_X[i], Str_Y[i]], activation, Alphas[i]))
# 4.1 Extract hidden layers to list
h = [i[0] for i in hwb_list]
# 5. Flatten and Fully Connect:
h_flat, flat_shape = tfs.flatten(h[NUM_LAYERS - 1])
h_fc = tfs.fully_connect_layer(h_flat, [flat_shape, UNITS_FC], [UNITS_FC], do, keep_prob, activation_fc, Alpha_fc)
# 6. Connect to form output:
y_conv = tfs.output_layer(h_fc, [UNITS_FC, NUMBER_CLASSES], [NUMBER_CLASSES])
# 7. Compute loss/cross-entropy for training step (using Adam gradient optimization):
train_step = tfs.train_loss(y, y_conv, learn_rate)
# 8. Output Node, compute accuracy [Softmax outputs, output int, accuracy]
outputs, prediction, accuracy = tfs.get_outputs(y, y_conv, output_node_name)
# 9. Initialize tensorflow for training & evaluation:
saver, init_op, config = tfs.tf_initialize()

# Enter Training Routine:
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # Print Model Information Before Starting.
    model_dims = tfs.get_model_dimensions(h, h_flat, h_fc, y_conv, NUM_LAYERS)
    filter_dims = tfs.get_filter_dimensions(W_x, W_y, Str_X, Str_Y, Alphas, NUM_LAYERS)
    print(model_dims, '\n', filter_dims)
    # Save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, Model_description + '.pbtxt', True)
    start_time_ms = tfs.current_time_ms()
    # Train Model:
    val_accuracy_rate = tfs.train(x, y, keep_prob, accuracy, train_step, x_train[:, :, 0], y_train, x_test[:, :, 0],
                                  y_test,
                                  keep_prob_feed, train_steps, train_batch_size=128)
    elapsed_time_ms = (tfs.current_time_ms() - start_time_ms)
    # Test Accuracy: (Test/Train Split)
    # tt_acc = tfs.test(sess, x, y, accuracy, x_test, y_test, keep_prob, test_type='Train-Split')
    # Validation Accuracy:
    # val_acc = tfs.test(sess, x, y, accuracy, x_val, y_val, keep_prob)
    # Confusion Matrix:
    conf_mat_v1, val_acc = tfs.confusion_matrix_test(sess, x, y, keep_prob, prediction, [1, *input_shape],
                                                     np.reshape(x_val[:, :, 0], [-1, 500, 1]), y_val, NUMBER_CLASSES)
    conf_mat_v2, val_acc2 = tfs.confusion_matrix_test(sess, x, y, keep_prob, prediction, [1, *input_shape],
                                                      np.reshape(x_val2[:, :, 0], [-1, 500, 1]), y_val2, NUMBER_CLASSES)
    print('Elapsed Time (ms): ', elapsed_time_ms)

    tfs.beep()
    # Save Statistics:
    # output_folder_name = EXPORT_DIRECTORY + 'S' + str(subject_number) + '/wlen' + str(win_len) + '/'
    output_folder_name = EXPORT_DIRECTORY + 'cnn2_stride_t2/'
    # TODO: create unique output directory
    # if not os.path.exists(output_folder_name):
    #     os.makedirs(output_folder_name)
    # stat_fn = 'stats_' + Model_description + '.mat'
    # user_input = input('Export Current Model?')
    # if user_input == "1" or user_input.lower() == "y":
    #     tfs.get_trained_vars(sess, output_folder_name + Model_description)
    #     tfs.save_statistics_v3(output_folder_name, val_accuracy_rate, Model_description, model_dims + filter_dims,
    #                            elapsed_time_ms, val_acc, conf_mat_v1, val_acc2, conf_mat_v2, stat_fn)
    #     tfs.get_all_activations(sess, x, keep_prob, [1, *input_shape], x_val, y_val,
    #                             output_folder_name + Model_description, h, h_flat, h_fc, y_conv)
    #     CHECKPOINT_FILE = EXPORT_DIRECTORY + Model_description + '.ckpt'
    #     saver.save(sess, CHECKPOINT_FILE)
    #     tfs.export_model([input_node_name, keep_prob_node_name], output_node_name,
    #                      EXPORT_DIRECTORY, Model_description)
