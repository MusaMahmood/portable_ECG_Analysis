import os as os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model, Model
from scipy.io import loadmat
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def load_mat(file_path, key, shape):
    x_array = loadmat(file_path).get(key)
    x_array = np.reshape(x_array, [x_array.shape[0], *shape])
    print("Loaded Data Shape: ", key, ': ', x_array.shape)
    return x_array


def prep_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# Save graph/model:
def export_model_keras(keras_model='model.h5', export_dir="graph", model_name="temp_model_name", sequential=True,
                       custom_objects=None):
    if os.path.isfile(keras_model):
        if custom_objects is None:
            model = load_model(keras_model)
        else:
            model = load_model(keras_model, custom_objects=custom_objects)
    else:
        return

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # Serialize the model and get its weights, for quick re-building.
    config = model.get_config()
    weights = model.get_weights()

    # Re-build a model where the learning phase is now hard-coded to 0.
    if sequential:
        new_model = Sequential.from_config(config, custom_objects=custom_objects)
    else:
        new_model = Model.from_config(config, custom_objects=custom_objects)

    new_model.set_weights(weights)

    temp_dir = "graph"
    checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "untrained_input_graph.pb"

    # Temporary save graph to disk without weights included.
    saver = tf.train.Saver()
    checkpoint_path = saver.save(K.get_session(), checkpoint_prefix, global_step=0,
                                 latest_filename=checkpoint_state_name)
    tf.train.write_graph(K.get_session().graph, temp_dir, input_graph_name)

    input_graph_path = os.path.join(temp_dir, input_graph_name)
    # input_saver_def_path = ""
    input_saver_def_path = None
    input_binary = False
    input_node_names = [node.op.name for node in model.inputs]
    output_node_names = [node.op.name for node in model.outputs]
    output_node_names_assigned = ['output']  # [node.op.name for node in model.outputs]
    print("Input layer name: ", input_node_names)
    print("Output layer name: ", output_node_names)
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = export_dir + '/frozen_' + model_name + '.pb'
    clear_devices = True  # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.

    # Embed weights inside the graph and save to disk.
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, checkpoint_path, *output_node_names,
                              restore_op_name, filename_tensor_name, output_graph_path, clear_devices, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(export_dir + '/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names,
                                                                         output_node_names, tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(export_dir + '/opt_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", export_dir + '/frozen_' + model_name + '.pb')
    print("2 - Android Optimized Model:", export_dir + '/opt_' + model_name + '.pb')

    print_graph_nodes(export_dir + '/frozen_' + model_name + '.pb')

    return model


def print_graph_nodes(filename):
    g = tf.GraphDef()
    g.ParseFromString(open(filename, 'rb').read())
    print()
    print(filename)
    print("=======================INPUT=========================")
    print([n for n in g.node if n.name.find('input') != -1])
    print("=======================OUTPUT========================")
    print([n for n in g.node if n.name.find('output') != -1])
    print("===================KERAS_LEARNING=====================")
    print([n for n in g.node if n.name.find('keras_learning_phase') != -1])
    print("======================================================")
    print()