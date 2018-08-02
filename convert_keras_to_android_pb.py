from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras.models import Sequential, load_model
from keras import backend as K
from keras.engine.saving import preprocess_weights_for_loading
import tensorflow as tf
import tf_shared as tfs
import os


def print_weights(weights, message=''):
    print(message)
    for w in weights:
        print(w.size)


# Using Function:
# model = tfs.export_model_keras(h5_filename, export_dir=tfs.prep_dir("graph"), model_name="temp_model_name")
model_name = 'rat_2cnn.c1d_s2s_64lr0.01ep40_v1'
h5_filename = model_name + '.h5'
alt_h5_filename = 'alt_' + h5_filename

if os.path.isfile(alt_h5_filename):
    k_model_alt = load_model(alt_h5_filename)
    k_config = k_model_alt.get_config()
    SWAP_CUDNN_LAYER = True
else:
    k_config = None
    SWAP_CUDNN_LAYER = False

# Load existing model.
export_dir = 'graph'
# Import graph that uses CuDNN
if os.path.isfile(h5_filename):
    model = load_model(h5_filename)
else:
    print("ERROR: Model Not Found")
    exit(-1)

# All new operations will be in test mode from now on.
K.set_learning_phase(0)

# Serialize the model and get its weights, for quick re-building.
original_config = model.get_config()
original_weights = model.get_weights()
# This method restores the model from the config, after freezing the model.
# If SWAP_CUDNN_LAYER, this means we have an optimized model (without CuDNN training components),
# and the weights have to be adjusted accordingly.
if SWAP_CUDNN_LAYER:
    output_model = Sequential.from_config(k_config)
    # print_weights(original_weights, 'CudnnLSTM_weights')
    weights_fixed = preprocess_weights_for_loading(output_model, original_weights)
    # print_weights(weights_fixed, 'FIXED: LSTM_weights')
    output_model.set_weights(weights_fixed)
else:
    # If we don't need to adjust any variables, then we roll with the original config/weights.
    output_model = Sequential.from_config(original_config)
    output_model.set_weights(original_weights)

# Re-build a model where the learning phase is now hard-coded to 0.

temp_dir = "graph"
checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "untrained_input_graph.pb"
output_graph_name = "final_output_graph.pb"

# Temporary save graph to disk without weights included.
saver = tf.train.Saver()
checkpoint_path = saver.save(K.get_session(), checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
tf.train.write_graph(K.get_session().graph, temp_dir, input_graph_name)

input_graph_path = os.path.join(temp_dir, input_graph_name)
input_saver_def_path = ""
input_binary = False
input_node_names = [node.op.name for node in output_model.inputs]
output_node_names = [node.op.name for node in output_model.outputs]
print("Input layer name: ", input_node_names)
print("Output layer name: ", output_node_names)
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = export_dir + '/frozen_' + model_name + '.pb'
clear_devices = True

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

tfs.print_graph_nodes(export_dir + '/frozen_' + model_name + '.pb')
tfs.print_graph_nodes(export_dir + '/opt_' + model_name + '.pb')
