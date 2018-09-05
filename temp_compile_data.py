from scipy.io import savemat

from tf_shared import load_data_v2, prep_dir, ind2vec

num_classes = 1
seq_length = 2000
input_length = seq_length
data_directory_name = 'extended_5_class/mit_bih_tlabeled_w8s_fixed'
relevant_data, y_data = load_data_v2('data/' + data_directory_name, [seq_length, 2], [seq_length, 5], 'relevant_data', 'Y')
key_x = 'relevant_data'
key_y = 'Y'

# y_data =

savemat(prep_dir('data/' + data_directory_name + '_all/') + 'all_data.mat', mdict={key_x: relevant_data,
                                                                                   key_y: y_data})

# modify labels:
