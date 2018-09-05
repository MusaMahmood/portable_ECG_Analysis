from scipy.io import savemat, loadmat

from tf_shared import load_data_v2, prep_dir, ind2vec

num_classes = 1
seq_length = 2000
input_length = seq_length
dir_x = 'ptb_ecg_1ch_temporal_labels/lead_v2'
# x_data, y_data = load_data_v2('data/' + dir_x, [seq_length, 1], [seq_length, 1], 'relevant_data', 'Y')
key_x = 'X'
key_y = 'Y'
x_data = loadmat(prep_dir('data/' + dir_x + '_all/') + 'all_x.mat').get(key_x)
y_data = loadmat(prep_dir('data/' + dir_x + '_all/') + 'all_x.mat').get(key_y)
print("Loaded Data Shape: X:", x_data.shape, " Y: ", y_data.shape)

y_data = ind2vec(y_data, dimensions=3)

savemat(prep_dir('data/' + dir_x + '_all/') + 'all_x.mat', mdict={key_x: x_data, key_y: y_data})


