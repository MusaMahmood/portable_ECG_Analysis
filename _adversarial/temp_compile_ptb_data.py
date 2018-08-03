import tf_shared as tfs
from scipy.io import savemat, loadmat

num_classes = 1
seq_length = 2000
input_length = seq_length
x_shape = [seq_length, 1]
y_shape = [seq_length, num_classes]
# dir_x = 'ptb_ecg_lead_convert/lead_v2'
# dir_y = 'ptb_ecg_lead_convert/lead_ii'
dir_x = 'flex_overlap'
dir_y = 'br_overlap'
x_lead_v3, y_v3 = tfs.load_data_v2('data/' + dir_x, [seq_length, 1], [1], 'relevant_data', 'Y')
x_lead_ii, y_ii = tfs.load_data_v2('data/' + dir_y, [seq_length, 1], [1], 'relevant_data', 'Y')
key_x = 'X'
key_y = 'Y'

savemat(tfs.prep_dir('data/' + dir_x + '_all/') + 'all_x.mat', mdict={key_x: x_lead_v3})
savemat(tfs.prep_dir('data/' + dir_y + '_all/') + 'all_y.mat', mdict={key_y: x_lead_ii})


