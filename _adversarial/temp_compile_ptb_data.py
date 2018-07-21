import tf_shared as tfs
from scipy.io import savemat, loadmat

num_classes = 1
seq_length = 2000
input_length = seq_length
x_shape = [seq_length, 1]
y_shape = [seq_length, num_classes]
x_lead_v3, y_v3 = tfs.load_data_v2('data/ptb_ecg_lead_convert/lead_v2', [seq_length, 1], [1], 'relevant_data', 'Y')
x_lead_ii, y_ii = tfs.load_data_v2('data/ptb_ecg_lead_convert/lead_ii', [seq_length, 1], [1], 'relevant_data', 'Y')
key_x = 'X'
key_y = 'Y'

savemat(tfs.prep_dir('data/ptb_ecg_lead_convert/lead_v2_all/') + 'all_x.mat', mdict={key_x: x_lead_v3})
savemat(tfs.prep_dir('data/ptb_ecg_lead_convert/lead_ii_all/') + 'all_y.mat', mdict={key_y: x_lead_ii})


