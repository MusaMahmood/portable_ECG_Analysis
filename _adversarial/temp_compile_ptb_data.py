from scipy.io import savemat

from tf_shared import load_data_v2, prep_dir

num_classes = 1
seq_length = 2000
input_length = seq_length
dir_x = 'ptb_ecg_2ch_half_overlap/lead_v23'
dir_y = 'ptb_ecg_2ch_half_overlap/lead_ii'
# dir_x = 'flex_overlap'
# dir_y = 'br_overlap'
x_lead_v3, y_v3 = load_data_v2('data/' + dir_x, [seq_length, 2], [1], 'relevant_data', 'Y')
x_lead_ii, y_ii = load_data_v2('data/' + dir_y, [seq_length, 1], [1], 'relevant_data', 'Y')
key_x = 'X'
key_y = 'Y'

savemat(prep_dir('data/' + dir_x + '_all/') + 'all_x.mat', mdict={key_x: x_lead_v3})
savemat(prep_dir('data/' + dir_y + '_all/') + 'all_y.mat', mdict={key_y: x_lead_ii})
