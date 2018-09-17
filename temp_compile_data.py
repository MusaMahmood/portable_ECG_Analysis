from scipy.io import savemat

from tf_shared import load_data_v2, prep_dir, ind2vec

data_directory_name = 'ptb_6class_temporal/lead_v2'
x_data, y_data = load_data_v2('data/' + data_directory_name, [2000, 1], [2000, 1], 'relevant_data', 'Y')

# New Keys:
key_x = 'X'
key_y = 'Y'

y_data = ind2vec(y_data)
print("Updated Data Shape: X:", x_data.shape, " Y: ", y_data.shape)

savemat(prep_dir('data/' + data_directory_name + '_all/') + 'all_data.mat', mdict={key_x: x_data, key_y: y_data})

# modify labels:
