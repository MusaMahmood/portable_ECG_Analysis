clr;
load('data_v2.mat');
for s = 1:size(x_val,1)
    sample = squeeze(x_val(s, :, :));
    y_label_real = squeeze(y_val(s, :, :));
    y_label_predict = squeeze(y_out(s, :, :));
    figure(1); subplot(2, 1, 1); plot(sample(:, 1)); subplot(2, 1, 2); plot(sample(:, 2));
    subplot(2, 1, 1); hold on; plot(y_label_real);title('Real Annot');
    subplot(2, 1, 2); hold on; plot(y_label_predict); title('Predicted Annot');
    xca = input('Continue ? \n'); clf(1);
end