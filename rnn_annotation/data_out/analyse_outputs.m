clr;
load('seq2seq_data_1ch_v4.mat'); % Great results (Conv2d + BiLSTM)
% load('origmodel_prescal_data_1ch_v0.mat'); % Original LSTM-only
% load('gru_prescal_data_1ch_v0.mat'); % Too much noise/confusion
% load('seq2seq_prescal_lstmU640.mat');
CH1_ONLY = size(x_val, 3) - 1; 
if exist('x_flex', 'var')
    for s = 1520:size(x_flex,1)
        sample =squeeze(x_flex(s, :));
        y_label = squeeze(y_flex(s, :, :));
        [~, yl2] = max(y_label, [], 2);
        figure(1); subplot(2, 1, 1); plot(sample); hold on; plot(yl2); xlim([0, 1300]);
        subplot(2, 1, 2); plot(y_label); title('Predicted Annot'); xlim([0, 1300]);
        xca = input('A1 Continue ? \n'); clf(1);
    end
end

if ~CH1_ONLY
    for s = 1:size(x_val,1)
        sample = squeeze(x_val(s, :));
        y_label_real = squeeze(y_val(s, :, :));
        y_label_predict = squeeze(y_out(s, :, :));
        figure(1); subplot(3, 1, 1); plot(sample);hold on;
        subplot(3, 1, 2); plot(y_label_real);title('Real Annot');
        subplot(3, 1, 3); hold on; plot(y_label_predict); title('Predicted Annot');
        xca = input('Continue ? \n'); clf(1);
    end
else
    for s = 1:size(x_val,1)
        sample = squeeze(x_val(s, :, :));
        y_label_real = squeeze(y_val(s, :, :));
        y_label_predict = squeeze(y_out(s, :, :));
        figure(1); subplot(2, 1, 1); plot(sample(:, 1)); subplot(2, 1, 2); plot(sample(:, 2));
        subplot(2, 1, 1); hold on; plot(y_label_real);title('Real Annot');
        subplot(2, 1, 2); hold on; plot(y_label_predict); title('Predicted Annot');
        xca = input('Continue ? \n'); clf(1);
    end
end
% 1ch version