clr;
load('seq2seq_prescal_lstmU32v2.mat'); % near perfect annotation. 
CH1_ONLY = size(x_val, 3) - 1; 
if exist('x_val', 'var')
    for s = 1:size(x_val,1)
        fprintf('Sample #: %d \n', s); 
        sample =squeeze(x_val(s, :, :));
        y_sample = squeeze(y_out(s, :, :));
        [~, yl2] = max(y_sample, [], 2);
        figure(1); clf(1); subplot(2, 1, 1); plot(sample); hold on; plot(yl2); xlim([0, 1300]);
        subplot(2, 1, 2); plot(y_sample); title('Predicted Annot'); xlim([0, 1300]);
        figure(2); subplot(3, 1, 1); plot(sample); subplot(3, 1, 2); plot(squeeze(y_prob(s, :, :)));
        subplot(3, 1, 3); plot(squeeze(y_val(s, :, :)));
        xca = input('A1 Continue ? \n'); 
    end
end

y_true = reshape(y_val, [14705*1000, 5]);
y_pred = reshape(y_out, [14705*1000, 5]);
sum(y_true)
sum(y_pred)
%{
if ~CH1_ONLY
    for s = 1:size(x_val,1)
        sample = squeeze(x_val(s, :));
        y_label_real = squeeze(y_val(s, :, :));
        y_label_predict = squeeze(y_out(s, :, :));
        figure(1); subplot(3, 1, 1); plot(sample);hold on;
        subplot(3, 1, 2); plot(y_label_real);title('Real Annot');
        subplot(3, 1, 3); hold on; plot(y_label_predict); title('Predicted Annot');
        xca = input('Continue ? \n'); c
lf(1);
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
%}

% 1ch version