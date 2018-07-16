clr;
% load('seq2seq_only_prescal_lstmU32v3.mat'); % decent
% load('cnn2layerU1024v0.mat'); % complete trash
% load('conv_seq2seq_prescal_lstmU32v0.mat');
% load('n2ch\conv_seq2seq_prescal_lstmU32lr0.01v0.mat');
load('n1ch\flex_normal.fixed.conv1d_seq2seq_64lr0.01ep40_v1.mat');
% %{
CH1_ONLY = size(x_val, 3) - 1; 
PLOT = 0
score = 0; miss = 0;
acc = zeros(size(x_val,1), 1);

if exist('x_val', 'var')
    ytrue_all = vec2ind(reshape(y_val, [4998*2000,5])' );
    ytest_all = vec2ind(reshape(y_out, [4998*2000,5])' );
    acc = sum(ytest_all == ytrue_all)/length(ytrue_all);
    for s = 1:1:size(x_val,1)
        y_sample = squeeze(y_out(s, :, :));
        ysi = vec2ind(y_sample');
        y_true = squeeze(y_val(s, :, :));
        yti = vec2ind(y_true');
        syy = ysi == yti; pct = sum(syy)/size(y_val, 2);
        fprintf('Sample #: %d  Acc: %1.3f \n', s, pct); 
        if pct >= 0.9
            score = score + 1;
            acc(s) = 1;
        else
            miss = miss + 1;
            acc(s) = 0;
            PLOT=1;
        end
        clear b yy yt
        if PLOT
            PLOT = 0;
            sample =squeeze(x_val(s, :, :));
            [~, yl2] = max(y_sample, [], 2);
            figure(2); subplot(4, 1, 1); plot(sample); title('Data');
            subplot(4, 1, 2); plot(squeeze(y_prob(s, :, :))); title('y prob');
            subplot(4, 1, 3); plot(squeeze(y_out(s, :, :))); title('y out');
            subplot(4, 1, 4); plot(squeeze(y_val(s, :, :))); title('y true vals');
            xca = input('A1 Continue ? \n'); 
        end
        %assess accuracty
        
    end
    fprintf('Correct: %d, Miss %d \n', score, miss); 
end
s = size(y_val);
y_true = reshape(y_val, [s(1)*s(2), 5]);
y_pred = reshape(y_out, [s(1)*s(2), 5]);
sum(y_true)
sum(y_pred)
y_true = vec2ind(y_true');
y_pred = vec2ind(y_pred');
C = confusionmat(y_true, y_pred);
figure(1); conf_heatmap(C);
%}
%{
if exist('x_flex', 'var')
    for s = 322:2:size(x_flex,1)  %5206
        fprintf('Sample #: %d \n', s); 
        sample = squeeze(x_flex(s, :, :));
        figure(2);
        subplot(3, 1, 1); plot(sample); title('Data');
        subplot(3, 1, 2); plot(squeeze(y_prob_flex(s, :, :))); title('y prob');
        subplot(3, 1, 3); plot(squeeze(y_out_flex(s, :, :))); title('y out');
        xca = input('A1 Continue ? \n'); 
    end
end
s = size(y_out_flex);
y_pred = reshape(y_out_flex, [s(1)*s(2), 5]);
sum(y_pred)
%}
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
function [y] = within_n_percent(x, x2, n)
    %check if x1 within n of x2
    y = false;
    x_up = n*x + x;
    x_down = x - n*x;
    y = x2 < x_up & x2 > x_down;
end

function [y] = within_n_points(x, x2, n)
    y = false;
    x_up = x+n;
    x_down = x-n;
    y = x2 < x_up & x2 > x_down;
end