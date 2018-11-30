%% analyze ptb outputs:
clr;
% load('incart_annotate\incart_annotate.mat');
load('incart_annotate\incartv2_ptb_predictions.mat');
idx = 1;
if exist('x_val', 'var')
    samples = size(x_val, 1);
    figure(1);
    for i = 1:samples
        fprintf('Sample #%d \n', i);
        total = sum(squeeze(y_val(i, :, :)));
        total2 = sum(squeeze(y_out(i, :, :)));
        if total(1)
            subplot(4, 1, 1); plot(x_val(i, :));
            subplot(4, 1, 2); plot(squeeze(y_prob(i, :, :)));
            subplot(4, 1, 3); plot(squeeze(y_out(i, :, :)));
            subplot(4, 1, 4); plot(squeeze(y_val(i, :, :))); title('true vals');
            accuracy = total2(1)/2000.0
            aaaaa = input('Continue? \n');
        end
    end
end