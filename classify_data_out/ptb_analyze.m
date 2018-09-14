%% analyze ptb outputs:
clr;
load('incart_annotate\incart_annotate.mat');
if exist('x_val', 'var')
    samples = size(x_val, 1);
    figure(1);
    for i = 1:samples
        fprintf('Sample #%d \n', i);
        subplot(3, 1, 1); plot(x_val(i, :));
        subplot(3, 1, 2); plot(squeeze(y_prob(i, :, :)));
        subplot(3, 1, 3); plot(squeeze(y_out(i, :, :)));
        aaaaa = input('Continue? \n');        
    end
end