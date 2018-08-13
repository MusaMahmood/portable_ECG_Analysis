clr;
load('rat_n2ch\realtime_rat_data.mat');
y_prob2 = y_prob;
load('rat_n2ch\rat_2cnn.c1d_dense.lr0.005ep50_v1.mat');
% %{
PLOT = 1;
samples = size(y_prob2,1);
[M, I] = max(y_prob2, [], 2);
if exist('x_test', 'var')
    for s = 1:samples
        y_sample = squeeze(y_prob2(s, :, :));
        fprintf('Sample #:%d y_out=[%1.3f, %1.3f]\n', s, y_sample(1), y_sample(2)); 
        if PLOT && (y_sample(1) > y_sample(2))
            sample = squeeze(x_test(s, :, :));
            figure(1); subplot(2, 1, 1); plot(sample(:, 1)); title('Data-ch1');
            subplot(2, 1, 2); plot(sample(:, 2)); title('Data-ch2');
            
            sample_old = squeeze(x_val(s, :, :));
            figure(2); subplot(2, 1, 1); plot(sample_old(:, 1));
            subplot(2, 1, 2); plot(sample_old(:, 2)); 
            xca = input('A1 Continue ? \n'); 
        end
    end
end

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