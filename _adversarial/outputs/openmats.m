clr; load('ecg_conversion_64lr0.01ep200_v1.mat');
% for f=1:3
%     figure(f)
%     for i=1:100
%         subplot(10, 10, i);
%         plot(x);
%     end
% end

for w = 1:size(x_val,1)
    figure(1);
    subplot(3, 1, 1);
    plot(x_val(w, :)); title('x-val-input');
    subplot(3, 1, 2); 
    plot(y_pred(w, :)); title('y-pred-output');
    subplot(3, 1, 3);
    plot(y_true(w, :)); title('y-troo');
    x = input('Continue \n');
end