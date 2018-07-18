clr; load('ecg_cycle_gan_v1_lr0.0005_90.mat');

for w = 1:size(x_val,1)
    figure(1);
    subplot(3, 2, 1);
    plot(x_val(w, :)); title('input-A');
    subplot(3, 2, 2); 
    plot(y_true(w, :)); title('input-B');
    subplot(3, 2, 3);
    plot(fake_B(w, :)); title('fake-B');
    subplot(3, 2, 4);
    plot(fake_A(w, :)); title('fake-A');
    subplot(3, 2, 5);
    plot(reconstr_A(w, :)); title('reconstr-A');
    subplot(3, 2, 6);
    plot(reconstr_B(w, :)); title('reconstr-B');
    x = input('Continue \n');
end