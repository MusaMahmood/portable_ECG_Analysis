clr;
% load('I:\_gan_data_backup\compute_engine\ptb_ecg_cycle_gan_leadv2_lr0.0002_r0_230.mat');
load('I:\_gan_data_backup\compute_engine\ptb_ecg_cycle_gan_leadv2_lr0.0002_r0_230.mat');
% load('I:\_gan_data_backup\compute_engine\test_ptb_ecg_cycle_gan_leadv2_lr0.0002_r0_[[430]]epochs.mat');
for w = 28430:size(x_val,1)
    fprintf('Sample %d \n', w)
    figure(1);
    subplot(3, 2, 1);
    plot(x_val(w, :)); title('input-A'); ylim([0, 1])
    subplot(3, 2, 2); 
    plot(y_true(w, :)); title('input-B'); ylim([0, 1])
    subplot(3, 2, 3);
    plot(fake_B(w, :)); title('fake-B'); ylim([0, 1])
    subplot(3, 2, 4);
    plot(fake_A(w, :)); title('fake-A'); ylim([0, 1])
    subplot(3, 2, 5);
    plot(reconstr_A(w, :)); title('reconstr-A'); ylim([0, 1])
    subplot(3, 2, 6);
    plot(reconstr_B(w, :)); title('reconstr-B'); ylim([0, 1])
    x = input('Continue \n');
end
% Restart & Redo R3, change LR to 2E-4 from 5E-4
% Need to explain everything in terms of reducing loss. Read original
% 1703.10593 ArXiv paper. 