clr;
% load('ecg_cycle_gan_v1_r2\ecg_cycle_gan_v1_r2_lr0.0005_100.mat');  %opt: 250-300
% load('ecg_cycle_gan_v1_r3\ecg_cycle_gan_v1_r3_lr0.0005_300.mat');
% load('ptb_ecg_cycle_gan_v1_lr0.0002_r0\test_ptb_ecg_cycle_gan_v1_lr0.0002_r0_40epochs.mat');
load('I:\_gan_data_backup\ptb_ecg_cycle_gan_v1_lr0.0002_r0\ptb_ecg_cycle_gan_v1_lr0.0002_r0_40.mat');
for w = 1:size(x_val,1)
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