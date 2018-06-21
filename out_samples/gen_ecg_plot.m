%
clr;
dir = '1ch_v2/';
dir = 'flexEcggan_new/';
dir = 'mit_db_ds_small_gan_new/';
list = 0:10:10000;
for i = 1:length(list)
    if exist([dir num2str(list(i)) '.mat'])
        load([dir num2str(list(i)) '.mat']);
        gen_all(i, :, :) = gen0;
    end
end
for j = 1
    offset = (j-1)*100;
    for i = 1:100
        figure(10 + j); subplot(10, 10, i);
        plot( squeeze(gen_all(i + offset, :, 1)) );
    end
end


%%
dir = 'flexEcggan_new/'; load([dir '200.mat']); figure(1); plot(gen0); 