clear; clc; close all;

print_levels = [0, 0.5, 1];

timestamp = datestr(now, 'yyyymmdd_HHMMSS');

[filename, pathname] = uigetfile('*.*', '选择彩色图像文件');
if isequal(filename, 0)
    disp('用户取消了选择');
    return;
end

originalImg = imread(fullfile(pathname, filename));
originalImg = im2double(originalImg);
[M, N, ~] = size(originalImg);

figure('Name', '原始图像及通道');
subplot(1,3,1); imshow(originalImg); title('原始彩色图像');

R_channel = originalImg(:,:,1);
G_channel = originalImg(:,:,2);
B_channel = originalImg(:,:,3);

subplot(1,3,2); imshow(R_channel); title('红色通道');
subplot(1,3,3); imshow(cat(3, R_channel, G_channel*0.8, B_channel*0.8)); title('通道分离演示');

holograms = cell(1,3);
quantized_amps = cell(1,3);
channel_names = {'Red', 'Green', 'Blue'};

if ~exist(timestamp, 'dir')
    mkdir(timestamp);
end

for c = 1:3
    rng(2025 + c); 
    channel = originalImg(:,:,c);
    
    rand_phase = exp(1i * 2 * pi * rand(M, N));
    obj_wave = channel .* rand_phase;
    
    hologram = fftshift(fft2(obj_wave));
    holograms{c} = hologram;
    
    phase_mat = angle(hologram);
    save(fullfile(timestamp, sprintf('%s_%s_phase.mat', timestamp, channel_names{c})), 'phase_mat');
    
    amp = abs(hologram);
    amp_norm = mat2gray(amp); 
    
    amp_flat = amp_norm(:);
    Npix = numel(amp_flat);
    target_counts = floor(Npix / 3) * [1 1 1]; 
    
    [~, idx_sorted] = sort(amp_flat);
    quantized_flat = zeros(size(amp_flat));
    start_idx = 1;
    for k = 1:3
        if k < 3
            end_idx = start_idx + target_counts(k) - 1;
        else
            end_idx = Npix;
        end
        quantized_flat(idx_sorted(start_idx:end_idx)) = print_levels(k);
        start_idx = end_idx + 1;
    end
    quantized_amp = reshape(quantized_flat, [M, N]);
    quantized_amps{c} = quantized_amp;
    
    imwrite(quantized_amp, fullfile(timestamp, sprintf('%s %s quantized hologram.tif', timestamp, channel_names{c})));
    writematrix(quantized_amp, fullfile(timestamp, sprintf('%s %s quantized hologram.xlsx', timestamp, channel_names{c})));
    
    imwrite(mat2gray(log(abs(hologram) + 1)), fullfile(timestamp, sprintf('%s %s hologram.png', timestamp, channel_names{c})));
end

disp(['全息图生成完毕，文件保存在文件夹: ', timestamp]);
clear; clc; close all;

print_levels = [0, 0.5, 1]; 

timestamp = input('请输入全息处理时的时间戳（例如 20250618_114348）: ','s');

channel_names = {'Red', 'Green', 'Blue'};
recons = cell(1,3);

for c = 1:3
    quantized_file = fullfile(timestamp, sprintf('%s %s quantized hologram.xlsx', timestamp, channel_names{c}));
    quantized_amp = readmatrix(quantized_file);
    
    phase_file = fullfile(timestamp, sprintf('%s_%s_phase.mat', timestamp, channel_names{c}));
    if ~exist(phase_file, 'file')
        error('缺少相位文件：%s，请确保color2生成时保存了相位', phase_file);
    end
    load(phase_file, 'phase_mat');
    
    if ~isequal(size(quantized_amp), size(phase_mat))
        error('量化幅度与相位矩阵大小不一致！');
    end
    
    hologram_complex = quantized_amp .* exp(1i * phase_mat);
    
    rec_img = abs(ifft2(ifftshift(hologram_complex)));
    rec_img = mat2gray(rec_img);
    recons{c} = rec_img;
end

color_recon = cat(3, recons{1}, recons{2}, recons{3});

figure('Name', '彩色全息重建图像');
imshow(color_recon);
title('基于三阶灰度量化及原始相位重建');

disp('重建完成。');



ref = originalImg(:,:,1);      
rec = recons{1};               

psnr_val = psnr(rec, ref);
ssim_val = ssim(rec, ref);

figure('Name', '红色通道重建质量指标', 'Color', 'w');

subplot(1,2,1);
bar(psnr_val, 'FaceColor', [0.2 0.6 0.9]);
ylim([0 50]); grid on;
ylabel('PSNR (dB)');
title('红色通道 PSNR');

subplot(1,2,2);
bar(ssim_val, 'FaceColor', [0.9 0.5 0.3]);
ylim([0 1]); grid on;
ylabel('SSIM');
title('红色通道 SSIM');

saveas(gcf, fullfile(timestamp, sprintf('%s Red PSNR_SSIM.png', timestamp)));
