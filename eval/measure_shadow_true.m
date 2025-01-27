clear; close all; clc

% Directories
shadowdir = '/Users/vitto/Desktop/outputs_from_me/g4lobiNet/';
SD = dir([shadowdir '/*.png']);
maskdir = '/Users/vitto/Desktop/Tesi/DATA/ISTD/test/test_B/';
MD = dir([maskdir '/*.png']);
freedir = '/Users/vitto/Desktop/Tesi/DATA/ISTD_adjusted/test_C_fixed/';
FD = dir([freedir '/*.png']);

% Initialize variables
total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae = zeros(1, numel(SD));
smae = zeros(1, numel(SD));
nmae = zeros(1, numel(SD));
ppsnr = zeros(1, numel(SD));
ppsnrs = zeros(1, numel(SD));
ppsnrn = zeros(1, numel(SD));
sssim = zeros(1, numel(SD));
sssims = zeros(1, numel(SD));
sssimn = zeros(1, numel(SD));
cform = makecform('srgb2lab');

for i = 1:numel(SD)
    % Read images
    sname = fullfile(shadowdir, SD(i).name);
    fname = fullfile(freedir, FD(i).name);
    mname = fullfile(maskdir, MD(i).name);
    s = imread(sname);
    f = imread(fname);
    m = imread(mname);

    % Convert to double and normalize
    f = double(f) / 255;
    s = double(s) / 255;
    m = imresize(m, [256 256]); % Resize masks
    s = imresize(s, [256 256]);
    f = imresize(f, [256 256]);

    % Define masks
    nmask = ~m; % Mask of non-shadow region
    smask = ~nmask; % Mask of shadow regions

    % Compute PSNR
    ppsnr(i) = psnr(s, f);
    ppsnrs(i) = psnr(s(repmat(smask, [1 1 3]) > 0), f(repmat(smask, [1 1 3]) > 0));
    ppsnrn(i) = psnr(s(repmat(nmask, [1 1 3]) > 0), f(repmat(nmask, [1 1 3]) > 0));

    % Compute SSIM
    sssim(i) = ssim(s, f);
    sssims(i) = ssim(s(repmat(smask, [1 1 3]) > 0), f(repmat(smask, [1 1 3]) > 0));
    sssimn(i) = ssim(s(repmat(nmask, [1 1 3]) > 0), f(repmat(nmask, [1 1 3]) > 0));

    % Convert to Lab color space
    f = applycform(f, cform);
    s = applycform(s, cform);

    % MAE calculations
    dist = abs(f - s);
    sdist = dist .* repmat(smask, [1 1 3]);
    ndist = dist .* repmat(nmask, [1 1 3]);
    
    sumsmask = sum(smask(:));
    sumnmask = sum(nmask(:));
    
    allmae(i) = sum(dist(:)) / (size(f, 1) * size(f, 2));
    smae(i) = sum(sdist(:)) / sumsmask;
    nmae(i) = sum(ndist(:)) / sumnmask;

    total_dists = total_dists + sum(sdist(:));
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sum(ndist(:));
    total_pixeln = total_pixeln + sumnmask;  

    disp(i);
end

fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n', mean(ppsnr), mean(ppsnrn), mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n', mean(sssim), mean(sssimn), mean(sssims));
fprintf('MAE(all,non-shadow,shadow):\n%f\t%f\t%f\n', mean(allmae), mean(nmae), mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n', mean(allmae), total_distn / total_pixeln, total_dists / total_pixels);
