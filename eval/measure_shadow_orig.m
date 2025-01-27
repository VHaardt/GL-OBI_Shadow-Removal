clear; close all; clc;

% 1. Modify the following directories before running

% Directories
shadowdir = '/Users/vitto/Desktop/outputs_from_me/SRD/g4lobiNet/';
maskdir = '/Users/vitto/Desktop/SRD/test_B/';
freedir = '/Users/vitto/Desktop/SRD/test_C/';

% Load image lists
SD = dir(fullfile(shadowdir, '*.jpg'));
MD = dir(fullfile(maskdir, '*.jpg'));
FD = dir(fullfile(freedir, '*.jpg'));

% Initialize arrays and variables
total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;

numImages = length(SD);
allmae = zeros(1, numImages);
smae = zeros(1, numImages);
nmae = zeros(1, numImages);
ppsnr = zeros(1, numImages);
ppsnrs = zeros(1, numImages);
ppsnrn = zeros(1, numImages);
sssim = zeros(1, numImages);
sssims = zeros(1, numImages);
sssimn = zeros(1, numImages);

% Color transformation
cform = makecform('srgb2lab');

for i = 1:numImages
    % Read images
    s = im2double(imread(fullfile(shadowdir, SD(i).name)));
    f = im2double(imread(fullfile(freedir, FD(i).name)));
    m = imread(strcat(maskdir, MD(i).name));

    % Create masks
    nmask = ~m;
    smask = ~nmask;

    % PSNR and SSIM calculations
    ppsnr(i) = psnr(s, f);
    ppsnrs(i) = psnr(s .* repmat(smask, [1, 1, 3]), f .* repmat(smask, [1, 1, 3]));
    ppsnrn(i) = psnr(s .* repmat(nmask, [1, 1, 3]), f .* repmat(nmask, [1, 1, 3]));
    sssim(i) = ssim(s, f);
    sssims(i) = ssim(s .* repmat(smask, [1, 1, 3]), f .* repmat(smask, [1, 1, 3]));
    sssimn(i) = ssim(s .* repmat(nmask, [1, 1, 3]), f .* repmat(nmask, [1, 1, 3]));

    % Color transformation to LAB space
    f = applycform(f, cform);
    s = applycform(s, cform);

    % MAE Calculations
    dist = abs(f - s);
    sdist = dist .* repmat(smask, [1, 1, 3]);
    ndist = dist .* repmat(nmask, [1, 1, 3]);

    sumsdist = sum(sdist(:));
    sumndist = sum(ndist(:));
    sumsmask = sum(smask(:));
    sumnmask = sum(nmask(:));

    % Per-image MAE
    allmae(i) = mean(dist(:));
    smae(i) = sumsdist / sumsmask;
    nmae(i) = sumndist / sumnmask;

    % Cumulative sums for non-shadow and shadow regions
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;

    disp(['Processing image: ', num2str(i)]);
end

% Final output
fprintf('PSNR (all, non-shadow, shadow):\n%f\t%f\t%f\n', mean(ppsnr), mean(ppsnrn), mean(ppsnrs));
fprintf('SSIM (all, non-shadow, shadow):\n%f\t%f\t%f\n', mean(sssim), mean(sssimn), mean(sssims));
fprintf('MAE (all, non-shadow, shadow):\n%f\t%f\t%f\n', mean(allmae), mean(nmae), mean(smae));
