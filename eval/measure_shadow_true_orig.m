clear; close all; clc

% Directories
shadowdir = '/Users/vitto/Desktop/outputs_from_me/SRD/g4lobiNet/';
SD = dir(fullfile(shadowdir, '*.jpg'));
maskdir = '/Users/vitto/Desktop/SRD/test_B/';
MD = dir(fullfile(maskdir, '*.jpg'));
freedir = '/Users/vitto/Desktop/SRD/test_C/';
FD = dir(fullfile(freedir, '*.jpg'));

% Initialize variables
total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
numImages = numel(SD);
allmae = zeros(1, numImages);
smae = zeros(1, numImages);
nmae = zeros(1, numImages);
ppsnr = zeros(1, numImages);
ppsnrs = zeros(1, numImages);
ppsnrn = zeros(1, numImages);
sssim = zeros(1, numImages);
sssims = zeros(1, numImages);
sssimn = zeros(1, numImages);
cform = makecform('srgb2lab');

for i = 1:numImages
    % Read images
    sname = fullfile(shadowdir, SD(i).name);
    fname = fullfile(freedir, FD(i).name);
    mname = fullfile(maskdir, MD(i).name);
    
    s = imread(sname);
    f = imread(fname);
    m = imread(mname);

    s = imresize(s, [640, 840]);
    f = imresize(f, [640, 840]);
    m = imresize(m, [640, 840]);

    % Resize the mask if necessary
    if size(m, 1) ~= size(s, 1) || size(m, 2) ~= size(s, 2)
        m = imresize(m, [size(s, 1), size(s, 2)]); % Resize mask to match shadow image
    end

    % Ensure the mask is binary
    m = m > 0; % Assuming mask is binary, modify as necessary if not

    % Convert to double and normalize
    f = double(f) / 255;
    s = double(s) / 255;
    m = double(m) / 255; % Normalize mask if needed, it might already be binary.

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
%fprintf('PP-Lab (all, non-shadow, shadow):\n%f\t%f\t%f\n\n', mean(allmae), total_distn / total_pixeln, total_dists / total_pixels);
