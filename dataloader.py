import os
import numpy as np
import random
import torch.utils.data as data
import cv2
from utils.utils_io import read_image, process_mask
import skimage


class Dataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, fix_color=True, masks_precomp=True):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        self.fix_color = fix_color
        self.masks_precomp = masks_precomp
        self.shadow_images = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_A')))
        self.shadow_free_images = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_C')))
        if self.masks_precomp:
            self.shadow_masks = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_B')))

    def __getitem__(self, index):
        shadow_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_A', self.shadow_images[index]))
        shadow_free_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_C', self.shadow_free_images[index]))

        if self.masks_precomp:
            shadow_mask = np.round(read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_B', self.shadow_masks[index]), grayscale=True))
        else:
            shadow_mask = self.compute_shadow_mask(shadow_image, shadow_free_image)

        name_img = self.shadow_images[index]

        if shadow_mask.shape[:2] != shadow_image.shape[:2]:
            shadow_mask = cv2.resize(shadow_mask, (shadow_image.shape[1], shadow_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.fix_color:
            shadow_free_image = self.color_alignment(shadow_image, shadow_mask, shadow_free_image)

        if self.augment:
            if random.random() < 0.5:
                shadow_image, shadow_free_image, shadow_mask = [cv2.flip(x, 1) for x in [shadow_image, shadow_free_image, shadow_mask]]
            if random.random() < 0.5:
                shadow_image, shadow_free_image, shadow_mask = [cv2.flip(x, 0) for x in [shadow_image, shadow_free_image, shadow_mask]]
            if random.random() < 0.5:
                shadow_image, shadow_free_image, shadow_mask = [cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE) for x in [shadow_image, shadow_free_image, shadow_mask]]

        if self.size is not None:
            shadow_image = cv2.resize(shadow_image, self.size)
            shadow_free_image = cv2.resize(shadow_free_image, self.size)
            shadow_mask = cv2.resize(shadow_mask, self.size, interpolation=cv2.INTER_NEAREST) 

        if len(shadow_mask.shape) == 2:
            shadow_mask = np.expand_dims(shadow_mask, axis=2)

        shadow_mask = process_mask(shadow_mask, min_size=100, max_size=100) ####

        mask_cont = self.contour(shadow_mask[:,:,0])

        shadow_image = np.transpose(shadow_image, (2, 0, 1))
        shadow_free_image = np.transpose(shadow_free_image, (2, 0, 1))
        shadow_mask = np.transpose(shadow_mask, (2, 0, 1))

        instance = {"shadow_image": shadow_image, 
                    "shadow_free_image": shadow_free_image, 
                    "shadow_mask": shadow_mask, 
                    "contour_mask": mask_cont,
                    "name": name_img}

        return instance  

    def contour(self, mask):
        """Generates a binary mask highlighting the expanded border region of the mask."""

        kernel_size = 40
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        padding = kernel_size // 2

        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        padded_mask = np.pad(mask, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        dilated_mask = cv2.dilate(padded_mask, kernel, iterations=1)
        border_mask2 = dilated_mask - padded_mask
        penumbra = border_mask2

        expanded = penumbra[padding:-padding, padding:-padding]
        expanded = (expanded > 0).astype(np.uint8)
        
        return expanded
    
    def color_alignment(self, shadow_image, shadow_mask, shadow_free_image):
        """Aligns the color of the shadow and shadow free images by histogram matching (without the mask)"""

        for ch in range(3):
            sh_im_mean = np.mean(shadow_image[shadow_mask == 0, ch])
            sh_im_std = np.std(shadow_image[shadow_mask == 0, ch])

            sh_free_im_mean = np.mean(shadow_free_image[:, :, ch])
            sh_free_im_std = np.std(shadow_free_image[:, :, ch])

            shadow_free_image[:, :, ch] = (shadow_free_image[:, :, ch] - sh_free_im_mean) / sh_free_im_std
            shadow_free_image[:, :, ch] = shadow_free_image[:, :, ch] * sh_im_std + sh_im_mean
        
            shadow_free_image[:, :, ch] = np.clip(shadow_free_image[:, :, ch], 0, 1)

        return shadow_free_image
    
    def compute_shadow_mask(self, shadow_image, shadow_free_image):
        """Computes shadow mask using otsu filter"""

        shadow_image = skimage.color.rgb2gray(shadow_image)
        shadow_free_image = skimage.color.rgb2gray(shadow_free_image)

        shadow_image = skimage.filters.gaussian(shadow_image, sigma=5)
        shadow_free_image = skimage.filters.gaussian(shadow_free_image, sigma=5)

        shadow_mask = np.abs(shadow_image - shadow_free_image)

        otsu_threshold = skimage.filters.threshold_otsu(shadow_mask)
        shadow_mask = np.where(shadow_mask > otsu_threshold, 1, 0).astype(np.float32)
        
        return shadow_mask

    def __len__(self):
        return len(self.shadow_images)