import os
import numpy as np
import random
import torch.utils.data as data
import cv2
from utils.utils_io import read_image, process_mask
import skimage
#from scipy.ndimage import label, find_objects, binary_closing, generate_binary_structure

import ipdb


class ISTDDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, fix_color=True):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        self.fix_color = fix_color
        self.shadow_images = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_A')))
        self.shadow_masks = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_B')))
        self.shadow_free_images = sorted(os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_C')))


    def __getitem__(self, index):
        shadow_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_A', self.shadow_images[index]))
        shadow_mask = np.round(read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_B', self.shadow_masks[index]), grayscale=True))
        shadow_free_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_C', self.shadow_free_images[index]))
        name_img = self.shadow_images[index]

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
        
        crop_coordinate = self.crop_region(shadow_mask) 

        img_exp = self.exposed_img(shadow_image, shadow_free_image, shadow_mask)

        shadow_image = np.transpose(shadow_image, (2, 0, 1))
        shadow_free_image = np.transpose(shadow_free_image, (2, 0, 1))
        shadow_mask = np.transpose(shadow_mask, (2, 0, 1))
        img_exp = np.transpose(img_exp, (2, 0, 1))

        instance = {"shadow_image": shadow_image, 
                    "shadow_free_image": shadow_free_image, 
                    "shadow_mask": shadow_mask, 
                    "crop_coordinate": crop_coordinate,
                    "exposed_image": img_exp,
                    "name": name_img}

        return instance
    
    def exposed_img(self, img_s, img_ns, mask):
        adjusted_image = img_s.copy()
        for c in range(img_s.shape[2]):
            # Extract the channel from the source and target images
            source_image_ch = img_s[:, :, c]
            target_image_ch = img_ns[:, :, c]

            # Apply the mask to select only masked pixels
            masked_source_pixels = source_image_ch[mask[:, :, 0] == 1]
            masked_target_pixels = target_image_ch[mask[:, :, 0] == 1]

            # Calculate the mean and std deviation for masked pixels
            mu = np.mean(masked_source_pixels)
            sd = np.std(masked_source_pixels)
            mu_t = np.mean(masked_target_pixels)
            sd_t = np.std(masked_target_pixels)

            adjusted_image_ch = adjusted_image[:, :, c].astype(np.float32)
            adjusted_image_ch = mu_t + (adjusted_image_ch - mu) * (sd_t/sd if sd != 0 else 1)
            adjusted_image[:, :, c] = np.clip(adjusted_image_ch, 0, 255)
        
        return adjusted_image

    
    def color_alignment(self, shadow_image, shadow_mask, shadow_free_image):
        """Aligns the color of the shadow and shadow free images by histogram matching (without the mask)"""
        # Compute mean and std of the shadow image (the shadow region is removed from the computation)


        for ch in range(3):
            sh_im_mean = np.mean(shadow_image[shadow_mask == 0, ch])
            sh_im_std = np.std(shadow_image[shadow_mask == 0, ch])

            # Align the shadow free image to the shadow image
            sh_free_im_mean = np.mean(shadow_free_image[:, :, ch])
            sh_free_im_std = np.std(shadow_free_image[:, :, ch])

            shadow_free_image[:, :, ch] = (shadow_free_image[:, :, ch] - sh_free_im_mean) / sh_free_im_std
            shadow_free_image[:, :, ch] = shadow_free_image[:, :, ch] * sh_im_std + sh_im_mean
        
            shadow_free_image[:, :, ch] = np.clip(shadow_free_image[:, :, ch], 0, 1) #aggiunta

        return shadow_free_image
    
    def crop_region(self, shadow_mask, m = 40):
        """Crop the image around the scattered points in the mask ensuring minimum size"""
        # Find all points where the mask is 1
        points = np.argwhere(shadow_mask == 1)
        
        # Extract the bounding box around all scattered points where mask is 1
        min_w = np.min(points[:, 1])
        min_h = np.min(points[:, 0])
        max_w = np.max(points[:, 1])
        max_h = np.max(points[:, 0])
        
        # Ensure the bounding box has a minimum size
        width = max_w - min_w + 1
        height = max_h - min_h + 1

        if width < m:
            pad_w = (m - width) // 2
            min_w = max(0, min_w - pad_w)
            max_w = min(shadow_mask.shape[1] - 1, max_w + pad_w + (m - width) % 2)
            
        if height < m:
            pad_h = (m - height) // 2
            min_h = max(0, min_h - pad_h)
            max_h = min(shadow_mask.shape[0] - 1, max_h + pad_h + (m - height) % 2)

        # Return the bounding box coordinates as a numpy array
        bbox = np.array([min_h, max_h, min_w, max_w])#np.array([min_h, max_h, min_w, max_w])
        return bbox


    def __len__(self):
        return len(self.shadow_images)
    
    
class SRDDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, fix_color=True, masks_precomp=False):
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

        if np.all(process_mask(np.expand_dims(shadow_mask, axis=2), min_size=100, max_size=100) == 0):
            shadow_mask = self.compute_shadow_mask(shadow_image, shadow_free_image)

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


        shadow_image = np.transpose(shadow_image, (2, 0, 1))
        shadow_free_image = np.transpose(shadow_free_image, (2, 0, 1))
        shadow_mask = np.transpose(shadow_mask, (2, 0, 1))

        instance = {"shadow_image": shadow_image, "shadow_free_image": shadow_free_image, "shadow_mask": shadow_mask}

        return instance
    
    def color_alignment(self, shadow_image, shadow_mask, shadow_free_image):
        """Aligns the color of the shadow and shadow free images by histogram matching (without the mask)"""
        # Compute mean and std of the shadow image (the shadow region is removed from the computation)


        for ch in range(3):
            sh_im_mean = np.mean(shadow_image[shadow_mask == 0, ch])
            sh_im_std = np.std(shadow_image[shadow_mask == 0, ch])

            # Align the shadow free image to the shadow image
            sh_free_im_mean = np.mean(shadow_free_image[:, :, ch])
            sh_free_im_std = np.std(shadow_free_image[:, :, ch])

            shadow_free_image[:, :, ch] = (shadow_free_image[:, :, ch] - sh_free_im_mean) / sh_free_im_std
            shadow_free_image[:, :, ch] = shadow_free_image[:, :, ch] * sh_im_std + sh_im_mean
        
            shadow_free_image[:, :, ch] = np.clip(shadow_free_image[:, :, ch], 0, 1) #aggiunta

        return shadow_free_image
    
    def compute_shadow_mask(self, shadow_image, shadow_free_image):
        # convert to grayscale
        shadow_image = skimage.color.rgb2gray(shadow_image)
        shadow_free_image = skimage.color.rgb2gray(shadow_free_image)

        # Blur the images
        shadow_image = skimage.filters.gaussian(shadow_image, sigma=5)
        shadow_free_image = skimage.filters.gaussian(shadow_free_image, sigma=5)

        shadow_mask = np.abs(shadow_image - shadow_free_image)

        # Binarize the mask using otsu thresholding
        otsu_threshold = skimage.filters.threshold_otsu(shadow_mask)
        shadow_mask = np.where(shadow_mask > otsu_threshold, 1, 0).astype(np.float32)
        
        return shadow_mask


    def __len__(self):
        return len(self.shadow_images)
    


class WSRDDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, masks_precomp=False):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        
        self.shadow_images = sorted(os.listdir(os.path.join(root_dir, 'shadow_affected')))
        self.shadow_free_images = sorted(os.listdir(os.path.join(root_dir, 'shadow_free')))
        self.masks_precomp = masks_precomp
        if self.masks_precomp:
            self.shadow_masks = sorted(os.listdir(os.path.join(root_dir, 'shadow_masks')))


    def __getitem__(self, index):
        shadow_image = read_image(os.path.join(self.root_dir, 'shadow_affected', self.shadow_images[index]))
        shadow_free_image = read_image(os.path.join(self.root_dir, 'shadow_free', self.shadow_free_images[index]))

        if self.masks_precomp:
            shadow_mask = read_image(os.path.join(self.root_dir, 'shadow_masks', self.shadow_masks[index]), grayscale=True)
        else:
            shadow_mask = self.compute_shadow_mask(shadow_image, shadow_free_image)
        

        if self.augment:
            if random.random() < 0.5:
                shadow_image = cv2.flip(shadow_image, 1)
                shadow_free_image = cv2.flip(shadow_free_image, 1)
                shadow_mask = cv2.flip(shadow_mask, 1)
            if random.random() < 0.5:
                shadow_image = cv2.flip(shadow_image, 0)
                shadow_free_image = cv2.flip(shadow_free_image, 0)
                shadow_mask = cv2.flip(shadow_mask, 0)
            if random.random() < 0.5:
                shadow_image = cv2.rotate(shadow_image, cv2.ROTATE_90_CLOCKWISE)
                shadow_free_image = cv2.rotate(shadow_free_image, cv2.ROTATE_90_CLOCKWISE)
                shadow_mask = cv2.rotate(shadow_mask, cv2.ROTATE_90_CLOCKWISE)
        
        if self.size is not None:
            shadow_image = cv2.resize(shadow_image, self.size)
            shadow_free_image = cv2.resize(shadow_free_image, self.size)
            shadow_mask = cv2.resize(shadow_mask, self.size, interpolation=cv2.INTER_NEAREST)

        
        if len(shadow_mask.shape) == 2:
            shadow_mask = np.expand_dims(shadow_mask, axis=2)

        shadow_mask = process_mask(shadow_mask, min_size=100, max_size=100) ####
        
        shadow_image = np.transpose(shadow_image, (2, 0, 1))
        shadow_free_image = np.transpose(shadow_free_image, (2, 0, 1))
        shadow_mask = np.transpose(shadow_mask, (2, 0, 1))
        
        instance = {"shadow_image": shadow_image, "shadow_free_image": shadow_free_image, "shadow_mask": shadow_mask}

        return instance

    def compute_shadow_mask(self, shadow_image, shadow_free_image):
        # convert to grayscale
        shadow_image = skimage.color.rgb2gray(shadow_image)
        shadow_free_image = skimage.color.rgb2gray(shadow_free_image)

        # Blur the images
        shadow_image = skimage.filters.gaussian(shadow_image, sigma=5)
        shadow_free_image = skimage.filters.gaussian(shadow_free_image, sigma=5)

        shadow_mask = np.abs(shadow_image - shadow_free_image)

        # Binarize the mask using otsu thresholding
        otsu_threshold = skimage.filters.threshold_otsu(shadow_mask)
        shadow_mask = np.where(shadow_mask > otsu_threshold, 1, 0).astype(np.float32)
        
        return shadow_mask

    def __len__(self):
        return len(self.shadow_images)
