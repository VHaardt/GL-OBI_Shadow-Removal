import os
import numpy as np
import random
import torch.utils.data as data
import cv2
from utils.utils_io import read_image
import skimage
from scipy.ndimage import label, find_objects, binary_closing, generate_binary_structure

import ipdb


class ISTDDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, fix_color=True):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        self.fix_color = fix_color
        self.shadow_images = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_A'))
        self.shadow_masks = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_B'))
        self.shadow_free_images = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_C'))


    def __getitem__(self, index):
        shadow_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_A', self.shadow_images[index]))
        shadow_mask = np.round(read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_B', self.shadow_masks[index]), grayscale=True))
        shadow_free_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_C', self.shadow_free_images[index]))

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

        shadow_mask = self.process_mask(shadow_mask, min_size=100, max_size=100) ####
        
        crop_coordinate = self.crop_region(shadow_mask) #######

        shadow_image = np.transpose(shadow_image, (2, 0, 1))
        shadow_free_image = np.transpose(shadow_free_image, (2, 0, 1))
        shadow_mask = np.transpose(shadow_mask, (2, 0, 1))

        instance = {"shadow_image": shadow_image, "shadow_free_image": shadow_free_image, "shadow_mask": shadow_mask, "crop_coordinate": crop_coordinate}

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
        

        return shadow_free_image
    
    def remove_small_objects(self, mask, min_size):
        labeled_mask, num_features = label(mask)
        output_mask = np.zeros_like(mask, dtype=bool)
        for i in range(1, num_features + 1):
            component = (labeled_mask == i)
            if np.sum(component) >= min_size:
                output_mask = np.logical_or(output_mask, component)
    
        return output_mask

    def fill_small_holes(self, mask, max_size):
        inverted_mask = np.logical_not(mask)
        labeled_holes, num_features = label(inverted_mask)
        output_mask = np.copy(mask)
        for i in range(1, num_features + 1):
            hole = (labeled_holes == i)
            if np.sum(hole) <= max_size:
                output_mask = np.logical_or(output_mask, hole)
        
        return output_mask

    def process_mask(self, mask, min_size=100, max_size=100):
        cleaned_mask = self.remove_small_objects(mask, min_size)
        final_mask = self.fill_small_holes(cleaned_mask, max_size)
        
        return final_mask
    
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
    
