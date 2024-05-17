import os
import numpy as np
import random
import torch.utils.data as data
import cv2
from utils.utils_io import read_image
import skimage



class ISTDDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False, fix_color=True):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        self.fix_color = fix_color
        self.shadow_images = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_A'))
        self.shadow_masks = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_B'))
        self.shadow_free_images = os.listdir(os.path.join(root_dir, root_dir.split('/')[-1] + '_C'))

        # self.shadow_images = self.shadow_images[:100]
        # self.shadow_masks = self.shadow_masks[:100]
        # self.shadow_free_images = self.shadow_free_images[:100]


    def __getitem__(self, index):
        shadow_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_A', self.shadow_images[index]))
        shadow_mask = np.round(read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_B', self.shadow_masks[index]), grayscale=True))
        shadow_free_image = read_image(os.path.join(self.root_dir, self.root_dir.split('/')[-1] + '_C', self.shadow_free_images[index]))

        if self.fix_color:
            shadow_free_image = self.color_alignment(shadow_image, shadow_mask, shadow_free_image)

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
        

        return shadow_free_image

    def __len__(self):
        return len(self.shadow_images) 
         
class SBUDataset(data.Dataset):
    def __init__(self, root_dir, size=None, aug=False):
        self.augment = aug
        self.size = size
        self.root_dir = root_dir
        self.shadow_images = os.listdir(os.path.join(root_dir, "ShadowImages"))
        self.shadow_masks = os.listdir(os.path.join(root_dir, "ShadowMasks"))

        # Sort the images
        self.shadow_images.sort()
        self.shadow_masks.sort()

        # self.shadow_images = self.shadow_images[:100]
        # self.shadow_masks = self.shadow_masks[:100]
