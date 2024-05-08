import numpy as np

def exposureRGB(img, R_a, R_b, G_a, G_b, B_a, B_b):
    rgb_img = img*1
    rgb_img[:, :, 0] = (rgb_img[:, :, 0]*R_a) + R_b
    rgb_img[:, :, 1] = (rgb_img[:, :, 1]*G_a) + G_b
    rgb_img[:, :, 2] = (rgb_img[:, :, 2]*B_a) + B_b
    rgb_img = np.clip(rgb_img, 0, 1)
    
    return rgb_img
