import numpy as np

def exposureRGB(img, R_a, R_b, G_a, G_b, B_a, B_b):
    rgb_img = img.clone()
    rgb_img[:, :, 0] = (rgb_img[:, :, 0]*R_a) + R_b
    rgb_img[:, :, 1] = (rgb_img[:, :, 1]*G_a) + G_b
    rgb_img[:, :, 2] = (rgb_img[:, :, 2]*B_a) + B_b
    
    rgb_img = np.clip(rgb_img, 0, 1)
    
    return rgb_img

def exposureRGB_Tens(img, R_a, R_b, G_a, G_b, B_a, B_b):
    rgb_img = img.clone()  # Cloning to prevent modification of the original tensor
    
    # Applying the exposure transformation to each channel
    rgb_img[:, 0, :, :] = (rgb_img[:, 0, :, :] * R_a) + R_b
    rgb_img[:, 1, :, :] = (rgb_img[:, 1, :, :] * G_a) + G_b
    rgb_img[:, 2, :, :] = (rgb_img[:, 2, :, :] * B_a) + B_b
    
    # Clipping the values to maintain the range [0, 1]
    rgb_img = torch.clamp(rgb_img, 0, 1)
    
    return rgb_img
