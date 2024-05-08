import numpy as np

def exposureRGB(img, R_a, R_b, G_a, G_b, B_a, B_b):
    rgb_img = img.clone()
    rgb_img[:, :, 0] = (rgb_img[:, :, 0]*R_a) + R_b
    rgb_img[:, :, 1] = (rgb_img[:, :, 1]*G_a) + G_b
    rgb_img[:, :, 2] = (rgb_img[:, :, 2]*B_a) + B_b
    
    rgb_img = np.clip(rgb_img, 0, 1)
    
    return rgb_img

def exposureRGB(inp, out_g):
    # Splitting the output tensor into individual parameter tensors
    R_a = out_g[:, 0, :, :]
    R_b = out_g[:, 1, :, :]
    G_a = out_g[:, 2, :, :]
    G_b = out_g[:, 3, :, :]
    B_a = out_g[:, 4, :, :]
    B_b = out_g[:, 5, :, :]

    # Applying the exposure transformation to the input image
    rgb_img = inp.clone()
    rgb_img[:, 0, :, :] = (inp[:, 0, :, :] * R_a) + R_b
    rgb_img[:, 1, :, :] = (inp[:, 1, :, :] * G_a) + G_b
    rgb_img[:, 2, :, :] = (inp[:, 2, :, :] * B_a) + B_b
    
    # Clipping the values to maintain the range [0, 1]
    rgb_img = torch.clamp(rgb_img, 0, 1)
    
    return rgb_img

    
    return rgb_img
