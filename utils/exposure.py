import numpy as np
import torch

def exposureRGB(img, out):
    R_a = out[:, 0].view(-1, 1, 1)
    R_b = out[:, 1].view(-1, 1, 1)
    G_a = out[:, 2].view(-1, 1, 1)
    G_b = out[:, 3].view(-1, 1, 1)
    B_a = out[:, 4].view(-1, 1, 1)
    B_b = out[:, 5].view(-1, 1, 1)
    
    rgb_img = torch.zeros_like(img)
    rgb_img[:, 0, :, :] = (img[:, 0, :, :]*R_a) + R_b
    rgb_img[:, 1, :, :] = (img[:, 1, :, :]*G_a) + G_b
    rgb_img[:, 2, :, :] = (img[:, 2, :, :]*B_a) + B_b
    
    #rgb_img = torch.clamp(rgb_img, 0, 1) #tolto in UNet
    
    return rgb_img


def exposureRGB_Tens(inp, out_g):
    # Splitting the output tensor into individual parameter tensors
    R_a = out_g[:, 0, :, :]
    R_b = out_g[:, 1, :, :]
    G_a = out_g[:, 2, :, :]
    G_b = out_g[:, 3, :, :]
    B_a = out_g[:, 4, :, :]
    B_b = out_g[:, 5, :, :]

    # Applying the exposure transformation to the input image
    rgb_img = torch.zeros_like(inp)
    rgb_img[:, 0, :, :] = (inp[:, 0, :, :] * R_a) + R_b
    rgb_img[:, 1, :, :] = (inp[:, 1, :, :] * G_a) + G_b
    rgb_img[:, 2, :, :] = (inp[:, 2, :, :] * B_a) + B_b
    
    # Clipping the values to maintain the range [0, 1]
    #rgb_img = torch.clamp(rgb_img, 0, 1) #tolto in UNet
    
    return rgb_img

def exposure_3ch(inp, out_g):
    # Splitting the output tensor into individual parameter tensors
    R = out_g[:, 0, :, :]
    G = out_g[:, 1, :, :]
    B = out_g[:, 2, :, :]


    # Applying the exposure transformation to the input image
    rgb_img = torch.zeros_like(inp)
    rgb_img[:, 0, :, :] = inp[:, 0, :, :] + R
    rgb_img[:, 1, :, :] = inp[:, 1, :, :] + G
    rgb_img[:, 2, :, :] = inp[:, 2, :, :] + B
    
    # Clipping the values to maintain the range [0, 1]
    #rgb_img = torch.clamp(rgb_img, 0, 1) #tolto in UNet
    
    return rgb_img
