import torch
from torchvision.transforms import v2
import torch.nn.functional as F

def dilate_erode_mask(mask, kernel_size):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device).float()
    padding = kernel_size // 2
    
    eroded_mask = F.conv2d(mask.float(), kernel, padding=padding)
    eroded_mask = torch.where(eroded_mask >= kernel_size**2, 1.0, 0.0)
    eroded_mask = eroded_mask[:, :, :mask.size(2), :mask.size(3)]

    dilated_mask = F.conv2d(mask.float(), kernel, padding=padding)
    dilated_mask = torch.where(dilated_mask > 0, 1.0, 0.0)
    dilated_mask = dilated_mask[:, :, :mask.size(2), :mask.size(3)]

    return dilated_mask, eroded_mask

def blur_image_border(image, border_mask, blur_kernel_size, sigma=1.0):
    blurred_combined = torch.empty_like(image) #image.clone()
    device = blurred_combined.device
    
    # Iterate over each image in the batch
    for i in range(image.size(0)):
        border_mask_i = border_mask[i, :, :, :].to(device)
        image_i = image[i, :, :, :].clone().to(device)
        blurrer = v2.GaussianBlur(kernel_size=(blur_kernel_size, blur_kernel_size), sigma=sigma)

        blurred_img = blurrer(image_i)

        blurred_combined[i, :, :, :] = image_i * (1 - border_mask_i) + blurred_img * border_mask_i

    blurred_combined = torch.clamp(blurred_combined, 0, 1)
    return blurred_combined