import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
from scipy.spatial import ConvexHull
from scipy import ndimage

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

def penumbra(mask_):
    mask = mask_.clone().squeeze()
    mask_np = mask.cpu().numpy().astype(np.uint8)
    labeled_mask, num_features = ndimage.label(mask_np)

    perimeters = []
    hull_coordinates = []

    for i in range(1, num_features + 1):
        region_points = np.argwhere(labeled_mask == i)
        hull = ConvexHull(region_points)
        hull_vertices = region_points[hull.vertices]

        perimeter = 0.0
        for j in range(len(hull_vertices)):
            point1 = hull_vertices[j]
            point2 = hull_vertices[(j + 1) % len(hull_vertices)]
            distance = np.linalg.norm(point1 - point2)
            perimeter += distance
        
        perimeters.append(perimeter)
        hull_coordinates.append(hull_vertices)
    sum_per = sum(perimeters)

    device = mask.device 
    kernel_size =  np.clip(int(sum_per / 50), 10, 30) #clip [10,30]
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).float().to(device)
    padding = kernel_size // 2

    mask = mask.expand(1, 1, -1, -1)
    
    eroded_mask = F.conv2d(mask.float(), kernel, padding=padding)
    eroded_mask = torch.where(eroded_mask >= kernel_size**2, 1.0, 0.0)
    eroded_mask = eroded_mask[:, :, :mask.size(2), :mask.size(3)]

    dilated_mask = F.conv2d(mask.float(), kernel, padding=padding)
    dilated_mask = torch.where(dilated_mask > 0, 1.0, 0.0)
    dilated_mask = dilated_mask[:, :, :mask.size(2), :mask.size(3)]

    border_mask1 = mask - eroded_mask
    border_mask2 = dilated_mask - mask
    penumbra = border_mask1 + border_mask2

    return penumbra