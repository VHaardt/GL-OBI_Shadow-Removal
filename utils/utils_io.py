import cv2
import numpy as np
from scipy.ndimage import label

def read_image(image_path, grayscale=False):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def remove_small_objects(mask, min_size):
    labeled_mask, num_features = label(mask)
    output_mask = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        if np.sum(component) >= min_size:
            output_mask = np.logical_or(output_mask, component)
    
    return output_mask

def fill_small_holes(mask, max_size):
    inverted_mask = np.logical_not(mask)
    labeled_holes, num_features = label(inverted_mask)
    output_mask = np.copy(mask)
    for i in range(1, num_features + 1):
        hole = (labeled_holes == i)
        if np.sum(hole) <= max_size:
            output_mask = np.logical_or(output_mask, hole)
    
    return output_mask


def process_mask(mask, min_size=100, max_size=100):
    cleaned_mask = remove_small_objects(mask, min_size)
    final_mask = fill_small_holes(cleaned_mask, max_size)
    return final_mask