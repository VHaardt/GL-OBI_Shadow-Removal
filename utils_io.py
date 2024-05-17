import cv2
import numpy as np
def read_image(image_path, grayscale=False):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img
