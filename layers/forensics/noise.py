import cv2
import numpy as np

def noise_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    noise = cv2.absdiff(img, blur)

    return {
        "noise_std": noise.std() / 255.0,
        "noise_map": noise
    }
