"""
import cv2
import numpy as np

def fft_feature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))

    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(f) + 1e-6)

    h, w = mag.shape
    center = mag[h//4:3*h//4, w//4:3*w//4]

    score = center.mean() / (mag.mean() + 1e-6)

    # Normalize + clamp to [0,1]
    score = float(min(max(score, 0.0), 1.0))
    return score
"""
import cv2
import numpy as np

def fft_feature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))

    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(f) + 1e-6)

    h, w = mag.shape

    # Low-frequency region (center)
    lf = mag[h//2-20:h//2+20, w//2-20:w//2+20].mean()

    # High-frequency regions (corners)
    hf = (
        mag[:h//8, :w//8].mean() +
        mag[:h//8, -w//8:].mean() +
        mag[-h//8:, :w//8].mean() +
        mag[-h//8:, -w//8:].mean()
    ) / 4.0

    score = hf / (lf + 1e-6)

    # Normalize to [0,1] using soft clamp
    score = min(score / 2.0, 1.0)

    return float(score)
