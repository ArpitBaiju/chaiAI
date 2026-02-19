import cv2
import numpy as np


def tamper_regions(image_path, threshold=30):
    img = cv2.imread(image_path)

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simple noise estimation (Laplacian)
    noise_map = cv2.Laplacian(gray, cv2.CV_64F)
    noise_map = np.absolute(noise_map)

    if noise_map is None or not isinstance(noise_map, np.ndarray):
        return []

    _, mask = cv2.threshold(
        noise_map.astype(np.uint8),
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:  # ignore tiny noise
            regions.append([int(x), int(y), int(w), int(h)])

    return regions
