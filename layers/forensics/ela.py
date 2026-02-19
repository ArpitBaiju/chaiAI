from PIL import Image, ImageChops
import numpy as np
import os

def ela_features(image_path, quality=90):
    img = Image.open(image_path).convert("RGB")
    tmp = "tmp.jpg"
    img.save(tmp, "JPEG", quality=quality)

    comp = Image.open(tmp)
    diff = ImageChops.difference(img, comp)
    diff_np = np.array(diff, dtype=np.float32)

    os.remove(tmp)

    return {
        "ela_mean": diff_np.mean() / 255.0,
        "ela_std": diff_np.std() / 255.0
    }
