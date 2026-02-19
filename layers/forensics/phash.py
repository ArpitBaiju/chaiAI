from PIL import Image
import imagehash

def compute_phash(image_path):
    return str(imagehash.phash(Image.open(image_path)))
