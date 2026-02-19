from PIL import Image

def check_metadata(image_path):
    img = Image.open(image_path)
    return not bool(img.info)
