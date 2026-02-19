from .ela import ela_features
from .fft import fft_feature
from .noise import noise_features
from .tamper import tamper_regions
from .metadata2 import metadata_flag

def extract_features(image_path):
    ela = ela_features(image_path)
    freq = fft_feature(image_path)
    noise = noise_features(image_path)
    regions = tamper_regions(noise["noise_map"])
    meta = int(metadata_flag(image_path))

    return [
        ela["ela_mean"],
        ela["ela_std"],
        freq,
        noise["noise_std"],
        len(regions),
        meta
    ], regions
