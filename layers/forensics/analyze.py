"""
import joblib

from features import extract_features
from phash import compute_phash

# Load trained ML model
model = joblib.load("model.pkl")


def analyze_image(image_path: str) -> dict:
    
    # Main forensic analysis entry point
    

    # Extract forensic features + tamper regions
    features, tamper_regions = extract_features(image_path)

    # ML-based manipulation probability
    manipulation_score = model.predict_proba([features])[0][1]

    # Safety clamp
    manipulation_score = float(
        max(0.0, min(1.0, manipulation_score))
    )

    return {
        "manipulation_score": manipulation_score,
        "frequency_score": float(features[2]),
        "tamper_regions": tamper_regions,
        "metadata_flag": bool(features[-1]),
        "phash": compute_phash(image_path)
    }
"""
"""
from .features import extract_features
from .phash import compute_phash
from .cnn_infer import cnn_manipulation_score


def analyze_forensics(image_path: str) -> dict:
    
#  Main forensic analysis entry point (ML-free, deterministic)
    

    # Extract forensic features and tamper regions
    features, tamper_regions = extract_features(image_path)

    # Unpack features for clarity
    ela_mean = features[0]
    ela_std = features[1]
    frequency_score = features[2]
    noise_std = features[3]
    tamper_count = features[4]
    metadata_flag = bool(features[5])

    # Rule-based manipulation score (conservative by design)
    # manipulation_score = min(
    #     0.5 * ela_mean +
    #     0.3 * noise_std +
    #     0.2 * min(tamper_count / 3, 1.0),
    #     1.0
    # )
    cnn_score = cnn_manipulation_score(image_path)


    return {
        "manipulation_score": cnn_score,
        "frequency_score": float(frequency_score),
        "tamper_regions": tamper_regions,
        "metadata_flag": metadata_flag,
        "phash": compute_phash(image_path)
    }

"""
from .cnn_infer import cnn_manipulation_score
from .fft import fft_feature
from .tamper import tamper_regions
from .metadata2 import check_metadata
from .phash import compute_phash


def analyze_forensics(image_path: str) -> dict:
    """
    Main forensic analysis entry point.
    Produces classical forensic signals and CNN-based manipulation probability.
    """

    manipulation_score = cnn_manipulation_score(image_path)
    frequency_score = fft_feature(image_path)
    regions = tamper_regions(image_path)
    metadata_flag = check_metadata(image_path)

    return {
        "manipulation_score": float(manipulation_score),
        "frequency_score": float(frequency_score),
        "tamper_regions": regions,
        "metadata_flag": metadata_flag,
        "phash": compute_phash(image_path)
    }
