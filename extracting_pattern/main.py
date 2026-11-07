import cv2
import numpy as np
from feature_extractor import extract_template_features

img = cv2.imread("legend_item_1.png")

features = extract_template_features(img)

feature_vector = np.concatenate([
    features["median_colour"] / 255.0,
    [features["pattern_density"]],
    features["fft_features"]
])
