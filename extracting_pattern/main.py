import cv2
import numpy as np

from feature_extractor import (
    extract_template_features
)

from legend_extractor import (
    LegendExtractor
)

legend = cv2.imread("legend.png")

if legend is None:
    raise FileNotFoundError(
        "Could not load legend.png"
    )

extractor = LegendExtractor()

boxes = extractor.extract_boxes(
    legend
)

# Debug image
debug = legend.copy()

all_feature_vectors = []

for i, box in enumerate(boxes):

    x, y, w, h = box["bbox"]

    cv2.rectangle(
        debug,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    features = extract_template_features(
        box["image"]
    )

    feature_vector = np.concatenate([
        features["median_colour"] / 255.0,
        [features["pattern_density"]],
        features["fft_features"]
    ])

    all_feature_vectors.append(
        feature_vector
    )

    print(f"\nBox {i}")
    print(
        "Location:",
        (x, y, w, h)
    )

    print(
        "Median colour:",
        features["median_colour"]
    )

    print(
        "Pattern density:",
        features["pattern_density"]
    )

    cv2.imwrite(
        f"box_{i}.png",
        box["image"]
    )

cv2.imwrite(
    "legend_debug.png",
    debug
)

print(
    f"\nExtracted {len(all_feature_vectors)} legend boxes"
)

print(
    "Saved legend_debug.png"
)