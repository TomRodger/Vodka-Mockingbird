# test_legend_extractor.py

import cv2
from legend_extractor import LegendExtractor

legend = cv2.imread("legend.png")

if legend is None:
    raise FileNotFoundError("legend.png not found")

extractor = LegendExtractor()

boxes = extractor.extract_boxes(legend)

print(f"Found {len(boxes)} boxes")

debug = legend.copy()

for i, box in enumerate(boxes):

    x, y, w, h = box["bbox"]

    print(f"{i}: ({x}, {y}, {w}, {h})")

    cv2.rectangle(
        debug,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        debug,
        str(i),
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite("legend_debug.png", debug)

print("Saved legend_debug.png")
