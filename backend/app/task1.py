import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys

# Remove PIL pixel safety limits for ultra-tall well logs
Image.MAX_IMAGE_PIXELS = None


def load_annotation(json_path):
    """Loads bounding box annotations from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["bounding_boxes"]


def draw_bounding_boxes(image_path, bounding_boxes, output_path="annotated.png"):
    """Draws all annotated bounding boxes on the document and saves the visualization."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for name, box in bounding_boxes.items():
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline="red", width=5)

    image.save(output_path)
    print(f"Saved annotated overview to {output_path}")


def crop_column(image_path, box):
    """Crops a region from the image given a bounding box [x1, y1, x2, y2]."""
    image = Image.open(image_path)
    return image.crop(tuple(box))


def detect_boundaries(crop_pil, std_multiplier=3.0, min_distance=10):
    """Detects horizontal interval boundaries based on mean row-to-row pixel differences."""
    # Ensure the PIL crop is in RGB format before converting to a NumPy array for OpenCV
    gray = cv2.cvtColor(np.array(crop_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    crop_height = gray.shape[0]

    # Mean absolute pixel difference between adjacent rows
    row_difference = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1)

    # Threshold for sharp horizontal visual changes
    threshold = row_difference.mean() + (std_multiplier * row_difference.std())
    raw_candidates = np.where(row_difference > threshold)[0]

    # Filter out closely adjacent noise peaks
    filtered_boundaries = []
    for y in raw_candidates:
        if not filtered_boundaries or (y - filtered_boundaries[-1]) >= min_distance:
            filtered_boundaries.append(int(y))

    # Include start (0) and end (crop_height) to frame all intervals
    full_boundaries = [0] + filtered_boundaries + [crop_height]
    return sorted(list(set(full_boundaries)))


def build_intervals(boundaries, crop_y_start):
    """Converts relative crop boundary y-coords to full-image y-coordinate intervals."""
    intervals = []
    for top, bottom in zip(boundaries[:-1], boundaries[1:]):
        intervals.append({
            "top_y": int(crop_y_start + top),
            "bottom_y": int(crop_y_start + bottom)
        })
    return intervals




def main():
    # Check if user provided the required file paths
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_path> <annotation_json_path> [output_prefix]")
        sys.exit(1)

    image_path = sys.argv[1]
    json_path = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "out"

    # 1. Load annotations
    bounding_boxes = load_annotation(json_path)

    # 2. Draw annotated overview
    draw_bounding_boxes(image_path, bounding_boxes, output_path=f"{prefix}_annotated.png")

    # 3. Crop target lithology column
    lith_box = bounding_boxes["interpreted_lithology_column"]
    crop_y_start = lith_box[1]
    lithology_crop = crop_column(image_path, lith_box)
    lithology_crop.save(f"{prefix}_crop.png")

    # 4. Detect boundaries & generate full-image intervals
    boundaries = detect_boundaries(lithology_crop)
    intervals = build_intervals(boundaries, crop_y_start)

    print(f"[{prefix}] Detected {len(intervals)} lithology intervals:")
    for interval in intervals[:5]:
        print(interval)

if __name__ == "__main__":
    main()