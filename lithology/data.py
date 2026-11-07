"""Shared loading helpers and dataset paths.

Everything in the pipeline uses pixel rows of the ORIGINAL full image, so these helpers
crop without resizing.
"""

import csv
import glob
import json
import os

from PIL import Image

# These scans are very tall (up to ~314,000 px); lift Pillow's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(REPO, "hack27", "project")                 # organisers' material
DATASET_IMAGES = os.path.join(PROJECT, "dataset", "images")
DATASET_ANNOTATIONS = os.path.join(PROJECT, "dataset", "bbox_annotation")
EXAMPLE_DIR = os.path.join(PROJECT, "example")
EXAMPLE_STEM = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"
EXAMPLE_IMAGE = os.path.join(EXAMPLE_DIR, EXAMPLE_STEM + ".tif")
EXAMPLE_ANNOTATION = os.path.join(EXAMPLE_DIR, EXAMPLE_STEM + ".json")
EXAMPLE_TRUTH = os.path.join(EXAMPLE_DIR, "example.csv")
OUT_DIR = os.path.join(REPO, "out")                                 # generated files (git-ignored)


def load_annotation(json_path):
    """Bounding boxes from an annotation JSON: {name: [x_min, y_min, x_max, y_max]}."""
    with open(json_path, "r") as f:
        return json.load(f)["bounding_boxes"]


def crop_column(image_path, box):
    """Crop a region [x1, y1, x2, y2] from a scan.

    Note: Pillow decodes the whole (compressed) page to crop it, ~3.6 GB peak on the
    tallest log. Strip-wise TIFF reading would fix that for every caller."""
    return Image.open(image_path).crop(tuple(box))


def find_image(stem):
    """Path of the dataset image whose name (without extension) is `stem` (.tif or .tiff)."""
    hits = [p for p in glob.glob(os.path.join(DATASET_IMAGES, "*"))
            if os.path.splitext(os.path.basename(p))[0] == stem]
    if not hits:
        raise FileNotFoundError(f"No image named {stem} in {DATASET_IMAGES}")
    return hits[0]


def read_intervals(path):
    """Intervals from a CSV with top_y, bottom_y and optionally lithology."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [dict(top_y=int(r["top_y"]), bottom_y=int(r["bottom_y"]), lithology=r.get("lithology"))
            for r in rows]
