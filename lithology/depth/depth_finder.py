"""Locate a depth ruler without using an annotation box.

Search several strips across the page, group vertically aligned numbers, then
check that those numbers form an increasing ruler. Read the winning narrow
strip again at full height to estimate its top and bottom. All returned boxes
and label positions use the ORIGINAL image coordinates.
"""

import argparse
import contextlib
import io
import json
from math import ceil, floor
from pathlib import Path
from statistics import median

from .extract_depths import create_ocr_engine, parse_depth, ruler_series, select_markers


class DepthColumnError(ValueError):
    """No reliable ruler, or several rulers need a person's choice."""

    def __init__(self, message, candidates=()):
        if candidates:
            choices = "; ".join(f"{i}: {c['bbox']}" for i, c in enumerate(candidates))
            message += f" Candidate boxes: {choices}"
        super().__init__(message)
        self.candidates = list(candidates)


def _starts(length, size, overlap):
    """Overlapping tiles preserve labels which straddle the edge of a crop."""
    if length <= size:
        return [0]
    return sorted(set(range(0, length - size + 1, size - overlap)) | {length - size})


def _read_tile(image, box, engine, rotation, band):
    """OCR one tile and undo both the padding and rotation on every text box."""
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = box
    crop = image.crop(box).convert("RGB")
    padded = Image.new("RGB", (crop.width + 40, crop.height + 40), "white")
    padded.paste(crop, (20, 20))
    view = padded.rotate(rotation, expand=True) if rotation else padded
    result = engine(np.asarray(view), use_cls=False)
    readings = []
    if result.txts is None:
        return readings
    original = np.asarray(padded)
    for points, text, confidence in zip(result.boxes, result.txts, result.scores):
        depth = parse_depth(text)
        if depth is None or confidence < 0.7:
            continue
        points = np.asarray(points, dtype=float)
        if rotation == 90:
            ox, oy = padded.width - points[:, 1], points[:, 0]
        elif rotation == 270:
            ox, oy = points[:, 1], padded.height - points[:, 0]
        else:
            ox, oy = points[:, 0], points[:, 1]
        x, y = float(ox.mean()) - 20 + x0, float(oy.mean()) - 20 + y0
        if not (x0 <= x < x1 and y0 <= y < y1):
            continue
        patch = original[max(0, floor(oy.min())):min(padded.height, ceil(oy.max()) + 1),
                         max(0, floor(ox.min())):min(padded.width, ceil(ox.max()) + 1)]
        ink = patch.min(axis=2) < 180
        colour = patch.max(axis=2).astype(int) - patch.min(axis=2).astype(int) > 45
        coloured = float((ink & colour).sum()) / max(1, int(ink.sum())) > 0.35
        readings.append(dict(pixel_x=x, pixel_y=y, depth=depth, confidence=float(confidence),
                             text=text, ink="colour" if coloured else "black", band=band,
                             left=max(x0, float(ox.min()) - 20 + x0),
                             right=min(x1, float(ox.max()) - 20 + x0),
                             label_height=float(oy.max() - oy.min()), rotation=rotation))
    return readings


def rank_columns(readings, image_size, series="primary"):
    """Turn OCR observations into ruler candidates; also usable without running OCR.

    A graph value or a repeated heading number is not enough: require four
    consistent, increasing labels spread over a substantial part of the page.
    The score ranks evidence; it is NOT a probability of being correct.
    """
    width, height = image_size
    groups = []
    for row in sorted(readings, key=lambda r: r["pixel_x"]):
        # A tight centre tolerance keeps neighbouring rulers separate. Wider
        # labels may vary slightly when their number of digits changes.
        tolerance = max(10, min(35, (row["right"] - row["left"]) * 0.4))
        group = next((g for g in groups if abs(median(r["pixel_x"] for r in g) - row["pixel_x"]) <= tolerance), None)
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    candidates = []
    for group in groups:
        try:
            markers = select_markers(ruler_series(group, series))
        except ValueError:
            continue
        span = markers[-1]["pixel_y"] - markers[0]["pixel_y"]
        if len(markers) < 4 or span < min(800, height * 0.2):
            continue
        # Distant sample bands rule out a small numeric table in the header.
        if len({r["band"] for r in markers}) < 2 and height > 2400:
            continue
        pad = max(8, median(r["label_height"] for r in markers) * 0.6)
        left = max(0, floor(min(r["left"] for r in markers) - pad))
        right = min(width, ceil(max(r["right"] for r in markers) + pad))
        candidates.append(dict(bbox=[left, 0, right, height], n_markers=len(markers),
                               span=round(span, 1),
                               score=round(len(markers) * (0.5 + span / height), 3),
                               rotation=int(median(r["rotation"] for r in markers)),
                               depth_range=[markers[0]["depth"], markers[-1]["depth"]]))
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def choose_column(candidates, candidate_index=None):
    """Refuse to silently select between two similarly supported rulers."""
    if not candidates:
        raise DepthColumnError("No reliable depth column found. Use the annotation box or inspect the scan.")
    if candidate_index is not None:
        if not 0 <= candidate_index < len(candidates):
            raise DepthColumnError("Candidate index is outside the available choices.", candidates)
        return candidates[candidate_index]
    if len(candidates) > 1 and candidates[1]["score"] >= 0.7 * candidates[0]["score"]:
        raise DepthColumnError(
            "Several possible depth rulers found. Check MD/TVD headings and select --depth-candidate INDEX (starting at 0).",
            candidates)
    return candidates[0]


def find_depth_column(image_path, engine=None, *, series="primary", candidate_index=None,
                      sample_bands=5, refine=True):
    """Return {bbox, candidates, ...} without reading ANY annotation.

    Uses sparse native-resolution tiles instead of shrinking an entire tall
    log until its numbers disappear. refine=True reads the chosen strip to
    locate the ruler's vertical extent. No files are written by this function.
    Image decoding still has Pillow's whole-page TIFF memory cost.
    """
    from PIL import Image
    from .extract_depths import read_ruler

    if sample_bands < 2:
        raise ValueError("Use at least two sample bands.")
    if series not in ("primary", "secondary", "all"):
        raise ValueError("Unknown ruler series.")
    engine = engine if engine is not None else create_ocr_engine()
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as image:
        tile = 1000
        band_height = min(tile, image.height)
        tops = sorted({round((image.height - band_height) * (i + 0.5) / sample_bands)
                       for i in range(sample_bands)})
        readings, candidates = [], []
        # Normal text first. Rotations are only needed if it finds no ruler.
        for rotation in (0, 90, 270):
            for band, top in enumerate(tops):
                for left in _starts(image.width, tile, 160):
                    box = (left, top, min(image.width, left + tile), top + band_height)
                    readings.extend(_read_tile(image, box, engine, rotation, band))
            candidates = rank_columns(readings, image.size, series)
            if candidates:
                break
        chosen = choose_column(candidates, candidate_index)
        bbox = list(chosen["bbox"])
        result = dict(bbox=bbox, candidates=candidates, series=series,
                      depth_kind="unconfirmed", warnings=["Check the heading: automatic location does not identify MD versus TVD."],
                      sampled_readings=len(readings))
        if refine:
            # Scan only the selected x strip, then expand by one typical label
            # gap so that cut-off first/last labels are not cropped away.
            with contextlib.redirect_stdout(io.StringIO()):
                raw = read_ruler(image, tuple(bbox), engine, 1200, chosen["rotation"])
            markers = select_markers(ruler_series(raw, series))
            gap = median(b["pixel_y"] - a["pixel_y"] for a, b in zip(markers, markers[1:]))
            bbox[1] = max(0, floor(markers[0]["pixel_y"] - gap))
            bbox[3] = min(image.height, ceil(markers[-1]["pixel_y"] + gap))
            result["n_markers"] = len(markers)
            result["depth_range"] = [markers[0]["depth"], markers[-1]["depth"]]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--depth-candidate", type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(find_depth_column(args.image, candidate_index=args.depth_candidate), indent=2))
    except DepthColumnError as error:
        if error.candidates:
            print(json.dumps({"candidates": error.candidates}, indent=2))
        parser.exit(1, f"Error: {error}\n")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
