import csv
import json
import sys
import warnings

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks

# Remove PIL pixel safety limits for ultra-tall well logs
Image.MAX_IMAGE_PIXELS = None


def load_annotation(json_path):
    """Loads bounding box annotations from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["bounding_boxes"]


def draw_bounding_boxes(image_path, bounding_boxes, output_path="annotated.png", max_height=4000):
    """Draws all annotated bounding boxes on a shrunken overview of the document.

    The page is converted to RGB a strip at a time and shrunk as it goes, so a
    300,000-px-tall log never has to exist at full size in memory (that would be ~1.6 GB).
    """
    image = Image.open(image_path)
    scale = min(1.0, max_height / image.height)
    out = Image.new("RGB", (max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    strip = 4000
    for ys in range(0, image.height, strip):
        ye = min(ys + strip, image.height)
        part = image.crop((0, ys, image.width, ye)).convert("RGB")
        top, bottom = round(ys * scale), round(ye * scale)
        out.paste(part.resize((out.width, max(1, bottom - top))), (0, top))
    draw = ImageDraw.Draw(out)
    for name, (x1, y1, x2, y2) in bounding_boxes.items():
        draw.rectangle((x1 * scale, y1 * scale, x2 * scale, y2 * scale), outline="red", width=3)
    out.save(output_path)
    print(f"Saved annotated overview to {output_path}")


def crop_column(image_path, box):
    """Crops a region from the image given a bounding box [x1, y1, x2, y2]."""
    image = Image.open(image_path)
    return image.crop(tuple(box))


# ---------------------------------------------------------------- boundary detection
#
# Why not compare adjacent rows: inside one rock the pattern itself changes sharply from
# row to row (every dash row in shale is a black/white edge), so row-to-row differences
# fire on pattern lines, not boundaries (160 intervals found on the example vs 17 real).
#
# Instead, two detectors that compare what is ABOVE a row with what is BELOW it:
#   colour:  median fill colour per row, compared over a few rows each side. Precise,
#            catches thin beds, and is how most colour documents separate rocks.
#   texture: pattern statistics summed over a band of rows each side, for documents
#            (or rocks) that differ only in pattern. Statistics per band:
#              density             fraction of pixels that are linework
#              horizontal run      mean length of linework runs along a row  (dashes: long)
#              vertical run        mean length down a column           (vertical bars: long)
#            These are first smoothed over one repeat period of the pattern, so a band
#            containing one dash row vs two doesn't look like a change of rock.


def _row_signals(crop, chunk=20000):
    """Per-row signals for the column. Works in chunks of rows so even the 300,000-row
    columns stay small in memory: only 1-D per-row arrays are kept for the whole column."""
    a = np.asarray(crop.convert("RGB"))
    H, W = a.shape[:2]
    t = max(3, W // 10)                         # skip the column's own border lines
    a = a[:, t:W - t]

    # pass 1: typical grey of each row (the fill), smoothed over 15 rows
    row_grey = np.concatenate([np.median(a[i:i + chunk].mean(axis=2) / 255, axis=1)
                               for i in range(0, H, chunk)])
    fill = median_filter(row_grey, size=15)

    ink_n, h_n, v_n, colour = (np.zeros(H), np.zeros(H), np.zeros(H), np.zeros((H, 3), np.float32))
    prev_last = None
    for i in range(0, H, chunk):
        c = a[i:i + chunk]
        g = c.mean(axis=2, dtype=np.float32) / 255
        f = fill[i:i + chunk, None]
        ink = g < np.maximum(0.6 * f, f - 0.2)          # linework relative to the local fill
        left = np.zeros_like(ink)
        left[:, 1:] = ink[:, :-1]
        up = np.zeros_like(ink)
        up[1:] = ink[:-1]
        if prev_last is not None:
            up[0] = prev_last
        prev_last = ink[-1]
        ink_n[i:i + chunk] = ink.sum(1)
        h_n[i:i + chunk] = (ink & ~left).sum(1)
        v_n[i:i + chunk] = (ink & ~up).sum(1)
        # Fill colour from NON-linework pixels only. A plain row median flips to black
        # whenever a dash row covers half the width, which fires on every dash in B&W logs.
        px = np.where(~ink[..., None], c.astype(np.float32) / 255, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")     # all-linework rows: "All-NaN slice"
            colour[i:i + chunk] = np.nanmedian(px, axis=1)
    bad = np.isnan(colour).any(axis=1)          # rows that are all linework: copy the row above
    for i in np.where(bad)[0]:
        colour[i] = colour[i - 1] if i > 0 else 1.0
    return dict(ink=ink_n, h_starts=h_n, v_starts=v_n,
                colour=median_filter(colour, size=(7, 1)), width=a.shape[1])


def _pattern_period(ink, lo=6, hi=120):
    """Repeat period of the pattern in rows: the FIRST strong autocorrelation peak.
    (The strongest peak is often a multiple, e.g. 75 instead of 25, which over-smooths.)"""
    x = ink - uniform_filter1d(ink, 151)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return 16
    ac = ac / ac[0]
    pk, _ = find_peaks(ac[:hi])
    pk = pk[pk >= lo]
    if len(pk) == 0 or ac[pk].max() < 0.2:
        return 16                               # no clear repeating pattern
    return int(next(p for p in pk if ac[p] >= 0.6 * ac[pk].max()))


def _peaks(score, window, thresh):
    """Rows where score >= thresh, strongest first, keeping no two within `window` rows.
    (One boundary produces a broad bump in the score, often with several local maxima.)"""
    cand = np.where((score >= thresh) & (score == maximum_filter1d(score, size=3)))[0]
    kept = []
    for y in cand[np.argsort(-score[cand])]:
        if all(abs(y - k) > window for k in kept):
            kept.append(int(y))
    return sorted(kept)


def detect_boundaries(crop_pil, band=32, texture_threshold=0.7, colour_rows=4,
                      colour_threshold=0.25, colour_gap=10, min_distance=None, edge_margin=20):
    """Detects horizontal interval boundaries in the interpreted lithology column.

    Returns crop-relative rows [0, b1, b2, ..., crop_height], as before.
      band:              rows compared on each side for texture (32 found 16/16 on the example)
      texture_threshold: texture change needed (log run-length + density units)
      colour_rows:       rows compared on each side for colour
      colour_gap:        two colour boundaries closer than this are one (thinnest bed kept)
      colour_threshold:  summed |RGB| change in 0-1 units (0.25 = clearly different fill)
      min_distance:      a texture boundary this close to a colour boundary is the same one
                         (default: band). Texture locates boundaries less precisely (+-band/2),
                         so a smaller value leaves slivers between the two detections.
      edge_margin:       ignore boundaries this close to the crop's top/bottom. Those are the
                         column's own border lines, which otherwise cut off thin slivers
                         (on the example: 4 px at the top, 15 px at the bottom).
    """
    if min_distance is None:
        min_distance = band
    s = _row_signals(crop_pil)
    H = len(s["ink"])

    # texture: band sums of period-smoothed statistics, above vs below each row
    P = _pattern_period(s["ink"])
    sm = {k: uniform_filter1d(s[k], size=P, mode="nearest") for k in ("ink", "h_starts", "v_starts")}
    cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in sm.items()}
    tex = np.zeros(H)
    if H > 2 * band:
        y = np.arange(band, H - band)

        def stats(a, b):
            i = cs["ink"][b] - cs["ink"][a]
            hr = np.log1p(i / np.maximum(cs["h_starts"][b] - cs["h_starts"][a], 1))
            vr = np.log1p(i / np.maximum(cs["v_starts"][b] - cs["v_starts"][a], 1))
            dens = i / (band * s["width"])
            return np.stack([hr, vr, 4 * dens], axis=1)

        tex[y] = np.abs(stats(y - band, y) - stats(y, y + band)).sum(axis=1)
    texture_rows = _peaks(tex, band, texture_threshold)

    # Refine each texture boundary to where the smoothed ink density changes fastest nearby.
    # Band sums only say "somewhere within +-band/2"; a pattern ending into blank paper was
    # found 33 px late on the example without this.
    grad = np.abs(np.gradient(sm["ink"]))
    half = band // 2
    texture_rows = [int(max(0, y - half) + np.argmax(grad[max(0, y - half): y + half + 1]))
                    for y in texture_rows]

    # colour: fill colour a few rows above vs below
    k = colour_rows
    col = np.zeros(H)
    if H > 2 * k:
        c = s["colour"]
        col[k:H - k] = np.abs(c[2 * k:] - c[:-2 * k]).sum(axis=1)
    colour_rows_found = _peaks(col, colour_gap, colour_threshold)

    # combine: colour boundaries are sharper, so they win when both fire close together
    found = sorted(colour_rows_found)
    for y in texture_rows:
        if all(abs(y - c) >= min_distance for c in found):
            found.append(y)
    found = sorted(set(y for y in found if edge_margin <= y <= H - edge_margin))
    return [0] + found + [H]


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

    # 2. Draw annotated overview (shrunk, memory-safe)
    draw_bounding_boxes(image_path, bounding_boxes, output_path=f"{prefix}_annotated.png")

    # 3. Crop target lithology column
    lith_box = bounding_boxes["interpreted_lithology_column"]
    crop_y_start = lith_box[1]
    lithology_crop = crop_column(image_path, lith_box)
    lithology_crop.save(f"{prefix}_crop.png")

    # 4. Detect boundaries & generate full-image intervals
    boundaries = detect_boundaries(lithology_crop)
    intervals = build_intervals(boundaries, crop_y_start)

    # 5. Save them for the clustering step (python clustering/cluster.py ... --intervals)
    with open(f"{prefix}_intervals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["top_y", "bottom_y"])
        w.writeheader()
        w.writerows(intervals)

    print(f"[{prefix}] Detected {len(intervals)} lithology intervals -> {prefix}_intervals.csv")
    for interval in intervals[:5]:
        print(interval)


if __name__ == "__main__":
    main()
