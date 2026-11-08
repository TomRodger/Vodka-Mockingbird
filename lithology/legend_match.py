"""Stage 3b: name each cluster by matching it to the document's legend swatches.

    python -m lithology.legend_match IMAGE ANNOTATION_JSON MERGED_CSV

Colour leads, texture breaks ties:
  - The legend and the column are on the same scan, drawn by the same software, so a rock's
    fill colour matches its swatch closely. Colour is the reliable signal.
  - Many legends have several rocks with the SAME fill (e.g. ardennes: Shale, Clay-shale,
    Marl, Sandy Clay-Shale are all green; black-and-white logs: every swatch is white).
    Only the drawn pattern separates those, so texture decides among colour-tied swatches.
  - If no swatch is close in colour, the answer is "unknown" rather than a forced wrong name.

Texture here is NOT the pooled FFT used for clustering. The legend draws a rock's symbol
differently from the column (example: column shale = rows of long dashes, legend shale =
staggered short dashes; column sandstone = dots in rows, legend = a dense dot grid), at a
different scale and spacing. FFT and template matching compare spacing and position, so they
fail (0/17 correct on the example). What survives is the SHAPE of the marks: horizontal
dashes, vertical ticks, dots, solid lines, grid. mark_shapes() describes a patch by the
fraction of its marks in each shape class, which ignores scale and spacing.
"""

import argparse
import csv
import re

import cv2
import numpy as np
from PIL import Image

from .data import load_annotation
from .features import FeatureExtractor, bgr_to_lab
from . import legend_extractor as lx

Image.MAX_IMAGE_PIXELS = None

COLOUR_MAX = 25.0    # Lab Delta-E: no swatch closer than this -> "unknown"
COLOUR_TIE = 8.0     # swatches within this Delta-E of the best are colour-tied: texture decides
# Legend entries that are drilling / log symbols, not rocks (ardennes lists casing, cores, DST).
NOT_ROCK = re.compile(r"casing|core|\bdst\b|\bfit\b|shoe|show|survey|symbol|gas|oil", re.I)
SHAPE_TIE = 0.15     # shape distances this close are a tie (then the commoner rock wins)
TRIM = 4             # px removed from patch edges (frames, column borders, boundary lines)
SAMPLE_ROWS = 512    # rows of a cluster's tallest interval used to describe it


def describe_patch(bgr, fe):
    """Fill colour (Lab) and pattern texture of a BGR patch, the same way for swatch and column."""
    if bgr.shape[0] > 2 * TRIM and bgr.shape[1] > 2 * TRIM:
        bgr = np.ascontiguousarray(bgr[TRIM:-TRIM, TRIM:-TRIM])
    f = fe.extract(bgr)
    return dict(lab=bgr_to_lab(f["median_colour"]), has_pattern=f["has_pattern"],
                shapes=mark_shapes(f["pattern_mask"]))


SHAPE_CLASSES = ("dot", "blob", "horizontal", "vertical", "long_line", "grid")


def mark_shapes(mask):
    """Fraction of the pattern's marks (connected ink components) in each shape class.

    dot: solid and roughly square (fills most of its box, at any scale); blob: roughly square
    but mostly empty inside its box (L, T, triangle, cross symbols);
    horizontal / vertical: at least 2:1 elongated; long_line: a thin line across most of the
    patch; grid: one mark spanning most of the patch (brick and lattice patterns).
    Marks running down a whole side are border or frame lines, not pattern, and are dropped."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    counts = dict.fromkeys(SHAPE_CLASSES, 0)
    for x, y, w, h, area in st[1:]:
        if area < 3:
            continue                                        # scan speckle
        if h > 0.6 * H and w < 0.15 * W and (x == 0 or x + w == W):
            continue                                        # column border / swatch frame
        if w > 0.6 * W and h <= max(3, 0.1 * H):
            k = "long_line"
        elif w > 0.5 * W or h > 0.5 * H:
            k = "grid"
        elif w >= 2 * h:
            k = "horizontal"
        elif h >= 2 * w:
            k = "vertical"
        elif area >= 0.6 * w * h:
            k = "dot"                                       # solid round-ish mark: any size
        else:
            k = "blob"
        counts[k] += 1
    total = sum(counts.values())
    return np.array([counts[k] / total if total else 0.0 for k in SHAPE_CLASSES])


# Rock names that appear in these legends (from the legend extractor's earlier version). Used
# only to fix OCR misreads ("Slity Sandstone" -> "Silty Sandstone"), never to rename a swatch.
KNOWN_NAMES = "ALTERED ZONE|ANDESITE|ANHYDRITE|BASALT|BENTONITE|BIOTITIZATION|BRECCIA|CALCARENITE|CALCAREOUS TUFF|CALCILUTITE|CARBONATES|CARBONACEOUS MAT|CARBONACEOUS SH|CEMENT CONTAM.|CHALK|CRYSTALLINE TUFF|CHERT - ARGILL|CHERT - GLASSY|CHERT - PORCEL|CHERT - TIGER STRIPE|CHERT - UNDIFF|CLAY|CLAY-MUDSTONE|CLYST-TUFFACEOUS|CHLORITIZATION|COAL|CONGLOMERATE|CONGL. SAND|CONGL. SANDSTONE|COQUINA|DACITE|DIATOMITE|DIORITE|DOLOSTONE|FELSIC SILIC DIKE|FOSSIL|GABBRO|GLASSY TUFF|GRANITE|GRANITE WASH|GRANODIORITE|GYPSUM|HALITE|HORNBL-QTZ-DIO|IGNEOUS (ACIDIC)|IGNEOUS (BASIC)|INTRUSIVES|KAOLINITIC|LIMESTONE|LITHIC TUFF|MARL - DOLO|MARL - CALC|METAMORPHICS|MUDSTONE|OBSIDIAN|PALEOSOL|PHOSPHATE|PORCELANITE|PORCELANEOUS CLYST|PYRITE|PYROCLASTICS|QUARTZ DIORITE|QUARTZ LATITE|QUARTZ MONZONITE|RECRYSTALLIZED CALCITE|RHYOLITE|SALT|SAND|SANDSTONE|SANDSTONE-TUFFACEOUS|SERICITIZATION|SERPENTINE|SHALE|SHALE TUFFACEOUS|SHELL FRAGMENTS|SIDERITE|SILICIFICATION|SILTSTONE|SILTST-TUFFACEOUS|TUFF|VOLCANICLASTICS SEDS|VOLCANICS|DOLOMITE|SILTY SANDSTONE|SAND AND SANDSTONE".split("|")
SNAP_CUTOFF = 0.85   # similarity needed to snap; legend-specific names ("Calcareous clay-shale")
                     # are not in the list and must be kept as written, so this stays strict


def clean_name(text):
    """OCR text -> legend name: drop junk symbols at the ends ("Calcareous sandstone™",
    "Calcareous clay-shale —"), then snap to a known rock name only if it is a near-certain
    misread. Otherwise the legend's own wording is kept: the brief asks for the legend's name."""
    import difflib
    t = re.sub(r"\s+", " ", re.sub(r"^[^A-Za-z]+|[^A-Za-z)]+$", "", text)).strip()
    if not t:
        return ""
    table = {k.lower(): k for k in KNOWN_NAMES}
    hit = difflib.get_close_matches(t.lower(), list(table), n=1, cutoff=SNAP_CUTOFF)
    return table[hit[0]].title() if hit and hit[0] != t.lower() else t


def tighten(img, x, y, w, h):
    """Shrink a swatch box to the largest blob of ink inside it. The extractor's boxes can be
    offset or padded with paper (ardennes: Siltstone's box reaches into Limestone, Dolomite's
    includes white margin), and paper in the box makes the fill colour read as white."""
    sub = img[y:y + h, x:x + w]
    ink = lx.ink_mask(sub).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))      # join a pattern's separate strokes
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    if n < 2:
        return x, y, w, h
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = stats[i, :4]
    return x + bx, y + by, bw, bh


def legend_swatches(page, legend_box, fe):
    """[(name, description)] for every named swatch the extractor finds in the legend box."""
    img = np.asarray(page.crop(tuple(legend_box)).convert("RGB"))[:, :, ::-1].copy()   # -> BGR
    labels = lx.ocr_labels(img) if lx.pytesseract else []
    pairs, _ = lx.detect(img, labels)
    out = []
    for box, name in pairs:
        name = clean_name(name or "")
        x, y, w, h = tighten(img, *box)
        if name and not NOT_ROCK.search(name) and w > 2 * TRIM and h > 2 * TRIM:
            out.append((name, describe_patch(img[y:y + h, x:x + w], fe)))
    return out


def describe_clusters(page, column_box, rows, fe):
    """{cluster_id: description}, from each cluster's TALLEST interval (most reliable texture)."""
    tallest = {}
    for r in rows:
        c, h = r["cluster_id"], r["bottom_y"] - r["top_y"]
        if c >= 0 and h > tallest.get(c, (0,))[0]:
            tallest[c] = (h, r["top_y"], r["bottom_y"])
    out = {}
    for c, (h, top, bottom) in tallest.items():
        mid, n = (top + bottom) // 2, min(h, SAMPLE_ROWS)
        a = max(top, mid - n // 2)
        crop = page.crop((column_box[0], a, column_box[2], a + n)).convert("RGB")
        out[c] = describe_patch(np.asarray(crop)[:, :, ::-1].copy(), fe)
    return out


def best_swatch(d, swatches):
    """Name for one cluster description, or "unknown". Colour leads; the mix of mark shapes
    decides among colour-tied swatches (L1 distance between shape fractions, 0..2)."""
    if not swatches:
        return "unknown"
    dc = np.array([np.linalg.norm(d["lab"] - s["lab"]) for _, s in swatches])
    if dc.min() > COLOUR_MAX:
        return "unknown"
    tied = [i for i in range(len(swatches)) if dc[i] <= dc.min() + COLOUR_TIE]

    def dt(i):
        s = swatches[i][1]
        if s["has_pattern"] != d["has_pattern"]:
            return 2.0                     # patterned vs plain: as different as it gets
        return float(np.abs(s["shapes"] - d["shapes"]).sum())
    # Swatches that tie on shape (example: Shale and Clay are both pure horizontal dashes) are
    # indistinguishable from the drawing. Prefer the more common rock: a stated prior, not a
    # measurement, and the only thing that can decide such a tie.
    best = min(dt(i) for i in tied)
    close = [i for i in tied if dt(i) <= best + SHAPE_TIE]
    return swatches[min(close, key=lambda i: (commonness(swatches[i][0]), dt(i)))][0]


# Most to least common in mudlogs. Used ONLY to break ties the drawing can't decide.
COMMON = ["shale", "sandstone", "siltstone", "limestone", "claystone", "clay", "dolomite",
          "marl", "anhydrite", "salt", "halite", "chalk", "coal"]


def commonness(name):
    n = name.lower()
    return next((i for i, c in enumerate(COMMON) if n == c), len(COMMON))


def name_clusters(image, annotation, rows):
    """Set row["lithology"] for every row. Returns the number of legend swatches found."""
    boxes = load_annotation(annotation)
    fe = FeatureExtractor()
    page = Image.open(image)
    try:
        swatches = legend_swatches(page, boxes["lithology_legend"], fe)
    except Exception:            # unreadable legend: everything stays "unknown", run continues
        swatches = []
    clusters = describe_clusters(page, boxes["interpreted_lithology_column"], rows, fe)
    names = {c: best_swatch(d, swatches) for c, d in clusters.items()}
    for r in rows:
        # Blank paper (cluster -1) is never matched: it would get the palest swatch's name.
        r["lithology"] = "no_data" if r["cluster_id"] < 0 else names.get(r["cluster_id"], "unknown")
    return len(swatches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("annotation")
    ap.add_argument("merged_csv", help="clustered intervals (top_y, bottom_y, cluster_id)")
    a = ap.parse_args()
    with open(a.merged_csv, newline="") as f:
        rows = [dict(top_y=int(r["top_y"]), bottom_y=int(r["bottom_y"]), cluster_id=int(r["cluster_id"]))
                for r in csv.DictReader(f)]
    n = name_clusters(a.image, a.annotation, rows)
    print(f"{n} legend swatches")
    seen = {}
    for r in rows:
        seen.setdefault(r["cluster_id"], r["lithology"])
    for c, name in sorted(seen.items()):
        print(f"  cluster {c:3d} -> {name}")


if __name__ == "__main__":
    main()
