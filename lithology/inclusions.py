"""Absorb partial-width inclusions: small symbols of a minor rock drawn INSIDE the main rock.

Mudlogs draw a minor component as a box covering part of the column width (Rio Lavaca: yellow
sandstone and brown conglomerate boxes, about half the width and ~10 rows tall, inside salt).
Segmentation cuts row by row on the column's overall fill, so each box becomes its own thin
interval and two false cuts. An interval is one rock across the depth range, so these should
be part of the surrounding rock. Touching intervals with the same legend name are also joined
(join_same_name): often the chunk itself got the surrounding rock's name. Measured on the Rio Lavaca label: boundary precision 0.44,
recall 1.00, i.e. more than half the cuts were on inclusions.

A real thin bed covers (nearly) the full width; an inclusion does not. So for each thin
interval sandwiched between two intervals of the same rock, we measure how much of the
column WIDTH differs from that rock, and absorb it only if that share is small.
"""

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MAX_H = 25          # only intervals thinner than this (rows) can be inclusions
REF_ROWS = 30       # rows of the surrounding rock sampled above and below
EDGE = 4            # px trimmed from each side (column border lines)
DARK_L = 35         # Lab L below this = linework (pattern, outlines): ignored, not fill
SAME_FILL = 20.0    # Delta-E: a pixel this close to the rock's fill is "the rock"
MIN_COVER = 0.1     # differs over less than this share: no colour evidence (B&W), keep
MAX_COVER = 0.9     # differs over less than this share of the width -> inclusion. Measured:
                    # real thin beds (mc761) all >= 0.95 (they run edge to edge); on Rio Lavaca
                    # and Callaway the thin sandwiched intervals have a gap between 0.9 and 0.95,
                    # and chunks of two overlapping boxes reach ~0.82 (Rio Lavaca row 19036).


def _lab(page, box, a, b):
    """Lab pixels of column rows a..b (EDGE trimmed), shape (rows, width, 3)."""
    rgb = np.asarray(page.crop((box[0] + EDGE, a, box[2] - EDGE, b)).convert("RGB"))
    if rgb.size == 0:
        return None
    # Same Lab scale as features.bgr_to_lab (L 0-100, a/b centred on 0), for a whole image.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    lab[:, :, 0] *= 100 / 255
    lab[:, :, 1:] -= 128
    return lab


def _fill(lab):
    """Median colour of the non-linework pixels."""
    px = lab.reshape(-1, 3)
    px = px[px[:, 0] >= DARK_L]
    return np.median(px, axis=0) if len(px) else None


def width_cover(page, box, above, thin, below):
    """Share of the column width where the thin interval is NOT the surrounding rock, or None
    when colour can't decide (e.g. black-and-white logs, where every fill is white)."""
    ref = []
    if above["bottom_y"] - above["top_y"] > 2:
        ref.append(_lab(page, box, max(above["top_y"], above["bottom_y"] - REF_ROWS), above["bottom_y"]))
    if below["bottom_y"] - below["top_y"] > 2:
        ref.append(_lab(page, box, below["top_y"], min(below["bottom_y"], below["top_y"] + REF_ROWS)))
    ref = [r for r in ref if r is not None]
    mid = _lab(page, box, thin["top_y"], thin["bottom_y"])
    if not ref or mid is None:
        return None
    rock = _fill(np.concatenate([r.reshape(-1, 3) for r in ref]))
    if rock is None:
        return None
    light = mid[:, :, 0] >= DARK_L                               # fill pixels, per (row, x)
    differs = (np.linalg.norm(mid - rock, axis=2) > SAME_FILL) & light
    n_light = light.sum(axis=0)
    col_diff = np.where(n_light > 0, differs.sum(axis=0) / np.maximum(n_light, 1), 0) > 0.5
    cover = float(col_diff.mean())
    # Almost nothing differs in colour: colour can't judge it (black-and-white logs, where a
    # real thin bed and its neighbours are all white). Keep it rather than guess. Not judged
    # on the interval's average colour: a half-width chunk's average is mostly the rock around it.
    return None if cover < MIN_COVER else cover


def absorb_inclusions(image, column_box, rows):
    """rows: [{top_y, bottom_y, cluster_id, lithology}] in order. Returns new rows with
    partial-width inclusions merged into the rock around them, and how many were absorbed."""
    page = Image.open(image)
    rows = join_same_name(rows)
    absorbed, i = 0, 1
    while i < len(rows) - 1:
        a, t, b = rows[i - 1], rows[i], rows[i + 1]
        name = a["lithology"]
        if (t["bottom_y"] - t["top_y"] < MAX_H and b["lithology"] == name
                and t["lithology"] != name and name not in ("unknown", "no_data")):
            cover = width_cover(page, column_box, a, t, b)
            if cover is not None and cover < MAX_COVER:
                a["bottom_y"] = b["bottom_y"]                  # above + inclusion + below = one
                del rows[i:i + 2]
                absorbed += 1
                continue                                       # re-test the new neighbour
        i += 1
    return join_same_name(rows), absorbed


def join_same_name(rows):
    """Join touching intervals that the legend gave the same name: the same rock, so a cut
    between them is not a boundary. Clustering often splits one rock into several clusters
    (ardennes: one green shale in 5); naming tells us they are one. The joined interval keeps
    the first piece's cluster_id. Unknown and no_data are never joined on name alone."""
    out = []
    for r in rows:
        r = dict(r)
        if (out and r["lithology"] == out[-1]["lithology"] and r["lithology"] not in ("unknown", "no_data")
                and r["top_y"] == out[-1]["bottom_y"]):
            out[-1]["bottom_y"] = r["bottom_y"]
        else:
            out.append(r)
    return out
