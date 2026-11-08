#!/usr/bin/env python3
"""
Find the legend block in a full well-log page using computer vision only (no OCR).

Idea: a legend is a regular grid of many same-sized swatch frames. We
  1. look only at the top part of the page (legends sit near the start of a log),
  2. find rectangular frames (ink along >= 3 of 4 sides) of plausible size,
  3. take the largest group of frames with near-identical size (the swatches),
  4. grow the box outwards until it hits long ruled lines (the legend's cell
     border) -- or, if the page has no ruling, by a margin that leaves room for
     the labels and heading.

Usage:
  python legend_finder.py log.tif legend_crop.png [--search-frac 0.35]

From code:
  from legend_finder import find_legend
  res = find_legend(img_bgr)            # None if nothing legend-like was found
  res["crop"], res["bbox"], res["confidence"]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# loading (big 1-bit TIFFs are normal for logs)
# ----------------------------------------------------------------------------
def load_image(path):
    """Read PNG/JPG/TIFF (incl. huge 1-bit or multi-page TIFF: first page) as BGR."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(str(path))
    im.seek(0)
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _side_coverage(ink, x, y, w, h):
    """Fraction of each side (top, bottom, left, right) of a box that carries ink."""
    t = max(1, int(0.08 * min(w, h)))
    sub = ink[y:y + h, x:x + w] > 0
    return (sub[:t].any(axis=0).mean(), sub[-t:].any(axis=0).mean(),
            sub[:, :t].any(axis=1).mean(), sub[:, -t:].any(axis=1).mean())


def _frames(ink, W, min_w_frac=0.008, max_w_frac=0.10, partial=False, k=3):
    """Rectangular frames of plausible swatch size: list of (x, y, w, h).

    partial=True also accepts frames cut off on one end (ink on only 2 opposite-ish sides)."""
    glued = cv2.dilate(ink, np.ones((k, k), np.uint8)) if k > 1 else ink
    n, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    lo, hi = min_w_frac * W, max_w_frac * W
    out = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if not (lo <= w <= hi) or h < 0.4 * lo or h > hi:
            continue
        if not 1.0 <= w / h <= (6.0 if partial else 3.5):
            continue
        sides = sorted(_side_coverage(ink, x, y, w, h), reverse=True)
        if sides[2] >= 0.75 or (partial and sides[1] >= 0.8):   # >= 3 sides (or 2, if cut off)
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _best_group(frames, min_count=6, w_tol=0.15, h_tol=0.2):
    """Largest set of frames whose width AND height agree within tolerance."""
    best = []
    for (_, _, w0, h0) in frames:
        grp = [f for f in frames if abs(f[2] - w0) <= w_tol * w0 and abs(f[3] - h0) <= h_tol * h0]
        if len(grp) > len(best):
            best = grp
    # a legend grid has several rows or columns; a single strip of frames is not one
    if len(best) >= min_count:
        xs = sorted({round(f[0] / max(f[2], 1)) for f in best})
        ys = sorted({round(f[1] / max(f[3], 1)) for f in best})
        if len(xs) >= 2 and len(ys) >= 2:
            return best
    return []


def _ray(line_mask, axis_lo, axis_hi, start, step, limit, cover=0.6):
    """Walk from `start` in direction `step` until a ruled line covering >= cover of the span."""
    pos = start
    while 0 <= pos < limit:
        seg = line_mask[pos, axis_lo:axis_hi] if step in (-1, 1) and line_mask.ndim == 2 else None
        if seg is not None and seg.size and (seg > 0).mean() >= cover:
            return pos
        pos += step
    return None


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------
def find_legend(img, search_frac=0.35, work_width=2000, return_debug=False):
    """Locate the legend. Returns dict(crop, bbox(x,y,w,h in original px), confidence, ...) or None."""
    H0, W0 = img.shape[:2]
    s = min(1.0, work_width / W0)                      # work on a ~1500 px wide copy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    H, W = small.shape
    ink = (small < 200).astype(np.uint8) * 255
    top = int(H * search_frac)
    ink_top = ink[:top]

    # Dilation closes speckled borders but fuses tightly packed swatches with their neighbours and
    # labels, so try it first and fall back to none.
    for k in (3, 1):
        frames = _frames(ink_top, W, k=k)
        grp = _best_group(frames)
        if grp:
            break
    debug = dict(scale=s, frames=frames, group=grp, top=top, small=small, k=k)
    if not grp:
        return (None, debug) if return_debug else None

    gx0 = min(f[0] for f in grp); gy0 = min(f[1] for f in grp)
    gx1 = max(f[0] + f[2] for f in grp); gy1 = max(f[1] + f[3] for f in grp)
    sw = float(np.median([f[2] for f in grp])); sh = float(np.median([f[3] for f in grp]))

    # ---- later rows of the same grid (the seed group may only be the top rows) ----
    band = ink.copy()
    band[:, :max(int(gx0 - 0.5 * sw), 0)] = 0
    band[:, int(gx1 + 0.5 * sw):] = 0
    last = gy1
    for f in sorted(_frames(band, W, partial=True, k=k), key=lambda f: f[1]):
        if f[1] < gy1 - 0.5 * sh or f[1] > last + 3.5 * sh:
            continue
        if abs(f[2] - sw) <= 0.15 * sw and 0.25 * sh <= f[3] <= 1.25 * sh:
            last = max(last, f[1] + f[3])
    gy1 = int(last)
    debug["grid_bottom"] = gy1

    # ---- grow to the ruled cell border (long lines, much longer than a swatch) ----
    span_w, span_h = gx1 - gx0, gy1 - gy0
    k_h = max(int(0.6 * span_w), 15); k_v = max(int(0.6 * span_h), 15)
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v)))

    def cast(mask, lo, hi, start, step, limit, along_rows):
        pos = start
        while 0 <= pos < limit:
            seg = mask[pos, lo:hi] if along_rows else mask[lo:hi, pos]
            if seg.size and (seg > 0).mean() >= 0.6:
                return pos
            pos += step
        return None

    # start just outside the swatches so the swatch frames' own edges are never hit
    m = 2   # start right outside the swatch grid; cell borders can be only a few px away
    up = cast(hl, gx0, gx1, gy0 - m, -1, H, True)
    dn = cast(hl, gx0, gx1, gy1 + m, +1, H, True)
    lf = cast(vl, gy0, gy1, gx0 - m, -1, W, False)
    rt = cast(vl, gy0, gy1, gx1 + m, +1, W, False)
    # ignore borders absurdly far away (a page-wide rule is not this legend's cell)
    far_x, far_y = 6 * sw, 8 * sh
    ruled = {
        "top": up is not None and gy0 - up <= far_y,
        "bottom": dn is not None and dn - gy1 <= far_y,
        "left": lf is not None and gx0 - lf <= far_x,
        "right": rt is not None and rt - gx1 <= far_x,
    }
    pad = 3   # step inside the border line
    x0 = lf + pad if ruled["left"] else gx0 - 0.6 * sw
    x1 = rt - pad if ruled["right"] else gx1 + 0.6 * sw
    y0 = up + pad if ruled["top"] else gy0 - 1.8 * sh            # room for the heading
    y1 = dn - pad if ruled["bottom"] else gy1 + 1.6 * sh         # room for wrapped labels
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), W), min(int(y1), H)

    # ---- back to original pixels ----
    bx0, by0, bx1, by1 = (int(round(v / s)) for v in (x0, y0, x1, y1))
    bx0, by0, bx1, by1 = max(bx0, 0), max(by0, 0), min(bx1, W0), min(by1, H0)
    crop = img[by0:by1, bx0:bx1]

    if (x1 - x0) * (y1 - y0) > 0.5 * W * H:           # that is not a legend block, that is the page
        return (None, debug) if return_debug else None
    conf = min(1.0, len(grp) / 12.0) * (0.6 + 0.1 * sum(ruled.values()))
    res = dict(crop=crop, bbox=(bx0, by0, bx1 - bx0, by1 - by0), confidence=float(conf),
               n_swatches=len(grp), ruled=ruled)
    debug.update(cell=(x0, y0, x1, y1), swatch_box=(gx0, gy0, gx1, gy1))
    return (res, debug) if return_debug else res


def draw_debug(debug, res):
    """Overlay on the working-size page: grey = top search band, blue = frames, red = swatch grid, green = result."""
    vis = cv2.cvtColor(debug["small"], cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, debug["top"]), (vis.shape[1], debug["top"]), (160, 160, 160), 1)
    for x, y, w, h in debug["frames"]:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 120, 0), 1)
    for x, y, w, h in debug["group"]:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    if "cell" in debug:
        x0, y0, x1, y1 = debug["cell"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 0), 2)
    return vis[:int(debug["top"] * 1.15)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image"); ap.add_argument("out_png")
    ap.add_argument("--search-frac", type=float, default=0.35,
                    help="only search this top fraction of the page (default 0.35)")
    a = ap.parse_args()
    img = load_image(a.image)
    res, dbg = find_legend(img, a.search_frac, return_debug=True)
    out = Path(a.out_png); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_name(out.stem + "_finder_debug.png")), draw_debug(dbg, res))
    if res is None:
        sys.exit("no legend-like block found (see *_finder_debug.png)")
    cv2.imwrite(str(out), res["crop"])
    x, y, w, h = res["bbox"]
    print(f"legend at x={x} y={y} w={w} h={h}  swatches={res['n_swatches']}  "
          f"ruled sides={[k for k, v in res['ruled'].items() if v]}  confidence={res['confidence']:.2f}")


if __name__ == "__main__":
    main()